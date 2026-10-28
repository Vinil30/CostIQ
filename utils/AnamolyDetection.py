"""
Anomaly Detection Agent (+ root-cause explanation).

Job: find the specific entities (in whatever dimensions the business has —
products, stores, suppliers, regions, campaigns…) and days that are costing
money, and put an honest figure on each.

How it works — all detection is deterministic statistics, no LLM:
- For every dimension the Business Analyst identified, each entity's
  cost share of revenue, average satisfaction score, late-fulfilment rate and
  cancel/return rate is compared with its peers using a robust z-score
  (median / MAD scaled so z ≈ σ). An entity is flagged at |z| ≥ 2 AND a
  practical gap from typical, and only with enough transactions to matter
  (the minimum adapts to how many transactions a typical entity has).
- Loss is computed from the rows themselves:
    excess cost      = cost paid above the typical cost share of revenue
    revenue at risk  = revenue from bad transactions (low score, late or
                       cancelled/returned) above the business's normal rate
  The headline total counts every row once, even if several of its
  dimensions are flagged.
- Days are flagged when revenue is ≥ 3σ (robust) away from the previous 28 days.
- The LLM only explains the entities it is given (root cause); it cannot add,
  remove or re-score anomalies.
"""

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from utils.DatasetAnalyser import BusinessAnalystAgent

Z_ENTITY = 2.0
Z_DAY = 3.0
MIN_PEERS = 8
MATERIALITY = 0.0002          # an entity must cost ≥ 0.02 % of total revenue to be reported
SIGNALS = {
    #  key        column         direction  practical gap   label
    "cost":     ("cost_share",     +1,        5.0),
    "quality":  ("avg_quality",    -1,        None),        # gap = 10 % of the score range
    "late":     ("late_rate",      +1,        10.0),
    "negative": ("negative_rate",  +1,        5.0),
}


def _robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    if not mad or not np.isfinite(mad):
        sd = x.std()
        mad = sd if sd and np.isfinite(sd) else np.nan
    return (x - med) / mad


def _wilson_lower(p: pd.Series, n: pd.Series, z: float = 1.96) -> pd.Series:
    """95 % lower confidence bound of a rate observed on n transactions (small samples get a wide interval)."""
    n = n.replace(0, np.nan)
    centre = p + z * z / (2 * n)
    spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - spread) / (1 + z * z / n)


def _sigma(z: float) -> str:
    return f"{'+' if z > 0 else '−'}{min(abs(z), 20):.1f}σ" if abs(z) <= 20 else f"{'>+' if z > 0 else '<−'}20σ"


def _r(x, n=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, n) if np.isfinite(x) else None


class AnomalyDetectionAgent:
    def __init__(self, api_key: Optional[str] = None, model: str = "openai/gpt-oss-120b", verbose: bool = False):
        self.verbose = verbose
        self.llm = None
        if api_key:
            from langchain_groq import ChatGroq
            self.llm = ChatGroq(api_key=api_key, model=model, temperature=0, max_retries=6)

    # ── One row per fact line: money + transaction-level signals ───────────
    @staticmethod
    def _lines(df: pd.DataFrame, s: Dict, grain: Dict) -> pd.DataFrame:
        oid, rev = s.get("order_id"), s["revenue"]
        L = pd.DataFrame(index=df.index)
        L["rev"] = pd.to_numeric(df[rev], errors="coerce").fillna(0)
        if oid and grain.get(rev) == oid:                      # revenue only exists per order: count it once
            L["rev"] = L["rev"].where(~df.duplicated(oid), 0)
        cost = pd.Series(0.0, index=df.index)
        for c in s.get("cost_columns") or []:
            v = pd.to_numeric(df[c], errors="coerce").fillna(0)
            if oid and grain.get(c) == oid:                    # order-level cost: allocate to lines by revenue
                order_rev = L.groupby(df[oid])["rev"].transform("sum").replace(0, np.nan)
                v = (v * L["rev"] / order_rev).fillna(0)
            cost += v
        L["cost"] = cost
        L["has_cost"] = bool(s.get("cost_columns"))
        for d in s.get("dimensions") or []:
            L[d] = df[d].astype(str)
        orders = BusinessAnalystAgent.order_frame(df, s, grain)
        sig = [c for c in ("_low_review", "_late", "_negative") if c in orders]
        rename = {"_late": "late", "_negative": "negative", "_low_review": "low_review"}
        if s.get("review"):
            sig.append(s["review"]); rename[s["review"]] = "quality"
        if oid:
            L["oid"] = df[oid].values
            L = L.merge(orders[[oid] + sig].rename(columns={oid: "oid", **rename}), on="oid", how="left")
        else:
            L["oid"] = np.arange(len(L))
            for c in sig:
                L[rename[c]] = orders[c].values
        bad_parts = [L[c] for c in ("low_review", "late", "negative") if c in L]
        L["bad"] = pd.concat(bad_parts, axis=1).max(axis=1) if bad_parts else np.nan
        return L

    # ── Entity-level detection for one dimension ───────────────────────────
    def _dimension(self, L: pd.DataFrame, dim: str, label: str, bench_cost: float, base_bad: float, quality_range: float, min_loss: float = 0.0):
        g = L.groupby(dim)
        st = pd.DataFrame({"revenue": g["rev"].sum(), "orders": g["oid"].nunique()})
        if L["has_cost"].iloc[0]:
            st["cost"] = g["cost"].sum()
            st["cost_share"] = st["cost"] / st["revenue"].replace(0, np.nan) * 100
        per_order = L.drop_duplicates([dim, "oid"]).groupby(dim)
        for src, dst, scale in (("quality", "avg_quality", 1), ("late", "late_rate", 100), ("negative", "negative_rate", 100), ("low_review", "low_quality_share", 100)):
            if src in L:
                st[dst] = per_order[src].mean() * scale
                st[f"n_{src}"] = per_order[src].count()
        if "quality" in L:
            st["sd_quality"] = per_order["quality"].std()

        need = int(np.clip(st["orders"].median(), 10, 50))     # adaptive support threshold
        st = st[(st["orders"] >= need) & (st["revenue"] > 0)]
        if len(st) < MIN_PEERS:
            return [], None, None, {}

        flags, base = pd.DataFrame(index=st.index), {"entities_compared": int(len(st)), "min_transactions": need}
        for key, (col, direction, gap) in SIGNALS.items():
            if col not in st or st[col].notna().sum() < MIN_PEERS:
                continue
            z = _robust_z(st[col]); med = st[col].median()
            gap = gap if gap is not None else 0.1 * quality_range
            st[f"z_{key}"] = z; base[col] = med
            flag = ((z * direction) >= Z_ENTITY) & ((st[col] - med) * direction >= gap)
            # The entity's own sample must also be significantly worse than the business overall.
            if key in ("late", "negative"):
                overall = float(L.drop_duplicates("oid")[key].mean())
                flag &= _wilson_lower(st[col] / 100, st[f"n_{key}"]) > overall
            elif key == "cost":
                flag &= st[col] >= 1.5 * med                    # materially above typical, not just statistically
            elif key == "quality":
                o = L.drop_duplicates("oid")["quality"]
                t = (st[col] - o.mean()) / (o.std() / np.sqrt(st["n_quality"].replace(0, np.nan)))
                flag &= t <= -2.58
            flags[key] = flag
        if flags.empty:
            return [], None, None, base
        flags = flags.fillna(False)

        cost_ids = set(flags.index[flags["cost"]]) if "cost" in flags else set()
        bad_cols = [k for k in ("quality", "late", "negative") if k in flags]
        bad_ids = set(flags.index[flags[bad_cols].any(axis=1)]) if bad_cols else set()
        row_cost_flag, row_bad_flag = L[dim].isin(cost_ids), L[dim].isin(bad_ids)

        excess = (L["cost"] - L["rev"] * bench_cost).clip(lower=0)
        risk = L["rev"] * (L["bad"].fillna(base_bad) - base_bad) if np.isfinite(base_bad) else pd.Series(0.0, index=L.index)
        ent_excess = excess.where(row_cost_flag, 0).groupby(L[dim]).sum()
        ent_risk = risk.where(row_bad_flag, 0).groupby(L[dim]).sum().clip(lower=0)

        # Materiality: statistically unusual but cheap entities are not reported (or counted in the total).
        material = set((ent_excess.add(ent_risk, fill_value=0) >= min_loss).pipe(lambda x: x[x].index))
        flags = flags[flags.index.isin(material)]
        row_cost_flag &= L[dim].isin(material)
        row_bad_flag &= L[dim].isin(material)
        base["min_reported_loss"] = round(min_loss, 2)

        out = []
        for eid, f in flags.iterrows():
            if not f.any():
                continue
            row = st.loc[eid]
            reasons = []
            if f.get("cost"):
                reasons.append(f"costs {row['cost_share']:.1f}% of revenue vs {base['cost_share']:.1f}% typical ({_sigma(row['z_cost'])})")
            if f.get("quality"):
                reasons.append(f"avg score {row['avg_quality']:.2f} vs {base['avg_quality']:.2f} typical ({_sigma(row['z_quality'])})")
            if f.get("late"):
                reasons.append(f"{row['late_rate']:.0f}% fulfilled late vs {base['late_rate']:.0f}% typical ({_sigma(row['z_late'])})")
            if f.get("negative"):
                reasons.append(f"{row['negative_rate']:.1f}% cancelled/returned vs {base['negative_rate']:.1f}% typical ({_sigma(row['z_negative'])})")
            ex, rk = float(ent_excess.get(eid, 0)), float(ent_risk.get(eid, 0))
            out.append({
                "type": label, "dimension": dim, "id": str(eid), "reasons": reasons,
                "signals": [k for k in flags.columns if f.get(k)],
                "estimated_loss": _r(ex + rk),
                "loss_breakdown": {"excess_cost": _r(ex), "revenue_at_risk": _r(rk)},
                "total_revenue": _r(row["revenue"]), "order_count": int(row["orders"]),
                "metrics": {k: _r(row.get(k), 2) for k in ("cost_share", "avg_quality", "low_quality_share", "late_rate", "negative_rate") if k in row},
                "z_scores": {k: _r(row.get(f"z_{k}"), 1) for k in SIGNALS if f"z_{k}" in row},
            })
        out.sort(key=lambda a: a["estimated_loss"] or 0, reverse=True)
        return out, row_cost_flag, row_bad_flag, {k: (_r(v) if isinstance(v, float) else v) for k, v in base.items()}

    # ── Day-level detection ────────────────────────────────────────────────
    def _days(self, df: pd.DataFrame, s: Dict, grain: Dict) -> List[Dict]:
        if not s.get("date"):
            return []
        o = BusinessAnalystAgent.order_frame(df, s, grain).dropna(subset=["_date"])
        if o.empty:
            return []
        daily = o.groupby(o["_date"].dt.normalize())[s["revenue"]].sum()
        daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"), fill_value=0.0)
        if len(daily) < 42:
            return []
        hist = daily.shift(1).rolling(28, min_periods=21)
        med = hist.median()
        mad = hist.apply(lambda w: np.median(np.abs(w - np.median(w))), raw=True) * 1.4826
        z = (daily - med) / mad.replace(0, np.nan)
        active = (daily.shift(1) > 0).rolling(28, min_periods=21).sum()   # judge a day only against a busy history
        out = []
        for day, zz in z.iloc[28:-1].items():                   # skip the warm-up window and the possibly partial last day
            exp = med[day]
            if not np.isfinite(zz) or abs(zz) < Z_DAY or not exp or exp <= 0 or not (active[day] >= 21):
                continue
            act = float(daily[day])
            if abs(act - exp) < 0.5 * exp:
                continue
            item = {"type": "time", "date": str(day.date()), "issue": "revenue_spike" if zz > 0 else "revenue_drop",
                    "deviation": _r(zz, 1), "actual_revenue": _r(act), "expected_revenue": _r(exp)}
            item["revenue_gain" if zz > 0 else "revenue_loss"] = _r(abs(act - exp))
            out.append(item)
        return sorted(out, key=lambda a: abs(a["deviation"]), reverse=True)[:10]

    # ── Root cause (explains, never invents) ───────────────────────────────
    def _root_cause(self, ranked: List[Dict], days: List[Dict], baselines: Dict, kpis: Dict) -> str:
        if not ranked and not days:
            return "No entity or day crossed the 2σ threshold — performance is in line with peers."
        top = [{k: a[k] for k in ("type", "id", "reasons", "estimated_loss", "loss_breakdown", "order_count")} for a in ranked[:8]]
        fallback = (f"{len(ranked)} entities flagged; largest: {top[0]['type']} {top[0]['id']} — {'; '.join(top[0]['reasons'])}." if top
                    else f"{len(days)} unusual days found.")
        if not self.llm:
            return fallback
        prompt = f"""
Statistically flagged anomalies in a business's transaction data (robust z-scores vs peers):
{json.dumps(top, indent=1, default=str)}
Unusual days: {json.dumps(days[:5], default=str)}
Typical values per dimension: {json.dumps(baselines, default=str)}
Business KPIs: {json.dumps({k: kpis.get(k) for k in ('cost_share', 'cost_breakdown', 'avg_quality', 'late_delivery_rate', 'cancel_rate', 'aov')}, default=str)}

Explain the most likely root causes in 3–4 markdown bullets. Connect related entities across
dimensions when their problems overlap. Reference only the entities and numbers above.
"""
        try:
            resp = self.llm.invoke([SystemMessage(content="You are a precise root-cause analyst."), HumanMessage(content=prompt)])
            return resp.content.strip() or fallback
        except Exception as e:
            print(f"⚠️ Root cause LLM failed: {e}")
            return fallback

    # ── Entry point ────────────────────────────────────────────────────────
    def analyze(self, df: pd.DataFrame, schema: Dict, column_grain: Optional[Dict] = None, kpis: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            if df is None or df.empty:
                return {"status": "error", "error": "No data provided."}
            s, grain = schema or {}, column_grain or {}
            if not s.get("revenue"):
                return {"status": "error", "error": "No revenue column — anomaly detection needs money to measure."}
            L = self._lines(df, s, grain)
            dims = [d for d in (s.get("dimensions") or []) if d in L]
            labels = s.get("dimension_labels") or {}

            # Typical cost share: median over entities of the most granular dimension (or overall share).
            bench_cost = 0.0
            if s.get("cost_columns") and L["rev"].sum() > 0:
                bench_cost = float(L["cost"].sum() / L["rev"].sum())
                if dims:
                    e = L.groupby(dims[0]).agg(r=("rev", "sum"), c=("cost", "sum"))
                    e = e[e["r"] > 0]
                    if len(e) >= MIN_PEERS:
                        bench_cost = float((e["c"] / e["r"]).median())
            order_bad = L.drop_duplicates("oid")["bad"]
            base_bad = float(order_bad.mean()) if order_bad.notna().any() else float("nan")
            q = L["quality"] if "quality" in L else pd.Series(dtype=float)
            q_range = float(q.max() - q.min()) if q.notna().any() else 4.0

            by_dim, cost_rows, bad_rows = {}, [], []
            baselines = {"typical_cost_share": _r(bench_cost * 100, 1), "bad_transaction_rate": _r(base_bad * 100, 1)}
            for d in dims:
                label = labels.get(d, d)
                found, rc, rb, base = self._dimension(L, d, label, bench_cost, base_bad, q_range, MATERIALITY * float(L["rev"].sum()))
                by_dim[label] = found
                if base:
                    baselines[label] = base
                if rc is not None:
                    cost_rows.append(rc); bad_rows.append(rb)

            total_excess = total_risk = 0.0
            if cost_rows:
                any_c = pd.concat(cost_rows, axis=1).any(axis=1)
                any_b = pd.concat(bad_rows, axis=1).any(axis=1)
                total_excess = float((L["cost"] - L["rev"] * bench_cost).clip(lower=0).where(any_c, 0).sum())
                if np.isfinite(base_bad):
                    total_risk = max(0.0, float((L["rev"] * (L["bad"].fillna(base_bad) - base_bad)).where(any_b, 0).sum()))

            days = self._days(df, s, grain)
            entities = sorted([a for v in by_dim.values() for a in v], key=lambda a: a["estimated_loss"] or 0, reverse=True)
            print(f"🔎 Anomalies: " + ", ".join(f"{len(v)} {k}" for k, v in by_dim.items()) + f", {len(days)} days")
            return {
                "status": "success",
                "method": f"robust z-score (median/MAD) ≥ {Z_ENTITY}σ vs peers with practical-gap and minimum-transaction guards; days ≥ {Z_DAY}σ vs the previous 28 days",
                "detected_schema": s,
                "anomalies": {"entities": entities[:40], "by_dimension": {k: v[:20] for k, v in by_dim.items()}, "time": days},
                "summary": {
                    "total_anomaly_loss": _r(total_excess + total_risk),
                    "excess_cost_total": _r(total_excess),
                    "revenue_at_risk_total": _r(total_risk),
                    "flagged_entities": len(entities),
                    "flagged_by_dimension": {k: len(v) for k, v in by_dim.items()},
                    "time_anomalies_count": len(days),
                    "dimensions_tested": [labels.get(d, d) for d in dims],
                    "baselines": baselines,
                },
                "root_cause_insights": self._root_cause(entities, days, baselines, kpis or {}),
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}
