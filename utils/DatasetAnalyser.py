"""
Business Analyst Agent.

Job: work out what the business looks like — the KPIs and the trend — for
ANY transactional dataset, whatever its column names.

How it works
- The LLM maps the dataset onto generic ROLES using column names, dtypes and
  sample values: the transaction id, the revenue column, any number of COST
  columns (shipping, ad spend, fees, discounts, refunds…), dates, status,
  a quality score, and up to four business DIMENSIONS to break results down
  by (product, store, region, supplier, channel, campaign…).
- Every role is validated against the data (money must be numeric, dates must
  parse, dimensions must have a sensible number of distinct values). Name
  patterns are only a fallback hint when the LLM's choice fails validation.
- Every KPI is computed deterministically with pandas at the right grain:
  order-level values that repeat on each line are de-duplicated first.
- Profit / margin are only reported when a real cost-of-goods column exists.
- The LLM then writes three insights from the computed KPIs only.
"""

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

ROLES = {
    "order_id":       "id of a transaction / order / invoice (repeats when an order has several lines)",
    "revenue":        "money earned per row (line price, sale amount). Must be money, never an id or a count",
    "payment_value":  "amount the customer paid, if different from revenue",
    "cogs":           "cost of goods / unit purchase cost. null unless the data really has it",
    "date":           "when the transaction happened",
    "delivered_date": "when it was actually delivered / fulfilled",
    "estimated_date": "when it was promised to be delivered",
    "status":         "transaction status (completed, cancelled, returned…)",
    "review":         "customer satisfaction score / rating",
    "customer":       "customer id",
}
# Fallback hints only — used when the LLM's choice fails validation.
HINTS = {
    "order_id": ["order_id", "invoice", "transaction_id", "order_no", "order_number", "receipt"],
    "revenue": ["revenue", "sales", "price", "amount", "subtotal", "net_sales", "total"],
    "payment_value": ["payment", "paid"],
    "cogs": ["cogs", "unit_cost", "cost_price", "purchase_cost", "product_cost"],
    "date": ["purchase", "order_date", "created", "transaction_date", "date", "timestamp", "time"],
    "delivered_date": ["delivered", "fulfilled", "shipped_at"],
    "estimated_date": ["estimated", "promised", "expected", "eta"],
    "status": ["status", "state"],
    "review": ["review", "rating", "score", "stars", "csat", "nps"],
    "customer": ["customer_unique", "customer", "client", "buyer"],
}
COST_HINTS = ["cost", "freight", "shipping", "delivery_fee", "fee", "commission", "discount", "refund", "ad_spend", "spend", "return"]
DIM_HINTS = ["product", "sku", "item", "seller", "vendor", "supplier", "store", "shop", "branch", "region", "state", "city",
             "category", "brand", "channel", "campaign", "warehouse", "carrier", "segment"]
NEGATIVE_WORDS = ("cancel", "unavailable", "refund", "return", "fail", "declin", "void")
NUMERIC_ROLES = {"revenue", "payment_value", "cogs", "review"}
DATE_ROLES = {"date", "delivered_date", "estimated_date"}


def _strip_fences(text: str) -> str:
    text = text.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _r(x, n=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return round(x, n) if np.isfinite(x) else None


def _tokens(col: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9]+", col.lower()) if t]


def to_dates(series: pd.Series) -> pd.Series:
    """Parse dates without assuming a locale.
    ISO / unambiguous formats parse directly. For a/b/yyyy style dates the column's own
    values decide the order: a first part above 12 means day-first, a second part above 12
    means month-first (one column always uses one convention)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    s = series.astype(str).where(series.notna())
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parts = s.dropna().head(5000).str.extract(r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-]\d{2,4}")
        if parts.notna().all(axis=1).mean() > 0.8:
            first, second = parts[0].astype(float), parts[1].astype(float)
            dayfirst = (first > 12).any() and not (second > 12).any()
            return pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, format="%d/%m/%Y" if dayfirst and s.dropna().str.match(r"^\d{1,2}/\d{1,2}/\d{4}$").all() else None)
        return pd.to_datetime(s, errors="coerce", format="mixed" if pd.to_datetime(s.head(500), errors="coerce").isna().mean() > 0.2 else None)


def dimension_label(col: str) -> str:
    """Human label for a dimension column: 'product_category_name' → 'product category'."""
    toks = [t for t in _tokens(col) if t not in ("id", "name", "code", "key", "unique", "english")]
    return " ".join(toks) or col


class BusinessAnalystAgent:
    _schema_cache: Dict[str, Dict] = {}

    def __init__(self, api_key: str, model: str = "openai/gpt-oss-120b", verbose: bool = False, use_cache: bool = True):
        self.verbose = verbose
        self.use_cache = use_cache
        self.llm = ChatGroq(api_key=api_key, model=model, temperature=0, max_retries=6)

    # ── Validation of role choices ─────────────────────────────────────────
    @staticmethod
    def _is_money(df: pd.DataFrame, col: str) -> bool:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            return False
        if "id" in _tokens(col):
            return False
        s = df[col].dropna()
        return len(s) > 0 and s.nunique() > 1

    def _valid(self, df: pd.DataFrame, role: str, col) -> bool:
        if not isinstance(col, str) or col not in df.columns or df[col].dropna().empty:
            return False
        s = df[col].dropna()
        if role in ("revenue", "payment_value", "cogs"):
            return self._is_money(df, col)
        if role == "review":
            return pd.api.types.is_numeric_dtype(s) and s.between(-10, 10).mean() > 0.95 and s.nunique() <= 21
        if role in DATE_ROLES:
            return to_dates(s.head(300)).notna().mean() > 0.8
        if role == "order_id":
            return s.nunique() > 1
        return True

    def _valid_dim(self, df: pd.DataFrame, col, taken: set) -> bool:
        if not isinstance(col, str) or col not in df.columns or col in taken:
            return False
        n, k = len(df), df[col].nunique()
        if k < 3 or k > max(0.6 * n, 3):
            return False
        if pd.api.types.is_float_dtype(df[col]) or self._valid(df, "date", col):
            return False
        return True

    def _hint(self, df: pd.DataFrame, role: str, taken: set) -> Optional[str]:
        for h in HINTS[role]:
            for c in df.columns:
                if c not in taken and h in c.lower() and self._valid(df, role, c):
                    return c
        return None

    # ── Role mapping (LLM + validation) ────────────────────────────────────
    def detect_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        key = hashlib.md5(",".join(sorted(df.columns)).encode()).hexdigest()
        if self.use_cache and key in self._schema_cache:
            return json.loads(json.dumps(self._schema_cache[key]))

        profile = {c: {"dtype": str(df[c].dtype), "distinct": int(df[c].nunique()), "sample": [str(v)[:40] for v in df[c].dropna().head(3)]}
                   for c in df.columns[:100]}
        prompt = f"""
You are mapping an unknown business dataset ({len(df):,} rows) onto generic roles.
Columns (dtype, distinct values, samples): {json.dumps(profile, default=str)}

Roles (use null when nothing fits — never guess): {json.dumps(ROLES, indent=1)}
Also return:
- "cost_columns": every numeric column that is money the business PAYS per row
  (shipping, freight, ad spend, fees, commissions, discounts, refunds). Not revenue, not cogs.
- "dimensions": up to 4 columns the business would manage separately and act on
  (e.g. product/SKU, seller/supplier, store, region, category, channel, campaign),
  most granular first. Not ids of individual transactions or customers, not dates.

Return ONLY JSON: {{"<role>": "<column or null>", ..., "cost_columns": [], "dimensions": []}}
"""
        llm = {}
        try:
            resp = self.llm.invoke([SystemMessage(content="Return only valid JSON. No markdown."), HumanMessage(content=prompt)])
            llm = json.loads(_strip_fences(resp.content))
        except Exception as e:
            print(f"⚠️ Analyst: LLM role mapping failed ({e}); using validated hints.")

        schema: Dict[str, Any] = {}
        taken = set()
        for role in ROLES:
            col = llm.get(role)
            schema[role] = col if (self._valid(df, role, col) and col not in taken) else self._hint(df, role, taken)
            if schema[role]:
                taken.add(schema[role])
        if not schema["revenue"] and schema["payment_value"]:
            schema["revenue"] = schema["payment_value"]
        if not schema["revenue"]:                                   # last resort: the largest money column
            money = [c for c in df.columns if c not in taken and self._is_money(df, c)]
            if money:
                schema["revenue"] = max(money, key=lambda c: pd.to_numeric(df[c], errors="coerce").abs().sum())
                taken.add(schema["revenue"])

        costs = [c for c in (llm.get("cost_columns") or []) if isinstance(c, str) and c not in taken and self._is_money(df, c)]
        if not costs:
            costs = [c for c in df.columns if c not in taken and self._is_money(df, c) and any(h in c.lower() for h in COST_HINTS)]
        schema["cost_columns"] = costs[:6]
        taken |= set(costs)

        dims = []
        excluded = taken | {schema.get("order_id"), schema.get("customer")}
        for c in (llm.get("dimensions") or []):
            if self._valid_dim(df, c, excluded | set(dims)):
                dims.append(c)
        if not dims:
            for h in DIM_HINTS:
                for c in df.columns:
                    if h in c.lower() and self._valid_dim(df, c, excluded | set(dims)):
                        dims.append(c)
        schema["dimensions"] = dims[:4]
        schema["dimension_labels"] = {c: dimension_label(c) for c in schema["dimensions"]}
        if self.use_cache:
            self._schema_cache[key] = json.loads(json.dumps(schema))
        return schema

    # ── Grain-aware helpers (shared with the anomaly agent) ────────────────
    @staticmethod
    def native(df: pd.DataFrame, col: str, grain: Dict) -> pd.Series:
        """`col` at its native grain: values repeated on every line of an order are taken once."""
        g = (grain or {}).get(col)
        return df.drop_duplicates(g)[col] if g and g in df.columns else df[col]

    @staticmethod
    def order_frame(df: pd.DataFrame, s: Dict, grain: Optional[Dict] = None) -> pd.DataFrame:
        """One row per transaction with revenue, cost and the satisfaction / fulfilment signals."""
        grain = grain or {}
        oid = s.get("order_id")
        money = [c for c in [s.get("revenue")] + list(s.get("cost_columns") or []) if c]
        firsts = [s[k] for k in ("date", "delivered_date", "estimated_date", "status", "review") if s.get(k)]
        if oid:
            agg = {c: "first" for c in firsts}
            for c in money:
                agg[c] = "first" if grain.get(c) == oid else "sum"
            o = df.groupby(oid, as_index=False).agg(agg) if agg else df[[oid]].drop_duplicates()
        else:
            o = df[firsts + money].copy()
        if s.get("date"):
            o["_date"] = to_dates(o[s["date"]])
        if s.get("delivered_date") and s.get("estimated_date"):
            d = to_dates(o[s["delivered_date"]])
            e = to_dates(o[s["estimated_date"]])
            o["_late"] = np.where(d.notna() & e.notna(), (d.dt.normalize() > e.dt.normalize()).astype(float), np.nan)
            if s.get("date"):
                o["_fulfil_days"] = (d - o["_date"]).dt.total_seconds() / 86400
        if s.get("status"):
            o["_negative"] = o[s["status"]].astype(str).str.lower().str.contains("|".join(NEGATIVE_WORDS)).astype(float)
        if s.get("review"):
            r = o[s["review"]]
            lo, hi = r.min(), r.max()
            cut = lo + (hi - lo) * 0.25                               # bottom quarter of the scale (≤ 2 on a 1–5 scale)
            o["_low_review"] = np.where(r.notna(), (r <= cut).astype(float), np.nan)
        return o

    # ── KPIs ───────────────────────────────────────────────────────────────
    def _metrics(self, df: pd.DataFrame, s: Dict, grain: Dict) -> Dict[str, Any]:
        m: Dict[str, Any] = {"row_count": len(df), "column_count": len(df.columns)}
        rev_col = s["revenue"]
        rev = pd.to_numeric(self.native(df, rev_col, grain), errors="coerce")
        o = self.order_frame(df, s, grain)
        orders = len(o) if s.get("order_id") else len(df)
        total_rev = float(rev.sum())
        m.update({"primary_metric_name": rev_col, "primary_metric_sum": _r(total_rev), "order_count": int(orders),
                  "aov": _r(total_rev / max(orders, 1)), "primary_metric_mean": _r(rev.mean()),
                  "primary_metric_median": _r(rev.median()), "primary_metric_max": _r(rev.max())})
        if s.get("order_id"):
            m["lines_per_order"] = _r(len(df) / max(orders, 1))

        breakdown, total_cost = {}, 0.0
        for c in s.get("cost_columns") or []:
            v = float(pd.to_numeric(self.native(df, c, grain), errors="coerce").sum())
            total_cost += v
            breakdown[c] = {"total": _r(v), "share_of_revenue": _r(v / total_rev * 100, 1) if total_rev else None}
        if breakdown:
            m["cost_breakdown"] = breakdown
            m["total_cost"] = _r(total_cost)
            m["cost_share"] = _r(total_cost / total_rev * 100, 1) if total_rev else None
            first = next(iter(breakdown))
            m["secondary_metric_name"], m["secondary_metric_sum"] = first, breakdown[first]["total"]
        if s.get("cogs") and total_rev:
            cogs = float(pd.to_numeric(self.native(df, s["cogs"], grain), errors="coerce").sum())
            m["profit"] = _r(total_rev - cogs - total_cost)
            m["profit_margin"] = _r((total_rev - cogs - total_cost) / total_rev * 100, 1)

        if "_low_review" in o:
            m["avg_quality"] = _r(o[s["review"]].mean())
            m["low_quality_percentage"] = _r(o["_low_review"].mean() * 100, 1)
        if "_late" in o:
            m["late_delivery_rate"] = _r(o["_late"].mean() * 100, 1)
        if "_fulfil_days" in o:
            m["avg_delivery_days"] = _r(o["_fulfil_days"].median(), 1)
        if "_negative" in o:
            m["cancel_rate"] = _r(o["_negative"].mean() * 100, 1)
        if s.get("customer"):
            m["unique_entities"] = int(df[s["customer"]].nunique())
        if "_date" in o and o["_date"].notna().any():
            m["date_start"] = o["_date"].min().strftime("%Y-%m-%d")
            m["date_end"] = o["_date"].max().strftime("%Y-%m-%d")
            m["date_range_days"] = int((o["_date"].max() - o["_date"].min()).days)
        return m

    def _trends(self, df: pd.DataFrame, s: Dict, grain: Dict) -> Dict[str, Any]:
        """Revenue month over month on COMPLETE months; partial first/last months are excluded."""
        if not s.get("date"):
            return {"trend": "insufficient_data", "message": "No date column."}
        o = self.order_frame(df, s, grain).dropna(subset=["_date"])
        if o.empty:
            return {"trend": "insufficient_data", "message": "No parseable dates."}
        rev = s["revenue"]
        o["_month"] = o["_date"].dt.to_period("M")
        monthly = o.groupby("_month").agg(revenue=(rev, "sum"), orders=("_date", "size"))
        monthly["active_days"] = o.groupby("_month")["_date"].apply(lambda x: x.dt.day.nunique())
        complete = monthly[monthly["active_days"] >= max(10, 0.6 * monthly["active_days"].median())]
        last = o["_date"].max()
        if len(complete) and complete.index[-1] == last.to_period("M") and last.day < last.days_in_month - 2:
            complete = complete.iloc[:-1]
        if len(complete) >= 2:
            a, b = complete.iloc[-2], complete.iloc[-1]
            pct = (b.revenue - a.revenue) / a.revenue * 100 if a.revenue else 0.0
            peak = complete["revenue"].idxmax()
            return {"trend": "increasing" if pct > 2 else "decreasing" if pct < -2 else "stable",
                    "trend_percentage": _r(pct), "basis": f"{complete.index[-1]} vs {complete.index[-2]} (complete months)",
                    "peak_date": str(peak), "peak_value": _r(complete.loc[peak, "revenue"]),
                    "avg_daily_value": _r(complete["revenue"].sum() / max(int(complete["active_days"].sum()), 1)),
                    "data_points": int(len(complete)),
                    "monthly": [{"month": str(i), "revenue": _r(r.revenue), "orders": int(r.orders)} for i, r in complete.tail(12).iterrows()]}
        daily = o.groupby(o["_date"].dt.date)[rev].sum()
        if len(daily) < 4:
            return {"trend": "insufficient_data", "message": f"Only {len(daily)} days of data."}
        half = len(daily) // 2
        a, b = daily.iloc[:half].mean(), daily.iloc[half:].mean()
        pct = (b - a) / a * 100 if a else 0.0
        return {"trend": "increasing" if pct > 2 else "decreasing" if pct < -2 else "stable", "trend_percentage": _r(pct),
                "basis": "second half vs first half of the period (average per day)", "peak_date": str(daily.idxmax()),
                "peak_value": _r(daily.max()), "data_points": int(len(daily))}

    def _insights(self, metrics: Dict, trends: Dict, schema: Dict) -> str:
        facts = {k: v for k, v in metrics.items() if v is not None and k != "column_count"}
        facts["trend"] = {k: trends.get(k) for k in ("trend", "trend_percentage", "basis", "peak_date") if trends.get(k) is not None}
        facts["dimensions_available"] = list((schema.get("dimension_labels") or {}).values())
        prompt = f"""
KPIs computed exactly from a business's transaction data (money in the data's currency):
{json.dumps(facts, indent=1, default=str)}

Write exactly 3 short markdown bullets: the key finding, the most important cost problem, and one concrete recommendation.
Use ONLY these numbers and column names; do not invent figures or assume an industry the data doesn't show.
"""
        try:
            resp = self.llm.invoke([SystemMessage(content="You are a concise business analyst."), HumanMessage(content=prompt)])
            return resp.content.strip()
        except Exception as e:
            print(f"⚠️ Analyst: insight generation failed ({e})")
            out = [f"- Revenue {metrics.get('primary_metric_sum', 0):,.0f} across {metrics.get('order_count', 0):,} transactions."]
            if metrics.get("cost_share") is not None:
                out.append(f"- Tracked costs are {metrics['cost_share']}% of revenue.")
            return "\n".join(out)

    # ── Entry point ────────────────────────────────────────────────────────
    def analyze(self, df: Union[pd.DataFrame, str], column_grain: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            if isinstance(df, str):
                df = pd.read_csv(df, low_memory=False)
            if not isinstance(df, pd.DataFrame) or df.empty:
                return {"status": "error", "error": "No data to analyse."}
            grain = column_grain or {}
            schema = self.detect_schema(df)
            print("🔍 Analyst roles:", {k: v for k, v in schema.items() if v and k != "dimension_labels"})
            if not schema.get("revenue"):
                return {"status": "warning", "message": "No revenue column could be identified.", "detected_schema": schema,
                        "metrics": {"row_count": len(df), "column_count": len(df.columns)}}
            metrics = self._metrics(df, schema, grain)
            trends = self._trends(df, schema, grain)
            return {"status": "success", "detected_schema": schema, "metrics": metrics, "trends": trends,
                    "insights": self._insights(metrics, trends, schema)}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}
