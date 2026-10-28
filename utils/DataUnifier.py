"""
Schema Unification Agent.

Job: turn the uploaded CSVs into ONE analysable table.

How it works
- The LLM reads a profile of every file (columns, types, sample values, row
  counts, which shared columns are unique keys) and plans the joins: which
  file is the fact table (the finest-grained table holding the money) and how
  every other file attaches to it.
- The plan is validated and executed deterministically with pandas. Every
  join is forced to be many-to-one: if the right-hand table repeats its key
  (e.g. several payments per order) it is aggregated to one row per key first,
  so the fact table never fans out and money is never double counted.
- Columns that come from a coarser table (one value per order repeated on
  every item row) are recorded in `column_grain`, so later agents can
  de-duplicate them before summing.
"""

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

SCORE_HINTS = ("score", "rating", "stars", "review")
TEXT_DROP_AVG_LEN = 60          # free-text columns (comments) are dropped
MAX_NULL_SHARE = 0.98           # near-empty columns are dropped


def _strip_fences(text: str) -> str:
    text = text.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


class SchemaUnificationPipeline:

    def __init__(self, api_key: str, model: str = "openai/gpt-oss-120b"):
        self.llm = ChatGroq(api_key=api_key, model=model, temperature=0, max_retries=6)

    # ── Profiling ──────────────────────────────────────────────────────────
    @staticmethod
    def _profile(name: str, df: pd.DataFrame, shared: List[str]) -> Dict[str, Any]:
        return {
            "rows": len(df),
            "columns": {c: str(t) for c, t in df.dtypes.items()},
            "sample": df.head(2).to_dict(orient="records"),
            "shared_columns": {c: {"unique": bool(df[c].is_unique), "distinct": int(df[c].nunique())} for c in shared},
        }

    @staticmethod
    def _shared_columns(dfs: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        names = list(dfs)
        out = {n: set() for n in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                common = set(dfs[a].columns) & set(dfs[b].columns)
                out[a] |= common
                out[b] |= common
        return {n: sorted(c) for n, c in out.items()}

    # ── Planning (LLM) ─────────────────────────────────────────────────────
    def _plan_with_llm(self, profiles: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt = f"""
You are a data engineer joining ecommerce exports into one table for cost analysis.

Files (name → profile):
{json.dumps(profiles, indent=2, default=str)[:12000]}

Choose:
1. "fact_table": the file at the FINEST grain that holds the transaction money
   (e.g. one row per order line with price and freight). Not a lookup table.
2. "joins": how every other useful file attaches, in order. Each join uses a
   column that already exists in the table built so far ("key") and a column
   in the joined file ("right_key"). Skip files that cannot be joined.

Return ONLY JSON:
{{"fact_table": "<file name>", "joins": [{{"table": "<file name>", "key": "<col>", "right_key": "<col>"}}], "reason": "<one sentence>"}}
"""
        try:
            resp = self.llm.invoke([
                SystemMessage(content="Return only valid JSON. No markdown."),
                HumanMessage(content=prompt),
            ])
            return json.loads(_strip_fences(resp.content))
        except Exception as e:
            print(f"⚠️ Unifier: LLM plan failed ({e}); using deterministic plan.")
            return None

    @staticmethod
    def _fallback_plan(dfs: Dict[str, pd.DataFrame], shared: Dict[str, List[str]]) -> Dict[str, Any]:
        """Fact table = largest file with numeric money-like columns; join others greedily on shared columns."""
        def money_cols(df):
            return [c for c in df.select_dtypes("number").columns if any(k in c.lower() for k in ("price", "value", "amount", "revenue", "sales", "freight", "cost"))]
        ranked = sorted(dfs, key=lambda n: (len(money_cols(dfs[n])) > 0, len(dfs[n])), reverse=True)
        fact = ranked[0]
        built_cols, joins, pending = set(dfs[fact].columns), [], [n for n in dfs if n != fact]
        progress = True
        while pending and progress:
            progress = False
            for n in list(pending):
                common = [c for c in shared[n] if c in built_cols and c in dfs[n].columns]
                if common:
                    key = next((c for c in common if dfs[n][c].is_unique), common[0])
                    joins.append({"table": n, "key": key, "right_key": key})
                    built_cols |= set(dfs[n].columns)
                    pending.remove(n)
                    progress = True
        return {"fact_table": fact, "joins": joins, "reason": "deterministic fallback"}

    def _validate_plan(self, plan: Optional[Dict], dfs: Dict[str, pd.DataFrame], shared) -> Dict[str, Any]:
        if not plan or plan.get("fact_table") not in dfs:
            return self._fallback_plan(dfs, shared)
        built = set(dfs[plan["fact_table"]].columns)
        joins, used = [], {plan["fact_table"]}
        for j in plan.get("joins", []):
            t, k, rk = j.get("table"), j.get("key"), j.get("right_key") or j.get("key")
            if t in dfs and t not in used and k in built and rk in dfs[t].columns:
                joins.append({"table": t, "key": k, "right_key": rk})
                built |= set(dfs[t].columns)
                used.add(t)
        # Any joinable file the LLM forgot is attached deterministically.
        fb = self._fallback_plan({n: dfs[n] for n in dfs}, shared)
        for j in fb["joins"]:
            if j["table"] not in used and j["key"] in built:
                joins.append(j)
                built |= set(dfs[j["table"]].columns)
                used.add(j["table"])
        return {"fact_table": plan["fact_table"], "joins": joins, "reason": plan.get("reason", "")}

    # ── Execution (deterministic) ──────────────────────────────────────────
    @staticmethod
    def _to_one_row_per_key(df: pd.DataFrame, key: str) -> pd.DataFrame:
        """Aggregate a table so `key` is unique: money/counts are summed, scores averaged, text takes the first value."""
        agg = {}
        for c in df.columns:
            if c == key:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                agg[c] = "mean" if any(h in c.lower() for h in SCORE_HINTS) else "sum"
            else:
                agg[c] = "first"
        return df.groupby(key, as_index=False).agg(agg)

    def unify(self, file_paths: List[str], output_dir: Optional[str] = None) -> Dict[str, Any]:
        try:
            dfs = {os.path.basename(p): pd.read_csv(p, low_memory=False) for p in file_paths}
        except Exception as e:
            return {"status": "error", "output_file": None, "error": f"Could not read CSV: {e}"}
        output_dir = output_dir or os.path.dirname(file_paths[0]) or "."
        out_path = os.path.join(output_dir, "unified_dataset.csv")

        shared = self._shared_columns(dfs)
        if len(dfs) == 1:
            plan = {"fact_table": next(iter(dfs)), "joins": [], "reason": "single file"}
        else:
            profiles = {n: self._profile(n, df, shared[n]) for n, df in dfs.items()}
            plan = self._validate_plan(self._plan_with_llm(profiles), dfs, shared)

        fact = plan["fact_table"]
        base = dfs[fact].copy()
        n_base = len(base)
        column_grain: Dict[str, Optional[str]] = {c: None for c in base.columns}
        joins_done = []
        print(f"\n📋 Unifier: fact table = {fact} ({n_base:,} rows)")

        for j in plan["joins"]:
            right = dfs[j["table"]]
            key, rkey = j["key"], j["right_key"]
            aggregated = not right[rkey].is_unique
            if aggregated:
                right = self._to_one_row_per_key(right, rkey)
            right = right.rename(columns={rkey: key}) if rkey != key else right
            new_cols = [c for c in right.columns if c != key and c not in base.columns]
            if not new_cols:
                continue
            before_keys = base[key].notna().sum()
            base = base.merge(right[[key] + new_cols], on=key, how="left", validate="many_to_one")
            matched = base[new_cols[0]].notna().sum()
            # Columns from this table repeat across fact rows that share `key` → record their grain.
            grain = None if base[key].is_unique else key
            for c in new_cols:
                column_grain[c] = grain
            joins_done.append({"table": j["table"], "key": key, "aggregated": aggregated,
                               "match_rate": round(100 * float(matched) / max(int(before_keys), 1), 1)})
            print(f"   + {j['table']} on {key}{' (aggregated to one row per key)' if aggregated else ''} → {joins_done[-1]['match_rate']}% matched")

        assert len(base) == n_base, "join fan-out detected"   # many_to_one guarantees this

        # Drop free text and near-empty columns; they add memory, not signal.
        dropped = []
        for c in list(base.columns):
            s = base[c]
            if s.isna().mean() > MAX_NULL_SHARE:
                dropped.append(c)
            elif s.dtype == object and s.dropna().astype(str).str.len().mean() > TEXT_DROP_AVG_LEN:
                dropped.append(c)
        base = base.drop(columns=dropped)
        for c in dropped:
            column_grain.pop(c, None)

        os.makedirs(output_dir, exist_ok=True)
        base.to_csv(out_path, index=False)
        print(f"🎯 Unified dataset: {len(base):,} rows × {len(base.columns)} columns → {out_path}")

        return {
            "status": "success",
            "output_file": out_path,
            "dataframe": base,
            "rows": len(base),
            "columns": list(base.columns),
            "fact_table": fact,
            "joins": joins_done,
            "column_grain": column_grain,
            "dropped_columns": dropped,
            "plan_reason": plan.get("reason", ""),
        }
