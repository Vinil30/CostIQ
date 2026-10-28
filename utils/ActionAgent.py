"""
Action Agent.

Job: turn the flagged anomalies into concrete, executable actions.

How it works
- The LLM is given the actual flagged entities (ids, reasons, measured loss
  breakdown), the store KPIs, the previous recommendations and the last
  validation verdict, and proposes up to three actions.
- Every proposal is checked: the target must be one of the flagged entities,
  the action type must be one we support, and the expected impact is capped at
  the loss measured for that target (an action cannot promise to recover more
  money than the anomaly is costing). Invalid proposals are dropped; if
  nothing valid remains, a rule-based action is derived from the top anomaly.
"""

import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

# Suggested action types (the LLM may use another snake_case type when none fits the business).
ACTION_TYPES = {
    "reduce_cost":           "renegotiate rates, change provider/carrier, cut fees or packaging for the target",
    "review_partner":        "pause, audit or put on probation a supplier / seller / store / partner",
    "fix_quality":           "investigate and fix what customers rate poorly",
    "improve_fulfilment":    "fix late fulfilment (handling time, carrier, promised dates)",
    "reduce_cancellations":  "fix the cause of cancellations / returns (stock, description, sizing…)",
    "adjust_pricing":        "reprice so costs are covered",
    "stop_spend":            "stop spend (ads, promotions) that drives revenue to a problem entity",
}
RECOVERY = {"excess_cost": 0.6, "revenue_at_risk": 0.3}   # conservative share of each measured loss a fix recovers


def _strip_fences(text: str) -> str:
    text = text.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


class ActionAgent:
    def __init__(self, model: str = "openai/gpt-oss-120b"):
        self.llm = ChatGroq(api_key=os.environ.get("GROQ_API_KEY"), model=model, temperature=0, max_retries=6)

    @staticmethod
    def _candidates(anomaly_output: Dict) -> List[Dict]:
        a = (anomaly_output or {}).get("anomalies", {}) or {}
        items = a.get("entities") or ((a.get("products") or []) + (a.get("sellers") or []) + (a.get("categories") or []))
        items = [i for i in items if isinstance(i, dict) and i.get("id")]
        return sorted(items, key=lambda i: i.get("estimated_loss") or 0, reverse=True)[:10]

    @staticmethod
    def _expected(target: Dict) -> float:
        b = target.get("loss_breakdown") or {}
        return round(sum((b.get(k) or 0) * f for k, f in RECOVERY.items()), 2)

    @staticmethod
    def _rule_action(t: Dict) -> Dict:
        sig = t.get("signals") or []
        text = " ".join(t.get("reasons", [])).lower()
        if "cost" in sig or "cost" in text or "freight" in text:
            kind = "reduce_cost"
        elif "late" in sig or "late" in text:
            kind = "improve_fulfilment"
        elif "negative" in sig or "cancel" in text:
            kind = "reduce_cancellations"
        else:
            kind = "fix_quality"
        return {"action_type": kind, "target": t["id"], "target_type": t["type"],
                "description": f"{ACTION_TYPES[kind][0].upper() + ACTION_TYPES[kind][1:]} — {t['type']} {t['id']}: {'; '.join(t.get('reasons', []))}.",
                "rationale": "Derived from the largest flagged anomaly."}

    def suggest_action(self, ba_output: Dict, anomaly_output: Dict, previous_actions: Optional[List] = None,
                       validation: Optional[Dict] = None) -> Dict[str, Any]:
        cands = self._candidates(anomaly_output)
        if not cands:
            return {"status": "success", "action": {"actions": [], "note": "No flagged anomalies — no action needed this run."}}
        by_id = {c["id"]: c for c in cands}
        kpis = {k: (ba_output or {}).get("metrics", {}).get(k) for k in ("primary_metric_sum", "order_count", "aov", "cost_share", "cost_breakdown", "avg_quality", "late_delivery_rate", "cancel_rate")}
        past = []
        for p in (previous_actions or [])[:3]:
            inner = p.get("action", p) if isinstance(p, dict) else {}
            for a in (inner.get("actions") if isinstance(inner, dict) and isinstance(inner.get("actions"), list) else [inner]):
                if isinstance(a, dict) and a.get("action_type"):
                    past.append({k: a.get(k) for k in ("action_type", "target")})
        prompt = f"""
Flagged anomalies (measured, sorted by loss):
{json.dumps([{k: c.get(k) for k in ('type', 'id', 'reasons', 'estimated_loss', 'loss_breakdown', 'order_count')} for c in cands], indent=1, default=str)}

Store KPIs: {json.dumps(kpis, default=str)}
Previous recommendations: {json.dumps(past, default=str)}
Last validation verdict: {json.dumps(validation or {}, default=str)[:600]}

Propose up to 3 actions, highest value first. Rules:
- "target" MUST be an id from the flagged list above; "target_type" its type.
- "action_type": prefer one of {json.dumps(ACTION_TYPES)}; use another short snake_case type only if none fits.
- Fit the action to what the data shows about this business (its dimensions and costs), not to a generic store.
- If the last verdict says a previous action was ineffective, do not repeat it on the same target.
- "description": one or two sentences a store operator can act on today.
Return ONLY JSON: {{"actions": [{{"action_type": "", "target": "", "target_type": "", "description": "", "priority": "high|medium|low", "rationale": ""}}]}}
"""
        proposed = []
        try:
            resp = self.llm.invoke([SystemMessage(content="You are an ecommerce operations manager. Return only valid JSON."), HumanMessage(content=prompt)])
            data = json.loads(_strip_fences(resp.content))
            proposed = data.get("actions", []) if isinstance(data, dict) else data
        except Exception as e:
            print(f"⚠️ Action agent: LLM proposal failed ({e}); using rule-based action.")

        actions, seen = [], set()
        for a in proposed if isinstance(proposed, list) else []:
            if not isinstance(a, dict):
                continue
            t = by_id.get(str(a.get("target")))
            kind = re.sub(r"[^a-z0-9]+", "_", str(a.get("action_type") or "").lower()).strip("_")[:40]
            if not t or not kind or (t["id"], kind) in seen:
                continue
            seen.add((t["id"], kind))
            actions.append({**a, "action_type": kind, "target_type": t["type"]})
        if not actions:
            actions = [self._rule_action(cands[0])]

        out = []
        for a in actions[:3]:
            t = by_id[a["target"]]
            prio = str(a.get("priority", "")).lower()
            out.append({
                "action_id": "act_" + uuid.uuid4().hex[:8],
                "action_type": a["action_type"],
                "target": t["id"],
                "target_type": t["type"],
                "description": str(a.get("description") or "").strip() or self._rule_action(t)["description"],
                "rationale": str(a.get("rationale") or "").strip(),
                "priority": prio if prio in ("high", "medium", "low") else "medium",
                "expected_impact": min(self._expected(t), t.get("estimated_loss") or 0),
                "target_loss": t.get("estimated_loss"),
                "evidence": t.get("reasons", []),
            })
        return {"status": "success", "action": {"actions": out}}
