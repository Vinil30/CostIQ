"""
Validation Agent.

Job: check whether the PREVIOUS run's recommended action actually worked,
using the data from THIS run.

How it works — the verdict is computed, not guessed:
- For every target of the previous action it compares the measured loss in
  the previous run with the loss in this run (0 if it is no longer flagged).
- It also compares store-level revenue, freight share and anomaly loss.
- Verdict:
    not_executed  the user has not marked the action as done
    no_new_data   this upload covers the same period as the previous one
    effective     target loss down ≥ 20 %
    ineffective   target loss flat or up
    neutral       in between
- The LLM only writes the one-line recommendation from these computed facts.
"""

import json
import os
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq


def _actions(previous_action: Dict) -> List[Dict]:
    if not isinstance(previous_action, dict):
        return []
    inner = previous_action.get("action", previous_action)
    if isinstance(inner, dict) and isinstance(inner.get("actions"), list):
        return [a for a in inner["actions"] if isinstance(a, dict)]
    return [inner] if isinstance(inner, dict) and inner.get("action_type") else []


def _loss_of(anomaly_output: Dict, target: str, ttype: str) -> float:
    """Measured loss of `target` in a run (0 when it is no longer flagged). Reads current and legacy layouts."""
    an = (anomaly_output or {}).get("anomalies", {}) or {}
    pool = an.get("entities") or [a for k in ("products", "sellers", "categories") for a in (an.get(k) or [])]
    for a in pool:
        if isinstance(a, dict) and str(a.get("id")) == str(target) and (not ttype or a.get("type") == ttype):
            return float(a.get("estimated_loss") or 0)
    return 0.0


def _num(d: Dict, *path):
    for p in path:
        d = (d or {}).get(p) if isinstance(d, dict) else None
    try:
        return float(d)
    except (TypeError, ValueError):
        return None


class ValidationAgent:
    def __init__(self, model: str = "openai/gpt-oss-120b"):
        self.llm = ChatGroq(api_key=os.environ.get("GROQ_API_KEY"), model=model, temperature=0, max_retries=6)

    def validate(self, previous_action: Dict, previous_ba: Dict, previous_anomaly: Dict,
                 current_ba: Dict, current_anomaly: Dict, action_taken: bool) -> Dict[str, Any]:
        acts = _actions(previous_action)
        if not acts:
            return {"status": "error", "error": "No previous action to validate."}

        targets, before_sum, after_sum = [], 0.0, 0.0
        for a in acts:
            b = _loss_of(previous_anomaly, a.get("target"), a.get("target_type"))
            n = _loss_of(current_anomaly, a.get("target"), a.get("target_type"))
            before_sum += b; after_sum += n
            targets.append({"action_type": a.get("action_type"), "target": a.get("target"), "target_type": a.get("target_type"),
                            "loss_before": round(b, 2), "loss_after": round(n, 2), "still_flagged": n > 0})

        def delta(*path):
            x, y = _num(previous_ba, *path), _num(current_ba, *path)
            return None if x is None or y is None else round(y - x, 2)
        rev_d = delta("metrics", "primary_metric_sum")
        fs_d = delta("metrics", "cost_share")
        margin_d = delta("metrics", "profit_margin")
        pl, cl = _num(previous_anomaly, "summary", "total_anomaly_loss"), _num(current_anomaly, "summary", "total_anomaly_loss")
        loss_d = None if pl is None or cl is None else round(cl - pl, 2)

        same_period = (_num(previous_ba, "metrics", "row_count") == _num(current_ba, "metrics", "row_count")
                       and (previous_ba or {}).get("metrics", {}).get("date_end") == (current_ba or {}).get("metrics", {}).get("date_end"))
        change = (after_sum - before_sum) / before_sum if before_sum else None
        if not action_taken:
            verdict = "not_executed"
        elif same_period:
            verdict = "no_new_data"
        elif change is not None and change <= -0.2:
            verdict = "effective"
        elif change is None or change >= 0:
            verdict = "ineffective"
        else:
            verdict = "neutral"

        facts = {"verdict": verdict, "action_marked_done": bool(action_taken), "targets": targets,
                 "target_loss_change_pct": None if change is None else round(change * 100, 1),
                 "revenue_delta": rev_d, "cost_share_delta_pts": fs_d, "anomaly_loss_delta": loss_d}
        recommendation = self._recommend(facts)
        return {"status": "success", "result": {
            "improved": verdict == "effective",
            "verdict": verdict,
            "action_taken": bool(action_taken),
            "targets": targets,
            "target_loss_change_pct": facts["target_loss_change_pct"],
            "revenue_delta": rev_d,
            "cost_share_delta": fs_d,
            "margin_delta": margin_d,
            "anomaly_loss_delta": loss_d,
            "recommendation": recommendation,
        }}

    def _recommend(self, facts: Dict) -> str:
        fallback = {
            "not_executed": "The previous action hasn't been marked as done yet — carry it out, then tick “I've done this” before the next upload.",
            "no_new_data": "This upload covers the same period as the last one, so the action's effect can't be measured yet. Upload data from after the change.",
            "effective": "The action reduced the targeted loss — keep it and apply the same fix to the next-largest anomaly.",
            "ineffective": "The targeted loss did not fall — try a different action on this target.",
            "neutral": "The targeted loss fell slightly — give it another cycle before deciding.",
        }[facts["verdict"]]
        if facts["verdict"] in ("not_executed", "no_new_data"):
            return fallback
        try:
            resp = self.llm.invoke([
                SystemMessage(content="You are a concise ecommerce analyst. One or two sentences, no markdown."),
                HumanMessage(content=f"Validation facts (computed): {json.dumps(facts, default=str)}\nWrite the recommended next step. Use only these numbers."),
            ])
            return resp.content.strip() or fallback
        except Exception:
            return fallback
