# validation_agent.py

import json
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
load_dotenv()

class ValidationAgent:
    def __init__(self, model: str = "openai/gpt-oss-120b"):
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.llm = ChatGroq(api_key=self.api_key, model=model, temperature=0)

    def _normalize_ba(self, ba: Dict) -> Dict:
        if not ba:
            return {"revenue": 0, "margin": 0, "trend": "unknown", "total_loss": 0}
        m = ba.get("metrics", {})
        t = ba.get("trends", {})
        kpis = ba.get("kpis", {})
        cost = ba.get("cost_leakages", {})
        return {
            "revenue": (
                ba.get("revenue")
                or kpis.get("total_revenue")
                or m.get("primary_metric_sum", 0)
            ),
            "margin": (
                ba.get("margin")
                or kpis.get("profit_margin")
                or m.get("profit_margin", 0)
            ),
            "trend": (
                ba.get("trend")
                or t.get("trend", "unknown")
            ),
            "total_loss": (
                ba.get("total_loss")
                or cost.get("total_loss")
                or m.get("cost_leakage", 0)
            ),
        }

    def _normalize_anomaly(self, anomaly: Dict) -> Dict:
        if not anomaly:
            return {"total_loss": 0, "problematic_count": 0}
        s = anomaly.get("summary", {})
        return {
            "total_loss": (
                anomaly.get("anomaly_loss")
                or s.get("total_anomaly_loss", 0)
            ),
            "problematic_count": (
                s.get("problematic_products", 0)
                + s.get("problematic_sellers", 0)
                + s.get("problematic_categories", 0)
            ),
        }

    def validate(
        self,
        previous_action: Dict,
        previous_ba: Dict,
        previous_anomaly: Dict,
        current_ba: Dict,
        current_anomaly: Dict,
        action_taken: bool,
    ) -> Dict[str, Any]:
        prev_ba = self._normalize_ba(previous_ba)
        curr_ba = self._normalize_ba(current_ba)
        prev_an = self._normalize_anomaly(previous_anomaly)
        curr_an = self._normalize_anomaly(current_anomaly)

        prompt = f"""
You are validating whether a business action improved key metrics.

PREVIOUS STATE:
- Revenue: ${prev_ba['revenue']:,.2f}
- Margin: {prev_ba['margin']:.1f}%
- Trend: {prev_ba['trend']}
- Anomaly Loss: ${prev_an['total_loss']:,.2f}
- Problematic Entities: {prev_an['problematic_count']}

ACTION TAKEN: {json.dumps(previous_action, default=str)}
Action Executed: {action_taken}

CURRENT STATE:
- Revenue: ${curr_ba['revenue']:,.2f}
- Margin: {curr_ba['margin']:.1f}%
- Trend: {curr_ba['trend']}
- Anomaly Loss: ${curr_an['total_loss']:,.2f}
- Problematic Entities: {curr_an['problematic_count']}

Return a JSON object with:
- improved: true/false
- revenue_delta: change in revenue
- margin_delta: change in margin
- anomaly_loss_delta: change in anomaly loss
- verdict: one of [effective, ineffective, neutral]
- recommendation: next step as a string

Return ONLY valid JSON, no markdown, no extra text.
"""
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a business analyst. Return only valid JSON, no markdown fences."),
                HumanMessage(content=prompt)
            ])
            content = response.content.strip().strip("```json").strip("```").strip()
            return {"status": "success", "result": json.loads(content)}
        except Exception as e:
            return {"status": "error", "error": str(e)}