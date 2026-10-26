# action_agent.py

import json
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
load_dotenv()

class ActionAgent:
    def __init__(self, model: str = "openai/gpt-oss-120b"):
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.llm = ChatGroq(api_key=self.api_key, model=model, temperature=0)

    def _normalize_ba(self, ba: Dict) -> Dict:
        """Flatten any BA shape into a simple dict."""
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
        """Flatten any anomaly shape into a simple dict."""
        a = anomaly.get("anomalies", {})
        s = anomaly.get("summary", {})
        return {
            "products": a.get("products", [])[:3],
            "sellers": a.get("sellers", [])[:3],
            "categories": a.get("categories", [])[:3],
            "total_loss": (
                anomaly.get("anomaly_loss")
                or s.get("total_anomaly_loss", 0)
            ),
        }

    def suggest_action(self, ba_output: Dict, anomaly_output: Dict) -> Dict[str, Any]:
        ba = self._normalize_ba(ba_output)
        an = self._normalize_anomaly(anomaly_output)

        prompt = f"""
Based on the following ecommerce analysis, suggest specific actions to fix cost leakages.

BUSINESS ANALYSIS:
- Revenue: ${ba['revenue']:,.2f}
- Profit Margin: {ba['margin']:.1f}%
- Trend: {ba['trend']}
- Total Loss: ${ba['total_loss']:,.2f}

ANOMALIES DETECTED:
Top Problematic Products:
{json.dumps(an['products'], indent=2, default=str)}

Top Problematic Sellers:
{json.dumps(an['sellers'], indent=2, default=str)}

Top Problematic Categories:
{json.dumps(an['categories'], indent=2, default=str)}

Return a JSON object with:
- action_id: unique identifier
- action_type: one of [pause_seller, adjust_pricing, clear_inventory, stop_ad_campaign, negotiate_freight]
- target: specific product_id, seller_id, or category
- description: what to do
- expected_impact: estimated savings in USD
- priority: high/medium/low

Return ONLY valid JSON, no markdown, no extra text.
"""
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are an operations manager. Return only valid JSON, no markdown fences."),
                HumanMessage(content=prompt)
            ])
            content = response.content.strip().strip("```json").strip("```").strip()
            return {"status": "success", "action": json.loads(content)}
        except Exception as e:
            return {"status": "error", "error": str(e)}