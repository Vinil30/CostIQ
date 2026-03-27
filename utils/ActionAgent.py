# action_agent.py

import json
from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


class ActionAgent:
    """
    Suggests actionable fixes based on anomalies and business metrics.
    """
    
    def __init__(self, api_key: str, model: str = "mixtral-8x7b-32768"):
        self.llm = ChatGroq(api_key=api_key, model=model, temperature=0)
    
    def suggest_action(self, ba_output: Dict, anomaly_output: Dict) -> Dict[str, Any]:
        """
        Generate action based on anomalies and business analysis.
        """
        
        prompt = f"""
        Based on the following ecommerce analysis, suggest specific actions to fix cost leakages.
        
        BUSINESS ANALYSIS:
        - Revenue: ${ba_output.get('kpis', {}).get('total_revenue', 0):,.2f}
        - Profit Margin: {ba_output.get('kpis', {}).get('profit_margin', 0):.1f}%
        - Total Loss: ${ba_output.get('cost_leakages', {}).get('total_loss', 0):,.2f}
        
        ANOMALIES DETECTED:
        Top Problematic Products:
        {json.dumps(anomaly_output.get('anomalies', {}).get('products', [])[:3], indent=2, default=str)}
        
        Top Problematic Sellers:
        {json.dumps(anomaly_output.get('anomalies', {}).get('sellers', [])[:3], indent=2, default=str)}
        
        Top Problematic Categories:
        {json.dumps(anomaly_output.get('anomalies', {}).get('categories', [])[:3], indent=2, default=str)}
        
        Return a JSON with:
        - action_id: unique identifier
        - action_type: one of [pause_seller, adjust_pricing, clear_inventory, stop_ad_campaign, negotiate_freight]
        - target: specific product_id, seller_id, or category
        - description: what to do
        - expected_impact: estimated savings in USD
        - priority: high/medium/low
        
        Return ONLY valid JSON.
        """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are an operations manager. Return only valid JSON."),
                HumanMessage(content=prompt)
            ])
            action = json.loads(response.content)
            return {"status": "success", "action": action}
        except Exception as e:
            return {"status": "error", "error": str(e)}