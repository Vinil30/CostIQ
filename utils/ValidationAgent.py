# validation_agent.py

import json
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


class ValidationAgent:
    """
    Validates if actions taken improved business metrics.
    LLM handles all analysis directly.
    """
    
    def __init__(self, api_key: str, model: str = "mixtral-8x7b-32768"):
        self.llm = ChatGroq(api_key=api_key, model=model, temperature=0)
    
    def validate(
        self,
        previous_action: Dict,
        previous_ba: Dict,
        previous_anomaly: Dict,
        current_ba: Dict,
        current_anomaly: Dict,
        action_taken: bool = True
    ) -> Dict[str, Any]:
        """
        LLM validates if action improved performance.
        """
        
        prompt = f"""
        Evaluate if the action taken was effective.
        
        PREVIOUS ACTION:
        {json.dumps(previous_action, indent=2, default=str)}
        
        PREVIOUS BUSINESS ANALYSIS:
        {json.dumps(previous_ba, indent=2, default=str)}
        
        PREVIOUS ANOMALIES:
        {json.dumps(previous_anomaly, indent=2, default=str)}
        
        CURRENT BUSINESS ANALYSIS:
        {json.dumps(current_ba, indent=2, default=str)}
        
        CURRENT ANOMALIES:
        {json.dumps(current_anomaly, indent=2, default=str)}
        
        Action Taken: {action_taken}
        
        Return JSON with:
        - effective: true/false
        - improvement_pct: percentage improvement (0-100)
        - feedback: brief assessment
        - recommendation: "continue" or "adjust" or "stop"
        - suggested_changes: what to modify if not effective
        
        Return ONLY valid JSON.
        """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a validation analyst. Return only valid JSON."),
                HumanMessage(content=prompt)
            ])
            result = json.loads(response.content)
            return {"status": "success", "validation": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}