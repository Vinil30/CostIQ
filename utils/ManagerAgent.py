from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool
import json


class ManagerAgent:
    def __init__(self, api_key: str, tools: List[BaseTool], model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key
        self.llm = ChatGroq(
            api_key=api_key,
            model=model,
            temperature=0
        ).bind_tools(tools)

    def run(self, state: Dict[str, Any]) -> AIMessage:
        loop_counter = state.get("loop_counter", 0)
        action_taken = state.get("action_taken", False)
        validation_done = state.get("validation_done", False)
        is_first_run = state.get("is_first_run", True)

        # ── Hard guards: decide WITHOUT calling the LLM ──────────────────────
        # Guard 1: both steps done → stop immediately
        if action_taken and validation_done:
            print("✅ Action + Validation complete. Stopping.")
            return AIMessage(content="Pipeline complete. Action taken and validated.")

        # Guard 2: too many iterations → stop
        if loop_counter >= 2:
            print(f"⚠️ loop_counter={loop_counter} — stopping before LLM call.")
            return AIMessage(content="Max iterations reached. Stopping.")

        # ── Determine the ONE correct next tool ──────────────────────────────
        # After action is taken, ONLY validation should ever be called.
        if action_taken and not validation_done:
            next_tool = "validation_agent_tool"
        elif is_first_run and not action_taken:
            next_tool = "action_agent_tool"
        else:
            # Nothing left to do
            print("🤖 Manager: No action needed.")
            return AIMessage(content="No further action required.")

        # ── Build a tightly constrained prompt ───────────────────────────────
        ba_output = state.get("ba_output", {})
        anomaly_output = state.get("anomaly_output", {})
        metrics = ba_output.get("metrics", {})
        trends = ba_output.get("trends", {})
        anomaly_summary = anomaly_output.get("summary", {})

        state_snapshot = f"""
Revenue:       ${metrics.get('primary_metric_sum', 0):,.0f}
Margin:        {metrics.get('profit_margin', 0):.1f}%
Trend:         {trends.get('trend', 'N/A')} ({trends.get('trend_percentage', 0)}%)
Anomaly Loss:  ${anomaly_summary.get('total_anomaly_loss', 0):,.0f}
Loop:          {loop_counter}/2
Action Taken:  {action_taken}
Validated:     {validation_done}
"""

        prompt = f"""
Current business state:
{state_snapshot}

You MUST call exactly ONE tool: `{next_tool}`.
Do not call any other tool. Do not skip. Do not explain. Just call `{next_tool}`.
"""

        messages = [
            SystemMessage(content=(
                "You are a manager agent. You will be told exactly which tool to call. "
                "Call that tool exactly once, with no additional reasoning or explanation."
            )),
            HumanMessage(content=prompt),
        ]

        response = self.llm.invoke(messages)

        # ── Post-call safety: strip unexpected extra tool calls ───────────────
        if hasattr(response, "tool_calls") and response.tool_calls:
            # Keep only the first call, and only if it matches the expected tool
            valid_calls = [
                tc for tc in response.tool_calls
                if tc["name"] == next_tool
            ]
            if not valid_calls:
                # LLM called the wrong tool — force stop
                print(f"⚠️ LLM called wrong tool(s): {[tc['name'] for tc in response.tool_calls]}. Stopping.")
                return AIMessage(content=f"Expected {next_tool} but got wrong tool. Stopping.")

            # Keep only the first valid call
            object.__setattr__(response, "tool_calls", valid_calls[:1]) if hasattr(response, "__setattr__") else None
            try:
                response.tool_calls = valid_calls[:1]
            except Exception:
                pass  # immutable — graph injection in manager_agent_node will handle it

            print(f"\n🤖 Manager: Calling {next_tool}")
        else:
            print("\n🤖 Manager: No tool calls in response.")

        return response