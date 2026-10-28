"""
Manager Agent (orchestrator + memory).

Job: decide which downstream agent runs next and carry memory between runs.

The policy is explicit and deterministic, so it is predictable and testable:
  1. If an earlier run recommended an action, validate it first against this
     run's data (using the user's "I've done this" flag).
  2. Then ask the Action agent for new actions, giving it the past
     recommendations and the fresh validation verdict.
  3. Stop.
The decision is emitted as a tool call that LangGraph's ToolNode executes, so
the graph stays an agent loop; no LLM is needed to pick between two fixed steps.
"""

import uuid
from typing import Any, Dict

from langchain_core.messages import AIMessage


class ManagerAgent:
    def __init__(self, api_key: str = None, tools=None):
        self.tools = {t.name for t in (tools or [])}

    @staticmethod
    def _call(name: str) -> AIMessage:
        return AIMessage(content=f"Manager → {name}", tool_calls=[{"name": name, "args": {}, "id": "call_" + uuid.uuid4().hex[:12], "type": "tool_call"}])

    def run(self, state: Dict[str, Any]) -> AIMessage:
        has_history = bool(state.get("previous_actions"))
        if has_history and not state.get("validation_done"):
            print("🧭 Manager: validating the previous run's action")
            return self._call("validation_agent_tool")
        if not state.get("action_done"):
            print("🧭 Manager: requesting new actions")
            return self._call("action_agent_tool")
        print("🧭 Manager: done")
        return AIMessage(content="Pipeline complete.")
