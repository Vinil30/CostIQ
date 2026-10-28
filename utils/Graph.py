"""
LangGraph pipeline: Unify → Analyse → Detect → Manager ⇄ (Validation, Action).

State flow
- DataTransformer   files → one table (+ column grain so money isn't double counted)
- DataAnalysis      table → roles, KPIs, trend, insights
- AnomalyDetection  table + roles → flagged entities / days with measured loss
- ManagerAgent      decides the next tool: validate the previous run's action
                    (if there is one), then request new actions, then stop
- tools             runs the chosen agent; process_tool_results records the outcome
"""

import json
import operator
import os
from typing import Annotated, Dict, List, Optional, Sequence, TypedDict

import pandas as pd
from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langsmith import traceable

from utils.ActionAgent import ActionAgent
from utils.ManagerAgent import ManagerAgent
from utils.ValidationAgent import ValidationAgent

MAX_MANAGER_STEPS = 4


class AgentState(TypedDict, total=False):
    api_key: str
    file_paths: List[str]
    unified_dataset_path: Optional[str]
    unified_dataframe: Optional[pd.DataFrame]
    column_grain: Dict
    unify_meta: Dict
    ba_output: Dict
    anomaly_output: Dict
    # Memory from earlier runs (newest first), supplied by the app
    previous_actions: Optional[List[Dict]]
    previous_bas: Optional[List[Dict]]
    previous_anomalies: Optional[List[Dict]]
    action_taken: Optional[bool]          # the user's "I've done this" flag for the previous action
    # Manager bookkeeping
    validation_done: bool
    action_done: bool
    validation_result: Optional[Dict]
    loop_counter: int
    messages: Annotated[Sequence[BaseMessage], operator.add]


# ── Tools the Manager can call ─────────────────────────────────────────────
@tool
def action_agent_tool(ba_output: Dict, anomaly_output: Dict, previous_actions: Optional[List[Dict]] = None,
                      validation: Optional[Dict] = None) -> Dict:
    """Recommend concrete actions for the flagged anomalies."""
    result = ActionAgent().suggest_action(ba_output, anomaly_output, previous_actions, validation)
    return {"status": "action_suggested", "action": result}


@tool
def validation_agent_tool(previous_action: Dict, previous_ba: Dict, previous_anomaly: Dict,
                          current_ba: Dict, current_anomaly: Dict, action_taken: bool) -> Dict:
    """Measure whether the previous run's action reduced the loss it targeted."""
    result = ValidationAgent().validate(previous_action, previous_ba, previous_anomaly, current_ba, current_anomaly, action_taken)
    return {"status": "validation_complete", "validation_result": result}


TOOLS = [action_agent_tool, validation_agent_tool]
tool_node = ToolNode(TOOLS)


# ── Pipeline nodes ─────────────────────────────────────────────────────────
@traceable(name="DataTransformationAgent")
def data_transformer(state: AgentState) -> Dict:
    from utils.DataUnifier import SchemaUnificationPipeline
    paths = state["file_paths"]
    result = SchemaUnificationPipeline(api_key=state["api_key"]).unify(paths, output_dir=os.path.dirname(paths[0]))
    if result.get("status") != "success":
        raise RuntimeError(f"Data unification failed: {result.get('error')}")
    return {
        "unified_dataset_path": result["output_file"],
        "unified_dataframe": result.pop("dataframe"),
        "column_grain": result.get("column_grain", {}),
        "unify_meta": {k: result.get(k) for k in ("fact_table", "joins", "rows", "dropped_columns", "plan_reason")},
        "validation_done": False, "action_done": False, "validation_result": None, "loop_counter": 0,
    }


@traceable(name="DataAnalysisAgent")
def data_analysis(state: AgentState) -> Dict:
    from utils.DatasetAnalyser import BusinessAnalystAgent
    result = BusinessAnalystAgent(api_key=state["api_key"], verbose=True).analyze(state["unified_dataframe"], state.get("column_grain"))
    result["data_profile"] = state.get("unify_meta", {})
    return {"ba_output": result}


@traceable(name="AnomalyDetectionAgent")
def anomaly_detection(state: AgentState) -> Dict:
    from utils.AnamolyDetection import AnomalyDetectionAgent
    ba = state.get("ba_output") or {}
    if ba.get("status") != "success":
        return {"anomaly_output": {"status": "error", "error": "Skipped: business analysis did not succeed."}}
    result = AnomalyDetectionAgent(api_key=state["api_key"], verbose=True).analyze(
        state["unified_dataframe"], schema=ba.get("detected_schema"), column_grain=state.get("column_grain"), kpis=ba.get("metrics"))
    return {"anomaly_output": result}


@traceable(name="ManagerAgent")
def manager_agent_node(state: AgentState) -> Dict:
    msg = ManagerAgent(tools=TOOLS).run(state)
    for call in getattr(msg, "tool_calls", []) or []:
        if call["name"] == "validation_agent_tool":
            call["args"] = {
                "previous_action": (state.get("previous_actions") or [{}])[0],
                "previous_ba": (state.get("previous_bas") or [{}])[0],
                "previous_anomaly": (state.get("previous_anomalies") or [{}])[0],
                "current_ba": state.get("ba_output") or {},
                "current_anomaly": state.get("anomaly_output") or {},
                "action_taken": bool(state.get("action_taken")),
            }
        elif call["name"] == "action_agent_tool":
            call["args"] = {
                "ba_output": state.get("ba_output") or {},
                "anomaly_output": state.get("anomaly_output") or {},
                "previous_actions": state.get("previous_actions") or [],
                "validation": state.get("validation_result") or {},
            }
    return {"messages": [msg], "loop_counter": state.get("loop_counter", 0) + 1}


@traceable(name="ProcessToolResults")
def process_tool_results(state: AgentState) -> Dict:
    for msg in reversed(state["messages"]):
        if not isinstance(msg, ToolMessage):
            continue
        try:
            parsed = json.loads(msg.content)
        except Exception:
            print(f"⚠️ Unreadable tool output: {str(msg.content)[:160]}")
            # Mark the step done anyway so the manager doesn't retry forever.
            return {"action_done": True} if getattr(msg, "name", "") == "action_agent_tool" else {"validation_done": True}
        if parsed.get("status") == "action_suggested":
            return {"action_done": True}
        if parsed.get("status") == "validation_complete":
            res = parsed.get("validation_result") or {}
            return {"validation_done": True, "validation_result": res.get("result", res)}
        break
    return {}


def should_continue(state: AgentState) -> str:
    if state.get("loop_counter", 0) > MAX_MANAGER_STEPS:
        return END
    last = state["messages"][-1] if state.get("messages") else None
    return "tools" if isinstance(last, AIMessage) and getattr(last, "tool_calls", None) else END


graph = StateGraph(AgentState)
graph.add_node("DataTransformer", data_transformer)
graph.add_node("DataAnalysis", data_analysis)
graph.add_node("AnomalyDetection", anomaly_detection)
graph.add_node("ManagerAgent", manager_agent_node)
graph.add_node("tools", tool_node)
graph.add_node("process_tool_results", process_tool_results)

graph.add_edge(START, "DataTransformer")
graph.add_edge("DataTransformer", "DataAnalysis")
graph.add_edge("DataAnalysis", "AnomalyDetection")
graph.add_edge("AnomalyDetection", "ManagerAgent")
graph.add_conditional_edges("ManagerAgent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "process_tool_results")
graph.add_edge("process_tool_results", "ManagerAgent")

app = graph.compile()
