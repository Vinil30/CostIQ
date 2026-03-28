from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain.tools import tool
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from typing import TypedDict, Optional, Dict, Annotated, Sequence
import operator
import os
import json
import pandas as pd
from langsmith import traceable
from utils.ActionAgent import ActionAgent
from utils.ValidationAgent import ValidationAgent
from utils.ManagerAgent import ManagerAgent


class AgentState(TypedDict):
    api_key: str
    file_paths: list[str]
    unified_dataset_path: Optional[str]
    unified_dataframe: Optional[pd.DataFrame]
    ba_output: dict
    anomaly_output: dict
    ba_schema: Optional[dict]
    previous_action: Optional[dict]
    previous_ba: Optional[dict]
    previous_anomaly: Optional[dict]
    action_taken: Optional[bool]
    validation_done: Optional[bool]
    messages: Annotated[Sequence[BaseMessage], operator.add]
    is_first_run: Optional[bool]
    loop_counter: Optional[int]


# ──────────────────────────────────────────────────────────────────────────────
# TOOLS
# ──────────────────────────────────────────────────────────────────────────────

@tool
def action_agent_tool(ba_output: Dict, anomaly_output: Dict, api_key: str) -> Dict:
    """Suggest an action based on business analysis and detected anomalies."""
    agent = ActionAgent()
    result = agent.suggest_action(ba_output, anomaly_output)
    return {
        "status": "action_suggested",
        "action": result,
        "message": "Action suggested successfully",
    }


@tool
def validation_agent_tool(
    previous_action: Dict,
    previous_ba: Dict,
    previous_anomaly: Dict,
    current_ba: Dict,
    current_anomaly: Dict,
    action_taken: bool,
    api_key: str,
) -> Dict:
    """Validate whether a previous action improved business metrics."""
    agent = ValidationAgent()
    result = agent.validate(
        previous_action,
        previous_ba,
        previous_anomaly,
        current_ba,
        current_anomaly,
        action_taken,
    )
    return {
        "status": "validation_complete",
        "validation_result": result,
        "message": "Validation completed",
    }


TOOLS = [action_agent_tool, validation_agent_tool]
tool_node = ToolNode(TOOLS)


# ──────────────────────────────────────────────────────────────────────────────
# ROUTING
# ──────────────────────────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """Route after ManagerAgent: go to tools only if there's a valid tool call."""
    # Hard stop by counter (matches ManagerAgent's own guard)
    if state.get("loop_counter", 0) >= 2:
        print(f"⚠️ Router: loop_counter={state.get('loop_counter')}. Ending.")
        return END

    # Hard stop when both phases are complete
    if state.get("action_taken") and state.get("validation_done"):
        print("✅ Router: Action + Validation complete. Ending.")
        return END

    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "tools"
    return END


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE NODES
# ──────────────────────────────────────────────────────────────────────────────

@traceable(name="DataTransformationAgent")
def data_transformer(state: AgentState) -> AgentState:
    from utils.DataUnifier import SchemaUnificationPipeline

    pipeline = SchemaUnificationPipeline(api_key=state["api_key"])
    result = pipeline.unify(state["file_paths"])

    unified_df = None
    if result["output_file"] and os.path.exists(result["output_file"]):
        unified_df = pd.read_csv(result["output_file"])

    return {
        **state,
        "unified_dataset_path": result["output_file"],
        "unified_dataframe": unified_df,
        "is_first_run": True,
        "loop_counter": 0,
        "action_taken": False,
        "validation_done": False,
    }


@traceable(name="DataAnalysisAgent")
def data_analysis(state: AgentState) -> AgentState:
    from utils.DatasetAnalyser import BusinessAnalystAgent

    agent = BusinessAnalystAgent(api_key=state["api_key"], verbose=True, use_cache=True)

    if state.get("unified_dataframe") is not None:
        result = agent.analyze(state["unified_dataframe"])
    elif state.get("unified_dataset_path"):
        result = agent.analyze(state["unified_dataset_path"])
    else:
        result = {"status": "error", "error": "No data available for analysis"}

    new_state = {**state, "ba_output": result}
    if result.get("status") == "success" and result.get("detected_schema"):
        new_state["ba_schema"] = result["detected_schema"]

    return new_state


def _translate_schema_for_anomaly(ba_schema: dict, df: pd.DataFrame, api_key: str) -> dict:
    if not ba_schema or df is None:
        return None

    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile", temperature=0)
    available_columns = df.columns.tolist()
    column_samples = {col: df[col].dropna().head(3).tolist() for col in available_columns}

    prompt = f"""
You are a data schema mapper. The Business Analyst identified this schema:
{json.dumps(ba_schema, indent=2)}

Actual available columns:
{json.dumps(column_samples, indent=2)}

Map to anomaly detector keys:
- revenue_col, cost_col, date_col, product_col, seller_col, category_col, review_col

Rules:
- Only use columns that actually exist.
- Set to null if no suitable column exists.
- revenue_col must be numeric money column (not an ID).

Return ONLY valid JSON, no markdown:
{{
  "revenue_col": "...", "cost_col": "...", "date_col": "...",
  "product_col": "...", "seller_col": "...", "category_col": "...", "review_col": "..."
}}
"""
    try:
        response = llm.invoke([
            SystemMessage(content="You are a precise data schema mapper. Return only valid JSON."),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip().replace("```json", "").replace("```", "").strip()
        translated = json.loads(raw)
        for key, col in translated.items():
            if col and col not in df.columns:
                translated[key] = None
        return translated
    except Exception as e:
        print(f"⚠️ Schema translation failed: {e}")
        return None


@traceable(name="AnomalyDetectionAgent")
def anomaly_detection(state: AgentState) -> AgentState:
    from utils.AnamolyDetection import AnomalyDetectionAgent

    agent = AnomalyDetectionAgent(api_key=state["api_key"], verbose=True)
    df = state.get("unified_dataframe")
    translated = _translate_schema_for_anomaly(state.get("ba_schema"), df, state["api_key"])

    if df is not None:
        result = agent.analyze(df=df, schema=translated)
    elif state.get("unified_dataset_path"):
        result = agent.analyze(df=state["unified_dataset_path"], schema=translated)
    else:
        result = {"status": "error", "error": "No data available"}

    return {**state, "anomaly_output": result}


@traceable(name="ManagerAgent")
def manager_agent_node(state: AgentState) -> dict:
    manager = ManagerAgent(api_key=state["api_key"], tools=TOOLS)

    serializable_state = {
        "api_key": state["api_key"],
        "ba_output": state.get("ba_output", {}),
        "anomaly_output": state.get("anomaly_output", {}),
        "previous_ba": state.get("previous_ba"),
        "previous_anomaly": state.get("previous_anomaly"),
        "previous_action": state.get("previous_action"),
        "action_taken": state.get("action_taken", False),
        "validation_done": state.get("validation_done", False),
        "is_first_run": state.get("is_first_run", True),
        "loop_counter": state.get("loop_counter", 0),
    }

    response = manager.run(serializable_state)

    # Inject full state into tool call args (so ToolNode gets correct data)
    if getattr(response, "tool_calls", None):
        for tool_call in response.tool_calls:
            tool_call["args"]["api_key"] = state["api_key"]
            if tool_call["name"] == "action_agent_tool":
                tool_call["args"]["ba_output"] = state.get("ba_output", {})
                tool_call["args"]["anomaly_output"] = state.get("anomaly_output", {})
            elif tool_call["name"] == "validation_agent_tool":
                tool_call["args"]["previous_ba"] = state.get("previous_ba", {})
                tool_call["args"]["previous_anomaly"] = state.get("previous_anomaly", {})
                tool_call["args"]["current_ba"] = state.get("ba_output", {})
                tool_call["args"]["current_anomaly"] = state.get("anomaly_output", {})
                tool_call["args"]["previous_action"] = state.get("previous_action", {})
                tool_call["args"]["action_taken"] = state.get("action_taken", False)
        print(f"✅ Injected state into {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": [response]}


@traceable(name="ProcessToolResults")
def process_tool_results(state: AgentState) -> AgentState:
    """
    Parse ToolMessages and update state flags.
    This is the single source of truth for action_taken / validation_done.
    """
    new_state = {**state}
    new_state["loop_counter"] = state.get("loop_counter", 0) + 1

    # Walk messages in reverse to find the most recent ToolMessages
    for msg in reversed(state["messages"]):
        if not isinstance(msg, ToolMessage):
            continue

        try:
            parsed = json.loads(msg.content)
        except Exception:
            # content might be a plain string on error
            print(f"⚠️ Could not parse ToolMessage content: {msg.content[:120]}")
            continue

        if not isinstance(parsed, dict):
            continue

        tool_name = getattr(msg, "name", None)

        if tool_name == "action_agent_tool" and parsed.get("status") == "action_suggested":
            new_state["action_taken"] = True
            new_state["validation_done"] = False
            new_state["is_first_run"] = False
            new_state["previous_action"] = parsed.get("action")
            # Snapshot current BA/anomaly as "previous" for the validation step
            new_state["previous_ba"] = state.get("ba_output")
            new_state["previous_anomaly"] = state.get("anomaly_output")
            print("✅ process_tool_results: action_taken=True")
            break  # Only process the most recent relevant message

        if tool_name == "validation_agent_tool" and parsed.get("status") == "validation_complete":
            new_state["validation_done"] = True
            new_state["action_taken"] = False   # reset so loop guard fires cleanly
            new_state["is_first_run"] = False
            new_state["previous_validation"] = parsed.get("validation_result")
            print("✅ process_tool_results: validation_done=True")
            break

    return new_state


# ──────────────────────────────────────────────────────────────────────────────
# GRAPH
# ──────────────────────────────────────────────────────────────────────────────

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

graph.add_conditional_edges(
    "ManagerAgent",
    should_continue,
    {"tools": "tools", END: END},
)

graph.add_edge("tools", "process_tool_results")
graph.add_edge("process_tool_results", "ManagerAgent")

app = graph.compile()