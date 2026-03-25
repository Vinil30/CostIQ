from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain.tools import tool
from typing import TypedDict
from langsmith import traceable

class AgentState(TypedDict):
    pass

@tool
def validation_agent():
    pass

@tool
def action_agent():
    pass


@traceable(name="DataTransformationAgent")
def data_transformer(state:AgentState)->AgentState:
    pass
@traceable(name="DataAnalysisAgent")
def data_analysis(state:AgentState)->AgentState:
    pass
@traceable(name="AnomolyDetectionAgent")
def anamoly_detection(state:AgentState)->AgentState:
    pass
@traceable(name="ManagerAgent")
def manager_agent(state:AgentState)->AgentState:
    pass

graph = StateGraph(AgentState)
graph.add_node("DataTransformer",data_transformer)
graph.add_node("DataAnalysis",data_analysis)
graph.add_node("AnamolyDetection",anamoly_detection)
graph.add_node("ManagerAgent",manager_agent)
graph.add_edge(START,"DataTransformer")
graph.add_edge("DataTransformer","DataAnalysis")
graph.add_edge("DataAnalysis","AnamolyDetection")
graph.add_edge("AnamolyDetection","ManagerAgent")
graph.add_edge("ManagerAgent",END)


