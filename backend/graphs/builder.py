from backend.agents.supervisor_agent import supervisor_agent
from backend.agents.analyst_agent import analysis_agent
from backend.agents.researcher_agent import researcher_agent
from backend.agents.writer_agent import writer_agent
from backend.agents.routers import intent_router, agent_router
from langgraph.graph import StateGraph, START, END
from backend.schemas.state_schema import State
from backend.models.normal_chatbot import handle_normal_chat
from backend.models.intent_classifier import classify_user_intent


graph = StateGraph(State)

graph.add_node("intent classifier", classify_user_intent)
graph.add_node("normal chatbot", handle_normal_chat)
graph.add_node("supervisor", supervisor_agent)
graph.add_node("researcher", researcher_agent)
graph.add_node("analyst", analysis_agent)
graph.add_node("writer", writer_agent)

graph.add_edge(START, "intent classifier")
graph.add_conditional_edges(
    "intent classifier",
    intent_router,
    {
    "normal chatbot": "normal chatbot",
    "supervisor": "supervisor",
    END:END
    })

for node in ["supervisor", "researcher", "analyst", "writer"]:
    graph.add_conditional_edges(
    node,
    agent_router,
    {
    "supervisor": "supervisor",
    "researcher": "researcher",
    "analyst": "analyst",
    "writer": "writer",
    END: END
    })

builder = graph.compile()