from langgraph.graph import END
from backend.schemas.state_schema import State
from typing import Literal


def agent_router(state:State)-> Literal["supervisor", "researcher", "analyst", "writer", "end"]:
    next_agent = state.get("next_agent", "")

    if "end" in next_agent or state.get("complete_task"):
        return END

    if next_agent in ["researcher", "analyst", "writer"]:
        return next_agent

    return "supervisor"

def intent_router(state:State)-> Literal["supervisor", "normal chatbot", "end"]:
    next_agent = state.get("next_agent", "supervisor")

    if "end" in next_agent or state.get("complete_task", False):
        return END
    elif next_agent == "normal chatbot":
        return "normal chatbot"
    else:
        return "supervisor"
