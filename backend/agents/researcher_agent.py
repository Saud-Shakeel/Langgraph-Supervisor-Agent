from backend.core.config import llm
from backend.schemas.state_schema import State
from typing import Dict
from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage

def researcher_agent(state:State)->Dict:
    task = state.get("current_task", "research topic")

    research_system_prompt = SystemMessage(content=
    """
    You are a research agent that understands the user's task and provides the research based on the mentioned topic.
    Be concise but thorough.
    """
    )
    research_user_prompt = HumanMessage(content=
    f"""
    As a research specialist, provide comprehensive information about: {task}

    Include:
    1. Key facts and background
    2. Current trends or developments
    3. Important statistics or data points
    4. Notable examples or case studies

    Be concise but thorough."
    """
    )
    research_response = llm.invoke([research_system_prompt, research_user_prompt])
    research_data = research_response.content
    agent_msg = f"🔍 Researcher: I've completed the research on '{task}'.\n\nKey findings:\n{research_data[:500]}..."

    return {
        "messages": [AIMessage(content=agent_msg)],
        "next_agent": "supervisor",
        "research_data": research_data
    }
