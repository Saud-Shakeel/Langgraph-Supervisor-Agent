from langchain.prompts import ChatPromptTemplate
from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage
from backend.schemas.state_schema import State
from backend.core.config import llm
from typing import Dict

def create_supervisor_chain():
    supervisor_chain = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="""
    You are a supervisor managing a team of agents:

    1. Researcher - Gathers information and data
    2. Analyst - Analyzes data and provides insights
    3. Writer - Creates reports and summaries

    Based on the current state and conversation, decide which agent should work next.
    If the task is complete, respond with 'DONE'.
    If the user asks for a task that has no concern with any of the agents in the team, then respond with
    'There is no agent dedicated for this task, do you anything else to work on?'

    Current state:
    - Has research data: {has_research}
    - Has analysis: {has_analysis}
    - Has report: {has_report}

    Respond with ONLY the agent name (researcher/analyst/writer) or 'DONE'.

    """
    ),
    (HumanMessage(content="{task}"))

])
    return supervisor_chain | llm


