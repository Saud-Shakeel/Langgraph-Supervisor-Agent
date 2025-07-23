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


def supervisor_agent(state:State)->Dict:
    messages = state["messages"][-1].content
    task = messages if messages else "No Task"
    has_research = bool(state.get("research_data", ""))
    has_analysis = bool(state.get("analysis", ""))
    has_report = bool(state.get("final_report", ""))

    chain = create_supervisor_chain()
    decision = chain.invoke({
        "has_research" : has_research,
        "has_analysis" : has_analysis,
        "has_report" : has_report,
        "task" : task
    })
    decision_text = decision.content.strip().lower()
    if "done" in decision_text or has_report:
        supervisor_msg = "✅ Supervisor: All tasks complete! Great work team."
        next_agent = "end"
    elif "researcher" in decision_text or not has_research:
        supervisor_msg = "📋 Supervisor: Let's start with research. Assigning to Researcher..."
        next_agent = "researcher"
    elif "analyst" in decision_text or (has_research and not has_analysis):
        supervisor_msg = "📋 Supervisor: Research done. Time for analysis. Assigning to Analyst..."
        next_agent = "analyst"
    elif "writer" in decision_text or (has_analysis and not has_report):
        supervisor_msg = "📋 Supervisor: Analysis complete. Let's create the report. Assigning to Writer..."
        next_agent = "writer"
    else:
        supervisor_msg = "✅ Supervisor: Task seems complete."
        next_agent = "end"

    return {
        "messages": [AIMessage(content=supervisor_msg)],
        "next_agent": next_agent,
        "current_task": task
    }
