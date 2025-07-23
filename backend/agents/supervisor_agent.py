from backend.schemas.state_schema import State
from typing import Dict
from backend.core.chain import create_supervisor_chain
from langchain.schema.messages import AIMessage


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