from backend.core.config import llm
from backend.schemas.state_schema import State
from typing import Dict
from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage

def analysis_agent(state:State)->Dict:
    task = state.get("current_task", "")
    research_data = state.get("research_data", "")

    analysis_system_prompt = SystemMessage(
        content="""
        You are an analysis agent that understands the research data and provides the relevant analysis based on the
        mentioned topic. Be concise but thorough.
        """
    )
    analysis_human_prompt = HumanMessage(
        content=f"""
        As a data analyst, analyze this research data and provide insights:

        Research Data:
        {research_data}

        Provide:
        1. Key insights and patterns
        2. Strategic implications
        3. Risks and opportunities
        4. Recommendations

        Focus on actionable insights related to: {task}
        """
    )
    analysis_response = llm.invoke([analysis_system_prompt, analysis_human_prompt])
    analysis = analysis_response.content
    agent_msg = f"📊 Analyst: I've completed the analysis.\n\nTop insights:\n{analysis[:400]}..."

    return {
        "messages": [AIMessage(content=agent_msg)],
        "next_agent": "supervisor",
        "analysis": analysis
    }