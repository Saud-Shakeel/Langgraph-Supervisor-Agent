from backend.core.config import llm
from backend.schemas.state_schema import State
from typing import Dict
from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage
from datetime import datetime

def writer_agent(state:State)->Dict:
    task = state.get("current_task", "")
    research_data = state.get("research_data", "")
    analysis = state.get("analysis", "")


    writer_system_prompt = SystemMessage(
        content="""
        You are a writer agent that understands the research data, analyzes the insights, and then writes a
        final report based on the mentioned topic. Be concise but thorough.
        """
    )
    writer_human_prompt = HumanMessage(
        content=f"""
        As a professional writer, create an executive report based on:

        Task: {task}

        Research Findings:
        {research_data[:1000]}

        Analysis:
        {analysis[:1000]}

        Create a well-structured report with:
        1. Executive Summary
        2. Key Findings
        3. Analysis & Insights
        4. Recommendations
        5. Conclusion

        Keep it professional and concise
        """
    )

    writer_response = llm.invoke([writer_system_prompt, writer_human_prompt])
    report = writer_response.content

    final_report = f"""
    📄 FINAL REPORT
    {'='*50}
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    Topic: {task}
    {'='*50}

    {report}

    {'='*50}
    Report compiled by Multi-Agent AI System powered by ChatGPT"""

    return {
        "messages": [AIMessage(content=f"✍️ Writer: Report complete! See below for the full document.")],
        "next_agent": "supervisor",
        "final_report": final_report,
        "complete_task": True
    }
