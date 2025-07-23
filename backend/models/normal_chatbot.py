from langchain.schema.messages import AIMessage, SystemMessage, HumanMessage
from typing import Dict
from backend.schemas.state_schema import State
from langchain.prompts import ChatPromptTemplate
from backend.core.config import llm

def handle_normal_chat(state:State)->Dict:
    user_input = state.get("current_task", "") or state["messages"][-1].content if state["messages"] else ""

    normal_chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="""
            You are a helpful AI assistant. Provide clear, accurate, and helpful responses to user queries.
            Be conversational and engaging while maintaining professionalism.
            If you do not know the answer or do not have the real-time information, answer accordingly. 
            """
        ),
        HumanMessage(content=user_input)
    ])

    response = llm.invoke(normal_chat_prompt.format_messages())
    return {
        "messages": [AIMessage(content=response.content)],
        "task_complete": True,
        "next_agent": "end"
    }