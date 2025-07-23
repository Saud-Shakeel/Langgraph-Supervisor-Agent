from backend.schemas.state_schema import State
from langchain.schema.messages import AIMessage, SystemMessage, HumanMessage
from typing import Dict
from langchain.prompts import ChatPromptTemplate
from backend.core.config import llm


def classify_user_intent(state:State)->Dict:
    user_input = state["messages"][-1].content if state["messages"] else ""

    classifier_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="""
            You are a classifier that determines if a user's request requires a multi-agent research system.

            Return "MULTI_AGENT" if the request involves:
            - Research on a specific topic
            - Analysis of data, trends, or phenomena
            - Creating reports or comprehensive summaries
            - Gathering information about companies, markets, technologies, etc.
            - Comparative studies or investigations
            - Any task requiring structured research → analysis → reporting workflow

            Return "NORMAL_CHAT" if the request is:
            - General questions or conversations
            - Simple explanations or definitions
            - Coding help or technical questions
            - Personal advice or opinions
            - Mathematical calculations
            - Creative writing requests
            - Casual conversation

            Respond with ONLY "MULTI_AGENT" or "NORMAL_CHAT".
            """
        ),
        HumanMessage(content=f"Classify this request: {user_input}")
    ])

    response = llm.invoke(classifier_prompt.format_messages())
    decision = response.content.strip().upper()

    if "MULTI_AGENT" in decision:
        next_node = "supervisor"
        classifier_msg = "🔄 Intent Classifier: Detected research/analysis request. Routing to multi-agent system..."
    else:
        next_node = "normal_chatbot"
        classifier_msg = "💬 Intent Classifier: Routing to normal chat..."

    return {
        "messages": [AIMessage(content=classifier_msg)],
        "next_agent": next_node,
        "current_task": user_input
    }