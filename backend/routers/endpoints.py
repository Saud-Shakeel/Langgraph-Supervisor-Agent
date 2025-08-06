from fastapi import APIRouter
from backend.schemas.chat_schema import ChatRequest, ChatResponse
from backend.graphs.builder import builder
from backend.models.intent_classifier import classify_user_intent
from backend.models.normal_chatbot import handle_normal_chat
from backend.cli.chatbot_cli import run_chatbot
from langchain.schema.messages import HumanMessage, AIMessage

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat_router(request: ChatRequest):
    user_input = request.message
    classify_intent = {"messages": [HumanMessage(content=user_input)]}
    current_intent = classify_user_intent(classify_intent)
    requires_multi_agent = current_intent.get("next_agent")

    if requires_multi_agent == "supervisor":
        multi_agent_state = {
            "research_data": "",
            "analysis": "",
            "final_report": "",
            "current_task": "",
            "next_agent": "",
            "complete_task": False,
            "messages": [HumanMessage(content=user_input)],
            "next_node": "start"
        }
        result_state = builder.invoke(multi_agent_state)
        final_reply = next((m.content for m in result_state["messages"] if isinstance(m, AIMessage)), "")
        return ChatResponse(reply=final_reply, report=result_state.get("final_report"))
    else:
        normal_intent = {"messages": [HumanMessage(content=user_input)]}
        response = handle_normal_chat(normal_intent)
        return ChatResponse(reply=response["messages"][-1].content)

@router.get("/")
def cli_router():
    run_chatbot()
    return {"message": "✅ CLI chatbot has been activated in terminal."}


