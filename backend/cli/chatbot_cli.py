from langchain.schema.messages import HumanMessage, AIMessage
from backend.models.intent_classifier import classify_user_intent
from backend.models.normal_chatbot import handle_normal_chat
from backend.graphs.builder import builder

def run_chatbot():
    print("🤖 Terminal Chatbot (CLI) is now running. Type 'exit' to quit.\n")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            print("👋 Exiting chatbot...")
            break

        # Step 1: Intent classification
        classify_intent = {"messages": [HumanMessage(content=user_input)]}
        current_intent = classify_user_intent(classify_intent)
        requires_multi_agent = current_intent.get("next_agent")

        # Step 2: Route to Multi-agent or Normal Chat
        if requires_multi_agent == "supervisor":
            multi_agent_state = {
                "messages": [HumanMessage(content=user_input)],
                "research_data": "",
                "analysis": "",
                "final_report": "",
                "current_task": "",
                "next_agent": "",
                "complete_task": False,
                "next_node": "start"
            }
            result_state = builder.invoke(multi_agent_state)
            final_reply = next((m.content for m in result_state["messages"] if isinstance(m, AIMessage)), "")
            print(f"🤖 Agent: {final_reply}\n")
        else:
            response = handle_normal_chat(classify_intent)
            final_reply = response["messages"][-1].content
            print(f"Assistant: {final_reply}\n")
