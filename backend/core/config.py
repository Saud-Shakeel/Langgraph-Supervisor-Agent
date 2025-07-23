from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(override=True)
llm = ChatOpenAI(model_name="gpt-4o-mini")
