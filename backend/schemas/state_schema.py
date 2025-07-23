from typing import TypedDict, List
from langchain.schema.messages import AnyMessage

class State(TypedDict):
    messages: List[AnyMessage]
    research_data:str = ""
    analysis:str = ""
    final_report:str = ""
    current_task:str = ""
    next_agent:str = ""
    complete_task:bool = False,
    next_node: str = ""