from pydantic import BaseModel, Field
from typing import Optional

class ChatRequest(BaseModel):
    message: str = Field(...,description="Query that the user enters to the LLM.")

class ChatResponse(BaseModel):
    reply: str = Field(...,description="Response from the LLM to the user.")
    report: Optional[str] = Field(default=None, description="""Goes through all the AI Agents, creates a final report
    for the required task, and returns it.""")