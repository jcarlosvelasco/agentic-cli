from typing import List

from pydantic import BaseModel

from src.llm.schema.ToolCall import ToolCall


class StreamLLMChatResponse(BaseModel):
    content: str | None = None
    tool_calls: List[ToolCall] = []
    done: bool = False
