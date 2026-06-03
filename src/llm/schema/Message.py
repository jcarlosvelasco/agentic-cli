from enum import StrEnum
from typing import List

from pydantic import BaseModel

from src.llm.schema.ToolCall import ToolCall


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    role: MessageRole
    content: str
    tool_calls: List[ToolCall] = []
    tool_call_id: str | None = None
