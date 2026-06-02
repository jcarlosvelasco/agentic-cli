from typing import List

from pydantic.main import BaseModel

from src.llm.schema.ToolCall import ToolCall


class LLMChatResponse(BaseModel):
    content: str
    tool_calls: List[ToolCall] = []

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0
