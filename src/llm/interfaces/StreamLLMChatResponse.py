from typing import List

from pydantic import BaseModel

from src.llm.schema.ToolCall import ToolCall


class StreamLLMChatResponse(BaseModel):
    content: str | None = None
    tool_calls: List[ToolCall] = []
    done: bool = False
    input_token_count: int = 0
    output_token_count: int = 0
