from typing import Any, List, TypedDict


class OllamaToolCallFunction(TypedDict):
    name: str
    arguments: dict[str, Any]


class OllamaToolCall(TypedDict):
    id: str
    function: OllamaToolCallFunction


class OllamaMessage(TypedDict):
    role: str
    content: str
    tools: List[OllamaToolCall]
    tool_call_id: str
