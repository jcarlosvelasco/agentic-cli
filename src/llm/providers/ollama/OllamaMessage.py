from typing import Any, List, NotRequired, TypedDict


class OllamaToolCallFunction(TypedDict):
    name: str
    arguments: dict[str, Any]


class OllamaToolCall(TypedDict):
    id: str
    function: OllamaToolCallFunction


class OllamaMessage(TypedDict):
    role: str
    content: str
    tool_calls: NotRequired[List[OllamaToolCall]]
    tool_call_id: NotRequired[str]
