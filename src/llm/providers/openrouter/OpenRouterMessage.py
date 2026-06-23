from typing import Any, List, NotRequired, TypedDict


class OpenRouterToolCallFunction(TypedDict):
    name: str
    arguments: dict[str, Any]


class OpenRouterToolCall(TypedDict):
    id: str
    function: OpenRouterToolCallFunction


class OpenRouterMessage(TypedDict):
    role: str
    content: str
    tool_calls: NotRequired[List[OpenRouterToolCall]]
    tool_call_id: NotRequired[str]
