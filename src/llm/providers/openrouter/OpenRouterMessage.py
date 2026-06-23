from typing import List, NotRequired, TypedDict


class OpenRouterToolCallFunction(TypedDict):
    name: str
    arguments: str


class OpenRouterToolCall(TypedDict):
    id: str
    type: str
    function: OpenRouterToolCallFunction


class OpenRouterMessage(TypedDict):
    role: str
    content: NotRequired[str | None]
    tool_calls: NotRequired[List[OpenRouterToolCall]]
    tool_call_id: NotRequired[str]
