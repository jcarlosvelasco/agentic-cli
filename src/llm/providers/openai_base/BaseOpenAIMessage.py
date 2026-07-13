from typing import List, NotRequired, TypedDict


class BaseOpenAIToolCallFunction(TypedDict):
    name: str
    arguments: str


class BaseOpenAIToolCall(TypedDict):
    id: str
    type: str
    function: BaseOpenAIToolCallFunction


class BaseOpenAIMessage(TypedDict):
    role: str
    content: NotRequired[str | None]
    tool_calls: NotRequired[List[BaseOpenAIToolCall]]
    tool_call_id: NotRequired[str]
