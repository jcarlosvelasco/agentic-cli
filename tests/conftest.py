import asyncio
import sys
from pathlib import Path
from typing import Any, AsyncIterator, List
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.interfaces.BaseLLMProvider import BaseLLMProvider
from src.llm.interfaces.StreamLLMChatResponse import StreamLLMChatResponse
from src.llm.schema.LLMChatResponse import LLMChatResponse
from src.llm.schema.Message import Message, MessageRole
from src.llm.schema.ToolCall import ToolCall
from src.tools.interfaces.Tool import Tool, ToolResult


class MockTool(Tool):
    def __init__(self, name: str, result: ToolResult | None = None):
        super().__init__(name=name, description=f"A mock tool: {name}")
        self._result = result or ToolResult(success=True, data=f"{name}_result")

    async def execute(self, args: dict[str, Any] | None) -> ToolResult:
        return self._result


class MockProvider(BaseLLMProvider[Any]):
    def __init__(
        self,
        chat_responses: list[LLMChatResponse] | None = None,
        stream_responses: list[list[StreamLLMChatResponse]] | None = None,
    ):
        self.chat_responses = chat_responses or []
        self.stream_responses = stream_responses or []
        self.chat_calls: list[tuple] = []
        self.stream_chat_calls: list[tuple] = []

    async def chat(
        self, messages: List[Message], tools: List[Tool], temperature: float = 0.0
    ) -> LLMChatResponse:
        self.chat_calls.append((messages, tools, temperature))
        if self.chat_responses:
            return self.chat_responses.pop(0)
        return LLMChatResponse(content="mock response")

    async def stream_chat(
        self, messages: List[Message], tools: List[Tool], temperature: float = 0.0
    ) -> AsyncIterator[StreamLLMChatResponse]:
        self.stream_chat_calls.append((messages, tools, temperature))
        responses = self.stream_responses.pop(0) if self.stream_responses else []
        for chunk in responses:
            yield chunk

    def format_messages(self, messages: List[Message]) -> list[dict[str, Any]]:
        return [{"role": m.role.value, "content": m.content} for m in messages]


@pytest.fixture
def sample_messages() -> list[Message]:
    return [
        Message(role=MessageRole.SYSTEM, content="You are a helpful AI."),
        Message(role=MessageRole.USER, content="Hello"),
        Message(role=MessageRole.ASSISTANT, content="Hi there!"),
        Message(role=MessageRole.USER, content="What is Python?"),
        Message(role=MessageRole.ASSISTANT, content="Python is a programming language."),
    ]


@pytest.fixture
def tool_call() -> ToolCall:
    return ToolCall(id="call_1", name="mock_tool", args={"query": "test"})


@pytest.fixture
def mock_tool() -> MockTool:
    return MockTool(name="mock_tool")


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
