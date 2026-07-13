import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)

from src.llm.providers.ollama.OllamaProvider import OllamaProvider
from src.llm.schema.ChatConnectionError import ChatConnectionError
from src.llm.schema.ChatResponseError import ChatResponseError
from src.llm.schema.ChatTimeoutError import ChatTimeoutError
from src.llm.schema.Message import Message, MessageRole
from src.llm.schema.ToolCall import ToolCall
from tests.conftest import MockTool


@pytest.fixture
def provider():
    return OllamaProvider(model="test-model", base_url="http://localhost:11434/v1")


@pytest.fixture
def mock_tool():
    return MockTool(name="get_weather")


class MockArgsTool(MockTool):
    def __init__(self):
        super().__init__(name="search")


from typing import AsyncIterator, List


async def _async_iter(items: list) -> AsyncIterator:
    for item in items:
        yield item


def _make_completion(content="Hello!", tool_calls=None):
    return ChatCompletion(
        id="chatcmpl-123",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content=content,
                    role="assistant",
                    tool_calls=tool_calls,
                ),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )


def _make_tool_call(id: str, name: str, arguments: str):
    return ChatCompletionMessageFunctionToolCall(
        id=id,
        function=Function(name=name, arguments=arguments),
        type="function",
    )


def _make_chunk(content="Hello", finish_reason=None, tool_calls=None):
    return ChatCompletionChunk(
        id="chatcmpl-123",
        choices=[
            ChunkChoice(
                delta=ChoiceDelta(
                    content=content,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
                index=0,
                logprobs=None,
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion.chunk",
    )


class TestFormatMessages:
    def test_simple_message(self, provider):
        msgs = [Message(role=MessageRole.USER, content="hello")]
        result = provider.format_messages(msgs)
        assert result == [{"role": "user", "content": "hello"}]

    def test_tool_calls_in_message(self, provider):
        tc = ToolCall(id="c1", name="tool", args={"k": "v"})
        msgs = [Message(role=MessageRole.ASSISTANT, content="", tool_calls=[tc])]
        result = provider.format_messages(msgs)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == ""
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["id"] == "c1"
        assert result[0]["tool_calls"][0]["function"]["name"] == "tool"
        assert result[0]["tool_calls"][0]["function"]["arguments"] == json.dumps(
            {"k": "v"}
        )

    def test_tool_call_id(self, provider):
        msgs = [Message(role=MessageRole.TOOL, content="result", tool_call_id="c1")]
        result = provider.format_messages(msgs)
        assert result[0]["tool_call_id"] == "c1"

    def test_mixed_roles(self, provider):
        msgs = [
            Message(role=MessageRole.SYSTEM, content="sys"),
            Message(role=MessageRole.USER, content="user"),
            Message(role=MessageRole.ASSISTANT, content="assistant"),
            Message(role=MessageRole.TOOL, content="tool", tool_call_id="c1"),
        ]
        result = provider.format_messages(msgs)
        assert len(result) == 4
        assert result[2]["role"] == "assistant"
        assert result[3]["tool_call_id"] == "c1"


class TestGetOpenaiSchema:
    def test_tool_without_args_schema(self, provider, mock_tool):
        schema = provider.get_openai_schema(mock_tool)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"
        assert "parameters" not in schema["function"]

    def test_tool_with_args_schema_but_no_json_schema(self, provider):
        tool = MockArgsTool()
        schema = provider.get_openai_schema(tool)
        assert schema["function"]["name"] == "search"


class TestChat:
    @patch("openai.resources.chat.completions.AsyncCompletions.create")
    async def test_successful_response(self, mock_create, provider):
        mock_create.return_value = AsyncMock(
            return_value=_make_completion(content="Hello!")
        )()

        result = await provider.chat(
            [Message(role=MessageRole.USER, content="hi")],
            [],
        )
        assert result.content == "Hello!"
        assert result.tool_calls == []

    @patch("openai.resources.chat.completions.AsyncCompletions.create")
    async def test_response_with_tool_calls(self, mock_create, provider):
        mock_create.return_value = AsyncMock(
            return_value=_make_completion(
                content="",
                tool_calls=[
                    _make_tool_call("call_123", "get_weather", '{"city": "London"}')
                ],
            )
        )()

        result = await provider.chat(
            [Message(role=MessageRole.USER, content="weather?")],
            [],
        )
        assert result.content == ""
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].args == {"city": "London"}

    @patch("openai.resources.chat.completions.AsyncCompletions.create")
    async def test_timeout_raises_chat_timeout_error(self, mock_create, provider):
        mock_create.return_value = AsyncMock(
            side_effect=APITimeoutError(request=MagicMock())
        )()

        with pytest.raises(ChatTimeoutError):
            await provider.chat(
                [Message(role=MessageRole.USER, content="hi")],
                [],
            )

    @patch("openai.resources.chat.completions.AsyncCompletions.create")
    async def test_connection_error_raises_chat_connection_error(
        self, mock_create, provider
    ):
        mock_create.return_value = AsyncMock(
            side_effect=APIConnectionError(request=MagicMock())
        )()

        with pytest.raises(ChatConnectionError):
            await provider.chat(
                [Message(role=MessageRole.USER, content="hi")],
                [],
            )

    @patch("openai.resources.chat.completions.AsyncCompletions.create")
    async def test_non_200_status_code(self, mock_create, provider):
        response = MagicMock()
        response.status_code = 400
        response.headers = {}
        response.json.return_value = {"error": "bad request"}
        mock_create.return_value = AsyncMock(
            side_effect=APIStatusError(
                message="bad request",
                response=response,
                body={"error": "bad request"},
            )
        )()

        with pytest.raises(ChatResponseError) as exc:
            await provider.chat(
                [Message(role=MessageRole.USER, content="hi")],
                [],
            )
        assert exc.value.status_code == 400


class TestStreamChat:
    @patch("openai.resources.chat.completions.AsyncCompletions.create")
    async def test_streaming_content_chunks(self, mock_create, provider):
        chunks_list = [
            _make_chunk(content="Hello", finish_reason=None),
            _make_chunk(content=" World", finish_reason=None),
            _make_chunk(content="", finish_reason="stop"),
        ]
        mock_create.return_value = AsyncMock(
            return_value=_async_iter(chunks_list)
        )()

        chunks = []
        async for chunk in provider.stream_chat(
            [Message(role=MessageRole.USER, content="hi")],
            [],
        ):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].content == "Hello"
        assert chunks[0].done is False
        assert chunks[1].content == " World"
        assert chunks[2].content == ""
        assert chunks[2].done is True

    @patch("openai.resources.chat.completions.AsyncCompletions.create")
    async def test_stream_with_tool_calls(self, mock_create, provider):
        chunks_list = [
            _make_chunk(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id="c1",
                        function=ChoiceDeltaToolCallFunction(
                            name="tool",
                            arguments=json.dumps({"k": "v"}),
                        ),
                        type="function",
                    )
                ],
            )
        ]
        mock_create.return_value = AsyncMock(
            return_value=_async_iter(chunks_list)
        )()

        chunks = []
        async for chunk in provider.stream_chat(
            [Message(role=MessageRole.USER, content="hi")],
            [],
        ):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].done is True
        assert len(chunks[0].tool_calls) == 1
        assert chunks[0].tool_calls[0].name == "tool"
        assert chunks[0].tool_calls[0].args == {"k": "v"}

    @patch("openai.resources.chat.completions.AsyncCompletions.create")
    async def test_stream_connection_error(self, mock_create, provider):
        mock_create.return_value = AsyncMock(
            side_effect=APIConnectionError(request=MagicMock())
        )()

        with pytest.raises(ChatConnectionError):
            async for _ in provider.stream_chat([], []):
                pass

    @patch("openai.resources.chat.completions.AsyncCompletions.create")
    async def test_stream_timeout_error(self, mock_create, provider):
        mock_create.return_value = AsyncMock(
            side_effect=APITimeoutError(request=MagicMock())
        )()

        with pytest.raises(ChatTimeoutError):
            async for _ in provider.stream_chat([], []):
                pass
