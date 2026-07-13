import json
from typing import AsyncIterator
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

from src.llm.providers.openrouter.OpenRouterProvider import OpenRouterProvider
from src.llm.schema.ChatConnectionError import ChatConnectionError
from src.llm.schema.ChatResponseError import ChatResponseError
from src.llm.schema.ChatTimeoutError import ChatTimeoutError
from src.llm.schema.Message import Message, MessageRole
from src.llm.schema.ToolCall import ToolCall


async def _async_iter(items: list) -> AsyncIterator:
    for item in items:
        yield item


@pytest.fixture
def provider():
    return OpenRouterProvider(
        model="test-model",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-test-key",
    )


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

    def test_tool_call_sets_content(self, provider):
        tc = ToolCall(id="c1", name="tool", args={"k": "v"})
        msgs = [Message(role=MessageRole.ASSISTANT, content="", tool_calls=[tc])]
        result = provider.format_messages(msgs)
        assert result[0]["content"] == ""
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["arguments"] == json.dumps(
            {"k": "v"}
        )

    def test_tool_call_serializes_args_to_json_string(self, provider):
        tc = ToolCall(id="c1", name="tool", args={"nested": {"a": 1}})
        msgs = [Message(role=MessageRole.ASSISTANT, content="", tool_calls=[tc])]
        result = provider.format_messages(msgs)
        args_str = result[0]["tool_calls"][0]["function"]["arguments"]
        assert json.loads(args_str) == {"nested": {"a": 1}}

    def test_tool_call_id(self, provider):
        msgs = [Message(role=MessageRole.TOOL, content="result", tool_call_id="c1")]
        result = provider.format_messages(msgs)
        assert result[0]["tool_call_id"] == "c1"
        assert result[0]["content"] == "result"

    def test_empty_content_keeps_string(self, provider):
        msgs = [Message(role=MessageRole.ASSISTANT, content="")]
        result = provider.format_messages(msgs)
        assert result[0]["content"] == ""


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
    async def test_tool_call_with_json_args(self, mock_create, provider):
        mock_create.return_value = AsyncMock(
            return_value=_make_completion(
                content=None,
                tool_calls=[
                    _make_tool_call("call_1", "get_weather", '{"city": "London"}')
                ],
            )
        )()

        result = await provider.chat(
            [Message(role=MessageRole.USER, content="weather?")],
            [],
        )
        assert len(result.tool_calls) == 1
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
    async def test_non_200_status(self, mock_create, provider):
        response = MagicMock()
        response.status_code = 401
        response.headers = {}
        response.json.return_value = {"error": "unauthorized"}
        mock_create.return_value = AsyncMock(
            side_effect=APIStatusError(
                message="unauthorized",
                response=response,
                body={"error": "unauthorized"},
            )
        )()

        with pytest.raises(ChatResponseError) as exc:
            await provider.chat(
                [Message(role=MessageRole.USER, content="hi")],
                [],
            )
        assert exc.value.status_code == 401


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
    async def test_tool_call_delta_accumulation(self, mock_create, provider):
        chunks_list = [
            _make_chunk(
                content=None,
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id="call_1",
                        function=ChoiceDeltaToolCallFunction(
                            name="get_weather",
                            arguments="",
                        ),
                        type="function",
                    )
                ],
            ),
            _make_chunk(
                content=None,
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        function=ChoiceDeltaToolCallFunction(
                            arguments='{"city": ',
                        ),
                    )
                ],
            ),
            _make_chunk(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        function=ChoiceDeltaToolCallFunction(
                            arguments='"London"}',
                        ),
                    )
                ],
            ),
        ]
        mock_create.return_value = AsyncMock(
            return_value=_async_iter(chunks_list)
        )()

        chunks = []
        async for chunk in provider.stream_chat([], []):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[2].done is True
        assert len(chunks[2].tool_calls) == 1
        assert chunks[2].tool_calls[0].name == "get_weather"
        assert chunks[2].tool_calls[0].args == {"city": "London"}

    @patch("openai.resources.chat.completions.AsyncCompletions.create")
    async def test_multiple_tool_indices(self, mock_create, provider):
        chunks_list = [
            _make_chunk(
                content=None,
                finish_reason="tool_calls",
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id="c1",
                        function=ChoiceDeltaToolCallFunction(
                            name="tool_a",
                            arguments='{"x": 1}',
                        ),
                        type="function",
                    ),
                    ChoiceDeltaToolCall(
                        index=1,
                        id="c2",
                        function=ChoiceDeltaToolCallFunction(
                            name="tool_b",
                            arguments='{"y": 2}',
                        ),
                        type="function",
                    ),
                ],
            )
        ]
        mock_create.return_value = AsyncMock(
            return_value=_async_iter(chunks_list)
        )()

        chunks = []
        async for chunk in provider.stream_chat([], []):
            chunks.append(chunk)

        assert len(chunks[0].tool_calls) == 2
        assert chunks[0].tool_calls[0].name == "tool_a"
        assert chunks[0].tool_calls[1].name == "tool_b"

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
