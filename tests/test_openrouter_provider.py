import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.llm.providers.openrouter.OpenRouterProvider import OpenRouterProvider
from src.llm.schema.ChatConnectionError import ChatConnectionError
from src.llm.schema.ChatResponseError import ChatResponseError
from src.llm.schema.ChatTimeoutError import ChatTimeoutError
from src.llm.schema.Message import Message, MessageRole
from src.llm.schema.ToolCall import ToolCall
from src.tools.interfaces.Tool import Tool, ToolResult


@pytest.fixture
def provider():
    return OpenRouterProvider(
        model="test-model",
        base_url="https://openrouter.ai/api",
        api_key="sk-test-key",
    )


class TestFormatMessages:
    def test_simple_message(self, provider):
        msgs = [Message(role=MessageRole.USER, content="hello")]
        result = provider.format_messages(msgs)
        assert result == [{"role": "user", "content": "hello"}]

    def test_tool_call_sets_content_to_none(self, provider):
        tc = ToolCall(id="c1", name="tool", args={"k": "v"})
        msgs = [Message(role=MessageRole.ASSISTANT, content="", tool_calls=[tc])]
        result = provider.format_messages(msgs)
        assert result[0]["content"] is None
        assert len(result[0]["tool_calls"]) == 1
        assert result[0]["tool_calls"][0]["function"]["arguments"] == '{"k": "v"}'

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

    def test_empty_content_sets_none(self, provider):
        msgs = [Message(role=MessageRole.ASSISTANT, content="")]
        result = provider.format_messages(msgs)
        assert result[0]["content"] is None


class TestChat:
    @patch("httpx.AsyncClient")
    async def test_successful_response(self, mock_client_class, provider):
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello!",
                    }
                }
            ]
        }
        mock_client.post.return_value = mock_response

        result = await provider.chat(
            [Message(role=MessageRole.USER, content="hi")],
            [],
        )
        assert result.content == "Hello!"
        assert result.tool_calls == []

    @patch("httpx.AsyncClient")
    async def test_tool_call_with_json_args(self, mock_client_class, provider):
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "London"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        mock_client.post.return_value = mock_response

        result = await provider.chat(
            [Message(role=MessageRole.USER, content="weather?")],
            [],
        )
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].args == {"city": "London"}

    @patch("httpx.AsyncClient")
    async def test_no_choices_raises_error(self, mock_client_class, provider):
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}
        mock_client.post.return_value = mock_response

        with pytest.raises(ChatResponseError) as exc:
            await provider.chat(
                [Message(role=MessageRole.USER, content="hi")],
                [],
            )

    @patch("httpx.AsyncClient")
    async def test_timeout_raises_chat_timeout_error(self, mock_client_class, provider):
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.post.side_effect = httpx.TimeoutException("timeout")

        with pytest.raises(ChatTimeoutError):
            await provider.chat(
                [Message(role=MessageRole.USER, content="hi")],
                [],
            )

    @patch("httpx.AsyncClient")
    async def test_connection_error_raises_chat_connection_error(
        self, mock_client_class, provider
    ):
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.post.side_effect = httpx.ConnectError("refused")

        with pytest.raises(ChatConnectionError):
            await provider.chat(
                [Message(role=MessageRole.USER, content="hi")],
                [],
            )

    @patch("httpx.AsyncClient")
    async def test_non_200_status(self, mock_client_class, provider):
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": "unauthorized"}
        mock_client.post.return_value = mock_response

        with pytest.raises(ChatResponseError) as exc:
            await provider.chat(
                [Message(role=MessageRole.USER, content="hi")],
                [],
            )
        assert exc.value.status_code == 401

    @patch("httpx.AsyncClient")
    async def test_error_in_response_body(self, mock_client_class, provider):
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"error": "rate limited"}
        mock_client.post.return_value = mock_response

        with pytest.raises(ChatResponseError):
            await provider.chat(
                [Message(role=MessageRole.USER, content="hi")],
                [],
            )


class TestStreamChat:
    async def _async_iter(self, items):
        for item in items:
            yield item

    def _build_mock_stream(self, lines: list[str]):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = lambda: self._async_iter(lines)

        mock_stream = MagicMock()
        mock_stream.__aenter__.return_value = mock_response
        return mock_stream

    @patch("httpx.AsyncClient")
    async def test_streaming_content_chunks(self, mock_client_class, provider):
        mock_client = MagicMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.stream.return_value = self._build_mock_stream([
            "data: " + json.dumps({
                "choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]
            }),
            "data: " + json.dumps({
                "choices": [{"delta": {"content": " World"}, "finish_reason": None}]
            }),
            "data: " + json.dumps({
                "choices": [{"delta": {"content": ""}, "finish_reason": "stop"}]
            }),
        ])

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

    @patch("httpx.AsyncClient")
    async def test_skips_non_data_lines(self, mock_client_class, provider):
        mock_client = MagicMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.stream.return_value = self._build_mock_stream([
            ": keep-alive",
            "data: " + json.dumps({
                "choices": [{"delta": {"content": "Hi"}, "finish_reason": "stop"}]
            }),
        ])

        chunks = []
        async for chunk in provider.stream_chat([], []):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].content == "Hi"

    @patch("httpx.AsyncClient")
    async def test_tool_call_delta_accumulation(self, mock_client_class, provider):
        mock_client = MagicMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.stream.return_value = self._build_mock_stream([
            "data: " + json.dumps({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "get_weather", "arguments": ""},
                        }],
                    },
                    "finish_reason": None,
                }]
            }),
            "data: " + json.dumps({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": '{"city": '},
                        }],
                    },
                    "finish_reason": None,
                }]
            }),
            "data: " + json.dumps({
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": '"London"}'},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }]
            }),
        ])

        chunks = []
        async for chunk in provider.stream_chat([], []):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[2].done is True
        assert len(chunks[2].tool_calls) == 1
        assert chunks[2].tool_calls[0].name == "get_weather"
        assert chunks[2].tool_calls[0].args == {"city": "London"}

    @patch("httpx.AsyncClient")
    async def test_multiple_tool_indices(self, mock_client_class, provider):
        mock_client = MagicMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.stream.return_value = self._build_mock_stream([
            "data: " + json.dumps({
                "choices": [{
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "c1", "function": {"name": "tool_a", "arguments": '{"x": 1}'}},
                            {"index": 1, "id": "c2", "function": {"name": "tool_b", "arguments": '{"y": 2}'}},
                        ],
                    },
                    "finish_reason": "tool_calls",
                }]
            }),
        ])

        chunks = []
        async for chunk in provider.stream_chat([], []):
            chunks.append(chunk)

        assert len(chunks[0].tool_calls) == 2
        assert chunks[0].tool_calls[0].name == "tool_a"
        assert chunks[0].tool_calls[1].name == "tool_b"

    @patch("httpx.AsyncClient")
    async def test_error_in_stream_response(self, mock_client_class, provider):
        mock_client = MagicMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.stream.return_value = self._build_mock_stream([
            "data: " + json.dumps({"error": "rate limited"}),
        ])

        with pytest.raises(ChatResponseError):
            async for _ in provider.stream_chat([], []):
                pass

    @patch("httpx.AsyncClient")
    async def test_skips_empty_choices(self, mock_client_class, provider):
        mock_client = MagicMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.stream.return_value = self._build_mock_stream([
            "data: " + json.dumps({"choices": []}),
            "data: " + json.dumps({
                "choices": [{"delta": {"content": "final"}, "finish_reason": "stop"}]
            }),
        ])

        chunks = []
        async for chunk in provider.stream_chat([], []):
            chunks.append(chunk)

        assert len(chunks) == 1
