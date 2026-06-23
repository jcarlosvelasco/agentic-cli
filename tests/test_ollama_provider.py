import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.llm.providers.ollama.OllamaProvider import OllamaProvider
from src.llm.schema.ChatConnectionError import ChatConnectionError
from src.llm.schema.ChatResponseError import ChatResponseError
from src.llm.schema.ChatTimeoutError import ChatTimeoutError
from src.llm.schema.Message import Message, MessageRole
from src.llm.schema.ToolCall import ToolCall



@pytest.fixture
def provider():
    return OllamaProvider(model="test-model", base_url="http://localhost:11434/api")


from tests.conftest import MockTool


@pytest.fixture
def mock_tool():
    return MockTool(name="get_weather")


class MockArgsTool(MockTool):
    def __init__(self):
        super().__init__(name="search")


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
        assert result[0]["tool_calls"][0]["function"]["arguments"] == {"k": "v"}

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


class TestGetOllamaSchema:
    def test_tool_without_args_schema(self, provider, mock_tool):
        schema = provider.get_ollama_schema(mock_tool)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"
        assert "parameters" not in schema["function"]

    def test_tool_with_args_schema_but_no_json_schema(self, provider):
        tool = MockArgsTool()
        schema = provider.get_ollama_schema(tool)
        assert schema["function"]["name"] == "search"


class TestChat:
    @patch("httpx.AsyncClient")
    async def test_successful_response(self, mock_client_class, provider):
        mock_client = MagicMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "Hello!",
            }
        }
        mock_client.post = AsyncMock(return_value=mock_response)

        result = await provider.chat(
            [Message(role=MessageRole.USER, content="hi")],
            [],
        )
        assert result.content == "Hello!"
        assert result.tool_calls == []

    @patch("httpx.AsyncClient")
    async def test_response_with_tool_calls(self, mock_client_class, provider):
        mock_client = MagicMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "London"},
                        },
                    }
                ],
            }
        }
        mock_client.post = AsyncMock(return_value=mock_response)

        result = await provider.chat(
            [Message(role=MessageRole.USER, content="weather?")],
            [],
        )
        assert result.content == ""
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].args == {"city": "London"}

    @patch("httpx.AsyncClient")
    async def test_timeout_raises_chat_timeout_error(self, mock_client_class, provider):
        mock_client = MagicMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

        with pytest.raises(ChatTimeoutError):
            await provider.chat(
                [Message(role=MessageRole.USER, content="hi")],
                [],
            )

    @patch("httpx.AsyncClient")
    async def test_connection_error_raises_chat_connection_error(
        self, mock_client_class, provider
    ):
        mock_client = MagicMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with pytest.raises(ChatConnectionError):
            await provider.chat(
                [Message(role=MessageRole.USER, content="hi")],
                [],
            )

    @patch("httpx.AsyncClient")
    async def test_non_200_status_code(self, mock_client_class, provider):
        mock_client = MagicMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "bad request"}
        mock_client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(ChatResponseError) as exc:
            await provider.chat(
                [Message(role=MessageRole.USER, content="hi")],
                [],
            )
        assert exc.value.status_code == 400

    @patch("httpx.AsyncClient")
    async def test_error_in_response_body(self, mock_client_class, provider):
        mock_client = MagicMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"error": "model not found"}
        mock_client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(ChatResponseError) as exc:
            await provider.chat(
                [Message(role=MessageRole.USER, content="hi")],
                [],
            )

    @patch("httpx.AsyncClient")
    async def test_invalid_json_response(self, mock_client_class, provider):
        mock_client = MagicMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("bad json")
        mock_client.post = AsyncMock(return_value=mock_response)

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
            json.dumps({"message": {"role": "assistant", "content": "Hello"}}),
            json.dumps({"message": {"role": "assistant", "content": " World"}}),
            json.dumps({"message": {"role": "assistant", "content": "", "done": "true"}}),
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
    async def test_stream_with_tool_calls(self, mock_client_class, provider):
        mock_client = MagicMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.stream.return_value = self._build_mock_stream([
            json.dumps({
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "tool", "arguments": {"k": "v"}},
                        }
                    ],
                    "done": "true",
                }
            }),
        ])

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

    @patch("httpx.AsyncClient")
    async def test_stream_non_200_error(self, mock_client_class, provider):
        mock_client = MagicMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.aiter_lines = lambda: self._async_iter([
            json.dumps({"error": "server error"}),
        ])
        mock_stream = MagicMock()
        mock_stream.__aenter__.return_value = mock_response
        mock_client.stream.return_value = mock_stream

        with pytest.raises(ChatResponseError):
            async for _ in provider.stream_chat([], []):
                pass

    @patch("httpx.AsyncClient")
    async def test_stream_invalid_json(self, mock_client_class, provider):
        mock_client = MagicMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        mock_client.stream.return_value = self._build_mock_stream([
            "not-json",
        ])

        with pytest.raises(ChatResponseError):
            async for _ in provider.stream_chat([], []):
                pass
