import json
from typing import Any, AsyncIterator, List

import httpx

from src.llm.interfaces.BaseLLMProvider import BaseLLMProvider, StreamLLMChatResponse
from src.llm.providers.ollama.OllamaMessage import OllamaMessage
from src.llm.schema.ChatConnectionError import ChatConnectionError
from src.llm.schema.ChatResponseError import ChatResponseError
from src.llm.schema.ChatTimeoutError import ChatTimeoutError
from src.llm.schema.LLMChatResponse import LLMChatResponse
from src.llm.schema.Message import Message
from src.llm.schema.ToolCall import ToolCall
from src.tools.interfaces.Tool import Tool


class OllamaProvider(BaseLLMProvider):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/api",
    ):
        self.model = model
        self.base_url = base_url

    def format_messages(
        self,
        messages: List[Message],
    ) -> List[OllamaMessage]:
        result: List[OllamaMessage] = []
        for message in messages:
            msg: OllamaMessage = {
                "role": message.role.value,
                "content": message.content,
            }
            if message.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "function": {"name": tc.name, "arguments": tc.args},
                    }
                    for tc in message.tool_calls
                ]
            if message.tool_call_id:
                msg["tool_call_id"] = message.tool_call_id

            result.append(msg)

        return result

    def get_ollama_schema(self, tool: Tool) -> dict[str, Any]:
        schema = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
            },
        }

        if tool.args_schema:
            json_schema = tool.args_schema.model_json_schema()
            schema["function"]["parameters"] = json_schema

        return schema

    async def chat(
        self, messages: List[Message], tools: List[Tool], temperature: float = 0.0
    ) -> LLMChatResponse:
        payload = {
            "model": self.model,
            "messages": self.format_messages(messages),
            "stream": False,
            "options": {
                "temperature": temperature,
            },
            "tools": [self.get_ollama_schema(tool) for tool in tools],
        }
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{self.base_url}/chat",
                    json=payload,
                )

        except httpx.TimeoutException as e:
            raise ChatTimeoutError("Request timeout") from e

        except httpx.ConnectError as e:
            raise ChatConnectionError("Cannot connect to Ollama") from e

        try:
            data = response.json()
        except Exception as e:
            raise ChatResponseError(
                "Invalid JSON response",
                response.status_code,
            ) from e

        if response.status_code != 200:
            raise ChatResponseError(
                data.get("error", "Unknown error"),
                response.status_code,
            )

        if "error" in data:
            raise ChatResponseError(
                data["error"],
                response.status_code,
            )

        message = data["message"]

        tool_calls: List[ToolCall] = []
        for call in message.get("tool_calls", []):
            tool_calls.append(
                ToolCall(
                    id=call["id"],
                    name=call["function"]["name"],
                    args=call["function"]["arguments"],
                )
            )

        return LLMChatResponse(
            content=data["message"]["content"], tool_calls=tool_calls
        )

    async def stream_chat(
        self, messages: List[Message], tools: List[Tool], temperature: float = 0.0
    ) -> AsyncIterator[StreamLLMChatResponse]:
        payload = {
            "model": self.model,
            "messages": self.format_messages(messages),
            "stream": True,
            "options": {
                "temperature": temperature,
            },
            "tools": [self.get_ollama_schema(tool) for tool in tools],
        }

        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST", f"{self.base_url}/chat", json=payload
            ) as response:
                async for line in response.aiter_lines():
                    try:
                        data = json.loads(line)
                    except Exception as e:
                        raise ChatResponseError(
                            "Invalid JSON response",
                            response.status_code,
                        ) from e

                    if response.status_code != 200:
                        raise ChatResponseError(
                            data.get("error", "Unknown error"),
                            response.status_code,
                        )

                    if "error" in data:
                        raise ChatResponseError(
                            data["error"],
                            response.status_code,
                        )

                    message = data["message"]

                    tool_calls: List[ToolCall] = []
                    for call in message.get("tool_calls", []):
                        tool_calls.append(
                            ToolCall(
                                id=call["id"],
                                name=call["function"]["name"],
                                args=call["function"]["arguments"],
                            )
                        )

                    if message.get("done") == "true":
                        yield StreamLLMChatResponse(
                            done=True,
                            content=message.get("content"),
                            tool_calls=tool_calls,
                        )
                        break
                    else:
                        yield StreamLLMChatResponse(
                            done=False,
                            content=message.get("content"),
                            tool_calls=tool_calls,
                        )
