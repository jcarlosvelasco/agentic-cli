import json
from this import s
from typing import Any, List

import httpx
from typing_extensions import AsyncIterator

from llm.interfaces.BaseLLMProvider import BaseLLMProvider
from llm.interfaces.StreamLLMChatResponse import StreamLLMChatResponse
from llm.providers.openrouter.OpenRouterMessage import OpenRouterMessage
from llm.schema.ChatConnectionError import ChatConnectionError
from llm.schema.ChatResponseError import ChatResponseError
from llm.schema.ChatTimeoutError import ChatTimeoutError
from llm.schema.LLMChatResponse import LLMChatResponse
from llm.schema.Message import Message
from llm.schema.ToolCall import ToolCall
from tools.interfaces.Tool import Tool


class OpenRouterProvider(BaseLLMProvider):
    def __init__(self, model: str, base_url: str, api_key: str):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    def format_messages(
        self,
        messages: List[Message],
    ) -> List[OpenRouterMessage]:
        result: List[OpenRouterMessage] = []
        for message in messages:
            msg: OpenRouterMessage = {
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

    def get_openrouter_schema(self, tool: Tool) -> dict[str, Any]:
        schema = {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": {},
        }

        if tool.args_schema:
            json_schema = tool.args_schema.model_json_schema()
            schema["parameters"] = json_schema

        return schema

    async def chat(
        self, messages: List[Message], tools: List[Tool], temperature: float = 0.0
    ) -> LLMChatResponse:
        payload = {
            "model": self.model,
            "messages": self.format_messages(messages),
            "stream": False,
            "temperature": temperature,
            "tools": [self.get_openrouter_schema(tool) for tool in tools],
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
            "temperature": temperature,
            "tools": [self.get_openrouter_schema(tool) for tool in tools],
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
