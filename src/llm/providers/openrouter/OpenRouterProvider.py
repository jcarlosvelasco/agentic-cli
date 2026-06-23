import json
from typing import Any, List

import httpx
from typing_extensions import AsyncIterator

from src.llm.interfaces.BaseLLMProvider import BaseLLMProvider
from src.llm.interfaces.StreamLLMChatResponse import StreamLLMChatResponse
from src.llm.providers.openrouter.OpenRouterMessage import OpenRouterMessage
from src.llm.schema.ChatConnectionError import ChatConnectionError
from src.llm.schema.ChatResponseError import ChatResponseError
from src.llm.schema.ChatTimeoutError import ChatTimeoutError
from src.llm.schema.LLMChatResponse import LLMChatResponse
from src.llm.schema.Message import Message
from src.llm.schema.ToolCall import ToolCall
from src.tools.interfaces.Tool import Tool


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
            }
            if message.tool_calls:
                msg["content"] = None
            elif message.content:
                msg["content"] = message.content
            else:
                msg["content"] = None
            if message.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
                    }
                    for tc in message.tool_calls
                ]
            if message.tool_call_id:
                msg["tool_call_id"] = message.tool_call_id

            result.append(msg)

        return result

    def get_openrouter_schema(self, tool: Tool) -> dict[str, Any]:
        parameters = {}

        if tool.args_schema:
            parameters = tool.args_schema.model_json_schema()

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            },
        }

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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers,
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

        choices = data.get("choices", [])
        if not choices:
            raise ChatResponseError("No choices in response", response.status_code)

        message = choices[0].get("message", {})
        content = message.get("content") or ""

        tool_calls: List[ToolCall] = []
        for call in message.get("tool_calls", []):
            tool_calls.append(
                ToolCall(
                    id=call["id"],
                    name=call["function"]["name"],
                    args=json.loads(call["function"]["arguments"]),
                )
            )

        return LLMChatResponse(content=content, tool_calls=tool_calls)

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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        tool_call_deltas: dict[int, dict[str, Any]] = {}

        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    payload_line = line.removeprefix("data: ")

                    try:
                        data = json.loads(payload_line)
                    except Exception as e:
                        raise ChatResponseError(
                            "Invalid JSON response",
                            response.status_code,
                        ) from e

                    if "error" in data:
                        raise ChatResponseError(
                            data["error"],
                            response.status_code,
                        )

                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    content = delta.get("content")
                    finish_reason = choices[0].get("finish_reason")

                    for call in delta.get("tool_calls", []):
                        idx = call["index"]
                        if idx not in tool_call_deltas:
                            tool_call_deltas[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        tc = tool_call_deltas[idx]
                        if call.get("id"):
                            tc["id"] = call["id"]
                        if call.get("function"):
                            if call["function"].get("name"):
                                tc["name"] = call["function"]["name"]
                            if call["function"].get("arguments"):
                                tc["arguments"] += call["function"]["arguments"]

                    done = finish_reason is not None

                    if done and tool_call_deltas:
                        tool_calls: List[ToolCall] = []
                        for idx in sorted(tool_call_deltas):
                            tc = tool_call_deltas[idx]
                            tool_calls.append(
                                ToolCall(
                                    id=tc["id"],
                                    name=tc["name"],
                                    args=json.loads(tc["arguments"]),
                                )
                            )
                        yield StreamLLMChatResponse(
                            done=True, content=content, tool_calls=tool_calls
                        )
                        break

                    if done:
                        yield StreamLLMChatResponse(
                            done=True, content=content, tool_calls=[]
                        )
                        break

                    yield StreamLLMChatResponse(
                        done=False, content=content, tool_calls=[]
                    )
