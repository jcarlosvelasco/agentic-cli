import json
from typing import Any, AsyncIterator, List

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.llm.interfaces.BaseLLMProvider import BaseLLMProvider
from src.llm.interfaces.StreamLLMChatResponse import StreamLLMChatResponse
from src.llm.schema.ChatConnectionError import ChatConnectionError
from src.llm.schema.ChatResponseError import ChatResponseError
from src.llm.schema.ChatTimeoutError import ChatTimeoutError
from src.llm.schema.LLMChatResponse import LLMChatResponse
from src.llm.schema.Message import Message
from src.llm.schema.ToolCall import ToolCall
from src.tools.interfaces.Tool import Tool


class OpenAICompatibleProvider(BaseLLMProvider[ChatCompletionMessageParam]):
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "not-needed",
    ):
        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    def format_messages(
        self, messages: List[Message]
    ) -> List[ChatCompletionMessageParam]:
        result: List[ChatCompletionMessageParam] = []
        for message in messages:
            msg: dict[str, Any] = {
                "role": message.role.value,
                "content": message.content,
            }
            if message.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.args),
                        },
                    }
                    for tc in message.tool_calls
                ]
            if message.tool_call_id:
                msg["tool_call_id"] = message.tool_call_id
            result.append(msg)
        return result

    def get_openai_schema(self, tool: Tool) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
            },
        }
        if tool.args_schema:
            schema["function"]["parameters"] = tool.args_schema.model_json_schema()
        return schema

    async def chat(
        self, messages: List[Message], tools: List[Tool], temperature: float = 0.0
    ) -> LLMChatResponse:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=self.format_messages(messages),
                tools=[self.get_openai_schema(t) for t in tools] or None,
                temperature=temperature,
            )
        except APITimeoutError as e:
            raise ChatTimeoutError("Request timeout") from e
        except APIConnectionError as e:
            raise ChatConnectionError("Cannot connect to provider") from e
        except APIStatusError as e:
            raise ChatResponseError(str(e), e.status_code) from e

        message = response.choices[0].message

        tool_calls: List[ToolCall] = [
            ToolCall(
                id=call.id,
                name=call.function.name,
                args=json.loads(call.function.arguments),
            )
            for call in (message.tool_calls or [])
        ]

        return LLMChatResponse(content=message.content, tool_calls=tool_calls)

    async def stream_chat(
        self, messages: List[Message], tools: List[Tool], temperature: float = 0.0
    ) -> AsyncIterator[StreamLLMChatResponse]:
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=self.format_messages(messages),
                tools=[self.get_openai_schema(t) for t in tools] or None,
                temperature=temperature,
                stream=True,
            )
        except APITimeoutError as e:
            raise ChatTimeoutError("Request timeout") from e
        except APIConnectionError as e:
            raise ChatConnectionError("Cannot connect to provider") from e

        tool_call_chunks: dict[int, dict[str, Any]] = {}

        async for chunk in stream:
            choice = chunk.choices[0]
            delta = choice.delta
            is_last = choice.finish_reason is not None

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    acc = tool_call_chunks.setdefault(
                        tc_delta.index, {"id": None, "name": "", "arguments": ""}
                    )
                    if tc_delta.id:
                        acc["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            acc["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            acc["arguments"] += tc_delta.function.arguments

            if is_last:
                tool_calls = [
                    ToolCall(
                        id=v["id"],
                        name=v["name"],
                        args=json.loads(v["arguments"]) if v["arguments"] else {},
                    )
                    for v in tool_call_chunks.values()
                ]
                yield StreamLLMChatResponse(
                    done=True, content=delta.content, tool_calls=tool_calls
                )
            else:
                yield StreamLLMChatResponse(
                    done=False, content=delta.content, tool_calls=[]
                )
