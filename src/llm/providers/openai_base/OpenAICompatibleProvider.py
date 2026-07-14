import json
from typing import Any, AsyncIterator, List, cast

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion_function_tool_param import (
    ChatCompletionFunctionToolParam,
)

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
            result.append(cast(ChatCompletionMessageParam, msg))
        return result

    def get_openai_schema(self, tool: Tool) -> ChatCompletionFunctionToolParam:
        schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
            },
        }
        if tool.args_schema:
            schema["function"]["parameters"] = tool.args_schema.model_json_schema()
        return cast(ChatCompletionFunctionToolParam, schema)

    async def chat(
        self, messages: List[Message], tools: List[Tool], temperature: float = 0.0
    ) -> LLMChatResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self.format_messages(messages),
            "temperature": temperature,
        }
        tool_schemas = [self.get_openai_schema(t) for t in tools]
        if tool_schemas:
            kwargs["tools"] = tool_schemas
        try:
            response = await self.client.chat.completions.create(**kwargs)
        except APITimeoutError as e:
            raise ChatTimeoutError("Request timeout") from e
        except APIConnectionError as e:
            raise ChatConnectionError("Cannot connect to provider") from e
        except APIStatusError as e:
            raise ChatResponseError(str(e), e.status_code) from e

        message = response.choices[0].message
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        tool_calls: List[ToolCall] = [
            ToolCall(
                id=call.id,
                name=call.function.name,
                args=json.loads(call.function.arguments),
            )
            for call in (message.tool_calls or [])
        ]

        return LLMChatResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            input_token_count=input_tokens,
            output_token_count=output_tokens,
        )

    async def stream_chat(
        self, messages: List[Message], tools: List[Tool], temperature: float = 0.0
    ) -> AsyncIterator[StreamLLMChatResponse]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self.format_messages(messages),
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        tool_schemas = [self.get_openai_schema(t) for t in tools]
        if tool_schemas:
            kwargs["tools"] = tool_schemas
        try:
            stream = await self.client.chat.completions.create(**kwargs)
        except APITimeoutError as e:
            raise ChatTimeoutError("Request timeout") from e
        except APIConnectionError as e:
            raise ChatConnectionError("Cannot connect to provider") from e

        tool_call_chunks: dict[int, dict[str, Any]] = {}
        finished = False

        async for chunk in stream:
            if not chunk.choices:
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens or 0
                    output_tokens = chunk.usage.completion_tokens or 0
                    if not finished:
                        finished = True
                        yield StreamLLMChatResponse(
                            done=True,
                            content=None,
                            tool_calls=[
                                ToolCall(
                                    id=v["id"],
                                    name=v["name"],
                                    args=json.loads(v["arguments"])
                                    if v["arguments"]
                                    else {},
                                )
                                for v in tool_call_chunks.values()
                            ],
                            input_token_count=input_tokens,
                            output_token_count=output_tokens,
                        )
                continue

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
                input_tokens = chunk.usage.prompt_tokens if chunk.usage else 0
                output_tokens = chunk.usage.completion_tokens if chunk.usage else 0
                if input_tokens or output_tokens:
                    finished = True
                yield StreamLLMChatResponse(
                    done=True,
                    content=delta.content,
                    tool_calls=tool_calls,
                    input_token_count=input_tokens,
                    output_token_count=output_tokens,
                )
            else:
                yield StreamLLMChatResponse(
                    done=False, content=delta.content, tool_calls=[]
                )
