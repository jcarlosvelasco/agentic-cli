import asyncio
from collections.abc import Callable
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field, computed_field
from pydantic.config import ConfigDict

from src.config.AppConfig import AppConfig
from src.llm.interfaces.BaseLLMProvider import BaseLLMProvider
from src.llm.schema.ChatResponseError import ChatResponseError
from src.llm.schema.Message import Message, MessageRole
from src.llm.schema.ToolCall import ToolCall
from src.llm.utils import retry_with_backoff
from src.memory.Session import Session
from src.shared.config import load_config
from src.shared.console import (
    LiveController,
    confirm_execution,
    display_assistant_message,
    display_tool_call,
    display_tool_result,
    display_warning,
    streaming_panel,
    thinking_spinner,
)
from src.tools.interfaces.Tool import Tool, ToolResult
from src.tools.ToolRunner import ToolRunner


class Agent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    provider: BaseLLMProvider
    tools: List[Tool]
    messages: List[Message] = []
    max_iterations: int = 10
    system_prompt: str | None = None
    session: Session

    runner: ToolRunner = Field(default_factory=ToolRunner)
    config: AppConfig = load_config()

    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @computed_field
    @property
    def tools_by_name(self) -> dict[str, Tool[Any]]:
        return {tool.name: tool for tool in self.tools}

    async def stream_run(self, task) -> tuple[str, dict[str, int]]:
        if self.system_prompt:
            self.append(Message(role=MessageRole.SYSTEM, content=self.system_prompt))
        self.append(Message(role=MessageRole.USER, content=task))
        return await self._stream_loop()

    async def _consume_stream_once(
        self,
        on_content: Callable[[str], None] | None = None,
    ) -> tuple[str, list[ToolCall], int, int]:
        stream = self.provider.stream_chat(self.messages, tools=self.tools)
        content = ""
        calls: list[ToolCall] = []
        input_tokens = 0
        output_tokens = 0

        if on_content:
            async for chunk in stream:
                if chunk.content:
                    content += chunk.content
                    on_content(content)
                if chunk.tool_calls:
                    calls = chunk.tool_calls
                if chunk.done:
                    input_tokens = chunk.input_token_count
                    output_tokens = chunk.output_token_count
                    break
        else:
            async with streaming_panel(self.name) as (update, ctrl):
                async for chunk in stream:
                    if chunk.content:
                        content += chunk.content
                        update(content)
                    if chunk.tool_calls:
                        calls = chunk.tool_calls
                    if chunk.done:
                        input_tokens = chunk.input_token_count
                        output_tokens = chunk.output_token_count
                        break

        return content, calls, input_tokens, output_tokens

    async def _stream_loop(
        self,
        on_content: Callable[[str], None] | None = None,
        ui_control: LiveController | None = None,
    ) -> tuple[str, dict[str, int]]:
        for _ in range(self.max_iterations):
            try:
                (
                    full_content,
                    last_tool_calls,
                    input_tokens,
                    output_tokens,
                ) = await retry_with_backoff(
                    lambda: self._consume_stream_once(on_content),
                    max_retries=self.config.llm.retry_max_attempts,
                    base_delay=self.config.llm.retry_base_delay,
                )
            except ChatResponseError as e:
                msg = f"LLM request failed after retries: {e.message}"
                if on_content:
                    on_content(f"\n⚠️ {msg}")
                else:
                    display_warning(msg)
                return msg, {"input_tokens": 0, "output_tokens": 0}

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            if last_tool_calls:
                self.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=full_content,
                        tool_calls=last_tool_calls,
                    )
                )

                async def execute_tool_call(
                    call: ToolCall,
                ) -> Tuple[str, ToolResult] | None:
                    tool = self.tools_by_name.get(call.name)

                    if not tool:
                        return

                    if on_content:
                        on_content(f"🔧 **{call.name}** `{call.args}`")
                    else:
                        display_tool_call(self.name, call.name, call.args)

                    if self.config.tools.confirm_execution and ui_control:
                        ui_control.pause()
                        confirmed = confirm_execution(call.name, call.args)
                        ui_control.resume()
                        if not confirmed:
                            return call.id, ToolResult(
                                success=False,
                                message="Tool execution cancelled by user",
                            )

                    result = await self.runner.run(
                        tool, call.args, should_confirm=False
                    )

                    if on_content:
                        on_content(f"🔧 **{call.name}** → `{result.data}`")
                    else:
                        display_tool_result(self.name, result.data)
                    return call.id, result

                results = await asyncio.gather(
                    *[execute_tool_call(call) for call in last_tool_calls]
                )

                for result in results:
                    if result is not None:
                        self.append(
                            Message(
                                role=MessageRole.TOOL,
                                content=str(result[1].data),
                                tool_call_id=result[0],
                            )
                        )

            else:
                self.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=full_content,
                    )
                )

                usage = {
                    "input_tokens": self.total_input_tokens,
                    "output_tokens": self.total_output_tokens,
                }
                return full_content, usage

        return "Max iterations reached", {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
        }

    async def _stream_chat(
        self,
        user_input: str,
        on_content: Callable[[str], None] | None = None,
        ui_control: LiveController | None = None,
    ) -> tuple[str, dict[str, int]]:
        self.append(Message(role=MessageRole.USER, content=user_input))

        return await self._stream_loop(on_content=on_content, ui_control=ui_control)

    async def run(self, task: str) -> Tuple[str, Dict[str, int]]:
        if self.system_prompt:
            self.append(Message(role=MessageRole.SYSTEM, content=self.system_prompt))

        self.append(Message(role=MessageRole.USER, content=task))

        return await self._loop()

    async def chat(self, user_input: str) -> Tuple[str, Dict[str, int]]:
        self.append(Message(role=MessageRole.USER, content=user_input))
        return await self._loop()

    async def _loop(self) -> Tuple[str, Dict[str, int]]:
        for _ in range(self.max_iterations):
            async with thinking_spinner():
                response = await retry_with_backoff(
                    lambda: self.provider.chat(self.messages, tools=self.tools),
                    max_retries=self.config.llm.retry_max_attempts,
                    base_delay=self.config.llm.retry_base_delay,
                )

            self.total_input_tokens += response.input_token_count
            self.total_output_tokens += response.output_token_count

            if response.has_tool_calls:
                if response.content:
                    display_assistant_message(self.name, response.content)

                self.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )

                async def execute_tool_call(
                    call: ToolCall,
                ) -> Tuple[str, ToolResult] | None:
                    display_tool_call(self.name, call.name, call.args)
                    tool = self.tools_by_name.get(call.name)

                    if not tool:
                        return

                    result = await self.runner.run(
                        tool, call.args, self.config.tools.confirm_execution
                    )
                    display_tool_result(self.name, result.data)
                    return call.id, result

                results = await asyncio.gather(
                    *[execute_tool_call(call) for call in response.tool_calls]
                )

                for result in results:
                    if result is not None:
                        self.append(
                            Message(
                                role=MessageRole.TOOL,
                                content=str(result[1].data),
                                tool_call_id=result[0],
                            )
                        )

            else:
                self.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=response.content,
                    )
                )

                usage = {
                    "input_tokens": response.input_token_count,
                    "output_tokens": response.output_token_count,
                }

                return response.content, usage

        return "Max iterations reached", {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
        }

    def append(self, msg: Message) -> None:
        self.messages.append(msg)
        self.session.append(msg)
