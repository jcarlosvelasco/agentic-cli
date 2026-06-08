import asyncio
from collections.abc import Callable
from typing import List, Tuple

from pydantic import BaseModel
from pydantic.config import ConfigDict

from src.llm.interfaces.BaseLLMProvider import BaseLLMProvider
from src.llm.schema.Message import Message, MessageRole
from src.llm.schema.ToolCall import ToolCall
from src.shared.console import (
    display_assistant_message,
    display_tool_call,
    display_tool_result,
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

    async def stream_run(self, task) -> str:
        if self.system_prompt:
            self.messages.append(
                Message(role=MessageRole.SYSTEM, content=self.system_prompt)
            )
        self.messages.append(Message(role=MessageRole.USER, content=task))
        return await self._stream_loop()

    async def _stream_loop(self, on_content: Callable[[str], None] | None = None) -> str:
        runner = ToolRunner()
        tools_by_name = {tool.name: tool for tool in self.tools}

        for _ in range(self.max_iterations):
            stream = self.provider.stream_chat(
                self.messages,
                tools=self.tools,
            )

            full_content = ""
            last_tool_calls: List[ToolCall] = []

            if on_content:
                async for chunk in stream:
                    if chunk.content:
                        full_content += chunk.content
                        on_content(full_content)

                    if chunk.tool_calls:
                        last_tool_calls = chunk.tool_calls

                    if chunk.done:
                        break
            else:
                async with streaming_panel(self.name) as update:
                    async for chunk in stream:
                        if chunk.content:
                            full_content += chunk.content
                            update(full_content)

                        if chunk.tool_calls:
                            last_tool_calls = chunk.tool_calls

                        if chunk.done:
                            break

            if last_tool_calls:
                self.messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=full_content,
                        tool_calls=last_tool_calls,
                    )
                )

                async def execute_tool_call(
                    call: ToolCall,
                ) -> Tuple[str, ToolResult] | None:
                    display_tool_call(self.name, call.name, call.args)
                    tool = tools_by_name.get(call.name)

                    # if tool and confirm_execution(call.name, call.args):
                    if tool:
                        result = await runner.run(tool, call.args)
                        display_tool_result(self.name, result.data)
                        return call.id, result

                results = await asyncio.gather(
                    *[execute_tool_call(call) for call in last_tool_calls]
                )

                for result in results:
                    if result is not None:
                        self.messages.append(
                            Message(
                                role=MessageRole.TOOL,
                                content=str(result[1].data),
                                tool_call_id=result[0],
                            )
                        )

            else:
                self.messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=full_content,
                    )
                )

                return full_content

        return "Max iterations reached"

    async def _stream_chat(self, user_input: str, on_content: Callable[[str], None] | None = None) -> str:
        self.messages.append(Message(role=MessageRole.USER, content=user_input))
        return await self._stream_loop(on_content=on_content)

    async def run(self, task: str) -> str:
        if self.system_prompt:
            self.messages.append(
                Message(role=MessageRole.SYSTEM, content=self.system_prompt)
            )
        self.messages.append(Message(role=MessageRole.USER, content=task))
        return await self._loop()

    async def chat(self, user_input: str) -> str:
        self.messages.append(Message(role=MessageRole.USER, content=user_input))
        return await self._loop()

    async def _loop(self) -> str:
        runner = ToolRunner()
        tools_by_name = {tool.name: tool for tool in self.tools}

        for _ in range(self.max_iterations):
            async with thinking_spinner():
                response = await self.provider.chat(self.messages, tools=self.tools)

            if response.has_tool_calls:
                if response.content:
                    display_assistant_message(self.name, response.content)

                self.messages.append(
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
                    tool = tools_by_name.get(call.name)

                    # if tool and confirm_execution(call.name, call.args):
                    if tool:
                        result = await runner.run(tool, call.args)
                        display_tool_result(self.name, result.data)
                        return call.id, result

                results = await asyncio.gather(
                    *[execute_tool_call(call) for call in response.tool_calls]
                )

                for result in results:
                    if result is not None:
                        self.messages.append(
                            Message(
                                role=MessageRole.TOOL,
                                content=str(result[1].data),
                                tool_call_id=result[0],
                            )
                        )

            else:
                self.messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=response.content,
                    )
                )
                return response.content

        return "Max iterations reached"
