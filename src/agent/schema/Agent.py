from typing import List

from pydantic import BaseModel
from pydantic.config import ConfigDict

from src.llm.interfaces.BaseLLMProvider import BaseLLMProvider
from src.llm.schema.Message import Message, MessageRole
from src.shared.console import (
    display_assistant_message,
    display_tool_call,
    display_tool_result,
    thinking_spinner,
)
from src.tools.interfaces.Tool import Tool
from src.tools.ToolRunner import ToolRunner


class Agent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    provider: BaseLLMProvider
    tools: List[Tool]
    messages: List[Message] = []
    max_iterations: int = 10
    system_prompt: str | None = None

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

                for call in response.tool_calls:
                    display_tool_call(call.name, call.args)
                    tool = tools_by_name.get(call.name)

                    # if tool and confirm_execution(call.name, call.args):
                    if tool:
                        result = await runner.run(tool, call.args)
                        if result is not None:
                            display_tool_result(result.data)

                            self.messages.append(
                                Message(
                                    role=MessageRole.TOOL,
                                    content=str(result.data),
                                    tool_call_id=call.id,
                                )
                            )
            else:
                display_assistant_message(self.name, response.content)
                self.messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=response.content,
                    )
                )
                return response.content

        return "Max iterations reached"
