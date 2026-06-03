import asyncio
from typing import List

from src.compaction.CompactionRunner import run_compaction
from src.compaction.CompactionStrategy import CompactionStrategy
from src.llm.interfaces.BaseLLMProvider import BaseLLMProvider
from src.llm.providers.ollama.OllamaProvider import OllamaProvider
from src.llm.schema.Message import Message, MessageRole
from src.shared.console import (
    confirm_execution,
    display_assistant_message,
    display_tool_call,
    display_tool_result,
    display_welcome,
    get_user_input,
    thinking_spinner,
)
from src.tools.interfaces.Tool import Tool
from src.tools.registry import get_tool_by_name, get_tools
from src.tools.ToolRunner import ToolRunner


async def main():
    display_welcome()
    messages: List[Message] = []
    tools = get_tools()
    runner = ToolRunner()
    provider = OllamaProvider(model="gemma4:e2b-mlx")

    while True:
        user_input = get_user_input()
        messages.append(Message(role=MessageRole.USER, content=user_input))
        messages = await agentLoop(messages, tools, runner, provider)


async def agentLoop(
    messages: List[Message],
    tools: List[Tool],
    runner: ToolRunner,
    provider: BaseLLMProvider,
) -> List[Message]:
    messages = await run_compaction(
        strategy=CompactionStrategy.SUMMARIZATION, messages=messages, provider=provider
    )

    while True:
        async with thinking_spinner():
            response = await provider.chat(messages, tools=tools)

        if response.has_tool_calls:
            if response.content:
                display_assistant_message(response.content)

            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )

            for call in response.tool_calls:
                display_tool_call(call.name, call.args)
                tool = get_tool_by_name(call.name)
                if tool and confirm_execution(call.name, call.args):
                    result = runner.run(tool, call.args)
                    if result is not None:
                        display_tool_result(result.data)
                        messages.append(
                            Message(
                                role=MessageRole.TOOL,
                                content=str(result.data),
                                tool_call_id=call.id,
                            )
                        )
        else:
            display_assistant_message(response.content)
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content,
                )
            )
            break

    return messages


if __name__ == "__main__":
    asyncio.run(main())
