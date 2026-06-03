import asyncio
from typing import List

from src.compaction.CompactionRunner import run_compaction
from src.compaction.CompactionStrategy import CompactionStrategy
from src.llm.interfaces.BaseLLMProvider import BaseLLMProvider
from src.llm.providers.ollama.OllamaProvider import OllamaProvider
from src.llm.schema.Message import Message, MessageRole
from src.tools.interfaces.Tool import Tool
from src.tools.registry import get_tool_by_name, get_tools
from src.tools.ToolRunner import ToolRunner


async def main():
    messages: List[Message] = []
    tools = get_tools()
    runner = ToolRunner()

    while True:
        user_input = input("You: ")
        messages.append(Message(role=MessageRole.USER, content=user_input))
        await agentLoop(messages, tools, runner)


async def agentLoop(messages: List[Message], tools: List[Tool], runner: ToolRunner):
    messages = await run_compaction(
        strategy=CompactionStrategy.SUMMARIZATION, messages=messages
    )

    provider: BaseLLMProvider = OllamaProvider(model="gemma4:e2b-mlx")

    while True:
        response = await provider.chat(messages, tools=tools)

        if response.has_tool_calls:
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )

            for call in response.tool_calls:
                tool = get_tool_by_name(call.name)
                if tool:
                    result = runner.run(tool, call.args)

                    messages.append(
                        Message(
                            role=MessageRole.TOOL,
                            content=str(result.data),
                            tool_call_id=call.id,
                        )
                    )
        else:
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=response.content,
                )
            )
            print("Agent:", response.content)
            break


if __name__ == "__main__":
    asyncio.run(main())
