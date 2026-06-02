import asyncio
from typing import List

from src.llm.interfaces.BaseLLMProvider import BaseLLMProvider
from src.llm.providers.ollama.OllamaProvider import OllamaProvider
from src.llm.schema.Message import Message, MessageRole
from src.tools.interfaces.Tool import Tool
from src.tools.registry.registry_core import get_tool_by_name, get_tools
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
    provider: BaseLLMProvider = OllamaProvider(model="gemma4:e2b-mlx")
    response = await provider.chat(messages, tools=tools)

    if response.has_tool_calls:
        for call in response.tool_calls:
            tool = get_tool_by_name(call.name)
            if tool:
                runner.run(tool, call.args)
    else:
        print(f"Agent: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
