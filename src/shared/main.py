import asyncio
from typing import List

from src.agent.schema.Agent import Agent
from src.llm.providers.ollama.OllamaProvider import OllamaProvider
from src.mcp.client import MCPClient
from src.shared.console import (
    display_welcome,
    get_user_input,
    streaming_panel,
)
from src.tools.interfaces.Tool import Tool
from src.tools.registry import ToolRegistry


async def get_tools(client: MCPClient, registry: ToolRegistry) -> List[Tool]:
    mcp_tools = await client.get_tools()
    return registry.get_tools() + mcp_tools


async def main():
    client = MCPClient()
    provider = OllamaProvider(model="gemma4:e2b-mlx")
    registry = ToolRegistry(provider=provider)

    display_welcome()

    tools = await get_tools(client, registry)

    agent = Agent(
        name="main",
        provider=provider,
        tools=tools,
        system_prompt="You are a coding assistant...",
    )

    while True:
        user_input = await get_user_input()

        async with streaming_panel(agent.name) as update:
            await agent._stream_chat(user_input, on_content=update)


if __name__ == "__main__":
    asyncio.run(main())
