import asyncio
from typing import List

from src.agent.schema.Agent import Agent
from src.llm.providers.ollama.OllamaProvider import OllamaProvider
from src.mcp.mpc_registry import MCPRegistry
from src.shared.console import (
    display_welcome,
    get_user_input,
    streaming_panel,
)
from src.tools.interfaces.Tool import Tool
from src.tools.registry import ToolRegistry


async def get_tools(mcp_registry: MCPRegistry, registry: ToolRegistry) -> List[Tool]:
    mcp_tools = await mcp_registry.load_all()
    return registry.get_tools() + mcp_tools


async def main():
    mcp_registry = MCPRegistry.from_file("mcp.json")

    provider = OllamaProvider(model="gemma4:e2b-mlx")
    registry = ToolRegistry(provider=provider)

    display_welcome()

    tools = await get_tools(mcp_registry, registry)

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
