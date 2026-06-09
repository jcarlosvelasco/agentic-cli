import asyncio
from os import path

from src.agent.schema.Agent import Agent
from src.llm.providers.ollama.OllamaProvider import OllamaProvider
from src.mcp.mpc_registry import MCPRegistry
from src.shared.console import (
    display_warning,
    display_welcome,
    get_user_input,
    streaming_panel,
)
from src.tools.registry import ToolRegistry


async def main():
    provider = OllamaProvider(model="gemma4:e2b-mlx")
    registry = ToolRegistry(provider=provider)
    mcp_registry: MCPRegistry | None = None

    mcp_path = "src/mcp/mcp.json"
    if not path.exists(mcp_path):
        display_warning("No MCP config found, skipping MCP setup.")
    else:
        mcp_registry = MCPRegistry.from_file(mcp_path)
        mcp_tools = await mcp_registry.load_all()
        for tool in mcp_tools:
            registry.register(tool)

    display_welcome()

    agent = Agent(
        name="main",
        provider=provider,
        tools=registry.get_tools(),
        system_prompt="You are a coding assistant...",
    )

    try:
        while True:
            user_input = await get_user_input()
            async with streaming_panel(agent.name) as update:
                await agent._stream_chat(user_input, on_content=update)
    finally:
        if mcp_registry is not None:
            await mcp_registry.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
