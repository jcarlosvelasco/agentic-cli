import asyncio
from os import path

from memory.preamble import preamble
from src.agent.schema.Agent import Agent
from src.llm.providers.ollama.OllamaProvider import OllamaProvider
from src.mcp.mpc_registry import MCPRegistry
from src.memory.interface.Session import Session
from src.memory.summarize import summarize
from src.shared.console import (
    display_warning,
    display_welcome,
    get_user_input,
    streaming_panel,
)
from src.tools.registry import ToolRegistry


async def main():
    provider = OllamaProvider(model="gemma4:e2b-mlx")
    session = Session()
    registry = ToolRegistry(provider=provider, session=session)
    mcp_registry: MCPRegistry | None = None
    memory = await preamble()

    print(f"Preparing agent with memory: {memory}")

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
        system_prompt=(
            f"You are a helpful coding assistant. Here is some memory from your recent sessions: {memory}"
        ),
        session=session,
    )

    try:
        while True:
            user_input = await get_user_input()
            async with streaming_panel(agent.name) as update:
                await agent._stream_chat(user_input, on_content=update)
    except KeyboardInterrupt:
        display_warning("Interrupted by user. Summarizing session...")
        session.save()
        await summarize(session.id, provider)
        display_warning("Session summarized. Exiting...")
    finally:
        if mcp_registry is not None:
            await mcp_registry.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
