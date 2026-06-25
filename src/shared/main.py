import asyncio
from os import path

from dotenv import load_dotenv

from src.shared.utils import build_system_prompt, compact
from src.agent.Agent import Agent
from src.llm.providers import create_provider
from src.mcp_integration.mcp_registry import MCPRegistry
from src.memory.Session import Session
from src.memory.summarize import summarize
from src.shared.config import load_config
from src.shared.console import (
    display_assistant_message,
    display_warning,
    display_welcome,
    get_user_input,
    streaming_panel,
)
from src.tools.registry import ToolRegistry


async def main():
    load_dotenv()
    config = load_config()

    provider = create_provider(config.llm)
    session = Session()
    registry = ToolRegistry(provider=provider, session=session, config=config)
    mcp_registry: MCPRegistry | None = None

    if config.mcp.enabled:
        mcp_path = config.mcp.config_file
        if not path.exists(mcp_path):
            display_warning("No MCP config found, skipping MCP setup.")
        else:
            mcp_registry = MCPRegistry.from_file(mcp_path)
            mcp_tools = await mcp_registry.load_all()
            for tool in mcp_tools:
                registry.register(tool)

    display_welcome()

    system_prompt = await build_system_prompt(config)

    agent = Agent(
        name="main",
        provider=provider,
        tools=registry.get_tools(),
        system_prompt=system_prompt,
        session=session,
    )

    try:
        while True:
            if config.compaction.enabled:
                await compact(
                    agent, config.compaction.strategy, provider, config, False
                )

            user_input = await get_user_input()

            if config.ui.streaming:
                async with streaming_panel(agent.name) as (update, ctrl):
                    await agent._stream_chat(
                        user_input, on_content=update, ui_control=ctrl
                    )
            else:
                result = await agent.chat(user_input)
                display_assistant_message(agent.name, result)
    except KeyboardInterrupt:
        if config.memory.enabled:
            display_warning("Interrupted by user. Summarizing session...")
            session.save()
            await summarize(session.id, provider)
            display_warning("Session summarized. Exiting...")
    finally:
        if mcp_registry is not None:
            await mcp_registry.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
