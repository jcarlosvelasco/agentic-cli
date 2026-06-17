import asyncio
from os import path

from compaction.CompactionRunner import run_compaction
from compaction.CompactionStrategy import CompactionStrategy
from config.AppConfig import AppConfig
from llm.interfaces.BaseLLMProvider import BaseLLMProvider
from memory.preamble import preamble
from shared.config import load_config
from src.agent.schema.Agent import Agent
from src.llm.providers.ollama.OllamaProvider import OllamaProvider
from src.mcp.mpc_registry import MCPRegistry
from src.memory.interface.Session import Session
from src.memory.summarize import summarize
from src.shared.console import (
    display_compacting,
    display_warning,
    display_welcome,
    get_user_input,
    streaming_panel,
)
from src.tools.registry import ToolRegistry


async def main():
    config = load_config()

    provider = OllamaProvider(model=config.llm.model, base_url=config.llm.base_url)
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
                await compact(agent, config.compaction.strategy, provider)

            user_input = await get_user_input()
            async with streaming_panel(agent.name) as update:
                await agent._stream_chat(user_input, on_content=update)
    except KeyboardInterrupt:
        if config.memory.enabled:
            display_warning("Interrupted by user. Summarizing session...")
            session.save()
            await summarize(session.id, provider)
            display_warning("Session summarized. Exiting...")
    finally:
        if mcp_registry is not None:
            await mcp_registry.cleanup()


async def build_system_prompt(config: AppConfig) -> str:
    system_prompt = "You are a helpful coding assistant"

    if config.memory.enable_preamble:
        memory = await preamble(config)
        system_prompt = f"{system_prompt}. Here is some memory from your recent sessions: {memory}\n\n"

    if config.memory.enabled:
        system_prompt = f"{system_prompt} You also have a 'recall' tool that can search ALL past sessions in more detail. "
        "When the user asks something not covered in the memory above, or asks for specifics "
        "that might be in a past conversation, call recall first before answering."

    return system_prompt


async def compact(
    agent: Agent,
    compaction_strategy: CompactionStrategy,
    provider: BaseLLMProvider,
):
    compacted = await run_compaction(
        strategy=compaction_strategy,
        messages=agent.messages,
        provider=provider,
    )
    if len(compacted) < len(agent.messages):
        display_compacting(len(agent.messages) - len(compacted))
        agent.messages = compacted


if __name__ == "__main__":
    asyncio.run(main())
