import asyncio
import os
from os import path

from dotenv import load_dotenv

from src.agent.schema.Agent import Agent
from src.compaction.CompactionRunner import run_compaction
from src.compaction.CompactionStrategy import CompactionStrategy
from src.config.AppConfig import AppConfig
from src.llm.interfaces.BaseLLMProvider import BaseLLMProvider
from src.llm.providers.openrouter.OpenRouterProvider import OpenRouterProvider
from src.mcp.mcp_registry import MCPRegistry
from src.memory.interface.Session import Session
from src.memory.preamble import preamble
from src.memory.summarize import summarize
from src.shared.config import load_config
from src.shared.console import (
    display_assistant_message,
    display_compacting,
    display_warning,
    display_welcome,
    get_user_input,
    streaming_panel,
)
from src.tools.registry import ToolRegistry


async def main():
    load_dotenv()
    config = load_config()

    if not config.llm.api_key:
        raise ValueError(f"API key not set for {config.llm.model}")

    provider: BaseLLMProvider = OpenRouterProvider(
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        base_url="https://openrouter.ai/api",
        api_key=os.getenv(config.llm.api_key, ""),
    )

    # provider = OllamaProvider(model=config.llm.model, base_url=config.llm.base_url)
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
