from src.agent.Agent import Agent
from src.compaction.CompactionRunner import run_compaction
from src.compaction.CompactionStrategy import CompactionStrategy
from src.config.AppConfig import AppConfig
from src.llm.interfaces.BaseLLMProvider import BaseLLMProvider
from src.memory.preamble import preamble
from src.shared.console import display_compacting


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
    config: AppConfig,
    api_mode: bool,
):
    compacted = await run_compaction(
        strategy=compaction_strategy,
        messages=agent.messages,
        provider=provider,
        config=config.compaction,
    )
    if len(compacted) < len(agent.messages):
        if not api_mode:
            display_compacting(len(agent.messages) - len(compacted))
        agent.messages = compacted
