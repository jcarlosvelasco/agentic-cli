import asyncio

from src.agent.schema.Agent import Agent
from src.llm.providers.ollama.OllamaProvider import OllamaProvider
from src.shared.console import (
    display_welcome,
    get_user_input,
    streaming_panel,
)
from src.tools.registry import ToolRegistry


async def main():
    display_welcome()
    provider = OllamaProvider(model="gemma4:e2b-mlx")

    registry = ToolRegistry(provider=provider)

    agent = Agent(
        name="main",
        provider=provider,
        tools=registry.get_tools(),
        system_prompt="You are a coding assistant...",
    )

    while True:
        user_input = get_user_input()

        async with streaming_panel(agent.name) as update:
            await agent._stream_chat(user_input, on_content=update)


if __name__ == "__main__":
    asyncio.run(main())
