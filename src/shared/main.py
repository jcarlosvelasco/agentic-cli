import asyncio
from typing import List

from src.llm.interfaces.BaseLLMProvider import BaseLLMProvider
from src.llm.providers.OllamaProvider import OllamaProvider
from src.llm.schema.Message import Message, MessageRole


async def main():
    messages: List[Message] = []

    while True:
        user_input = input("You: ")
        messages.append(Message(role=MessageRole.USER, content=user_input))
        await agentLoop(messages)


async def agentLoop(messages: List[Message]):
    provider: BaseLLMProvider = OllamaProvider(model="gemma4:e2b-mlx")
    response = await provider.chat(messages)
    print(f"Agent: {response.content}")


if __name__ == "__main__":
    asyncio.run(main())
