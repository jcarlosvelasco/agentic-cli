from abc import abstractmethod
from typing import Protocol

from src.llm.schema.LLMChatResponse import LLMChatResponse


class BaseLLMProvider(Protocol):
    @abstractmethod
    async def chat(self, prompt: str) -> LLMChatResponse:
        pass
