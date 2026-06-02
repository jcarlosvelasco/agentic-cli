from abc import abstractmethod
from typing import List, Protocol

from src.llm.schema.LLMChatResponse import LLMChatResponse
from src.llm.schema.Message import Message


class BaseLLMProvider(Protocol):
    @abstractmethod
    async def chat(self, messages: List[Message]) -> LLMChatResponse:
        pass
