from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from src.llm.schema.LLMChatResponse import LLMChatResponse
from src.llm.schema.Message import Message
from src.tools.interfaces.Tool import Tool

MessageT = TypeVar("MessageT")


class BaseLLMProvider(ABC, Generic[MessageT]):
    @abstractmethod
    async def chat(
        self, messages: List[Message], tools: List[Tool], temperature: float = 0.0
    ) -> LLMChatResponse:
        pass

    @abstractmethod
    def format_messages(self, messages: List[Message]) -> List[MessageT]:
        pass
