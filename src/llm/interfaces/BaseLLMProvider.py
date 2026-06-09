from abc import ABC, abstractmethod
from typing import AsyncIterator, Generic, List, TypeVar

from src.llm.interfaces.StreamLLMChatResponse import StreamLLMChatResponse
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
    async def stream_chat(
        self, messages: List[Message], tools: List[Tool], temperature: float = 0.0
    ) -> AsyncIterator[StreamLLMChatResponse]:
        if False:
            yield

    @abstractmethod
    def format_messages(self, messages: List[Message]) -> List[MessageT]:
        pass
