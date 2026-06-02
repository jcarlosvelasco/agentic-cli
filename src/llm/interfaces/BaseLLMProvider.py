from abc import ABC, abstractmethod
from typing import List

from src.llm.schema.LLMChatResponse import LLMChatResponse
from src.llm.schema.Message import Message
from src.tools.interfaces.Tool import Tool


class BaseLLMProvider(ABC):
    @abstractmethod
    async def chat(
        self, messages: List[Message], tools: List[Tool], temperature: float = 0.0
    ) -> LLMChatResponse:
        pass
