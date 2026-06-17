from abc import ABC, abstractmethod

from pydantic import BaseModel

from src.llm.schema.Message import Message


class MemoryManager(ABC, BaseModel):
    @abstractmethod
    async def summarize(self, args: list[Message]) -> str:
        pass
