from abc import ABC, abstractmethod
from typing import List

from pydantic.main import BaseModel

from src.llm.schema.Message import Message


class Compaction(BaseModel, ABC):
    @abstractmethod
    async def compact(self, messages: List[Message]) -> List[Message]:
        pass
