from abc import ABC, abstractmethod
from typing import List

from pydantic import ConfigDict
from pydantic.main import BaseModel

from src.llm.schema.Message import Message


class Compaction(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    async def compact(self, messages: List[Message]) -> List[Message]:
        pass
