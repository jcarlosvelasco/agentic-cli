from typing import List

from pydantic.main import BaseModel

from src.compaction.Compaction import Compaction
from src.llm.schema.Message import Message


class SlidingWindow(Compaction, BaseModel):
    window_size: int

    async def compact(self, messages: List[Message]) -> List[Message]:
        if len(messages) <= self.window_size:
            return messages

        print(f"Compacting {len(messages)} messages...")
        messages = messages[-self.window_size :]
        return messages
