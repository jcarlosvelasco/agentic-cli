from typing import List

from src.compaction.Compaction import Compaction
from src.llm.schema.Message import Message, MessageRole


class SlidingWindow(Compaction):
    window_size: int

    async def compact(self, messages: List[Message]) -> List[Message]:
        if len(messages) <= self.window_size:
            return messages

        system_messages = [m for m in messages if m.role == MessageRole.SYSTEM]
        non_system = [m for m in messages if m.role != MessageRole.SYSTEM]

        messages = non_system[-self.window_size :]
        return system_messages + messages
