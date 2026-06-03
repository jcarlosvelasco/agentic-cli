from typing import List

from src.compaction.CompactionStrategy import CompactionStrategy
from src.compaction.strategies.SlidingWindow import SlidingWindow
from src.compaction.strategies.Summarization import Summarization
from src.llm.providers.ollama.OllamaProvider import OllamaProvider
from src.llm.schema.Message import Message


async def run_compaction(
    strategy: CompactionStrategy, messages: List[Message]
) -> List[Message]:
    match strategy:
        case CompactionStrategy.SLIDING_WINDOW:
            sliding_window_compaction = SlidingWindow(window_size=20)
            messages = await sliding_window_compaction.compact(messages)
            print(f"Compacted messages: {messages}")
            return messages
        case CompactionStrategy.SUMMARIZATION:
            summarization = Summarization(
                threshold=20,
                keep_last_n=6,
                provider=OllamaProvider(model="gemma4:e2b-mlx"),
            )
            messages = await summarization.compact(messages)
            print(f"Compacted messages: {messages}")
            return messages
        case CompactionStrategy.NONE:
            return messages
