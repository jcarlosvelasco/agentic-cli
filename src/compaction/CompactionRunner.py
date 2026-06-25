from typing import List

from config.CompactionConfig import CompactionConfig
from src.compaction.CompactionStrategy import CompactionStrategy
from src.compaction.strategies.SlidingWindow import SlidingWindow
from src.compaction.strategies.Summarization import Summarization
from src.llm.interfaces.BaseLLMProvider import BaseLLMProvider
from src.llm.schema.Message import Message


async def run_compaction(
    strategy: CompactionStrategy,
    messages: List[Message],
    provider: BaseLLMProvider,
    config: CompactionConfig,
) -> List[Message]:
    match strategy:
        case CompactionStrategy.SLIDING_WINDOW:
            sliding_window_compaction = SlidingWindow(
                window_size=config.sliding_window_size
            )
            messages = await sliding_window_compaction.compact(messages)
            return messages
        case CompactionStrategy.SUMMARIZATION:
            summarization = Summarization(
                threshold=config.summarization_threshold,
                keep_last_n=config.summarization_keep,
                provider=provider,
            )
            messages = await summarization.compact(messages)
            return messages
        case CompactionStrategy.NONE:
            return messages
