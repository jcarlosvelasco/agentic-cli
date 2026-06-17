from pydantic import BaseModel

from compaction.CompactionStrategy import CompactionStrategy


class CompactionConfig(BaseModel):
    enabled: bool = True
    strategy: CompactionStrategy = CompactionStrategy.NONE
    sliding_window_size: int = 20
    summarization_threshold: int = 20
    summarization_keep: int = 6
