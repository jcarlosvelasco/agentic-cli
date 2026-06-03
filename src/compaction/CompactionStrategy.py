from enum import StrEnum


class CompactionStrategy(StrEnum):
    NONE = "none"
    SLIDING_WINDOW = "sliding_window"
    SUMMARIZATION = "summarization"
