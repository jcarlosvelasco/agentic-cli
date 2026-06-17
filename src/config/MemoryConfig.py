from pydantic.main import BaseModel


class MemoryConfig(BaseModel):
    enabled: bool = True
    max_recent_sessions: int = 5
    enable_preamble: bool = True
