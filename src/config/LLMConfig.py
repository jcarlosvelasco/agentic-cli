from pydantic.main import BaseModel


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "gemma4:e2b-mlx"
    base_url: str = "http://localhost:11434/v1"
    api_key: str | None = None
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0
