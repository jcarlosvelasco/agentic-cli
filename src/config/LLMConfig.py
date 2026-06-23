from pydantic.main import BaseModel


class LLMConfig(BaseModel):
    model: str = "gemma4:e2b-mlx"
    base_url: str = "http://localhost:11434/api"
    api_key: str | None = None
