from pydantic.main import BaseModel


class LLMChatResponse(BaseModel):
    content: str
