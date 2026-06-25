from pydantic import BaseModel


class ChatBody(BaseModel):
    query: str
    session_id: str | None = None
