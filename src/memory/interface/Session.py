from datetime import datetime

from pydantic import BaseModel

from src.llm.schema.Message import Message


class Session(BaseModel):
    messages: list[Message] = []
    created_at: datetime = datetime.now()

    @property
    def id(self) -> str:
        return self.created_at.strftime("%Y%m%d%H%M%S")

    def append(self, message: Message) -> None:
        self.messages.append(message)
