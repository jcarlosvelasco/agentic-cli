import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from memory.utils import get_session_folder_path
from src.llm.schema.Message import Message


class Session(BaseModel):
    messages: list[Message] = []
    created_at: str = datetime.now().isoformat()

    @property
    def id(self) -> str:
        return self.created_at

    def append(self, message: Message) -> None:
        self.messages.append(message)

    def save(self) -> None:
        data = self.model_dump()
        folder = get_session_folder_path()

        if not Path(folder).exists():
            Path(folder).mkdir(parents=True)

        with open(f"{folder}/session_{self.id}.json", "w") as f:
            json.dump(data, f)
