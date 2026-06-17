import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from src.memory.utils import (
    get_session_folder_path,
    get_session_index_file_path,
)


class SessionIndex(BaseModel):
    created_at: datetime
    summary: str
    tags: list[str]
    session_path: str

    @property
    def id(self) -> str:
        return self.created_at.strftime("%Y%m%d%H%M%S")

    @classmethod
    def create(cls) -> None:
        index_path = get_session_index_file_path()
        formatted_index_file_path = Path(index_path)

        if not formatted_index_file_path.exists():
            formatted_index_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(index_path, "w") as f:
                json.dump([], f)

    @classmethod
    def update(cls, session_id: str, summary: str, tags: list[str]) -> None:
        index_path = get_session_index_file_path()
        formatted_index_file_path = Path(index_path)

        if not formatted_index_file_path.exists():
            cls.create()

        entries: list[dict] = []
        with open(index_path) as f:
            entries = json.load(f)

        existing = [
            e for e in entries if e.get("session_path", "").endswith(session_id)
        ]

        entry = {
            "summary": summary,
            "tags": tags,
            "session_path": str(Path(get_session_folder_path()) / session_id),
        }

        if existing:
            for i, e in enumerate(entries):
                if e.get("session_path", "").endswith(session_id):
                    entries[i] = entry
                    break
        else:
            entries.append(entry)

        with open(index_path, "w") as f:
            json.dump(entries, f, indent=2)
