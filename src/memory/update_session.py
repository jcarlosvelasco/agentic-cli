import json
from pathlib import Path

from src.memory.utils import (
    get_session_folder_path,
    get_session_index_file_path,
)


def update_session_index(session_id: str, summary: str, tags: list[str]) -> None:
    index_path = get_session_index_file_path()
    formatted_index_path = Path(index_path)

    if not formatted_index_path.exists():
        formatted_index_path.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    if formatted_index_path.stat().st_size > 0:
        with open(index_path) as f:
            entries = json.load(f)

    existing = [e for e in entries if e.get("session_path", "").endswith(session_id)]

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
