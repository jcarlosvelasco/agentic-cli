import json
from pathlib import Path

from memory.utils import get_session_index_file_path


async def preamble() -> str:
    index_path = get_session_index_file_path()
    index_path_obj = Path(index_path)
    if not index_path_obj.exists() or index_path_obj.stat().st_size == 0:
        print(f"No memory found at {index_path}")
        return ""

    with open(index_path) as f:
        entries: list[dict] = json.load(f)

    recent = entries[-5:]
    if not recent:
        print("No recent session summaries found")
        return ""

    lines = ["## Recent session summaries"]

    for e in recent:
        date = Path(e.get("session_path", "")).stem
        summary = e.get("summary", "")
        tags = ", ".join(e.get("tags", []))
        lines.append(f"- [{date}] {summary}")
        if tags:
            lines.append(f"  tags: {tags}")

    return "\n".join(lines)
