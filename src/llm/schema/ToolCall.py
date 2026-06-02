from typing import Any

from pydantic import BaseModel


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any]
