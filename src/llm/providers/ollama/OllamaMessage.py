from typing import TypedDict


class OllamaMessage(TypedDict):
    role: str
    content: str
