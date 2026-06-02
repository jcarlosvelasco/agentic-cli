from enum import StrEnum


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message:
    def __init__(self, role: MessageRole, content: str):
        self.role = role
        self.content = content
