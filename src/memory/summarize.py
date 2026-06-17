from pathlib import Path

from src.llm.interfaces.BaseLLMProvider import BaseLLMProvider
from src.llm.schema.Message import Message, MessageRole
from src.memory.interface.Session import Session
from src.memory.interface.SessionIndex import SessionIndex
from src.memory.utils import get_session_folder_path


async def summarize(session_id: str, provider: BaseLLMProvider) -> None:
    min_messages = 2

    save_path = get_session_folder_path()

    session_path = Path(save_path) / f"session_{session_id}.json"

    session = Session.model_validate_json(session_path.read_text())
    if len(session.messages) < min_messages:
        return

    transcript = "\n".join(
        f"{m.role}: {m.content}" for m in session.messages if m.content
    )

    summary_prompt = (
        "Summarize the following coding-agent session in one paragraph "
        "followed by 3-5 single-word tags on a separate line starting with 'Tags:'. "
        "Focus on what was accomplished, decisions made, and key facts learned.\n\n"
        f"{transcript}"
    )

    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content=summary_prompt),
    ]

    response = await provider.chat(messages, tools=[])
    content = response.content.strip()

    if "Tags:" in content:
        parts = content.split("Tags:")
        summary = parts[0].strip()
        tags = [t.strip() for t in parts[1].strip().split(",")]
    else:
        summary = content
        tags = []

    SessionIndex.update(session_id, summary, tags)
