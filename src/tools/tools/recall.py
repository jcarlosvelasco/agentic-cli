import json
from pathlib import Path

from pydantic import BaseModel, Field

from src.llm.interfaces.BaseLLMProvider import BaseLLMProvider
from src.llm.schema.Message import Message, MessageRole
from src.memory.utils import get_session_folder_path, get_session_index_file_path
from src.tools.interfaces.Tool import Tool, ToolResult


class RecallToolArgs(BaseModel):
    query: str = Field(description="The topic or question to search past sessions for")


class RecallTool(Tool[RecallToolArgs]):
    def __init__(self, provider: BaseLLMProvider):
        super().__init__(
            name="recall",
            description=(
                "Searches past session summaries and explicit memories."
                "When the user asks anything that could be answered by previous sessions "
                "(e.g., their name, preferences, past decisions, what you worked on before, "
                "or any fact that might have been established earlier)"
            ),
            args_schema=RecallToolArgs,
        )
        self._provider = provider

    async def execute(self, args: RecallToolArgs | None) -> ToolResult:
        if not isinstance(args, RecallToolArgs):
            return ToolResult(success=False, message="Invalid arguments")

        index_path = get_session_index_file_path()
        if not Path(index_path).exists():
            return ToolResult(success=False, message="No session history found")

        with open(index_path) as f:
            entries: list[dict] = json.load(f)

        if not entries:
            return ToolResult(success=False, message="No session history found")

        sessions_dir = get_session_folder_path()
        candidates = []

        for entry in entries:
            session_path = entry.get("session_path", "")
            summary = entry.get("summary", "")
            tags = entry.get("tags", [])

            session_file = Path(f"{session_path}.json")
            if not session_file.exists():
                session_file = (
                    Path(sessions_dir) / f"session_{Path(session_path).name}.json"
                )
            if not session_file.exists():
                continue

            session_data = json.loads(session_file.read_text())
            transcript = "\n".join(
                f"{m['role']}: {m['content']}"
                for m in session_data.get("messages", [])
                if m.get("content")
            )

            candidates.append(
                {
                    "path": str(session_file),
                    "summary": summary,
                    "tags": tags,
                    "transcript": transcript,
                }
            )

        search_prompt = (
            "You are a memory retriever. Given a query and a list of past conversation summaries, "
            "identify which sessions are relevant. For each relevant session, explain briefly why.\n\n"
            f"Query: {args.query}\n\n"
            "Sessions:\n"
            + "\n---\n".join(
                f"Session {i}:\nSummary: {c['summary']}\nTags: {', '.join(c['tags'])}"
                for i, c in enumerate(candidates)
            )
            + "\n\nReturn the indices (0-based) of relevant sessions separated by commas (e.g., '0,2'). "
            "If none are relevant, return 'NONE'."
        )

        messages = [
            Message(role=MessageRole.SYSTEM, content="You are a memory retriever."),
            Message(role=MessageRole.USER, content=search_prompt),
        ]

        response = await self._provider.chat(messages, tools=[])
        content = response.content.strip()

        if content.upper() == "NONE":
            return ToolResult(
                success=True,
                message="No relevant past sessions found.",
                data=None,
            )

        indices = []
        for part in content.split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part)
                if 0 <= idx < len(candidates):
                    indices.append(idx)

        if not indices:
            return ToolResult(
                success=True,
                message="No relevant past sessions found.",
                data=None,
            )

        results = []
        for i in indices:
            c = candidates[i]
            results.append(
                {
                    "summary": c["summary"],
                    "tags": c["tags"],
                    "transcript": c["transcript"],
                }
            )

        return ToolResult(
            success=True,
            message=f"Found {len(results)} relevant session(s).",
            data=results,
        )
