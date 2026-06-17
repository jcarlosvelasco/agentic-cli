from llm.interfaces.BaseLLMProvider import BaseLLMProvider
from pydantic import BaseModel
from tools.interfaces.Tool import Tool, ToolResult


class RecallToolArgs(BaseModel):

class RecallTool(Tool[RecallToolArgs]):
    def __init__(self, provider: BaseLLMProvider):
        super().__init__(
            name="recall",
            description=(
                "Searches past session summaries and explicit memories. Use this when you need context from previous conversations - decisions, facts, preferences, or past work on a topic."
            ),
            args_schema=RecallToolArgs
        )

    async def execute(self, args: RecallToolArgs | None) -> ToolResult:
