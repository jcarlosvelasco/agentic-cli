import subprocess

from pydantic import BaseModel, Field

from src.tools.interfaces.Tool import Tool, ToolResult


class BashToolArgs(BaseModel):
    command: str = Field(description="The bash command to execute")


class BashTool(Tool[BashToolArgs]):
    def __init__(self):
        super().__init__(
            name="bash", description="Executes bash commands", args_schema=BashToolArgs
        )

    def execute(self, args: BashToolArgs | None) -> ToolResult:
        if isinstance(args, BashToolArgs):
            subprocess.run(["bash", "-c", args.command])
            return ToolResult(success=True)
        return ToolResult(success=False, message="Invalid arguments")
