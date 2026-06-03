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
            result = subprocess.run(
                ["bash", "-c", args.command], capture_output=True, text=True
            )
            return ToolResult(success=True, data={"output": result.stdout})
        return ToolResult(success=False, message="Invalid arguments")
