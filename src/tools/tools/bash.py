import asyncio

from pydantic import BaseModel, Field

from src.tools.interfaces.Tool import Tool, ToolResult


class BashToolArgs(BaseModel):
    command: str = Field(description="The bash command to execute")


class BashTool(Tool[BashToolArgs]):
    def __init__(self):
        super().__init__(
            name="bash", description="Executes bash commands", args_schema=BashToolArgs
        )

    async def execute(self, args: BashToolArgs | None) -> ToolResult:
        if not isinstance(args, BashToolArgs):
            return ToolResult(success=False, message="Invalid arguments")

        try:
            process = await asyncio.create_subprocess_exec(
                "bash", "-c", args.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            output = stdout.decode()
            stderr_text = stderr.decode()
            if stderr_text:
                output += stderr_text

            return ToolResult(
                success=process.returncode == 0,
                data={"output": output},
                message=stderr_text if stderr_text else None,
            )
        except Exception as e:
            return ToolResult(success=False, message=str(e))
