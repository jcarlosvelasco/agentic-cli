from pydantic import BaseModel, Field

from src.tools.interfaces.Tool import Tool, ToolResult


class ReadFileArgs(BaseModel):
    path: str = Field(description="The path to the file to read")


class ReadFileTool(Tool[ReadFileArgs]):
    def __init__(self):
        super().__init__(
            name="read_file",
            description="Reads the contents of a file",
            args_schema=ReadFileArgs,
        )

    async def execute(self, args: ReadFileArgs | None) -> ToolResult:
        if not args:
            return ToolResult(success=False, message="Invalid arguments")

        try:
            with open(args.path, "r") as f:
                content = f.read()
            return ToolResult(success=True, data=content)
        except Exception as e:
            return ToolResult(success=False, message=str(e))
