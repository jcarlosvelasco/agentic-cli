from pydantic import Field
from pydantic.main import BaseModel

from src.tools.interfaces.Tool import Tool, ToolResult


class WriteFileArgs(BaseModel):
    file_path: str = Field(description="The path to the file to write")
    content: str = Field(description="The content to write to the file")


class WriteFileTool(Tool[WriteFileArgs]):
    def __init__(self):
        super().__init__(
            name="write_file",
            description="Writes content to a file",
            args_schema=WriteFileArgs,
        )

    async def execute(self, args: WriteFileArgs | None) -> ToolResult:
        if not args:
            return ToolResult(success=False, message="Invalid arguments")

        try:
            with open(args.file_path, "w") as f:
                f.write(args.content)
            return ToolResult(success=True, message="File written successfully")
        except Exception as e:
            return ToolResult(success=False, message=str(e))
