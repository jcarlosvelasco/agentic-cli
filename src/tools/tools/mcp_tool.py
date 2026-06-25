from typing import Type

from dotenv import load_dotenv
from mcp.types import TextContent
from pydantic import BaseModel

from mcp import ClientSession
from src.tools.interfaces.Tool import Tool, ToolResult

load_dotenv()


class MCPTool(Tool[BaseModel]):
    def __init__(
        self,
        name: str,
        description: str,
        args_schema: Type[BaseModel],
        session: ClientSession,
    ):
        super().__init__(name=name, description=description, args_schema=args_schema)
        self._session = session

    async def execute(self, args: BaseModel | None) -> ToolResult:
        if args is None:
            return ToolResult(success=False, message="No arguments provided")

        raw_args = args.model_dump(exclude_none=True)

        try:
            response = await self._session.call_tool(self.name, raw_args)
            text_output = "\n".join(
                block.text
                for block in response.content
                if isinstance(block, TextContent)
            )
            return ToolResult(success=True, data=text_output)
        except Exception as e:
            return ToolResult(success=False, message=str(e))
