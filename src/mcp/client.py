from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Type, cast

from dotenv import load_dotenv
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent
from pydantic.main import BaseModel, create_model

from mcp import ClientSession
from src.tools.interfaces.Tool import Tool, ToolResult

load_dotenv()


def build_args_schema(input_schema: Dict[str, Any]) -> Type[BaseModel]:
    """Convert a JSON Schema dict (from MCP) into a Pydantic BaseModel subclass."""
    properties: Dict[str, Any] = input_schema.get("properties", {})
    required: List[str] = input_schema.get("required", [])

    field_definitions: Dict[str, Any] = {}
    for field_name, field_info in properties.items():
        json_type = field_info.get("type", "string")
        python_type: type = _json_type_to_python(json_type)

        if field_name in required:
            field_definitions[field_name] = (python_type, ...)
        else:
            field_definitions[field_name] = (python_type | None, None)

    return create_model("MCPToolArgs", **field_definitions)


def _json_type_to_python(json_type: str) -> type[Any]:
    mapping: dict[str, type[Any]] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return mapping.get(json_type, cast(type[Any], Any))


class MCPTool(Tool[BaseModel]):
    """Wraps a remote MCP tool as a local Tool[BaseModel]."""

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


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def get_tools(self) -> List[Tool]:
        if self.session is None:
            raise RuntimeError("Not connected. Call connect_to_server() first.")

        response = await self.session.list_tools()
        tool_list: List[Tool] = []

        for mcp_tool in response.tools:
            schema = mcp_tool.inputSchema or {}
            args_schema = build_args_schema(schema)

            tool = MCPTool(
                name=mcp_tool.name,
                description=mcp_tool.description or "",
                args_schema=args_schema,
                session=self.session,
            )
            tool_list.append(tool)

        return tool_list

    async def cleanup(self) -> None:
        await self.exit_stack.aclose()
