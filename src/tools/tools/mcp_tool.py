from typing import Any, Dict, List, Type, cast

from dotenv import load_dotenv
from mcp.types import TextContent
from pydantic import BaseModel, create_model

from mcp import ClientSession
from src.tools.interfaces.Tool import Tool, ToolResult

load_dotenv()


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


def build_args_schema(input_schema: Dict[str, Any]) -> Type[BaseModel]:
    properties: Dict[str, Any] = input_schema.get("properties", {})
    required: List[str] = input_schema.get("required", [])

    field_definitions: Dict[str, Any] = {}
    for field_name, field_info in properties.items():
        json_type = field_info.get("type", "string")
        python_type = _json_type_to_python(json_type)

        if field_name in required:
            field_definitions[field_name] = (python_type, ...)
        else:
            field_definitions[field_name] = (python_type | None, None)

    return create_model("MCPToolArgs", **field_definitions)


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
