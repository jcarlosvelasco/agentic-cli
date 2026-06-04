from typing import Any

from src.tools.interfaces.Tool import Tool, ToolResult


class ToolRunner:
    async def run(self, tool: Tool, tool_input: Any | None) -> ToolResult:
        args = None
        if tool.args_schema and isinstance(tool_input, dict):
            args = tool.args_schema(**tool_input)
        else:
            args = tool_input

        result = await tool.execute(args)
        return result
