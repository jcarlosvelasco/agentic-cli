from typing import Any

from src.tools.interfaces.Tool import Tool, ToolResult


class ToolRunner:
    def run(self, tool: Tool, tool_input: Any | None) -> ToolResult:
        args = None
        if tool.args_schema and isinstance(tool_input, dict):
            args = tool.args_schema(**tool_input)
        else:
            args = tool_input

        return tool.execute(args)
