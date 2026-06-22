from typing import Any

from src.shared.console import confirm_execution
from src.tools.interfaces.Tool import Tool, ToolResult


class ToolRunner:
    async def run(
        self, tool: Tool, tool_input: Any | None, should_confirm: bool
    ) -> ToolResult:
        if should_confirm:
            confirmed = confirm_execution(tool.name, tool_input)
            if not confirmed:
                return ToolResult(
                    success=False,
                    message="Tool execution cancelled by user",
                    data=None,
                )

        args = None
        if tool.args_schema and isinstance(tool_input, dict):
            args = tool.args_schema(**tool_input)
        else:
            args = tool_input

        result = await tool.execute(args)
        return result
