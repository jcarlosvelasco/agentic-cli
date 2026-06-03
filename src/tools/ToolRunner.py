from typing import Any

from src.tools.interfaces.Tool import Tool, ToolResult


class ToolRunner:
    def run(self, tool: Tool, tool_input: Any | None) -> ToolResult:
        print(f"\nExecute tool {tool.name} with input: {tool_input}")
        confirm = input("¿Execute? (y/n): ").strip().lower()

        if confirm != "y":
            print("Canceled execution")
            return ToolResult(success=False, message="Canceled execution")

        args = None
        if tool.args_schema and isinstance(tool_input, dict):
            args = tool.args_schema(**tool_input)
        else:
            args = tool_input

        return tool.execute(args)
