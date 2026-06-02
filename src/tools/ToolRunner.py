from typing import Any

from src.tools.interfaces.Tool import Tool


class ToolRunner:
    def run(self, tool: Tool, tool_input: Any | None):
        print(f"\nExecute tool {tool.name} with input: {tool_input}")
        confirm = input("¿Execute? (y/n): ").strip().lower()

        if confirm != "y":
            print("Canceled execution")
            return None

        return tool.execute(tool_input)
