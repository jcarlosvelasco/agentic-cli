from typing import List

from memory.interface.Session import Session
from src.llm.interfaces.BaseLLMProvider import BaseLLMProvider
from src.tools.interfaces.Tool import Tool
from src.tools.tools.bash import BashTool
from src.tools.tools.launch_subagent import LaunchSubagentTool
from src.tools.tools.read_file import ReadFileTool
from src.tools.tools.weather import WeatherTool
from src.tools.tools.web_search import WebSearchTool
from src.tools.tools.write_file import WriteFileTool


class ToolRegistry:
    def __init__(self, provider: BaseLLMProvider, session: Session):
        self._tools: List[Tool] = [
            BashTool(),
            WeatherTool(),
            LaunchSubagentTool(provider=provider, session=session),
            WebSearchTool(),
            ReadFileTool(),
            WriteFileTool(),
        ]
        self._by_name = {tool.name: tool for tool in self._tools}

    def register(self, tool: Tool) -> None:
        self._tools.append(tool)
        self._by_name[tool.name] = tool

    def get_tools(self) -> List[Tool]:
        return self._tools

    def get_tool_by_name(self, name: str) -> Tool | None:
        return self._by_name.get(name)
