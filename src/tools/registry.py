from typing import List

from src.llm.interfaces.BaseLLMProvider import BaseLLMProvider
from src.tools.interfaces.Tool import Tool
from src.tools.tools.bash import BashTool
from src.tools.tools.launch_subagent import LaunchSubagentTool
from src.tools.tools.weather import WeatherTool
from src.tools.tools.web_search import WebSearchTool


class ToolRegistry:
    def __init__(self, provider: BaseLLMProvider):
        self._tools: List[Tool] = [
            BashTool(),
            WeatherTool(),
            LaunchSubagentTool(provider=provider),
            WebSearchTool(),
        ]
        self._by_name = {tool.name: tool for tool in self._tools}

    def get_tools(self) -> List[Tool]:
        return self._tools

    def get_tool_by_name(self, name: str) -> Tool | None:
        return self._by_name.get(name)
