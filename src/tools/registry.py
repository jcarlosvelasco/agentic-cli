from typing import List

from src.config.AppConfig import AppConfig
from src.memory.Session import Session
from src.llm.interfaces.BaseLLMProvider import BaseLLMProvider
from src.tools.interfaces.Tool import Tool
from src.tools.tools.bash import BashTool
from src.tools.tools.launch_subagent import LaunchSubagentTool
from src.tools.tools.read_file import ReadFileTool
from src.tools.tools.recall import RecallTool
from src.tools.tools.weather import WeatherTool
from src.tools.tools.web_search import WebSearchTool
from src.tools.tools.write_file import WriteFileTool


class ToolRegistry:
    def __init__(self, provider: BaseLLMProvider, session: Session, config: AppConfig):
        self._config = config
        self._tools: List[Tool] = [
            BashTool(),
            RecallTool(provider=provider),
            WeatherTool(),
            LaunchSubagentTool(provider=provider, session=session, config=config),
            WebSearchTool(),
            ReadFileTool(),
            WriteFileTool(),
        ]
        self._by_name = {tool.name: tool for tool in self._tools}

    def register(self, tool: Tool) -> None:
        self._tools.append(tool)
        self._by_name[tool.name] = tool

    def get_tools(self) -> List[Tool]:
        if self._config.tools.enabled:
            return self._tools

        return []

    def get_tool_by_name(self, name: str) -> Tool | None:
        return self._by_name.get(name)
