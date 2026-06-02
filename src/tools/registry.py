from typing import List

from src.tools.interfaces.Tool import Tool
from src.tools.tools.bash import BashTool
from src.tools.tools.weather import WeatherTool

TOOLS: List[Tool] = [BashTool(), WeatherTool()]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


def get_tools() -> list[Tool]:
    return TOOLS


def get_tool_by_name(name: str) -> Tool | None:
    return TOOLS_BY_NAME.get(name)
