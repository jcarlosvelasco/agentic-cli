from typing import List

from src.tools.interfaces.Tool import Tool
from src.tools.registry.bash import BashTool
from src.tools.registry.weather import WeatherTool

TOOLS: List[Tool] = [
    BashTool(),
    WeatherTool(),
]


def get_tools() -> list[Tool]:
    return TOOLS
