from typing import List

from src.tools.interfaces.Tool import Tool
from src.tools.registry.bash import BashTool

TOOLS: List[Tool] = [
    BashTool(),
]


def get_tools() -> list[Tool]:
    return TOOLS
