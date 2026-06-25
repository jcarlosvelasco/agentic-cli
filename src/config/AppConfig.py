from pydantic.fields import Field
from pydantic.main import BaseModel

from src.config.CompactionConfig import CompactionConfig
from src.config.LLMConfig import LLMConfig
from src.config.MCPConfig import MCPConfig
from src.config.MemoryConfig import MemoryConfig
from src.config.ToolsConfig import ToolsConfig
from src.config.UIConfig import UIConfig


class AppConfig(BaseModel):
    mode: str = "cli"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
