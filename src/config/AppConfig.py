from pydantic.fields import Field
from pydantic.main import BaseModel

from config.CompactionConfig import CompactionConfig
from config.LLMConfig import LLMConfig
from config.MCPConfig import MCPConfig
from config.MemoryConfig import MemoryConfig
from config.ToolsConfig import ToolsConfig


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
