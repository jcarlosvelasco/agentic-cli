from pydantic import BaseModel


class MCPConfig(BaseModel):
    enabled: bool = True
    config_file: str = "src/mcp_integration/mcp.json"
