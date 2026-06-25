from pathlib import Path
from typing import List

from src.mcp_integration.client import MCPClient
from src.mcp_integration.mcp_config import MCPServerConfig, load_mcp_config
from src.tools.interfaces.Tool import Tool


class MCPRegistry:
    def __init__(self, configs: list[MCPServerConfig]):
        self._configs = configs
        self._clients: list[MCPClient] = []

    @classmethod
    def from_file(cls, path: str | Path = "mcp.json") -> "MCPRegistry":
        return cls(load_mcp_config(path))

    async def load_all(self) -> List[Tool]:
        all_tools: List[Tool] = []

        for config in self._configs:
            client = MCPClient()
            try:
                await client.connect_to_command(
                    command=config.command,
                    args=config.args,
                    env=config.env,
                )
                tools = await client.get_tools(server_name=config.name)
                all_tools.extend(tools)
                self._clients.append(client)
            except Exception as e:
                print(f"[MCP] Failed to load server '{config.name}': {e}")
                await client.cleanup()

        return all_tools

    async def cleanup(self) -> None:
        for client in self._clients:
            await client.cleanup()
