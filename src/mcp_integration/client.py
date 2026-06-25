import asyncio
import os
from contextlib import AsyncExitStack
from typing import List, Optional

from dotenv import load_dotenv

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from src.mcp_integration.utils import build_args_schema
from src.tools.interfaces.Tool import Tool
from src.tools.tools.mcp_tool import MCPTool

load_dotenv()


class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_command(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
    ) -> None:
        merged_env = {**os.environ, **(env or {})}
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=merged_env,
        )
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        await self.session.initialize()
        response = await self.session.list_tools()
        print(f"[MCP] Connected with tools: {[t.name for t in response.tools]}")

    async def get_tools(self, server_name: str) -> List[Tool]:
        if self.session is None:
            raise RuntimeError("Not connected. Call connect_to_command() first.")

        response = await self.session.list_tools()
        tool_list: List[Tool] = []

        for mcp_tool in response.tools:
            schema = mcp_tool.inputSchema or {}
            args_schema = build_args_schema(schema)

            tool = MCPTool(
                name=f"{server_name}.{mcp_tool.name}",
                description=mcp_tool.description or "",
                args_schema=args_schema,
                session=self.session,
            )
            tool_list.append(tool)

        return tool_list

    async def cleanup(self) -> None:
        try:
            await self.exit_stack.aclose()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[MCP] Cleanup warning: {e}")
