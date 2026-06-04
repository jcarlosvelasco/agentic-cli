from typing import List, Optional

from pydantic import Field
from pydantic.main import BaseModel

from src.llm.interfaces.BaseLLMProvider import BaseLLMProvider
from src.tools.interfaces.Tool import Tool, ToolResult

DEFAULT_SYSTEM_PROMPT = """You are a specialized subagent. Complete the given task
using only the tools available to you. Be concise — return only the result,
no explanations unless asked."""


class LaunchSubagentArgs(BaseModel):
    task: str = Field(description="The task for the subagent to complete")
    tools: List[str] = Field(
        description="List of tool names the subagent can use",
        examples=[["read_file", "search_in_files"]],
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt to specialize the subagent",
    )
    max_iterations: int = Field(
        default=10,
        description="Max iterations before stopping",
    )


class LaunchSubagentTool(Tool[LaunchSubagentArgs]):
    def __init__(self, provider: BaseLLMProvider):
        super().__init__(
            name="launch_subagent",
            description=(
                "Launch a subagent to complete a specific task. "
                "The subagent runs its own loop with a restricted set of tools. "
                "Use this to parallelize independent tasks."
            ),
            args_schema=LaunchSubagentArgs,
        )

        self._provider = provider

    async def execute(self, args: LaunchSubagentArgs | None) -> ToolResult:
        from src.agent.schema.Agent import Agent
        from src.tools.registry import ToolRegistry

        if not args:
            return ToolResult(success=False, message="Invalid arguments")

        registry = ToolRegistry(provider=self._provider)
        all_tools_by_name = {tool.name: tool for tool in registry.get_tools()}

        subagent_tools = []
        missing = []

        for tool_name in args.tools:
            tool = all_tools_by_name.get(tool_name)
            if tool:
                subagent_tools.append(tool)
            else:
                missing.append(tool_name)

        if missing:
            return ToolResult(
                success=False, message=f"Error: tools not found: {missing}"
            )

        subagent = Agent(
            name=f"subagent_{args.task[:20]}",
            provider=self._provider,
            tools=subagent_tools,
            system_prompt=args.system_prompt,
        )
        result = await subagent.run(args.task)
        return ToolResult(success=True, data=result)
