from dotenv import load_dotenv
from pydantic import BaseModel, Field
from tavily import TavilyClient
from tavily.client import os

from src.tools.interfaces.Tool import Tool, ToolResult

load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=tavily_api_key)


class WebSearchArgs(BaseModel):
    query: str = Field(description="The search query to execute")
    topic: str | None = Field(description="The topic to search within", default=None)


class WebSearchTool(Tool[WebSearchArgs]):
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for information",
            args_schema=WebSearchArgs,
        )

    async def execute(self, args: WebSearchArgs | None) -> ToolResult:
        if not isinstance(args, WebSearchArgs):
            return ToolResult(success=False, message="Invalid arguments")

        results = tavily.search(
            query=args.query,
            max_results=8,
            topic=args.topic,
        )

        return ToolResult(success=True, data=results)
