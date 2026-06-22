import httpx
from pydantic import BaseModel, Field

from src.tools.interfaces.Tool import Tool, ToolResult


class WeatherToolArgs(BaseModel):
    city: str = Field(default="Madrid", description="City name to get weather for")


class WeatherTool(Tool[WeatherToolArgs]):
    def __init__(self):
        super().__init__(
            name="weather",
            description="Gets the current weather for a given city",
            args_schema=WeatherToolArgs,
        )

    async def execute(self, args: WeatherToolArgs | None) -> ToolResult:
        if not isinstance(args, WeatherToolArgs):
            return ToolResult(success=False, message="Invalid arguments")
        city = args.city
        url = f"https://wttr.in/{city}?format=%C+%t+%w+%h"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(url)
                return ToolResult(success=True, data=response.text.strip())
        except httpx.HTTPError as e:
            return ToolResult(success=False, message=f"Error fetching weather: {e}")
