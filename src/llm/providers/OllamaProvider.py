import httpx

from src.llm.interfaces.BaseLLMProvider import BaseLLMProvider
from src.llm.schema.ChatConnectionError import ChatConnectionError
from src.llm.schema.ChatResponseError import ChatResponseError
from src.llm.schema.ChatTimeoutError import ChatTimeoutError
from src.llm.schema.LLMChatResponse import LLMChatResponse


class OllamaProvider(BaseLLMProvider):
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/api",
        temperature: float = 0.0,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

    async def chat(self, prompt: str) -> LLMChatResponse:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{self.base_url}/chat",
                    json=payload,
                )

        except httpx.TimeoutException as e:
            raise ChatTimeoutError("Request timeout") from e

        except httpx.ConnectError as e:
            raise ChatConnectionError("Cannot connect to Ollama") from e

        try:
            data = response.json()
        except Exception as e:
            raise ChatResponseError(
                "Invalid JSON response",
                response.status_code,
            ) from e

        if response.status_code != 200:
            raise ChatResponseError(
                data.get("error", "Unknown error"),
                response.status_code,
            )

        if "error" in data:
            raise ChatResponseError(
                data["error"],
                response.status_code,
            )

        return LLMChatResponse(content=data["message"]["content"])
