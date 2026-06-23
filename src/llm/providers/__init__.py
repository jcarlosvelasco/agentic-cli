import os
from typing import Any

from src.config.LLMConfig import LLMConfig
from src.llm.interfaces.BaseLLMProvider import BaseLLMProvider
from src.llm.providers.ollama.OllamaProvider import OllamaProvider
from src.llm.providers.openrouter.OpenRouterProvider import OpenRouterProvider


def create_provider(config: LLMConfig) -> BaseLLMProvider[Any]:
    if config.provider == "ollama":
        return OllamaProvider(model=config.model, base_url=config.base_url)

    if config.provider == "openrouter":
        api_key = os.getenv(config.api_key) if config.api_key else None
        if not api_key:
            raise ValueError(
                f"OpenRouter API key not found. Set {config.api_key} in .env"
            )
        return OpenRouterProvider(
            model=config.model,
            base_url=config.base_url,
            api_key=api_key,
        )

    raise ValueError(f"Unknown provider: {config.provider}")
