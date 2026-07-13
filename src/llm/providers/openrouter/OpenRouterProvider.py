from llm.providers.openai_base.OpenAICompatibleProvider import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    def __init__(self, model: str, base_url: str, api_key: str):
        super().__init__(model=model, base_url=base_url, api_key=api_key)
