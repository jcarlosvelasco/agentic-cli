from llm.providers.openai_base.OpenAICompatibleProvider import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    def __init__(self, model: str, base_url: str):
        super().__init__(model=model, base_url=base_url, api_key="ollama")
