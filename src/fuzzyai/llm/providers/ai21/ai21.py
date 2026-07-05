import logging
from typing import Any, AsyncGenerator, Optional

import aiohttp

from fuzzyai.llm.providers.base import register_provider
from fuzzyai.llm.providers.enums import LLMProvider
from fuzzyai.llm.providers.openai.openai import OpenAIProvider

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30)


class AI21Provider(OpenAIProvider):

    def __init__(self, model: str, **extra: Any) -> None:
        super().__init__(model, **extra)
        self._api_key = self._api_key or ""
        
    def _get_endpoint(self) -> str:
        return "https://api.ai21.com/studio/v1"
    
    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
    
    def _get_request_body(self, prompt: str, **extra: Any) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            **extra
        }

    async def generate(self, prompt: str, **extra: Any) -> AsyncGenerator[Any, None]:
        import aiohttp
        
        body = self._get_request_body(prompt, **extra)
        headers = self._get_headers()
        
        async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
            async with session.post(f"{self._get_endpoint()}/chat/completions", headers=headers, json=body) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"AI21 API error: {response.status} - {error_text}")
                    raise RuntimeError(f"AI21 API error: {response.status}")
                
                data = await response.json()
                for choice in data.get("choices", []):
                    yield choice.get("message", {}).get("content", "")

    @staticmethod
    def supported_models() -> list[str]:
        return ["jamba-1.5-mini", "jamba-1.5-large"]

    @classmethod
    def get_supported_models(cls) -> list[str]:
        return cls.supported_models()


register_provider(LLMProvider.AI21, AI21Provider)
