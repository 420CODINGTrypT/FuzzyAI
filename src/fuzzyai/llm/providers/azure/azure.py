import logging
import os
from typing import Any, AsyncGenerator

import aiohttp

from fuzzyai.llm.providers.base import BaseLLMProvider, register_provider
from fuzzyai.llm.providers.enums import LLMProvider

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30)


class AzureProvider(BaseLLMProvider):
    
    def __init__(self, model: str, **extra: Any) -> None:
        super().__init__(model, **extra)
        self._api_key = self._extra.get("api_key") or os.environ.get("AZURE_OPENAI_API_KEY", "")
        self._endpoint = self._extra.get("endpoint") or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        
    def _get_url(self) -> str:
        return f"{self._endpoint}/openai/deployments/{self._model}/chat/completions?api-version=2024-02-01"
    
    async def generate(self, prompt: str, **extra: Any) -> AsyncGenerator[Any, None]:
        headers = {
            "api-key": self._api_key,
            "Content-Type": "application/json"
        }
        
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": extra.get("max_tokens", 100)
        }
        
        async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
            async with session.post(self._get_url(), headers=headers, json=body) as response:
                if response.status != 200:
                    logger.error(f"Azure API error: {response.status}")
                    raise RuntimeError(f"Azure API error: {response.status}")
                
                data = await response.json()
                for choice in data.get("choices", []):
                    yield type('Response', (), {'response': choice.get("message", {}).get("content", "")})()
    
    async def close(self) -> None:
        pass

    @staticmethod
    def supported_models() -> list[str]:
        return ["gpt-4", "gpt-35-turbo"]
    
    @classmethod
    def get_supported_models(cls) -> list[str]:
        return cls.supported_models()


register_provider(LLMProvider.AZURE, AzureProvider)
