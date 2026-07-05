import logging
from typing import Any, AsyncGenerator

import aiohttp

from fuzzyai.llm.providers.base import BaseLLMProvider, register_provider
from fuzzyai.llm.providers.enums import LLMProvider

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30)


class AnthropicProvider(BaseLLMProvider):
    
    API_URL = "https://api.anthropic.com/v1/messages"
    
    def __init__(self, model: str, **extra: Any) -> None:
        super().__init__(model, **extra)
        self._api_key = self._extra.get("api_key") or ""
        if not self._api_key:
            import os
            self._api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    
    async def generate(self, prompt: str, **extra: Any) -> AsyncGenerator[Any, None]:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": extra.get("max_tokens", 1000)
        }
        
        async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
            async with session.post(self.API_URL, headers=headers, json=body) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Anthropic API error: {response.status}")
                    raise RuntimeError(f"Anthropic API error: {response.status}")
                
                data = await response.json()
                content = data.get("content", [])
                if content:
                    yield type('Response', (), {'response': content[0].get("text", "")})()
    
    async def close(self) -> None:
        pass

    @staticmethod
    def supported_models() -> list[str]:
        return [
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1"
        ]
    
    @classmethod
    def get_supported_models(cls) -> list[str]:
        return cls.supported_models()


register_provider(LLMProvider.ANTHROPIC, AnthropicProvider)
