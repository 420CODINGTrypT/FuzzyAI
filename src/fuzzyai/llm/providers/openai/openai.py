import logging
import os
from typing import Any, AsyncGenerator

import aiohttp

from fuzzyai.llm.providers.base import BaseLLMProvider, register_provider
from fuzzyai.llm.providers.enums import LLMProvider

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30)


class OpenAIProvider(BaseLLMProvider):
    
    API_URL = "https://api.openai.com/v1/chat/completions"
    
    def __init__(self, model: str, **extra: Any) -> None:
        super().__init__(model, **extra)
        self._api_key = self._extra.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    
    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
    
    def _get_request_body(self, prompt: str, **extra: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }
        max_tokens = extra.get("max_tokens")
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if "max_completion_tokens" in extra:
            body["max_completion_tokens"] = extra["max_completion_tokens"]
        return body
    
    async def generate(self, prompt: str, **extra: Any) -> AsyncGenerator[Any, None]:
        headers = self._get_headers()
        body = self._get_request_body(prompt, **extra)
        
        async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
            async with session.post(self.API_URL, headers=headers, json=body) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"OpenAI API error: {response.status}")
                    raise RuntimeError(f"OpenAI API error: {response.status}")
                
                data = await response.json()
                for choice in data.get("choices", []):
                    yield type('Response', (), {'response': choice.get("message", {}).get("content", "")})()
    
    async def close(self) -> None:
        pass

    @staticmethod
    def supported_models() -> list[str]:
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "o1-preview",
            "o1-mini",
            "o3-mini"
        ]
    
    @classmethod
    def get_supported_models(cls) -> list[str]:
        return cls.supported_models()


register_provider(LLMProvider.OPENAI, OpenAIProvider)
