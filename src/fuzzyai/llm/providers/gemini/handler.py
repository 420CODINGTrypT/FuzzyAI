import logging
import os
from typing import Any, AsyncGenerator

import aiohttp

from fuzzyai.llm.providers.base import BaseLLMProvider, register_provider
from fuzzyai.llm.providers.enums import LLMProvider

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30)


class GeminiProvider(BaseLLMProvider):
    
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    
    def __init__(self, model: str, **extra: Any) -> None:
        super().__init__(model, **extra)
        self._api_key = self._extra.get("api_key") or os.environ.get("GEMINI_API_KEY", "")
    
    async def generate(self, prompt: str, **extra: Any) -> AsyncGenerator[Any, None]:
        # API key sent as secure header, not URL query parameter
        headers = {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json"
        }
        
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": extra.get("max_tokens", 100)
            }
        }
        
        url = f"{self.API_URL}/{self._model}:generateContent"
        
        async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
            async with session.post(url, headers=headers, json=body) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Gemini API error: {response.status}")
                    raise RuntimeError(f"Gemini API error: {response.status}")
                
                data = await response.json()
                candidates = data.get("candidates", [])
                for candidate in candidates:
                    content = candidate.get("content", {})
                    parts = content.get("parts", [])
                    for part in parts:
                        yield type('Response', (), {'response': part.get("text", "")})()
    
    async def close(self) -> None:
        pass

    @staticmethod
    def supported_models() -> list[str]:
        return ["gemini-pro", "gemini-1.5-pro-latest"]
    
    @classmethod
    def get_supported_models(cls) -> list[str]:
        return cls.supported_models()


register_provider(LLMProvider.GEMINI, GeminiProvider)
