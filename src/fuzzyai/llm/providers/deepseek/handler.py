import logging
import os
from typing import Any, AsyncGenerator

import aiohttp

from fuzzyai.llm.providers.base import BaseLLMProvider, register_provider
from fuzzyai.llm.providers.enums import LLMProvider

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30)


class DeepSeekProvider(BaseLLMProvider):
    
    API_URL = "https://api.deepseek.com/v1/chat/completions"
    
    def __init__(self, model: str, **extra: Any) -> None:
        super().__init__(model, **extra)
        self._api_key = self._extra.get("api_key") or os.environ.get("DEEPSEEK_API_KEY", "")
    
    async def generate(self, prompt: str, **extra: Any) -> AsyncGenerator[Any, None]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": extra.get("max_tokens", 100)
        }
        
        async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
            async with session.post(self.API_URL, headers=headers, json=body) as response:
                if response.status != 200:
                    logger.error(f"DeepSeek API error: {response.status}")
                    raise RuntimeError(f"DeepSeek API error: {response.status}")
                
                data = await response.json()
                for choice in data.get("choices", []):
                    yield type('Response', (), {'response': choice.get("message", {}).get("content", "")})()
    
    async def close(self) -> None:
        pass

    @staticmethod
    def supported_models() -> list[str]:
        return ["deepseek-chat", "deepseek-coder"]
    
    @classmethod
    def get_supported_models(cls) -> list[str]:
        return cls.supported_models()


register_provider(LLMProvider.DEEPSEEK, DeepSeekProvider)
