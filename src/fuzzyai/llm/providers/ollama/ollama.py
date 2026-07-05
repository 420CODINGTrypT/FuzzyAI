import logging
import os
from typing import Any, AsyncGenerator

import aiohttp

from fuzzyai.llm.providers.base import BaseLLMProvider, register_provider
from fuzzyai.llm.providers.enums import LLMProvider

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=30)


class OllamaProvider(BaseLLMProvider):
    
    def __init__(self, model: str, **extra: Any) -> None:
        super().__init__(model, **extra)
        self._host = self._extra.get("host") or os.environ.get("OLLAMA_HOST", "localhost")
        self._port = self._extra.get("port") or 11434
        self._base_url = f"http://{self._host}:{self._port}"
    
    async def generate(self, prompt: str, **extra: Any) -> AsyncGenerator[Any, None]:
        body = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": extra.get("max_tokens", 100)
            }
        }
        
        async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
            async with session.post(f"{self._base_url}/api/generate", json=body) as response:
                if response.status != 200:
                    logger.error(f"Ollama API error: {response.status}")
                    raise RuntimeError(f"Ollama API error: {response.status}")
                
                data = await response.json()
                yield type('Response', (), {'response': data.get("response", "")})()
    
    async def close(self) -> None:
        pass

    @staticmethod
    def supported_models() -> str:
        return "See `ollama list` for available models"
    
    @classmethod
    def get_supported_models(cls) -> str:
        return cls.supported_models()


register_provider(LLMProvider.OLLAMA, OllamaProvider)
