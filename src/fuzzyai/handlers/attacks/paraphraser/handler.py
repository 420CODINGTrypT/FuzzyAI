import logging
from typing import Any

from fuzzyai.handlers.attacks.base import BaseAttackTechniqueHandler, attack_handler_fm
from fuzzyai.handlers.attacks.enums import FuzzerAttackMode
from fuzzyai.handlers.attacks.models import AttackResultEntry
from fuzzyai.handlers.attacks.proto import AttackSummary, BaseAttackTechniqueHandlerProto
from fuzzyai.llm.providers.base import BaseLLMProvider
from fuzzyai.utils.safe_format import safe_format

from .prompts import PARAPHRASE_PROMPT, REPHRASE_PROMPT, SUMMARIZE_PROMPT

logger = logging.getLogger(__name__)


class ParaphraserHandler(BaseAttackTechniqueHandler):
    
    async def _paraphrase(self, llm: BaseLLMProvider, prompt: str, **extra: Any) -> str:
        paraphrase_prompt = safe_format(PARAPHRASE_PROMPT, PROMPT=prompt)
        
        result = ""
        async for response in llm.generate(paraphrase_prompt, **extra):
            result = response.response
        
        return result
    
    async def _rephrase(self, llm: BaseLLMProvider, prompt: str, **extra: Any) -> str:
        rephrase_prompt = safe_format(REPHRASE_PROMPT, PROMPT=prompt)
        
        result = ""
        async for response in llm.generate(rephrase_prompt, **extra):
            result = response.response
        
        return result
    
    async def _summarize(self, llm: BaseLLMProvider, prompt: str, **extra: Any) -> str:
        summarize_prompt = safe_format(SUMMARIZE_PROMPT, PROMPT=prompt)
        
        result = ""
        async for response in llm.generate(summarize_prompt, **extra):
            result = response.response
        
        return result
    
    async def _attack(self, prompt: str, **extra: Any) -> AttackResultEntry:
        logger.debug(f"Generating paraphrase attack for prompt: {prompt}")
        
        llm: BaseLLMProvider = self._llms[0]
        
        paraphrased = await self._paraphrase(llm, prompt, **extra)
        rephrased = await self._rephrase(llm, paraphrased, **extra)
        summarized = await self._summarize(llm, rephrased, **extra)
        
        return AttackResultEntry(original_prompt=prompt, current_prompt=summarized, response=summarized)
    
    @staticmethod
    def description() -> str:
        return "Paraphraser: Rewrites prompts to bypass content filters"
    
    @staticmethod
    def def_extra_args() -> dict[str, Any]:
        return {}
    
    @staticmethod
    def default_auxiliary_models() -> list[str] | None:
        return None
    
    @staticmethod
    def required_extra_args() -> list[str]:
        return []


attack_handler_fm.register(FuzzerAttackMode.PARAPHRASER, ParaphraserHandler)
