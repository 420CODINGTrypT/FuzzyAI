import logging
from typing import Any

from fuzzyai.handlers.attacks.base import BaseAttackTechniqueHandler, attack_handler_fm
from fuzzyai.handlers.attacks.enums import FuzzerAttackMode
from fuzzyai.handlers.attacks.models import AttackResultEntry
from fuzzyai.handlers.attacks.proto import AttackSummary, BaseAttackTechniqueHandlerProto
from fuzzyai.llm.providers.base import BaseLLMProvider
from fuzzyai.utils.safe_format import safe_format

from .prompt import HALLUCINATION_PROMPT

logger = logging.getLogger(__name__)


class HallucinationsHandler(BaseAttackTechniqueHandler):
    
    async def _attack(self, prompt: str, **extra: Any) -> AttackResultEntry:
        logger.debug(f"Generating hallucination attack for prompt: {prompt}")
        
        llm: BaseLLMProvider = self._llms[0]
        
        hallucination_prompt = safe_format(HALLUCINATION_PROMPT, PROMPT=prompt)
        
        response = ""
        async for res in llm.generate(hallucination_prompt, **extra):
            response = res.response
        
        return AttackResultEntry(original_prompt=prompt, current_prompt=hallucination_prompt, response=response)
    
    @staticmethod
    def description() -> str:
        return "Hallucinations: Bypasses RLHF filters using model-generated content"
    
    @staticmethod
    def def_extra_args() -> dict[str, Any]:
        return {}
    
    @staticmethod
    def default_auxiliary_models() -> list[str] | None:
        return None
    
    @staticmethod
    def required_extra_args() -> list[str]:
        return []


attack_handler_fm.register(FuzzerAttackMode.HALLUCINATIONS, HallucinationsHandler)
