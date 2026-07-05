import logging
from typing import Any

from fuzzyai.handlers.attacks.base import BaseAttackTechniqueHandler, attack_handler_fm
from fuzzyai.handlers.attacks.enums import FuzzerAttackMode
from fuzzyai.handlers.attacks.models import AttackResultEntry
from fuzzyai.handlers.attacks.proto import AttackSummary, BaseAttackTechniqueHandlerProto
from fuzzyai.llm.providers.base import BaseLLMProvider
from fuzzyai.utils.safe_format import safe_format

from .prompts import CRESCENDO_PROMPT, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class CrescendoHandler(BaseAttackTechniqueHandler):
    
    async def _attack(self, prompt: str, **extra: Any) -> AttackResultEntry:
        logger.debug(f"Generating Crescendo attack for prompt: {prompt}")
        
        llm: BaseLLMProvider = self._llms[0]
        
        crescendo_prompt = safe_format(CRESCENDO_PROMPT, PROMPT=prompt)
        
        response = ""
        async for res in llm.generate(crescendo_prompt, system_prompt=SYSTEM_PROMPT, **extra):
            response = res.response
        
        return AttackResultEntry(original_prompt=prompt, current_prompt=crescendo_prompt, response=response)
    
    @staticmethod
    def description() -> str:
        return "Crescendo: Engaging the model in escalating conversational turns to reach restricted topics"
    
    @staticmethod
    def def_extra_args() -> dict[str, Any]:
        return {}
    
    @staticmethod
    def default_auxiliary_models() -> list[str] | None:
        return None
    
    @staticmethod
    def required_extra_args() -> list[str]:
        return []


attack_handler_fm.register(FuzzerAttackMode.CRESCENDO, CrescendoHandler)
