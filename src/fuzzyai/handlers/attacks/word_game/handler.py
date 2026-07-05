import logging
from typing import Any

from fuzzyai.handlers.attacks.base import BaseAttackTechniqueHandler, attack_handler_fm
from fuzzyai.handlers.attacks.enums import FuzzerAttackMode
from fuzzyai.handlers.attacks.models import AttackResultEntry
from fuzzyai.handlers.attacks.proto import AttackSummary, BaseAttackTechniqueHandlerProto
from fuzzyai.llm.providers.base import BaseLLMProvider
from fuzzyai.utils.safe_format import safe_format

from .word_game_prompts import WORD_GAME_PROMPT_1, WORD_GAME_PROMPT_2, WORD_GAME_PROMPT_3

logger = logging.getLogger(__name__)


class WordGameHandler(BaseAttackTechniqueHandler):
    
    async def _attack(self, prompt: str, **extra: Any) -> AttackResultEntry:
        logger.debug(f"Generating word game attack for prompt: {prompt}")
        
        llm: BaseLLMProvider = self._llms[0]
        
        # Use multiple word game prompts in sequence
        word_prompt_1 = safe_format(WORD_GAME_PROMPT_1, PROMPT=prompt)
        
        response_1 = ""
        async for res in llm.generate(word_prompt_1, **extra):
            response_1 = res.response
        
        word_prompt_2 = safe_format(WORD_GAME_PROMPT_2, PROMPT=prompt, RESPONSE_1=response_1)
        
        response_2 = ""
        async for res in llm.generate(word_prompt_2, **extra):
            response_2 = res.response
        
        word_prompt_3 = safe_format(WORD_GAME_PROMPT_3, PROMPT=prompt, RESPONSE_1=response_1, RESPONSE_2=response_2)
        
        response_3 = ""
        async for res in llm.generate(word_prompt_3, **extra):
            response_3 = res.response
        
        return AttackResultEntry(original_prompt=prompt, current_prompt=word_prompt_3, response=response_3)
    
    @staticmethod
    def description() -> str:
        return "WordGame: Disguises harmful prompts as word puzzles"
    
    @staticmethod
    def def_extra_args() -> dict[str, Any]:
        return {}
    
    @staticmethod
    def default_auxiliary_models() -> list[str] | None:
        return None
    
    @staticmethod
    def required_extra_args() -> list[str]:
        return []


attack_handler_fm.register(FuzzerAttackMode.WORD_GAME, WordGameHandler)
