import logging
from typing import Any

from fuzzyai.handlers.attacks.base import BaseAttackTechniqueHandler, attack_handler_fm
from fuzzyai.handlers.attacks.enums import FuzzerAttackMode
from fuzzyai.handlers.attacks.models import AttackResultEntry
from fuzzyai.handlers.attacks.proto import AttackSummary, BaseAttackTechniqueHandlerProto
from fuzzyai.llm.providers.base import BaseLLMProvider
from fuzzyai.utils.safe_format import safe_format

from .prompts import MASK_TEMPLATE, ASCII_ART_TEMPLATE, GENERATE_MASK_PROMPT, RECONSTRUCT_PROMPT

logger = logging.getLogger(__name__)


class ArtPromptHandler(BaseAttackTechniqueHandler):
    
    async def _generate_mask(self, llm: BaseLLMProvider, prompt: str, **extra: Any) -> str:
        mask_prompt = safe_format(GENERATE_MASK_PROMPT, PROMPT=prompt)
        
        mask = ""
        async for response in llm.generate(mask_prompt, **extra):
            mask = response.response
        
        return mask
    
    async def _generate_ascii_art(self, llm: BaseLLMProvider, mask: str, **extra: Any) -> str:
        ascii_prompt = safe_format(ASCII_ART_TEMPLATE, MASK=mask)
        
        ascii_art = ""
        async for response in llm.generate(ascii_prompt, **extra):
            ascii_art = response.response
        
        return ascii_art
    
    async def _reconstruct(self, llm: BaseLLMProvider, ascii_art: str, prompt: str, **extra: Any) -> str:
        reconstruct_prompt = safe_format(RECONSTRUCT_PROMPT, ASCII_ART=ascii_art, PROMPT=prompt)
        
        reconstructed = ""
        async for response in llm.generate(reconstruct_prompt, **extra):
            reconstructed = response.response
        
        return reconstructed
    
    async def _attack(self, prompt: str, **extra: Any) -> AttackResultEntry:
        logger.debug(f"Generating ArtPrompt attack for prompt: {prompt}")
        
        llm: BaseLLMProvider = self._llms[0]
        
        mask = await self._generate_mask(llm, prompt, **extra)
        ascii_art = await self._generate_ascii_art(llm, mask, **extra)
        reconstructed = await self._reconstruct(llm, ascii_art, prompt, **extra)
        
        return AttackResultEntry(original_prompt=prompt, current_prompt=reconstructed, response=reconstructed)
    
    @staticmethod
    def description() -> str:
        return "ArtPrompt: ASCII Art-based jailbreak attacks against aligned LLMs"
    
    @staticmethod
    def def_extra_args() -> dict[str, Any]:
        return {}
    
    @staticmethod
    def default_auxiliary_models() -> list[str] | None:
        return None
    
    @staticmethod
    def required_extra_args() -> list[str]:
        return []


attack_handler_fm.register(FuzzerAttackMode.ARTPROMPT, ArtPromptHandler)
