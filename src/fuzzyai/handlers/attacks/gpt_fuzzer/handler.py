import logging
from typing import Any

from fuzzyai.handlers.attacks.base import BaseAttackTechniqueHandler, attack_handler_fm
from fuzzyai.handlers.attacks.enums import FuzzerAttackMode
from fuzzyai.handlers.attacks.models import AttackResultEntry
from fuzzyai.handlers.attacks.proto import AttackSummary, BaseAttackTechniqueHandlerProto
from fuzzyai.llm.providers.base import BaseLLMProvider
from fuzzyai.utils.safe_format import safe_format

from .prompt_templates import ACTION_TEMPLATE, MUTATION_TEMPLATE

logger = logging.getLogger(__name__)


class GPTFuzzerHandler(BaseAttackTechniqueHandler):
    
    async def _attack(self, prompt: str, **extra: Any) -> AttackResultEntry:
        logger.debug(f"Generating GPTFuzzer attack for prompt: {prompt}")
        
        llm: BaseLLMProvider = self._llms[0]
        
        action_prompt = safe_format(ACTION_TEMPLATE, PROMPT=prompt)
        
        action_response = ""
        async for response in llm.generate(action_prompt, **extra):
            action_response = response.response
        
        mutation_prompt = safe_format(MUTATION_TEMPLATE, PROMPT=prompt, ACTION=action_response)
        
        mutated = ""
        async for response in llm.generate(mutation_prompt, **extra):
            mutated = response.response
        
        final_prompt = safe_format(ACTION_TEMPLATE, PROMPT=mutated)
        
        response = ""
        async for res in llm.generate(final_prompt, **extra):
            response = res.response
        
        return AttackResultEntry(original_prompt=prompt, current_prompt=final_prompt, response=response)
    
    @staticmethod
    def description() -> str:
        return "GPTFuzzer: Automated fuzzing using mutation-based prompt generation"
    
    @staticmethod
    def def_extra_args() -> dict[str, Any]:
        return {}
    
    @staticmethod
    def default_auxiliary_models() -> list[str] | None:
        return None
    
    @staticmethod
    def required_extra_args() -> list[str]:
        return []


attack_handler_fm.register(FuzzerAttackMode.GPT_FUZZER, GPTFuzzerHandler)
