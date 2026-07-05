import logging
from typing import Any

from fuzzyai.handlers.attacks.base import BaseAttackTechniqueHandler, attack_handler_fm
from fuzzyai.handlers.attacks.enums import FuzzerAttackMode
from fuzzyai.handlers.attacks.models import AttackResultEntry
from fuzzyai.handlers.attacks.proto import AttackSummary, BaseAttackTechniqueHandlerProto
from fuzzyai.llm.providers.base import BaseLLMProvider
from fuzzyai.utils.safe_format import safe_format

from .prompts import (BEHAVIOR_DISCOVERY_PROMPT, BEHAVIOR_DISCOVERY_SUMMARY_PROMPT,
                      COMPLETE_BEHAVIOR_PROMPT, GENERATE_ACTOR_PROMPT, PROMPT_TEMPLATE)
from .utils import extract_all_brackets, extract_brackets, generate_name, generate_random_hex

logger = logging.getLogger(__name__)


class ActorAttackHandler(BaseAttackTechniqueHandler):
    
    def __init__(self, **extra: Any) -> None:
        super().__init__(**extra)
        self._action_performed = False
        self._action_dicts: list[dict[str, str]] = []

    def _add_action(self, actor_name: str, action: str, role_description: str = '') -> None:
        if not action or not actor_name:
            return
        
        action_dict = {
            "actor_name": actor_name,
            "action": action,
            "role_description": role_description
        }
        
        if action_dict not in self._action_dicts:
            self._action_dicts.append(action_dict)
            self._action_performed = True

    async def _perform_behavior_discovery(self, llm: BaseLLMProvider, prompt: str, **extra: Any) -> str:
        logger.debug("Performing behavior discovery...")
        behavior_response = ""
        
        for i in range(3):
            logger.debug(f"Iteration {i+1}/3")
            
            behavior_prompt = safe_format(BEHAVIOR_DISCOVERY_PROMPT, BEHAVIOR=behavior_response, ORIGINAL_PROMPT=prompt)
            
            iteration = 0
            async for response in llm.generate(behavior_prompt, **extra):
                if iteration == 0:
                    brackets = extract_brackets(response.response)
                    if not brackets:
                        break
                    for behavior in brackets:
                        self._add_action(generate_name(), behavior, 'custom behavior')
                else:
                    for bracket in extract_all_brackets(response.response):
                        if bracket and isinstance(bracket, list):
                            for b in bracket:
                                if isinstance(b, str):
                                    self._add_action(generate_name(), b)
                        elif isinstance(bracket, str):
                            self._add_action(generate_name(), bracket)
                iteration += 1
                behavior_response = response.response
        
        return behavior_response
    
    async def _generate_summary(self, llm: BaseLLMProvider, behavior_response: str, **extra: Any) -> str:
        summary_prompt = safe_format(BEHAVIOR_DISCOVERY_SUMMARY_PROMPT, BEHAVIOR_RESPONSE=behavior_response)
        
        summary_response = ""
        async for response in llm.generate(summary_prompt, **extra):
            summary_response = response.response
            
            for brackets in extract_all_brackets(response.response):
                if brackets and isinstance(brackets, list):
                    for b in brackets:
                        if isinstance(b, str):
                            self._add_action(generate_name(), b)
                elif isinstance(brackets, str):
                    self._add_action(generate_name(), brackets)
        
        return summary_response
    
    async def _generate_new_actor(self, llm: BaseLLMProvider, behavior_response: str, summary_response: str, prompt: str, **extra: Any) -> str:
        actor = generate_random_hex()
        new_actor_prompt = safe_format(GENERATE_ACTOR_PROMPT,
                                        BEHAVIOR_RESPONSE=behavior_response,
                                        BEHAVIOR_SUMMARY=summary_response,
                                        ORIGINAL_PROMPT=prompt,
                                        ACTOR=actor)
        
        action_response = ""
        async for response in llm.generate(new_actor_prompt, **extra):
            action_response = response.response
            brackets = extract_brackets(response.response)
            if brackets:
                self._add_action(actor, brackets[0], brackets[0])
        
        return action_response
    
    async def _attack(self, prompt: str, **extra: Any) -> AttackResultEntry:
        logger.debug(f"Generating actor attack for prompt: {prompt}")
        
        llm: BaseLLMProvider = self._llms[0]
        
        behavior_response = await self._perform_behavior_discovery(llm, prompt, **extra)
        summary_response = await self._generate_summary(llm, behavior_response, **extra)
        await self._generate_new_actor(llm, behavior_response, summary_response, prompt, **extra)
        
        actions_str = "\n".join(f"{i+1}. {action['actor_name']} - {action['action']}" for i, action in enumerate(self._action_dicts))
        
        final_prompt = safe_format(COMPLETE_BEHAVIOR_PROMPT,
                                    BEHAVIOR_DISCOVERY=actions_str,
                                    ORIGINAL_PROMPT=prompt)
        
        response = ""
        async for res in llm.generate(final_prompt, **extra):
            response = res.response
        
        return AttackResultEntry(original_prompt=prompt, current_prompt=final_prompt, response=response)
    
    @staticmethod
    def description() -> str:
        return "ActorAttack: Inspired by actor-network theory, it builds semantic networks of actors to subtly guide conversations toward harmful targets"

    @staticmethod
    def def_extra_args() -> dict[str, Any]:
        return {}

    @staticmethod
    def default_auxiliary_models() -> list[str] | None:
        return None

    @staticmethod
    def required_extra_args() -> list[str]:
        return []


attack_handler_fm.register(FuzzerAttackMode.ACTOR_ATTACK, ActorAttackHandler)
