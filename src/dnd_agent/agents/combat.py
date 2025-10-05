"""
Combat agent.

This agent encapsulates turn order, attack resolution and damage
calculations.  The full combat engine is beyond the scope of this
refactoring, so this class currently exposes no tools.  Future work
could add tools such as ``validate_action`` or ``process_turn`` and
delegate to a dedicated combat subsystem.
"""

from __future__ import annotations

from typing import Any, Dict

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from .base import BaseAgent


class CombatAgent(BaseAgent):
    """Agent responsible for managing combat actions."""

    def __init__(self, deps: Any, model_config: Dict[str, Any]):
        super().__init__(name="Combat Agent", deps=deps, model_config=model_config)

    def _create_agent(self) -> Agent:
        model = OpenAIChatModel(self.model_config["model_name"])
        agent = Agent(
            model,
            deps_type=type(self.deps),
            system_prompt=(
                "You manage combat in the D&D campaign, including turn order and attack resolution.\n"
                "Future versions of this agent may provide specialised tools such as validate_action and process_turn."
            ),
            retries=2,
        )
        # No tools registered yet
        return agent