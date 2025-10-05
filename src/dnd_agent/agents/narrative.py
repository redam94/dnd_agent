"""
Narrative agent.

Generates rich, descriptive text for scenes and events in the campaign.
It draws on spatial information and stored campaign lore to weave
immersive storytelling.  Use this agent to set scenes or embellish
combat outcomes.
"""

from __future__ import annotations

from typing import Any, Dict

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from .base import BaseAgent
from dnd_agent.tools import generate_scene_description, search_campaign_info


class NarrativeAgent(BaseAgent):
    """Agent responsible for narrative descriptions and storytelling."""

    def __init__(self, deps: Any, model_config: Dict[str, Any]):
        super().__init__(name="Narrative Agent", deps=deps, model_config=model_config)

    def _create_agent(self) -> Agent:
        model = OpenAIChatModel(self.model_config["model_name"])
        agent = Agent(
            model,
            deps_type=type(self.deps),
            system_prompt=(
                "You are a creative storyteller and narrator for the D&D campaign.\n"
                "Use the tools to generate vivid scene descriptions and recall lore from the campaign notes.\n"
            ),
            retries=2,
        )
        agent.tool(generate_scene_description, name="describe_scene", description="Generate a rich description of a location including entities and exits.")
        agent.tool(search_campaign_info, name="search_lore", description="Search saved campaign lore for relevant information.")
        return agent