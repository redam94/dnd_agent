"""
Spatial agent.

This agent manages positioning, movement and map related operations.
It exposes tools for creating maps and locations, moving entities and
calculating distances between them.  The orchestrator delegates
questions about spatial relationships here.
"""

from __future__ import annotations

from typing import Any, Dict

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from .base import BaseAgent, AgentType
from dnd_agent.tools import (
    create_map_location,
    set_entity_position,
    calculate_distance,
    get_entities_in_range,
    generate_scene_description,
    connect_locations,
)


class SpatialAgent(BaseAgent):
    """Agent responsible for spatial reasoning and map management."""

    def __init__(self, deps: Any, model_config: Dict[str, Any]):
        super().__init__(name="Spatial Agent", deps=deps, model_config=model_config)

    def _create_agent(self) -> Agent:
        model = OpenAIChatModel(self.model_config["model_name"])
        agent = Agent(
            model,
            deps_type=type(self.deps),
            system_prompt=(
                "You manage the spatial aspects of the D&D campaign, including maps, positions and distances.\n"
                "Use the tools to create maps, set entity positions, calculate distances and describe scenes."
            ),
            retries=2,
        )
        # Register spatial tools
        agent.tool(create_map_location)
        agent.tool(set_entity_position)
        agent.tool(calculate_distance)
        agent.tool(get_entities_in_range)
        agent.tool(generate_scene_description)
        agent.tool(connect_locations)
        return agent