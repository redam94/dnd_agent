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
        agent.tool(create_map_location, name="create_map", description="Create a map with given dimensions and description.")
        agent.tool(set_entity_position, name="set_position", description="Set the position of an entity on a map or location.")
        agent.tool(calculate_distance, name="distance", description="Calculate the distance between two entities.")
        agent.tool(get_entities_in_range, name="entities_in_range", description="List all entities within a given range of a central entity.")
        agent.tool(generate_scene_description, name="describe_scene", description="Generate a detailed description of a location, including entities and exits.")
        agent.tool(connect_locations, name="connect_locations", description="Create a connection between two locations.")
        return agent