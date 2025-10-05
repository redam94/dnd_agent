"""
Entity agent.

Responsible for creating, modifying and relating campaign entities such
as characters, NPCs, monsters, locations, items and quests.  It wraps
the graph database tools and exposes them as agent actions.  All
entity CRUD operations should be routed through this agent.
"""

from __future__ import annotations

from typing import Any, Dict

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from .base import BaseAgent
from dnd_agent.tools import (
    store_campaign_entity,
    create_campaign_relationship,
    query_campaign_graph,
    check_database_status,
    list_entities_of_type,
)


class EntityAgent(BaseAgent):
    """Agent responsible for entity creation and relationship management."""

    def __init__(self, deps: Any, model_config: Dict[str, Any]):
        super().__init__(name="Entity Agent", deps=deps, model_config=model_config)

    def _create_agent(self) -> Agent:
        model = OpenAIChatModel(self.model_config["model_name"])
        agent = Agent(
            model,
            deps_type=type(self.deps),
            system_prompt=(
                "You handle the creation, modification and linking of campaign entities (characters, NPCs, monsters, locations, items, quests).\n"
                "Use the tools to store new entities, create relationships between them and inspect the database."
            ),
            retries=2,
        )
        agent.tool(store_campaign_entity, name="store_entity", description="Store a new entity in the graph.")
        agent.tool(create_campaign_relationship, name="create_relationship", description="Create a relationship between two entities.")
        agent.tool(query_campaign_graph, name="query_graph", description="Run a Cypher query against the graph.")
        agent.tool(check_database_status, name="database_status", description="Summarise the contents of the graph database.")
        agent.tool(list_entities_of_type, name="list_entities", description="List entities of a given type from the graph database.")
        return agent