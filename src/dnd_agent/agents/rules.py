"""
Rules agent.

Handles queries about the D&D 5e ruleset.  It delegates lookups to
the DnD 5e API client and can query the campaign graph for spell,
item and monster entities stored there.  By isolating rules logic in
its own agent, the orchestrator can direct all rule related
questions here without conflating them with narrative or combat
concerns.
"""

from __future__ import annotations

from typing import Any, Dict

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from .base import BaseAgent, AgentType
from dnd_agent.tools import lookup_dnd_resource, query_campaign_graph, list_entities_of_type


class RulesAgent(BaseAgent):
    """Agent responsible for D&D 5e rules lookups and validation."""

    def __init__(self, deps: Any, model_config: Dict[str, Any]):
        super().__init__(name="Rules Agent", deps=deps, model_config=model_config)

    def _create_agent(self) -> Agent:
        model = OpenAIChatModel(self.model_config["model_name"])
        agent = Agent(
            model,
            deps_type=type(self.deps),
            system_prompt=(
                "You are a D&D 5e rules expert.\n"
                "Use the available tools to look up spells, items, monsters and other rules.\n"
                "When answering questions reference official rules and avoid speculation."
            ),
            retries=2,
        )
        # Register relevant tools
        agent.tool(lookup_dnd_resource, name="lookup_dnd_resource", description="Look up spells, monsters, items and classes.")
        agent.tool(query_campaign_graph, name="query_graph", description="Execute a Cypher query on the campaign graph.")
        agent.tool(list_entities_of_type, name="list_entities", description="List entities of a given type from the graph database.")
        return agent