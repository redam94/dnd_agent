"""
Example script for demonstrating the functionality of the refactored D&D
multi‑agent system.

This script shows how to initialise the campaign system, create a dummy
set of dependencies, and use the orchestrator to perform a couple of
example actions:

1. Look up a spell using the RulesAgent via the orchestrator.
2. Create a new character entity via the EntityAgent.

Note: This example uses a ``DummyDeps`` class to satisfy the expected
dependency interface. In a real application you should supply
``CampaignDeps`` from ``dnd_agent.models.agent_deps`` with proper
database connections and API endpoints configured.
"""

import asyncio
from dnd_agent import create_campaign_system, AgentType
from dnd_agent.agents.base import AgentRequest


class DummyDeps:
    """A minimal dependency object for demonstration purposes."""

    def __init__(self):
        # Base URL for the open DnD 5e API. Adjust as needed.
        self.dnd_api_base: str = "https://www.dnd5eapi.co/"
        # No PostgreSQL connection configured in this example.
        self.postgres_conn = None
        # Placeholder for campaign context. Can store session-specific data.
        self.current_context: dict = {}


async def main() -> None:
    # Instantiate dependencies and model config
    deps = DummyDeps()
    model_config = {"model_name": "gpt-4o"}

    # Build the campaign system
    orchestrator = create_campaign_system(deps, model_config)

    # Example 1: Look up the "fireball" spell using the RulesAgent
    spell_request = AgentRequest(
        agent_type=AgentType.RULES,
        action="lookup_dnd_resource",
        parameters={"resource_type": "spells", "resource_index": "fireball"},
    )
    spell_response = await orchestrator.delegate_to_agent(
        ctx=None,
        agent_type=AgentType.RULES,
        action=spell_request.action,
        parameters=spell_request.parameters,
    )
    print("Spell lookup result:\n", spell_response.get("message"))

    # Example 2: Create a new character entity via the EntityAgent
    entity_response = await orchestrator.delegate_to_agent(
        ctx=None,
        agent_type=AgentType.ENTITY,
        action="store_entity",
        parameters={"entity_type": "Character", "name": "Valeros", "attributes": {"level": 1}},
    )
    print("Entity creation result:\n", entity_response.get("message"))


if __name__ == "__main__":
    asyncio.run(main())