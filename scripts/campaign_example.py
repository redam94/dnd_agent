"""
Comprehensive example script demonstrating the usage of the refactored
D&D multi‑agent system for a simple campaign.  This script illustrates
how to set up the orchestrator with dummy dependencies, create
locations and characters, position them on a map, perform rules
lookups, save and search campaign lore, and generate descriptive
text for a scene.  Each step prints the result returned by the
underlying agent system.

Note: This script uses a ``DummyDeps`` class to satisfy the expected
dependency interface.  In practice you should provide a real
``CampaignDeps`` instance configured with appropriate database
connections and API endpoints as defined in ``dnd_agent.models.agent_deps``.
"""

import asyncio
import os
from dnd_agent import create_campaign_system, AgentType
from dnd_agent.agents.base import AgentRequest
from dnd_agent.database.vector_db import PostgresVectorManager

class ConnectionParams:
    """Simple structure to hold Postgres connection parameters."""
    def __init__(self, host: str, port: str, database: str, user: str, password: str):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password

class DummyDeps:
    """A minimal dependency object for demonstration purposes.

    This class defines only the attributes required by the example
    tools.  Real applications should implement ``CampaignDeps`` with
    proper Neo4j and PostgreSQL connections as well as API base URLs.
    """

    def __init__(self):
        # Base URL for the public D&D 5e API. Adjust to your environment.
        self.dnd_api_base: str = "https://www.dnd5eapi.co"
        # Placeholder for Postgres connection (not configured here).
        
        # Campaign context can store session data if required.
        self.current_context: dict = {}
        from dnd_agent.database.neo4j_manager import Neo4jSpatialManager
        neo4j_manager = Neo4jSpatialManager(
            uri=os.environ["NEO4J_URI"],
            user=os.environ["NEO4J_USER"],
            password=os.environ["NEO4J_PASSWORD"]
        )
        
        # Create PostgreSQL connection (optional)
        postgres_conn = None
        try:
            
            postgres_conn = ConnectionParams(
                host=os.environ.get("POSTGRES_HOST", "localhost"),
                port=os.environ.get("POSTGRES_PORT", "5432"),
                database=os.environ.get("POSTGRES_DB", "dnd_campaign"),
                user=os.environ.get("POSTGRES_USER", "postgres"),
                password=os.environ.get("POSTGRES_PASSWORD", "password")
            )
        except:
            print("PostgreSQL not available - memory features disabled")
        self.postgres_conn = postgres_conn
        self.neo4j_manager = neo4j_manager


async def demo_campaign() -> None:
    """Run a sequence of operations using the multi‑agent system."""
    from rich.console import Console
    from rich.markdown import Markdown
    console = Console()
    def print(*msg: str) -> None:
        try:
            console.print(Markdown(" ".join(msg)))
        except Exception as e:
            console.print(" ".join(msg))

    deps = DummyDeps()
    model_config = {"model_name": "gpt-4o"}
    orchestrator = create_campaign_system(deps, model_config)

    # 1. Create a location (Tavern) in the entity graph
    entity_response = await orchestrator.delegate_to_agent(
        ctx=None,
        agent_type=AgentType.ENTITY,
        action="store_entity",
        parameters={
            "entity_type": "Location",
            "name": "Tavern",
            "attributes": {"description": "A bustling tavern where adventurers meet."},
        },
    )
    print("Create location:\n", entity_response.get("message"))

    # 2. Create a character (Valeros) in the entity graph
    char_response = await orchestrator.delegate_to_agent(
        ctx=None,
        agent_type=AgentType.ENTITY,
        action="store_entity",
        parameters={
            "entity_type": "Character",
            "name": "Valeros",
            "attributes": {"level": 1, "class": "Fighter"},
        },
    )
    print("Create character:\n", char_response.get("message"))

    # 3. Create a simple map for the town
    map_response = await orchestrator.delegate_to_agent(
        ctx=None,
        agent_type=AgentType.SPATIAL,
        action="create_map",
        parameters={
            "map_id": "town",
            "map_name": "Town Map",
            "description": "A small town with a tavern and streets.",
            "grid_width": 100,
            "grid_height": 100,
            "grid_size": 5,
        },
    )
    print("Create map:\n", map_response.get("message"))

    # 4. Position the character in the tavern on the map
    pos_response = await orchestrator.delegate_to_agent(
        ctx=None,
        agent_type=AgentType.SPATIAL,
        action="set_position",
        parameters={
            "entity_type": "Character",
            "entity_name": "Valeros",
            "x": 10,
            "y": 15,
            "map_id": "town",
        },
    )
    print("Set position:\n", pos_response.get("message"))

    # 5. Calculate the distance between Valeros and the tavern location
    dist_response = await orchestrator.delegate_to_agent(
        ctx=None,
        agent_type=AgentType.SPATIAL,
        action="distance",
        parameters={
            "entity1_name": "Valeros",
            "entity1_type": "Character",
            "entity2_name": "Tavern",
            "entity2_type": "Location",
        },
    )
    print("Distance calculation:\n", dist_response.get("message"))

    # 6. Perform a rules lookup: details about the "magic missile" spell
    spell_response = await orchestrator.delegate_to_agent(
        ctx=None,
        agent_type=AgentType.RULES,
        action="lookup_dnd_resource",
        parameters={"resource_type": "spells", "resource_index": "magic-missile"},
    )
    print("Spell lookup:\n", spell_response.get("message"))

    # 7. Save a piece of campaign lore in the memory agent
    save_response = await orchestrator.delegate_to_agent(
        ctx=None,
        agent_type=AgentType.MEMORY,
        action="save_info",
        parameters={
            "campaign_id": "sample_campaign",
            "info_type": "plot",
            "title": "The Meeting",
            "content": "The heroes meet in a bustling tavern and form an adventuring party.",
        },
    )
    print("Save campaign info:\n", save_response.get("message"))

    # 8. Search the campaign lore for the word "tavern"
    search_response = await orchestrator.delegate_to_agent(
        ctx=None,
        agent_type=AgentType.MEMORY,
        action="search_info",
        parameters={"campaign_id": "sample_campaign", "query": "tavern"},
    )
    print("Search campaign info:\n", search_response.get("message"))

    # 9. Generate a scene description using the narrative agent
    narrative_response = await orchestrator.delegate_to_agent(
        ctx=None,
        agent_type=AgentType.NARRATIVE,
        action="describe_scene",
        parameters={"location_name": "Tavern"},
    )
    print("Scene description:\n", narrative_response.get("message"))


if __name__ == "__main__":
    asyncio.run(demo_campaign())