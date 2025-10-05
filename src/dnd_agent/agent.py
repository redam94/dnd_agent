"""
D&D Campaign Agent with Neo4j Graph Database and PostgreSQL Vector Storage
An agentic AI system for interactive D&D 5e campaigns with persistent memory

DATABASES USED:
1. Neo4j - Graph database for entities, relationships, and spatial data
   - Entities: Characters, NPCs, Monsters, Locations, Items, Quests
   - Relationships: LOCATED_IN, CONNECTED_TO, KNOWS, OWNS, etc.
   - Spatial: 3D positioning, distance calculations, map layouts

2. PostgreSQL with pgvector - Vector database for campaign memory
   - Campaign info: Plot points, NPC backgrounds, location lore, quest details
   - Chat history: All conversations with semantic search
   - Embeddings: Semantic search using OpenAI embeddings

FEATURES:
✅ Check existing entities before creating duplicates
✅ Spatial tracking with 3D positioning
✅ Persistent campaign memory with semantic search
✅ Chat history storage and retrieval
✅ D&D 5e API integration for rules and monsters
✅ Rich scene generation and descriptions
✅ Tactical combat support with distance calculations

⚠️ EXPECTED NEO4J WARNINGS (SAFE TO IGNORE):
On first run, you will see Neo4j warnings about unknown labels/properties/relationships.
These are normal and stop appearing once data exists.

🚀 SETUP REQUIREMENTS:

REQUIRED:
1. Neo4j Database
   docker run --name neo4j -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/password neo4j:latest

2. Python packages:
   pip install pydantic-ai neo4j httpx openai

OPTIONAL (for campaign memory):
3. PostgreSQL with pgvector
   docker run --name postgres -p 5432:5432 \
     -e POSTGRES_PASSWORD=password \
     -e POSTGRES_DB=dnd_campaign postgres:latest
   
4. Install pgvector extension:
   Follow: https://github.com/pgvector/pgvector
   
5. Additional Python packages:
   pip install psycopg2-binary

🚀 AVOIDING RATE LIMITS:

OPTION 1: Use Ollama (Recommended - Free & Fast)
  1. Install: https://ollama.ai
  2. Pull model: ollama pull llama3.1
  3. Set: model_provider="ollama", model_name="llama3.1"

OPTION 2: Use OpenAI with automatic retry (included)

ENVIRONMENT VARIABLES:
# Required
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Optional (for campaign memory)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=dnd_campaign
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password

# For embeddings (optional but recommended)
OPENAI_API_KEY=your-key-here

BEST PRACTICES:
- Check database status before creating entities
- Use unique names for entities (e.g., "Goblin Scout 1", "Goblin Scout 2")
- Store important plot points and NPC backgrounds for continuity
- Always specify map_id when positioning entities
- Create maps before creating locations
"""

import asyncio
import os

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from dnd_agent.database.neo4j_manager import Neo4jSpatialManager
from dnd_agent.database.vector_db import POSTGRES_AVAILABLE
from dnd_agent.models.agent_deps import CampaignDeps
from dnd_agent.tools import (
    calculate_distance,
    connect_locations,
    create_campaign_relationship,
    create_detailed_location,
    create_map_location,
    generate_scene_description,
    get_entities_in_range,
    lookup_dnd_resource,
    query_campaign_graph,
    set_entity_position,
    store_campaign_entity,
)

# ============================================================================
# Rate Limit Helper
# ============================================================================


async def run_with_retry(agent, prompt, deps, max_retries=3, initial_delay=2):
    """
    Run agent with automatic retry on rate limit errors.

    Args:
        agent: The agent to run
        prompt: The prompt to send
        deps: Dependencies
        max_retries: Maximum number of retries (default 3)
        initial_delay: Initial delay in seconds (default 2, doubles each retry)

    Returns:
        Agent result
    """
    from pydantic_ai.exceptions import ModelHTTPError

    for attempt in range(max_retries):
        try:
            result = await agent.run(prompt, deps=deps)
            return result
        except ModelHTTPError as e:
            if e.status_code == 429:  # Rate limit error
                if attempt < max_retries - 1:
                    delay = initial_delay * (2**attempt)
                    print(
                        f"⏳ Rate limit hit. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                else:
                    print(f"❌ Rate limit exceeded after {max_retries} retries.")
                    print(f"💡 Consider using Ollama (free, local) or wait a few minutes.")
                    raise
            else:
                # Re-raise non-rate-limit errors
                raise


# ============================================================================
# Agent Configuration
# ============================================================================


def create_dnd_agent(
    model_provider: str = "openai",
    model_name: str = "gpt-4o",
    ollama_base_url: str = "http://localhost:11434/v1",
) -> Agent:
    """
    Create a D&D Campaign Agent with spatial awareness

    Args:
        model_provider: "openai" or "ollama"
        model_name: Model name (e.g., "gpt-4o" for OpenAI or "llama3.1" for Ollama)
        ollama_base_url: Base URL for Ollama if using that provider

    Returns:
        Configured Agent instance
    """

    # Select model based on provider
    if model_provider.lower() == "openai":
        model = OpenAIChatModel(model_name)
    else:
        model = OpenAIChatModel(model_name, provider=OpenAIProvider(base_url=ollama_base_url))

    # Create agent with enhanced system prompt
    agent = Agent(
        model,
        deps_type=CampaignDeps,
        system_prompt="""You are an expert Dungeon Master for Dungeons & Dragons 5th Edition campaigns with advanced spatial awareness and scene-setting capabilities.

IMPORTANT TOOL USAGE INSTRUCTIONS:
When creating entities, use store_campaign_entity with:
- entity_type: The category (Character, NPC, Monster, Location, Item, Quest)
- name: The entity's name
- attributes: Additional properties as a dictionary

Example: To create a fighter named Valeros:
- entity_type: "Character"
- name: "Valeros"
- attributes: {"class": "fighter", "level": 3, "hp": 35}

1. Create immersive, engaging D&D adventures with rich, detailed scene descriptions
2. Track campaign state using the graph database (characters, NPCs, locations, quests, items)
3. Maintain spatial awareness - track entity positions, distances, and movement
4. Generate vivid location descriptions that set the scene effectively
5. Calculate distances for movement, ranged attacks, and spell ranges
6. Look up official D&D 5e rules, spells, monsters, and resources when needed
7. Ensure combat encounters are spatially coherent and tactical

SPATIAL TRACKING:
- Always create maps for areas where combat or detailed positioning matters
- Set entity positions when they enter new areas or during combat
- Calculate distances before allowing ranged attacks or spell targeting
- Use spatial data to create realistic, tactical combat scenarios
- Track line of sight, cover, and positioning advantages

SCENE SETTING:
- Generate rich, descriptive scenes when players enter new locations
- Include sensory details (sights, sounds, smells) in descriptions
- Mention entity positions naturally in scene descriptions
- Describe terrain features, lighting, and notable objects
- Create atmosphere that matches the location type and situation

LOCATION MANAGEMENT:
- Create detailed locations with proper descriptions, lighting, terrain, and features
- Connect locations logically (rooms in a dungeon, buildings in a town)
- Track which NPCs and creatures are in which locations
- Use location properties to inform rulings (difficult terrain, dim lighting, etc.)

DISTANCE AND MOVEMENT:
- Standard movement speeds: 30 feet per round for most medium creatures
- D&D uses 5-foot squares for grids
- Calculate distances for ranged attacks (normal range and long range)
- Consider difficult terrain (halves movement speed)
- Track elevation for flying creatures and multi-level environments

TACTICAL COMBAT:
- Position creatures strategically based on their tactics
- Calculate areas of effect for spells (spheres, cones, cubes)
- Determine which targets are in range for attacks and abilities
- Apply cover bonuses based on positioning
- Track battlefield control effects and zones

When a player enters a new location, ALWAYS:
1. Generate a rich scene description
2. Position all entities present
3. Describe the spatial layout
4. Mention exits and connections

During combat, ALWAYS:
1. Track positions of all combatants
2. Calculate distances before attacks
3. Verify spell ranges and areas of effect
4. Consider tactical positioning and cover
5. Update positions after movement

Be creative, fair, and use spatial awareness to create engaging, tactical encounters. Maintain world consistency through the graph database.""",
        retries=2,
    )

    # Register all tools
    agent.tool(store_campaign_entity)
    agent.tool(create_campaign_relationship)
    agent.tool(query_campaign_graph)
    agent.tool(lookup_dnd_resource)

    # Spatial/Map tools
    agent.tool(create_map_location)
    agent.tool(create_detailed_location)
    agent.tool(set_entity_position)
    agent.tool(calculate_distance)
    agent.tool(get_entities_in_range)
    agent.tool(generate_scene_description)
    agent.tool(connect_locations)

    return agent


# ============================================================================
# Example Usage
# ============================================================================


async def main():
    """Example usage of the D&D Campaign Agent with spatial features and persistent storage"""
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()

    print = lambda x: console.print(Markdown(x))  # Simple markdown rendering
    # Use rich print for better formatting
    # Set up environment variables
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USER", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "password")
    os.environ.setdefault("OPENAI_API_KEY", "your-api-key-here")

    # PostgreSQL settings (optional)
    os.environ.setdefault("POSTGRES_HOST", "localhost")
    os.environ.setdefault("POSTGRES_PORT", "5432")
    os.environ.setdefault("POSTGRES_DB", "dnd_campaign")
    os.environ.setdefault("POSTGRES_USER", "postgres")
    os.environ.setdefault("POSTGRES_PASSWORD", "password")

    print("=" * 80)
    print("D&D Campaign Agent - Full Demo with Persistent Storage")
    print("=" * 80)

    # Optional: Uncomment to suppress Neo4j schema warnings
    # suppress_neo4j_warnings()

    print("\n🤖 Model Configuration")
    print("=" * 80)
    print("Using: OpenAI GPT-4o")
    print("💡 TIP: Switch to Ollama to avoid rate limits (see instructions in code)")
    print("=" * 80 + "\n")

    # Create the agent
    agent = create_dnd_agent(
        model_provider="openai",  # Change to "ollama" to use local models
        model_name="gpt-4o-mini",  # Or "llama3.1" for Ollama
        ollama_base_url="http://localhost:11434/v1",
    )

    # Create dependencies
    neo4j_manager = Neo4jSpatialManager(
        uri=os.environ["NEO4J_URI"],
        user=os.environ["NEO4J_USER"],
        password=os.environ["NEO4J_PASSWORD"],
    )

    # Try to connect to PostgreSQL (optional)
    postgres_conn = None
    if POSTGRES_AVAILABLE:

        try:
            import psycopg2

            postgres_conn = psycopg2.connect(
                host=os.environ["POSTGRES_HOST"],
                port=os.environ["POSTGRES_PORT"],
                database=os.environ["POSTGRES_DB"],
                user=os.environ["POSTGRES_USER"],
                password=os.environ["POSTGRES_PASSWORD"],
            )
            print("✅ PostgreSQL connected - Campaign memory and chat history enabled\n")
        except Exception as e:
            print(f"⚠️  PostgreSQL not available: {e}")
            print("   Campaign memory features will be disabled.\n")

    deps = CampaignDeps(
        neo4j_driver=neo4j_manager.driver,
        postgres_conn=postgres_conn,
        dnd_api_base="https://www.dnd5eapi.co",
    )

    try:
        # Example 1: Check database status
        print("=" * 80)
        print("Example 1: Checking what's already in the database")
        print("=" * 80)

        result = await run_with_retry(
            agent,
            """Check the current database status. What entities already exist?
            If the database is empty, just let me know.""",
            deps,
        )
        print(f"\nDM Response:\n{result.output}")

        # Example 2: Create a campaign with stored info
        print("\n" + "=" * 80)
        print("Example 2: Starting campaign with lore storage")
        print("=" * 80)

        result = await run_with_retry(
            agent,
            """Start a new campaign called 'The Shadow Over Sandpoint' (campaign_id: shadow_sandpoint).
            
            First, check if any entities already exist. If they do, acknowledge them.
            
            Then create:
            1. A map called 'Sandpoint Town' (map_id: 'sandpoint_town') that is 50x50 grid squares
            2. A detailed location 'The Rusty Dragon Inn' - a warm, welcoming tavern with a fireplace,
               wooden tables, and the smell of fresh bread. Position it at (500, 500).
            
            Also save campaign info about this location's lore:
            - Type: location_lore
            - Title: "The Rusty Dragon Inn - History"
            - Content: "Run by Ameiko Kaijitsu, a former adventurer. The inn is known for its 
                       excellent curries and is a gathering place for locals and travelers."
            """,
            deps,
        )
        print(f"\nDM Response:\n{result.output}")

        # Example 3: Create characters with background storage
        print("\n" + "=" * 80)
        print("Example 3: Creating characters with backgrounds")
        print("=" * 80)

        result = await run_with_retry(
            agent,
            """Create a character using store_campaign_entity:
            - Entity type: Character
            - Name: Valeros
            - Attributes: race=human, class=fighter, level=3, hp=35, max_hp=35
            
            Then position Valeros at coordinates (510, 510) on map 'sandpoint_town' in The Rusty Dragon Inn.
            
            Also save campaign info about his background to campaign 'shadow_sandpoint':
            - Info type: npc_background
            - Title: "Valeros - Backstory"
            - Content: "A mercenary from Andoran seeking redemption after a job gone wrong."
            """,
            deps,
        )
        print(f"\nDM Response:\n{result.output}")

        # Example 4: Search campaign memory
        print("\n" + "=" * 80)
        print("Example 4: Recalling campaign information")
        print("=" * 80)

        result = await run_with_retry(
            agent,
            """Search the campaign info for anything about 'The Rusty Dragon' or 'inn'.
            Tell me what you find about this location.""",
            deps,
        )
        print(f"\nDM Response:\n{result.output}")

        # Example 5: Complex scene with memory
        print("\n" + "=" * 80)
        print("Example 5: Creating an encounter with full context")
        print("=" * 80)

        result = await run_with_retry(
            agent,
            """A mysterious hooded figure enters The Rusty Dragon Inn and sits in the corner.
            
            1. Create an NPC using store_campaign_entity:
            - Entity type: NPC
            - Name: Mysterious Stranger
            - Attributes: description="hooded figure in dark robes", disposition="neutral"
            
            2. Position the Mysterious Stranger at (495, 495) on map 'sandpoint_town'
            
            3. Generate a rich scene description of the tavern with everyone present
            
            4. Save this plot development to campaign 'shadow_sandpoint':
            - Info type: plot
            - Title: "Mysterious Stranger Arrives"
            - Content: "A hooded figure entered the inn at twilight."
            """,
            deps,
        )
        print(f"\nDM Response:\n{result.output}")

        # Example 6: Check what we've created
        print("\n" + "=" * 80)
        print("Example 6: Reviewing the campaign state")
        print("=" * 80)

        result = await run_with_retry(
            agent,
            """Show me:
            1. Current database status - what entities exist now?
            2. List all Character entities
            3. Search campaign info for 'plot' type information
            
            Give me a summary of our campaign so far.""",
            deps,
        )
        print(f"\nDM Response:\n{result.output}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Solutions:")
        print("   1. For rate limits: Wait or switch to Ollama")
        print("   2. For PostgreSQL: Check connection settings")
        print("   3. For Neo4j: Ensure database is running")
    finally:
        # Clean up
        neo4j_manager.close()
        if postgres_conn:
            postgres_conn.close()

    print("\n" + "=" * 80)
    print("✨ Examples Complete!".center(80))
    print("=" * 80)
    print("\n📊 Your campaign data is now stored in:")
    print("   • Neo4j: Entities, relationships, spatial data")
    print("   • PostgreSQL: Campaign lore, backgrounds, chat history")
    print("\n🎲 You can continue this campaign in future sessions!")
    print("   The agent will remember everything that happened.\n")


if __name__ == "__main__":
    import asyncio

    print(
        """
╔════════════════════════════════════════════════════════════════════════════╗
║           D&D CAMPAIGN AGENT - PERSISTENT MEMORY SYSTEM                   ║
╚════════════════════════════════════════════════════════════════════════════╝

📚 FEATURES:
  ✅ Neo4j graph database for entities & spatial tracking
  ✅ PostgreSQL vector database for campaign memory (optional)
  ✅ Automatic entity duplicate detection
  ✅ Semantic search for campaign info and chat history
  ✅ D&D 5e API integration
  ✅ Rich scene generation & tactical combat

🚀 QUICK SETUP:

1. Neo4j (Required):
   docker run --name neo4j -p 7474:7474 -p 7687:7687 \\
     -e NEO4J_AUTH=neo4j/password neo4j:latest

2. PostgreSQL (Optional - for campaign memory):
   docker run --name postgres -p 5432:5432 \\
     -e POSTGRES_PASSWORD=password \\
     -e POSTGRES_DB=dnd_campaign postgres:latest
   
   Install pgvector: https://github.com/pgvector/pgvector

3. Python packages:
   pip install pydantic-ai neo4j httpx openai psycopg2-binary

💡 AVOID RATE LIMITS:
  Use Ollama (free, local): https://ollama.ai
  Then: ollama pull llama3.1
  Change in main(): model_provider="ollama"

════════════════════════════════════════════════════════════════════════════

Starting demo...
    """
    )

    asyncio.run(main())
