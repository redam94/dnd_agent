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

import os
import httpx
import asyncio
from typing import Optional, Any, Dict, List, Tuple
import json

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIChatModel

from dnd_agent.database.neo4j_manager import Neo4jSpatialManager, suppress_neo4j_warnings
from dnd_agent.database.vector_db import PostgresVectorManager, POSTGRES_AVAILABLE, OPENAI_EMBEDDINGS_AVAILABLE
from dnd_agent.models.agent_deps import CampaignDeps
from dnd_agent.models.location_handling import Position, MapLocation, EntityPosition, BattleMap


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
                    delay = initial_delay * (2 ** attempt)
                    print(f"⏳ Rate limit hit. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    print(f"❌ Rate limit exceeded after {max_retries} retries.")
                    print(f"💡 Consider using Ollama (free, local) or wait a few minutes.")
                    raise
            else:
                # Re-raise non-rate-limit errors
                raise


# ============================================================================
# D&D 5e API Client
# ============================================================================

class DnD5eAPIClient:
    """Client for D&D 5e API"""
    
    def __init__(self, base_url: str = "https://www.dnd5eapi.co"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)
    
    def close(self):
        self.client.close()
    
    def get_resource(self, endpoint: str, index: str = None) -> Dict[str, Any]:
        """Get a resource from the D&D 5e API"""
        if index:
            url = f"{self.base_url}/api/2014/{endpoint}/{index}"
        else:
            url = f"{self.base_url}/api/2014/{endpoint}"
        
        try:
            response = self.client.get(url)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            return {"error": str(e)}


# ============================================================================
# Agent Tools - Original
# ============================================================================

async def store_campaign_entity(
    ctx: RunContext[CampaignDeps],
    entity_type: str,  # Changed from 'label' to 'entity_type' for clarity
    name: str,         # Make name explicit as a separate parameter
    attributes: Optional[Dict[str, Any]] = None  # Changed from 'properties' to 'attributes'
) -> str:
    """
    Store a campaign entity to the graph database.
    
    Args:
        entity_type: The type of entity - must be one of: Character, NPC, Monster, Location, Item, Quest
        name: The name of the entity (e.g., "Valeros", "Goblin Scout", "Rusty Sword")
        attributes: Optional additional properties like level, hp, description, etc.
    
    Returns:
        Confirmation message
    """
    try:
        neo4j_manager = Neo4jSpatialManager(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password")
        )
        
        # Check if entity already exists
        if neo4j_manager.entity_exists(entity_type, name):
            neo4j_manager.close()
            return f"⚠️  {entity_type} '{name}' already exists in the database. Use a different name or update the existing entity."
        
        # Combine name with other attributes
        properties = attributes or {}
        properties['name'] = name
        
        result = neo4j_manager.create_node(entity_type, properties)
        neo4j_manager.close()
        
        return f"✅ Successfully created {entity_type} '{name}' with properties: {json.dumps(result, indent=2)}"
    except Exception as e:
        return f"❌ Error creating entity: {str(e)}"


async def create_campaign_relationship(
    ctx: RunContext[CampaignDeps],
    from_entity: str,
    from_entity_type: str,
    to_entity: str,
    to_entity_type: str,
    relationship: str,
    properties: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a relationship between two campaign entities.
    
    Args:
        from_entity: Name/ID of the source entity
        from_entity_type: Type of source entity (Character, Location, etc.)
        to_entity: Name/ID of the target entity
        to_entity_type: Type of target entity
        relationship: Type of relationship (KNOWS, LOCATED_IN, OWNS, QUESTS_FOR, etc.)
        properties: Optional properties for the relationship
    
    Returns:
        Confirmation message
    """
    try:
        neo4j_manager = Neo4jSpatialManager(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password")
        )
        
        success = neo4j_manager.create_relationship(
            from_node_label=from_entity_type,
            from_node_property="name",
            from_node_value=from_entity,
            to_node_label=to_entity_type,
            to_node_property="name",
            to_node_value=to_entity,
            relationship_type=relationship,
            properties=properties or {}
        )
        neo4j_manager.close()
        
        if success:
            return f"Successfully created relationship: ({from_entity})-[{relationship}]->({to_entity})"
        else:
            return f"Failed to create relationship. Make sure both entities exist."
    except Exception as e:
        return f"Error creating relationship: {str(e)}"


async def query_campaign_graph(
    ctx: RunContext[CampaignDeps],
    query_description: str,
    cypher_query: Optional[str] = None
) -> str:
    """
    Query the campaign graph database to retrieve information about entities and relationships.
    
    Args:
        query_description: Description of what you're looking for
        cypher_query: Optional direct Cypher query to execute
    
    Returns:
        Query results as JSON string
    """
    try:
        neo4j_manager = Neo4jSpatialManager(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password")
        )
        
        if cypher_query:
            result = neo4j_manager.query_graph(cypher_query)
        else:
            # General query - get recent nodes
            result = neo4j_manager.query_graph(
                "MATCH (n) RETURN n ORDER BY id(n) DESC LIMIT 10"
            )
        
        neo4j_manager.close()
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return f"Error querying graph: {str(e)}"


async def lookup_dnd_resource(
    ctx: RunContext[CampaignDeps],
    resource_type: str,
    resource_index: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None
) -> str:
    """
    Look up D&D 5e rules, monsters, spells, classes, equipment, and other game resources.
    
    Args:
        resource_type: Type of resource (spells, monsters, classes, equipment, rules, etc.)
        resource_index: Specific resource index to look up (e.g., 'fireball', 'goblin')
        filters: Optional filters for list queries (e.g., {'level': [1,2]} for spells)
    
    Returns:
        Resource information as JSON string
    """
    try:
        client = DnD5eAPIClient(base_url=ctx.deps.dnd_api_base)
        
        if resource_index:
            result = client.get_resource(resource_type, resource_index)
        else:
            result = client.get_resource(resource_type)
        
        client.close()
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return f"Error looking up D&D resource: {str(e)}"


# ============================================================================
# Agent Tools - Spatial/Map Functions
# ============================================================================

async def create_map_location(
    ctx: RunContext[CampaignDeps],
    map_id: str,
    map_name: str,
    description: str,
    grid_width: int,
    grid_height: int,
    grid_size: int = 5
) -> str:
    """
    Create a new map/area for tracking positions and distances.
    
    Args:
        map_id: Unique identifier for the map
        map_name: Name of the map/area
        description: Detailed description of the map
        grid_width: Width in grid squares
        grid_height: Height in grid squares
        grid_size: Size of each grid square in feet (default 5 for D&D)
    
    Returns:
        Confirmation message
    """
    try:
        neo4j_manager = Neo4jSpatialManager(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password")
        )
        
        map_data = {
            "map_id": map_id,
            "name": map_name,
            "description": description,
            "grid_width": grid_width,
            "grid_height": grid_height,
            "grid_size": grid_size,
            "width_feet": grid_width * grid_size,
            "height_feet": grid_height * grid_size
        }
        
        result = neo4j_manager.create_map(map_data)
        neo4j_manager.close()
        
        return f"Successfully created map '{map_name}' ({grid_width}x{grid_height} squares, {grid_width*grid_size}x{grid_height*grid_size} feet): {json.dumps(result, indent=2)}"
    except Exception as e:
        return f"Error creating map: {str(e)}"


async def create_detailed_location(
    ctx: RunContext[CampaignDeps],
    name: str,
    description: str,
    location_type: str,
    map_id: str,
    x: Optional[float] = None,
    y: Optional[float] = None,
    z: Optional[float] = 0.0,
    size_width: Optional[float] = None,
    size_height: Optional[float] = None,
    terrain: str = "normal",
    lighting: str = "bright",
    features: Optional[List[str]] = None
) -> str:
    """
    Create a detailed location with full spatial and descriptive properties.
    
    Args:
        name: Name of the location
        description: Rich, detailed description for scene setting
        location_type: Type (room, outdoor, dungeon, tavern, forest, etc.)
        map_id: ID of the map this location is on
        x: X coordinate on the map (optional)
        y: Y coordinate on the map (optional)
        z: Elevation/Z coordinate (optional, default 0)
        size_width: Width in feet (optional)
        size_height: Height in feet (optional)
        terrain: Terrain type (normal, difficult, water, etc.)
        lighting: Lighting conditions (bright, dim, dark, etc.)
        features: List of notable features (furniture, hazards, cover, etc.)
    
    Returns:
        Confirmation message
    """
    try:
        neo4j_manager = Neo4jSpatialManager(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password")
        )
        
        location_data = {
            "name": name,
            "description": description,
            "location_type": location_type,
            "terrain": terrain,
            "lighting": lighting,
            "features": features or []
        }
        
        if size_width:
            location_data["size_width"] = size_width
        if size_height:
            location_data["size_height"] = size_height
        
        position = (x, y, z) if x is not None and y is not None else None
        result = neo4j_manager.create_location_with_position(location_data, map_id, position)
        neo4j_manager.close()
        
        return f"Successfully created location '{name}' on map '{map_id}': {json.dumps(result, indent=2)}"
    except Exception as e:
        return f"Error creating location: {str(e)}"


async def set_entity_position(
    ctx: RunContext[CampaignDeps],
    entity_name: str,
    entity_type: str,
    x: float,
    y: float,
    z: float = 0.0,
    map_id: Optional[str] = None,
    location_name: Optional[str] = None
) -> str:
    """
    Set the position of a character, NPC, monster, or object on the map.
    
    Args:
        entity_name: Name of the entity to position
        entity_type: Type of entity (Character, NPC, Monster, Object)
        x: X coordinate in feet
        y: Y coordinate in feet
        z: Z coordinate/elevation in feet (default 0)
        map_id: ID of the map (REQUIRED - must specify either map_id or location_name)
        location_name: Name of location entity is in (optional, will get map_id from location)
    
    Returns:
        Confirmation message
    """
    try:
        neo4j_manager = Neo4jSpatialManager(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password")
        )
        
        success = neo4j_manager.set_entity_position(
            entity_type=entity_type,
            entity_name=entity_name,
            x=x,
            y=y,
            z=z,
            map_id=map_id,
            location_name=location_name
        )
        neo4j_manager.close()
        
        if success:
            position_str = f"({x}, {y}, {z})"
            location_str = f" in {location_name}" if location_name else ""
            map_str = f" on map {map_id}" if map_id else ""
            return f"Successfully positioned {entity_name} at {position_str}{location_str}{map_str}"
        else:
            return f"Failed to position entity. Make sure {entity_name} exists."
    except Exception as e:
        return f"Error setting position: {str(e)}"


async def calculate_distance(
    ctx: RunContext[CampaignDeps],
    entity1_name: str,
    entity1_type: str,
    entity2_name: str,
    entity2_type: str
) -> str:
    """
    Calculate the distance between two entities for movement and ranged attacks.
    Important for determining if attacks can reach, movement costs, etc.
    
    Args:
        entity1_name: Name of first entity
        entity1_type: Type of first entity (Character, NPC, Monster, etc.)
        entity2_name: Name of second entity
        entity2_type: Type of second entity
    
    Returns:
        Distance in feet
    """
    try:
        neo4j_manager = Neo4jSpatialManager(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password")
        )
        
        distance = neo4j_manager.calculate_distance(
            entity1_type=entity1_type,
            entity1_name=entity1_name,
            entity2_type=entity2_type,
            entity2_name=entity2_name
        )
        neo4j_manager.close()
        
        if distance is not None:
            # Provide D&D context
            squares = distance / 5  # D&D uses 5-foot squares
            context = []
            
            if distance <= 5:
                context.append("melee range")
            if distance <= 30:
                context.append("close range for most ranged weapons")
            elif distance <= 120:
                context.append("long range for many ranged weapons")
            
            context_str = f" ({', '.join(context)})" if context else ""
            return f"Distance between {entity1_name} and {entity2_name}: {distance} feet ({squares} squares){context_str}"
        else:
            return f"Could not calculate distance. Entities may not be on the same map or have positions set."
    except Exception as e:
        return f"Error calculating distance: {str(e)}"


async def get_entities_in_range(
    ctx: RunContext[CampaignDeps],
    entity_name: str,
    entity_type: str,
    range_feet: float,
    target_types: Optional[List[str]] = None
) -> str:
    """
    Get all entities within a certain range for spells, attacks, and abilities.
    Critical for area effects, targeting, and tactical decisions.
    
    Args:
        entity_name: Name of the entity at the center
        entity_type: Type of the central entity
        range_feet: Range in feet to search
        target_types: Optional list of entity types to filter (e.g., ["Character", "Monster"])
    
    Returns:
        List of entities in range with distances
    """
    try:
        neo4j_manager = Neo4jSpatialManager(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password")
        )
        
        entities = neo4j_manager.get_entities_in_range(
            entity_type=entity_type,
            entity_name=entity_name,
            range_feet=range_feet,
            target_types=target_types
        )
        neo4j_manager.close()
        
        if entities:
            result = f"Entities within {range_feet} feet of {entity_name}:\n"
            for ent in entities:
                result += f"  - {ent['name']} ({ent['entity_type']}): {ent['distance']} feet away\n"
            return result
        else:
            return f"No entities found within {range_feet} feet of {entity_name}"
    except Exception as e:
        return f"Error finding entities in range: {str(e)}"


async def generate_scene_description(
    ctx: RunContext[CampaignDeps],
    location_name: str
) -> str:
    """
    Generate a detailed scene description for a location including all entities present.
    Use this to set the scene and provide rich, immersive descriptions to players.
    
    Args:
        location_name: Name of the location to describe
    
    Returns:
        Detailed scene description with entity positions
    """
    try:
        neo4j_manager = Neo4jSpatialManager(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password")
        )
        
        scene_data = neo4j_manager.get_location_scene(location_name)
        neo4j_manager.close()
        
        if not scene_data:
            return f"Location '{location_name}' not found."
        
        location = scene_data["location"]
        entities = scene_data["entities"]
        
        # Build rich description
        description = f"\n{'='*60}\n"
        description += f"LOCATION: {location.get('name', 'Unknown')}\n"
        description += f"{'='*60}\n\n"
        description += f"{location.get('description', 'No description available.')}\n\n"
        
        # Add location details
        description += f"Type: {location.get('location_type', 'Unknown')}\n"
        description += f"Lighting: {location.get('lighting', 'Unknown')}\n"
        description += f"Terrain: {location.get('terrain', 'Normal')}\n"
        
        if location.get('features'):
            description += f"Notable Features: {', '.join(location['features'])}\n"
        
        if location.get('size_width') and location.get('size_height'):
            description += f"Size: {location['size_width']} x {location['size_height']} feet\n"
        
        # Add entities present
        if entities:
            description += f"\n--- Entities Present ---\n"
            for ent in entities:
                if ent.get('x') is not None and ent.get('y') is not None:
                    description += f"  • {ent['name']} ({ent['type']}) at position ({ent['x']}, {ent['y']})\n"
                else:
                    description += f"  • {ent['name']} ({ent['type']})\n"
        
        # Add connections
        if scene_data.get('connected_locations'):
            description += f"\n--- Exits ---\n"
            for conn in scene_data['connected_locations']:
                description += f"  → {conn}\n"
        
        description += f"\n{'='*60}\n"
        
        return description
    except Exception as e:
        return f"Error generating scene description: {str(e)}"


async def connect_locations(
    ctx: RunContext[CampaignDeps],
    location1: str,
    location2: str,
    connection_type: str = "CONNECTED_TO",
    distance: Optional[float] = None,
    description: Optional[str] = None
) -> str:
    """
    Create a connection between two locations (doors, passages, roads, etc.).
    Important for tracking how locations relate and calculating travel.
    
    Args:
        location1: Name of first location
        location2: Name of second location
        connection_type: Type of connection (CONNECTED_TO, DOOR, PASSAGE, ROAD, etc.)
        distance: Distance between locations in feet (optional)
        description: Description of the connection (optional)
    
    Returns:
        Confirmation message
    """
    try:
        neo4j_manager = Neo4jSpatialManager(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password")
        )
        
        success = neo4j_manager.create_location_connection(
            location1=location1,
            location2=location2,
            connection_type=connection_type,
            distance=distance,
            description=description
        )
        neo4j_manager.close()
        
        if success:
            dist_str = f" (distance: {distance} feet)" if distance else ""
            desc_str = f" - {description}" if description else ""
            return f"✅ Successfully connected {location1} to {location2} via {connection_type}{dist_str}{desc_str}"
        else:
            return f"❌ Failed to connect locations. Make sure both locations exist."
    except Exception as e:
        return f"❌ Error connecting locations: {str(e)}"


# ============================================================================
# Agent Tools - Database Inspection and Management
# ============================================================================

async def check_database_status(
    ctx: RunContext[CampaignDeps]
) -> str:
    """
    Check what entities currently exist in the Neo4j database.
    Use this before creating new entities to avoid duplicates.
    
    Returns:
        Summary of database contents
    """
    try:
        neo4j_manager = Neo4jSpatialManager(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password")
        )
        
        summary = neo4j_manager.get_database_summary()
        neo4j_manager.close()
        
        if summary['total_nodes'] == 0:
            return "📊 Database is empty. No entities exist yet."
        
        result = "📊 **Database Status**\n\n"
        result += f"**Total Nodes:** {summary['total_nodes']}\n"
        result += f"**Total Relationships:** {summary['total_relationships']}\n\n"
        
        if summary['node_counts']:
            result += "**Entities by Type:**\n"
            for label, count in summary['node_counts'].items():
                result += f"  • {label}: {count}\n"
        
        if summary['relationship_counts']:
            result += "\n**Relationships by Type:**\n"
            for rel_type, count in summary['relationship_counts'].items():
                result += f"  • {rel_type}: {count}\n"
        
        return result
    except Exception as e:
        return f"❌ Error checking database: {str(e)}"


async def list_entities_of_type(
    ctx: RunContext[CampaignDeps],
    entity_type: str,
    limit: int = 20
) -> str:
    """
    List all entities of a specific type from Neo4j database.
    Use this to see what characters, locations, NPCs, etc. already exist.
    
    Args:
        entity_type: Type of entity to list (Character, Location, NPC, Monster, etc.)
        limit: Maximum number of entities to return (default 20)
    
    Returns:
        List of entities with their properties
    """
    try:
        neo4j_manager = Neo4jSpatialManager(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password")
        )
        
        entities = neo4j_manager.get_all_entities_by_type(entity_type, limit)
        neo4j_manager.close()
        
        if not entities:
            return f"📋 No {entity_type} entities found in database."
        
        result = f"📋 **{entity_type} Entities ({len(entities)}):**\n\n"
        for entity in entities:
            name = entity.get('name', entity.get('id', 'Unknown'))
            result += f"**{name}**\n"
            for key, value in entity.items():
                if key != 'name' and value is not None:
                    result += f"  • {key}: {value}\n"
            result += "\n"
        
        return result
    except Exception as e:
        return f"❌ Error listing entities: {str(e)}"


# ============================================================================
# Agent Tools - Campaign Info Storage (PostgreSQL)
# ============================================================================

async def save_campaign_info(
    ctx: RunContext[CampaignDeps],
    campaign_id: str,
    info_type: str,
    title: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save campaign information to PostgreSQL vector database for long-term storage and semantic search.
    Use this for: plot developments, NPC backgrounds, location lore, quest details, world history, etc.
    
    Args:
        campaign_id: Unique identifier for the campaign
        info_type: Type of info (plot, npc_background, location_lore, quest, world_history, etc.)
        title: Title/summary of the information
        content: Detailed content
        metadata: Optional additional metadata
    
    Returns:
        Confirmation message
    """
    if not ctx.deps.postgres_conn:
        return "⚠️  PostgreSQL not configured. Campaign info storage disabled."
    
    try:
        pg_manager = PostgresVectorManager(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "dnd_campaign"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "password")
        )
        
        info_id = pg_manager.store_campaign_info(
            campaign_id=campaign_id,
            info_type=info_type,
            title=title,
            content=content,
            metadata=metadata
        )
        pg_manager.close()
        
        return f"✅ Saved campaign info (ID: {info_id}) - Type: {info_type}, Title: '{title}'"
    except Exception as e:
        return f"❌ Error saving campaign info: {str(e)}"


async def search_campaign_info(
    ctx: RunContext[CampaignDeps],
    campaign_id: str,
    query: str,
    info_type: Optional[str] = None,
    limit: int = 5
) -> str:
    """
    Search campaign information using semantic search.
    Use this to recall plot points, NPC details, location lore, etc.
    
    Args:
        campaign_id: Campaign identifier
        query: What to search for (semantic search)
        info_type: Optional filter by type (plot, npc_background, location_lore, etc.)
        limit: Maximum number of results (default 5)
    
    Returns:
        Relevant campaign information
    """
    if not ctx.deps.postgres_conn:
        return "⚠️  PostgreSQL not configured. Campaign info search disabled."
    
    try:
        pg_manager = PostgresVectorManager(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "dnd_campaign"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "password")
        )
        
        results = pg_manager.search_campaign_info(
            campaign_id=campaign_id,
            query=query,
            info_type=info_type,
            limit=limit
        )
        pg_manager.close()
        
        if not results:
            return f"🔍 No campaign info found matching: '{query}'"
        
        output = f"🔍 **Campaign Info Search Results** (Query: '{query}')\n\n"
        for i, result in enumerate(results, 1):
            output += f"**{i}. {result['title']}** ({result['info_type']})\n"
            output += f"{result['content']}\n"
            if result.get('metadata'):
                output += f"_Metadata: {result['metadata']}_\n"
            output += "\n"
        
        return output
    except Exception as e:
        return f"❌ Error searching campaign info: {str(e)}"


async def recall_chat_history(
    ctx: RunContext[CampaignDeps],
    campaign_id: str,
    session_id: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 20
) -> str:
    """
    Retrieve or search chat history from PostgreSQL.
    Use this to recall previous conversations and maintain continuity.
    
    Args:
        campaign_id: Campaign identifier
        session_id: Optional specific session to retrieve
        query: Optional search query for semantic search
        limit: Maximum number of messages (default 20)
    
    Returns:
        Chat history
    """
    if not ctx.deps.postgres_conn:
        return "⚠️  PostgreSQL not configured. Chat history disabled."
    
    try:
        pg_manager = PostgresVectorManager(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "dnd_campaign"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "password")
        )
        
        if query:
            # Semantic search
            results = pg_manager.search_chat_history(
                campaign_id=campaign_id,
                query=query,
                session_id=session_id,
                limit=limit
            )
        else:
            # Regular retrieval
            results = pg_manager.get_chat_history(
                campaign_id=campaign_id,
                session_id=session_id,
                limit=limit
            )
        
        pg_manager.close()
        
        if not results:
            return "💬 No chat history found."
        
        output = f"💬 **Chat History** ({len(results)} messages)\n\n"
        for msg in results:
            timestamp = msg['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            role_emoji = "🧙" if msg['role'] == 'assistant' else "👤"
            output += f"{role_emoji} **{msg['role'].title()}** ({timestamp}):\n"
            output += f"{msg['content']}\n\n"
        
        return output
    except Exception as e:
        return f"❌ Error retrieving chat history: {str(e)}"


# ============================================================================
# Agent Configuration
# ============================================================================

def create_dnd_agent(
    model_provider: str = "openai",
    model_name: str = "gpt-4o",
    ollama_base_url: str = "http://localhost:11434/v1"
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
        retries=2
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
    
    print = lambda x: console.print(Markdown(x)) # Simple markdown rendering
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
    #suppress_neo4j_warnings()
    
    print("\n🤖 Model Configuration")
    print("=" * 80)
    print("Using: OpenAI GPT-4o")
    print("💡 TIP: Switch to Ollama to avoid rate limits (see instructions in code)")
    print("=" * 80 + "\n")
    
    # Create the agent
    agent = create_dnd_agent(
        model_provider="openai",  # Change to "ollama" to use local models
        model_name="gpt-4o-mini",      # Or "llama3.1" for Ollama
        ollama_base_url="http://localhost:11434/v1"
    )
    
    # Create dependencies
    neo4j_manager = Neo4jSpatialManager(
        uri=os.environ["NEO4J_URI"],
        user=os.environ["NEO4J_USER"],
        password=os.environ["NEO4J_PASSWORD"]
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
                password=os.environ["POSTGRES_PASSWORD"]
            )
            print("✅ PostgreSQL connected - Campaign memory and chat history enabled\n")
        except Exception as e:
            print(f"⚠️  PostgreSQL not available: {e}")
            print("   Campaign memory features will be disabled.\n")
    
    deps = CampaignDeps(
        neo4j_driver=neo4j_manager.driver,
        postgres_conn=postgres_conn,
        dnd_api_base="https://www.dnd5eapi.co"
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
            deps
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
            deps
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
            deps
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
            deps
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
            deps
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
            deps
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
    
    print("""
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
    """)
    
    asyncio.run(main())