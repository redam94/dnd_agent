import json
import os
from typing import Any, Dict, List, Optional

from pydantic_ai import RunContext

from dnd_agent.database.api_client import DnD5eAPIClient
from dnd_agent.database.neo4j_manager import Neo4jSpatialManager
from dnd_agent.database.vector_db import PostgresVectorManager
from dnd_agent.models.agent_deps import CampaignDeps


async def store_campaign_entity(
    ctx: RunContext[CampaignDeps],
    entity_type: str,  # Changed from 'label' to 'entity_type' for clarity
    name: str,  # Make name explicit as a separate parameter
    attributes: Optional[Dict[str, Any]] = None,  # Changed from 'properties' to 'attributes'
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
            password=os.getenv("NEO4J_PASSWORD", "password"),
        )

        # Check if entity already exists
        if neo4j_manager.entity_exists(entity_type, name):
            neo4j_manager.close()
            return f"⚠️  {entity_type} '{name}' already exists in the database. Use a different name or update the existing entity."

        # Combine name with other attributes
        properties = attributes or {}
        properties["name"] = name

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
    properties: Optional[Dict[str, Any]] = None,
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
            password=os.getenv("NEO4J_PASSWORD", "password"),
        )

        success = neo4j_manager.create_relationship(
            from_node_label=from_entity_type,
            from_node_property="name",
            from_node_value=from_entity,
            to_node_label=to_entity_type,
            to_node_property="name",
            to_node_value=to_entity,
            relationship_type=relationship,
            properties=properties or {},
        )
        neo4j_manager.close()

        if success:
            return f"Successfully created relationship: ({from_entity})-[{relationship}]->({to_entity})"
        else:
            return f"Failed to create relationship. Make sure both entities exist."
    except Exception as e:
        return f"Error creating relationship: {str(e)}"


async def query_campaign_graph(
    ctx: RunContext[CampaignDeps], query_description: str, cypher_query: Optional[str] = None
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
            password=os.getenv("NEO4J_PASSWORD", "password"),
        )

        if cypher_query:
            result = neo4j_manager.query_graph(cypher_query)
        else:
            # General query - get recent nodes
            result = neo4j_manager.query_graph("MATCH (n) RETURN n ORDER BY id(n) DESC LIMIT 10")

        neo4j_manager.close()
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return f"Error querying graph: {str(e)}"


async def lookup_dnd_resource(
    ctx: RunContext[CampaignDeps],
    resource_type: str,
    resource_index: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
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
    grid_size: int = 5,
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
            password=os.getenv("NEO4J_PASSWORD", "password"),
        )

        map_data = {
            "map_id": map_id,
            "name": map_name,
            "description": description,
            "grid_width": grid_width,
            "grid_height": grid_height,
            "grid_size": grid_size,
            "width_feet": grid_width * grid_size,
            "height_feet": grid_height * grid_size,
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
    features: Optional[List[str]] = None,
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
            password=os.getenv("NEO4J_PASSWORD", "password"),
        )

        location_data = {
            "name": name,
            "description": description,
            "location_type": location_type,
            "terrain": terrain,
            "lighting": lighting,
            "features": features or [],
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
    location_name: Optional[str] = None,
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
            password=os.getenv("NEO4J_PASSWORD", "password"),
        )

        success = neo4j_manager.set_entity_position(
            entity_type=entity_type,
            entity_name=entity_name,
            x=x,
            y=y,
            z=z,
            map_id=map_id,
            location_name=location_name,
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
    entity2_type: str,
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
            password=os.getenv("NEO4J_PASSWORD", "password"),
        )

        distance = neo4j_manager.calculate_distance(
            entity1_type=entity1_type,
            entity1_name=entity1_name,
            entity2_type=entity2_type,
            entity2_name=entity2_name,
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
    target_types: Optional[List[str]] = None,
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
            password=os.getenv("NEO4J_PASSWORD", "password"),
        )

        entities = neo4j_manager.get_entities_in_range(
            entity_type=entity_type,
            entity_name=entity_name,
            range_feet=range_feet,
            target_types=target_types,
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


async def generate_scene_description(ctx: RunContext[CampaignDeps], location_name: str) -> str:
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
            password=os.getenv("NEO4J_PASSWORD", "password"),
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

        if location.get("features"):
            description += f"Notable Features: {', '.join(location['features'])}\n"

        if location.get("size_width") and location.get("size_height"):
            description += f"Size: {location['size_width']} x {location['size_height']} feet\n"

        # Add entities present
        if entities:
            description += f"\n--- Entities Present ---\n"
            for ent in entities:
                if ent.get("x") is not None and ent.get("y") is not None:
                    description += (
                        f"  • {ent['name']} ({ent['type']}) at position ({ent['x']}, {ent['y']})\n"
                    )
                else:
                    description += f"  • {ent['name']} ({ent['type']})\n"

        # Add connections
        if scene_data.get("connected_locations"):
            description += f"\n--- Exits ---\n"
            for conn in scene_data["connected_locations"]:
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
    description: Optional[str] = None,
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
            password=os.getenv("NEO4J_PASSWORD", "password"),
        )

        success = neo4j_manager.create_location_connection(
            location1=location1,
            location2=location2,
            connection_type=connection_type,
            distance=distance,
            description=description,
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


async def check_database_status(ctx: RunContext[CampaignDeps]) -> str:
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
            password=os.getenv("NEO4J_PASSWORD", "password"),
        )

        summary = neo4j_manager.get_database_summary()
        neo4j_manager.close()

        if summary["total_nodes"] == 0:
            return "📊 Database is empty. No entities exist yet."

        result = "📊 **Database Status**\n\n"
        result += f"**Total Nodes:** {summary['total_nodes']}\n"
        result += f"**Total Relationships:** {summary['total_relationships']}\n\n"

        if summary["node_counts"]:
            result += "**Entities by Type:**\n"
            for label, count in summary["node_counts"].items():
                result += f"  • {label}: {count}\n"

        if summary["relationship_counts"]:
            result += "\n**Relationships by Type:**\n"
            for rel_type, count in summary["relationship_counts"].items():
                result += f"  • {rel_type}: {count}\n"

        return result
    except Exception as e:
        return f"❌ Error checking database: {str(e)}"


async def list_entities_of_type(
    ctx: RunContext[CampaignDeps], entity_type: str, limit: int = 20
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
            password=os.getenv("NEO4J_PASSWORD", "password"),
        )

        entities = neo4j_manager.get_all_entities_by_type(entity_type, limit)
        neo4j_manager.close()

        if not entities:
            return f"📋 No {entity_type} entities found in database."

        result = f"📋 **{entity_type} Entities ({len(entities)}):**\n\n"
        for entity in entities:
            name = entity.get("name", entity.get("id", "Unknown"))
            result += f"**{name}**\n"
            for key, value in entity.items():
                if key != "name" and value is not None:
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
    metadata: Optional[Dict[str, Any]] = None,
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
            password=os.getenv("POSTGRES_PASSWORD", "password"),
        )

        info_id = pg_manager.store_campaign_info(
            campaign_id=campaign_id,
            info_type=info_type,
            title=title,
            content=content,
            metadata=metadata,
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
    limit: int = 5,
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
            password=os.getenv("POSTGRES_PASSWORD", "password"),
        )

        results = pg_manager.search_campaign_info(
            campaign_id=campaign_id, query=query, info_type=info_type, limit=limit
        )
        pg_manager.close()

        if not results:
            return f"🔍 No campaign info found matching: '{query}'"

        output = f"🔍 **Campaign Info Search Results** (Query: '{query}')\n\n"
        for i, result in enumerate(results, 1):
            output += f"**{i}. {result['title']}** ({result['info_type']})\n"
            output += f"{result['content']}\n"
            if result.get("metadata"):
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
    limit: int = 20,
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
            password=os.getenv("POSTGRES_PASSWORD", "password"),
        )

        if query:
            # Semantic search
            results = pg_manager.search_chat_history(
                campaign_id=campaign_id, query=query, session_id=session_id, limit=limit
            )
        else:
            # Regular retrieval
            results = pg_manager.get_chat_history(
                campaign_id=campaign_id, session_id=session_id, limit=limit
            )

        pg_manager.close()

        if not results:
            return "💬 No chat history found."

        output = f"💬 **Chat History** ({len(results)} messages)\n\n"
        for msg in results:
            timestamp = msg["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            role_emoji = "🧙" if msg["role"] == "assistant" else "👤"
            output += f"{role_emoji} **{msg['role'].title()}** ({timestamp}):\n"
            output += f"{msg['content']}\n\n"

        return output
    except Exception as e:
        return f"❌ Error retrieving chat history: {str(e)}"
