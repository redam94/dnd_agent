"""
Spatial tools.

This module groups functions that operate on the spatial graph of the
campaign.  Use them to create maps, set entity positions, calculate
distances, query nearby entities, generate scene descriptions and
connect locations.  All functions return human readable strings and
never raise exceptions.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from pydantic_ai import RunContext

from dnd_agent.database.neo4j_manager import Neo4jSpatialManager
from dnd_agent.models.agent_deps import CampaignDeps


async def _create_neo4j_manager() -> Neo4jSpatialManager:
    return Neo4jSpatialManager(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
    )


async def create_map_location(
    ctx: RunContext[CampaignDeps],
    map_id: str,
    map_name: str,
    description: str,
    grid_width: int,
    grid_height: int,
    grid_size: int = 5,
) -> str:
    """Create a new map node and associate it with a location entity."""
    try:
        manager = await _create_neo4j_manager()
        success = manager.create_map_location(
            map_id=map_id,
            map_name=map_name,
            description=description,
            grid_width=grid_width,
            grid_height=grid_height,
            grid_size=grid_size,
        )
        manager.close()
        if success:
            return f"✅ Created map '{map_name}' ({map_id})"
        return f"❌ Failed to create map '{map_name}'."
    except Exception as exc:  # noqa: BLE001
        return f"Error creating map: {exc}"


async def set_entity_position(
    ctx: RunContext[CampaignDeps],
    entity_type: str,
    entity_name: str,
    x: float,
    y: float,
    z: Optional[float] = None,
    map_id: Optional[str] = None,
    location_name: Optional[str] = None,
) -> str:
    """Set the position of an entity in the spatial graph."""
    try:
        manager = await _create_neo4j_manager()
        success = manager.set_entity_position(
            entity_type=entity_type,
            entity_name=entity_name,
            x=x,
            y=y,
            z=z,
            map_id=map_id,
            location_name=location_name,
        )
        manager.close()
        if success:
            pos_str = f"({x}, {y}, {z})" if z is not None else f"({x}, {y})"
            loc_str = f" in {location_name}" if location_name else ""
            map_str = f" on map {map_id}" if map_id else ""
            return f"Successfully positioned {entity_name} at {pos_str}{loc_str}{map_str}"
        return f"Failed to position entity. Make sure {entity_name} exists."
    except Exception as exc:  # noqa: BLE001
        return f"Error setting position: {exc}"


async def calculate_distance(
    ctx: RunContext[CampaignDeps],
    entity1_name: str,
    entity1_type: str,
    entity2_name: str,
    entity2_type: str,
) -> str:
    """Calculate the distance between two entities and provide D&D context."""
    try:
        manager = await _create_neo4j_manager()
        distance = manager.calculate_distance(
            entity1_type=entity1_type,
            entity1_name=entity1_name,
            entity2_type=entity2_type,
            entity2_name=entity2_name,
        )
        manager.close()
        if distance is None:
            return f"Could not calculate distance. Entities may not be on the same map or have positions set."
        squares = distance / 5.0
        context: List[str] = []
        if distance <= 5:
            context.append("melee range")
        if distance <= 30:
            context.append("close range for most ranged weapons")
        elif distance <= 120:
            context.append("long range for many ranged weapons")
        ctx_str = f" ({', '.join(context)})" if context else ""
        return f"Distance between {entity1_name} and {entity2_name}: {distance} feet ({squares} squares){ctx_str}"
    except Exception as exc:  # noqa: BLE001
        return f"Error calculating distance: {exc}"


async def get_entities_in_range(
    ctx: RunContext[CampaignDeps],
    entity_name: str,
    entity_type: str,
    range_feet: float,
    target_types: Optional[List[str]] = None,
) -> str:
    """Return a list of entities within a given range of a central entity."""
    try:
        manager = await _create_neo4j_manager()
        entities = manager.get_entities_in_range(
            entity_type=entity_type,
            entity_name=entity_name,
            range_feet=range_feet,
            target_types=target_types,
        )
        manager.close()
        if not entities:
            return f"No entities found within {range_feet} feet of {entity_name}"
        result_lines = [f"Entities within {range_feet} feet of {entity_name}:"]
        for ent in entities:
            result_lines.append(f"  - {ent['name']} ({ent['entity_type']}): {ent['distance']} feet away")
        return "\n".join(result_lines)
    except Exception as exc:  # noqa: BLE001
        return f"Error finding entities in range: {exc}"


async def generate_scene_description(ctx: RunContext[CampaignDeps], location_name: str) -> str:
    """Generate a rich description of a location including entities and exits."""
    try:
        manager = await _create_neo4j_manager()
        scene_data = manager.get_location_scene(location_name)
        manager.close()
        if not scene_data:
            return f"Location '{location_name}' not found."
        location = scene_data.get("location", {})
        entities = scene_data.get("entities", [])
        description = f"\n{'='*60}\n"
        description += f"LOCATION: {location.get('name', 'Unknown')}\n"
        description += f"{'='*60}\n\n"
        description += f"{location.get('description', 'No description available.')}\n\n"
        description += f"Type: {location.get('location_type', 'Unknown')}\n"
        description += f"Lighting: {location.get('lighting', 'Unknown')}\n"
        description += f"Terrain: {location.get('terrain', 'Normal')}\n"
        if location.get('features'):
            description += f"Notable Features: {', '.join(location['features'])}\n"
        if location.get('size_width') and location.get('size_height'):
            description += f"Size: {location['size_width']} x {location['size_height']} feet\n"
        if entities:
            description += "\n--- Entities Present ---\n"
            for ent in entities:
                if ent.get('x') is not None and ent.get('y') is not None:
                    description += f"  • {ent['name']} ({ent['type']}) at position ({ent['x']}, {ent['y']})\n"
                else:
                    description += f"  • {ent['name']} ({ent['type']})\n"
        if scene_data.get('connected_locations'):
            description += "\n--- Exits ---\n"
            for conn in scene_data['connected_locations']:
                description += f"  → {conn}\n"
        description += f"\n{'='*60}\n"
        return description
    except Exception as exc:  # noqa: BLE001
        return f"Error generating scene description: {exc}"


async def connect_locations(
    ctx: RunContext[CampaignDeps],
    location1: str,
    location2: str,
    connection_type: str = "CONNECTED_TO",
    distance: Optional[float] = None,
    description: Optional[str] = None,
) -> str:
    """Create a link between two location nodes."""
    try:
        manager = await _create_neo4j_manager()
        success = manager.create_location_connection(
            location1=location1,
            location2=location2,
            connection_type=connection_type,
            distance=distance,
            description=description,
        )
        manager.close()
        if success:
            dist_str = f" (distance: {distance} feet)" if distance else ""
            desc_str = f" - {description}" if description else ""
            return f"✅ Successfully connected {location1} to {location2} via {connection_type}{dist_str}{desc_str}"
        return f"❌ Failed to connect locations. Make sure both locations exist."
    except Exception as exc:  # noqa: BLE001
        return f"❌ Error connecting locations: {exc}"