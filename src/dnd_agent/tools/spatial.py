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
    location_name: Optional[str] = None,
) -> str:
    """Create a new map node and optionally link it to a location entity.

    The ``map_id`` and ``map_name`` identify the new map. ``description``
    describes the map contents, and ``grid_width``, ``grid_height`` and
    ``grid_size`` define its dimensions. If ``location_name`` is provided
    a relationship will be created from the corresponding ``Location`` node
    to the new ``Map`` node using the ``HAS_MAP`` relationship.  Linking
    maps to locations allows callers to navigate between them in the graph.

    Returns a human‑friendly status message indicating success or failure.
    """
    try:
        manager = await _create_neo4j_manager()
        # Attempt to create the map using the available method on the manager.
        try:
            # Older versions of Neo4jSpatialManager expose create_map() which
            # expects a single dictionary of map properties.  Newer versions
            # may offer create_map_location() which takes explicit kwargs.  We
            # detect and use whichever is available.
            if hasattr(manager, "create_map_location"):
                success = manager.create_map_location(
                    map_id=map_id,
                    map_name=map_name,
                    description=description,
                    grid_width=grid_width,
                    grid_height=grid_height,
                    grid_size=grid_size,
                )
            else:
                # create_map returns a dict when creation succeeds.
                result = manager.create_map(
                    dict(
                        map_id=map_id,
                        map_name=map_name,
                        description=description,
                        grid_width=grid_width,
                        grid_height=grid_height,
                        grid_size=grid_size,
                    )
                )
                success = bool(result)
        except Exception:
            success = False

        # If map creation succeeded and a location name was provided, link them.
        if success and location_name:
            try:
                manager.create_relationship(
                    from_node_label="Location",
                    from_node_property="name",
                    from_node_value=location_name,
                    to_node_label="Map",
                    to_node_property="map_id",
                    to_node_value=map_id,
                    relationship_type="HAS_MAP",
                    properties={},
                )
            except Exception:
                # Ignore relationship failures but continue closing the manager
                pass
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
        # If the position was updated, also record the explicit spatial relationships.
        if success:
            try:
                # Link the entity to its map via an ON_MAP relationship if a map_id is provided
                if map_id:
                    manager.create_relationship(
                        from_node_label=entity_type,
                        from_node_property="name",
                        from_node_value=entity_name,
                        to_node_label="Map",
                        to_node_property="map_id",
                        to_node_value=map_id,
                        relationship_type="ON_MAP",
                        properties={},
                    )
                # Link the entity to a specific location via an AT_LOCATION relationship
                if location_name:
                    manager.create_relationship(
                        from_node_label=entity_type,
                        from_node_property="name",
                        from_node_value=entity_name,
                        to_node_label="Location",
                        to_node_property="name",
                        to_node_value=location_name,
                        relationship_type="AT_LOCATION",
                        properties={},
                    )
            except Exception:
                # Ignore failures when creating relationships
                pass
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


# -----------------------------------------------------------------------------
# Location creation
# -----------------------------------------------------------------------------
async def create_location(
    ctx: RunContext[CampaignDeps],
    location_name: str,
    map_id: str,
    description: str,
    location_type: Optional[str] = None,
    x: Optional[float] = None,
    y: Optional[float] = None,
    z: Optional[float] = None,
) -> str:
    """Create a new location on a given map with optional position.

    This helper wraps :meth:`Neo4jSpatialManager.create_location_with_position` and
    returns a user‑friendly message.  At minimum you must specify a
    ``location_name`` and the ``map_id`` of the map it belongs to.  A
    description and an optional ``location_type`` can be provided to
    enrich the node.  If ``x`` and ``y`` (and optionally ``z``) are
    provided, the location will be assigned a precise position on the
    map.  Coordinates are measured in feet.

    When successful, a ``Location`` node is created with the supplied
    properties and an ``ON_MAP`` relationship is established to the map.
    On failure, a short error message is returned.
    """
    try:
        manager = await _create_neo4j_manager()
        location_data: Dict[str, Any] = {
            "name": location_name,
            "description": description,
        }
        if location_type:
            location_data["location_type"] = location_type
        # Determine whether to include a position tuple
        position: Optional[tuple] = None
        if x is not None and y is not None:
            # If z is omitted, default to 0.0
            position = (x, y, z if z is not None else 0.0)

        try:
            result = manager.create_location_with_position(
                location_data=location_data, map_id=map_id, position=position
            )
            success = bool(result)
        except Exception:
            success = False
        manager.close()
        if success:
            pos_desc = (
                f" at ({x}, {y}{f', {z}' if z is not None else ''})"
                if position
                else ""
            )
            return f"✅ Created location '{location_name}' on map {map_id}{pos_desc}"
        return f"❌ Failed to create location '{location_name}'. Make sure the map exists."
    except Exception as exc:  # noqa: BLE001
        return f"Error creating location: {exc}"


# -----------------------------------------------------------------------------
# Move entity into a location
# -----------------------------------------------------------------------------
async def move_entity_to_location(
    ctx: RunContext[CampaignDeps],
    entity_type: str,
    entity_name: str,
    location_name: str,
) -> str:
    """Move an existing entity into a location, adopting that location's position.

    This function looks up the specified location's coordinates and then calls
    ``Neo4jSpatialManager.set_entity_position`` to update the entity.  After
    setting the position, it also establishes the appropriate ``AT_LOCATION`` and
    ``ON_MAP`` relationships (delegated to ``set_entity_position`` logic).

    Use this when you want to move a character, NPC or other entity into a
    location rather than specifying explicit coordinates.  If the location
    does not have a position, the entity is placed at (0, 0, 0).
    """
    try:
        manager = await _create_neo4j_manager()
        # Fetch the location's scene to obtain its position and current map
        scene = manager.get_location_scene(location_name)
        if not scene or "location" not in scene:
            manager.close()
            return f"❌ Location '{location_name}' not found."
        location_data = scene["location"]
        # Extract coordinates if available, otherwise default to 0.0
        x = location_data.get("x")
        y = location_data.get("y")
        z = location_data.get("z")
        # Provide defaults if any coordinate is missing
        x = float(x) if x is not None else 0.0
        y = float(y) if y is not None else 0.0
        z = float(z) if z is not None else 0.0
        # Call the manager to update the entity's position
        success = manager.set_entity_position(
            entity_type=entity_type,
            entity_name=entity_name,
            x=x,
            y=y,
            z=z,
            map_id=location_data.get("map_id"),
            location_name=location_name,
        )
        # Use our linking logic on success
        if success:
            try:
                # Link entity to map
                map_id = location_data.get("map_id")
                if map_id:
                    manager.create_relationship(
                        from_node_label=entity_type,
                        from_node_property="name",
                        from_node_value=entity_name,
                        to_node_label="Map",
                        to_node_property="map_id",
                        to_node_value=map_id,
                        relationship_type="ON_MAP",
                        properties={},
                    )
                # Link entity to location
                manager.create_relationship(
                    from_node_label=entity_type,
                    from_node_property="name",
                    from_node_value=entity_name,
                    to_node_label="Location",
                    to_node_property="name",
                    to_node_value=location_name,
                    relationship_type="AT_LOCATION",
                    properties={},
                )
            except Exception:
                pass
        manager.close()
        if success:
            return f"✅ Moved {entity_name} to {location_name} at ({x}, {y}, {z})"
        return f"❌ Failed to move {entity_name} to {location_name}. Make sure the entity exists."
    except Exception as exc:  # noqa: BLE001
        return f"Error moving entity: {exc}"