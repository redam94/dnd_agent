"""
Graph database tools.

These functions provide a simple interface to the Neo4j graph used
internally by the campaign to store entities, relationships and spatial
information.  Each function returns a string suitable for presentation
to the end user and will never raise an exception – any exceptions are
captured and returned as error messages.

Functions accept a :class:`pydantic_ai.RunContext` with a generic
``CampaignDeps`` type.  The context may provide database connection
information via environment variables if not explicitly configured.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from pydantic_ai import RunContext

from dnd_agent.database.neo4j_manager import Neo4jSpatialManager
from dnd_agent.models.agent_deps import CampaignDeps


async def _create_neo4j_manager() -> Neo4jSpatialManager:
    """Create a Neo4j manager using environment variables.

    We centralise creation here to avoid duplicating URI/user/password
    retrieval throughout the module.  If you need to customise the
    configuration at runtime, consider extending ``CampaignDeps`` to
    carry a pre‑built manager instead.
    """
    return Neo4jSpatialManager(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
    )


async def store_campaign_entity(
    ctx: RunContext[CampaignDeps],
    entity_type: str,
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a new entity in the graph.

    The ``entity_type`` must match one of the allowed labels in the
    domain (e.g. Character, NPC, Monster, Location, Item, Quest).  If
    an entity with the same name already exists, a warning is returned
    instead of creating a duplicate.
    """
    try:
        manager = await _create_neo4j_manager()
        # Check for duplicates
        if manager.entity_exists(entity_type, name):
            manager.close()
            return f"⚠️  {entity_type} '{name}' already exists in the database. Use a different name or update the existing entity."
        props = attributes.copy() if attributes else {}
        props["name"] = name
        result = manager.create_node(entity_type, props)
        manager.close()
        return f"✅ Successfully created {entity_type} '{name}' with properties: {json.dumps(result, indent=2)}"
    except Exception as exc:  # noqa: BLE001 broad exception capturing
        return f"❌ Error creating entity: {exc}"


async def create_campaign_relationship(
    ctx: RunContext[CampaignDeps],
    from_entity: str,
    from_entity_type: str,
    to_entity: str,
    to_entity_type: str,
    relationship: str,
    properties: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a relationship between two entities.

    Relationships are directed and optionally carry properties.  Common
    relationship types include ``KNOWS``, ``LOCATED_IN``, ``OWNS`` and
    ``QUESTS_FOR``.  If either entity does not exist a failure
    message is returned.
    """
    try:
        manager = await _create_neo4j_manager()
        success = manager.create_relationship(
            from_node_label=from_entity_type,
            from_node_property="name",
            from_node_value=from_entity,
            to_node_label=to_entity_type,
            to_node_property="name",
            to_node_value=to_entity,
            relationship_type=relationship,
            properties=properties or {},
        )
        manager.close()
        if success:
            return f"Successfully created relationship: ({from_entity})-[{relationship}]->({to_entity})"
        return f"Failed to create relationship. Make sure both entities exist."
    except Exception as exc:  # noqa: BLE001
        return f"Error creating relationship: {exc}"


async def query_campaign_graph(
    ctx: RunContext[CampaignDeps], query_description: str, cypher_query: Optional[str] = None
) -> str:
    """Execute a Cypher query against the graph.

    If ``cypher_query`` is not provided a simple default query will
    return the most recent nodes.  The ``query_description`` is
    included in the response for context but not executed.
    """
    try:
        manager = await _create_neo4j_manager()
        if cypher_query:
            result = manager.query_graph(cypher_query)
        else:
            result = manager.query_graph("MATCH (n) RETURN n ORDER BY id(n) DESC LIMIT 10")
        manager.close()
        return json.dumps(result, indent=2, default=str)
    except Exception as exc:  # noqa: BLE001
        return f"Error querying graph: {exc}"


async def check_database_status(ctx: RunContext[CampaignDeps]) -> str:
    """Return a human‑friendly summary of the graph database contents."""
    try:
        manager = await _create_neo4j_manager()
        summary = manager.get_database_summary()
        manager.close()
        if summary.get("total_nodes", 0) == 0:
            return "Database is empty. No entities exist yet."
        result = "**Database Status**\n\n"
        result += f"**Total Nodes:** {summary['total_nodes']}\n"
        result += f"**Total Relationships:** {summary['total_relationships']}\n\n"
        node_counts = summary.get("node_counts")
        if node_counts:
            result += "**Entities by Type:**\n"
            for label, count in node_counts.items():
                result += f"  • {label}: {count}\n"
        rel_counts = summary.get("relationship_counts")
        if rel_counts:
            result += "\n**Relationships by Type:**\n"
            for rel_type, count in rel_counts.items():
                result += f"  • {rel_type}: {count}\n"
        return result
    except Exception as exc:  # noqa: BLE001
        return f"❌ Error checking database: {exc}"


async def list_entities_of_type(ctx: RunContext[CampaignDeps], entity_type: str, limit: int = 20) -> str:
    """List all entities of a given type up to the specified limit."""
    try:
        manager = await _create_neo4j_manager()
        entities = manager.get_all_entities_by_type(entity_type, limit)
        manager.close()
        if not entities:
            return f"No {entity_type} entities found in database."
        output = f"**{entity_type} Entities ({len(entities)}):**\n\n"
        for entity in entities:
            name = entity.get("name", entity.get("id", "Unknown"))
            output += f"**{name}**\n"
            for key, value in entity.items():
                if key != "name" and value is not None:
                    output += f"  • {key}: {value}\n"
            output += "\n"
        return output
    except Exception as exc:  # noqa: BLE001
        return f"❌ Error listing entities: {exc}"