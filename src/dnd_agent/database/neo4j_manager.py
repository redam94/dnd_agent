"""
Neo4j Spatial manager for DnD Agent
===================================

This module provides an abstraction over the Neo4j driver for common
operations used by the DnD agent.  It encapsulates the logic for
creating maps, locations and entities, positioning them in a spatial
graph and querying distances and relationships.  The manager is
designed to be stateless; each method opens a new session on the
driver and cleans up after itself.

Two convenience helpers are provided: ``create_map_location``
conveniently creates a map and links it to a location (if provided),
and ``create_location_with_position`` creates a location on an
existing map at a given coordinate.  The manager also exposes
``create_relationship`` for arbitrary relationships between nodes.

You may suppress Neo4j schema warnings during development by calling
``suppress_neo4j_warnings`` before performing operations.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase

__all__ = ["Neo4jSpatialManager", "suppress_neo4j_warnings"]


def suppress_neo4j_warnings() -> None:
    """Optionally suppress Neo4j schema warnings for a cleaner output."""
    warnings.filterwarnings("ignore", category=UserWarning, module="neo4j")
    print("ℹ️  Neo4j schema warnings suppressed for cleaner output.\n")


class Neo4jSpatialManager:
    """Manages Neo4j graph database operations with spatial awareness."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._create_spatial_indexes()

    def _create_spatial_indexes(self) -> None:
        """Create common indexes used by spatial queries."""
        with self.driver.session() as session:
            session.run("CREATE INDEX location_name IF NOT EXISTS FOR (n:Location) ON (n.name)")
            session.run("CREATE INDEX character_name IF NOT EXISTS FOR (n:Character) ON (n.name)")
            session.run("CREATE INDEX map_id IF NOT EXISTS FOR (n:Map) ON (n.map_id)")

    def close(self) -> None:
        """Close the underlying Neo4j driver."""
        self.driver.close()

    # ------------------------------------------------------------------
    # Map creation
    # ------------------------------------------------------------------
    def create_map(self, map_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a map node.  Returns the node properties if created."""
        with self.driver.session() as session:
            query = """
            CREATE (m:Map $properties)
            RETURN m
            LIMIT 1
            """
            map_data['name'] = map_data.get('map_name', 'Unnamed Map')
            result = session.run(query, properties=map_data)
            if not result.peek():
                return {}
            record = result.single()
            return dict(record["m"]) if record else {}

    def create_map_location(
        self,
        map_id: str,
        map_name: str,
        description: str,
        grid_width: int,
        grid_height: int,
        grid_size: int = 5,
        location_name: Optional[str] = None,
    ) -> bool:
        """Create a map node and optionally link it to an existing location.

        If ``location_name`` is provided and a ``Location`` node with that
        name exists, a ``HAS_MAP`` relationship will be created from the
        location to the new map.  Returns ``True`` when the map is
        created successfully; otherwise returns ``False``.
        """
        # Create the map node
        map_data = {
            "map_id": map_id,
            "map_name": map_name,
            "description": description,
            "grid_width": grid_width,
            "grid_height": grid_height,
            "grid_size": grid_size,
        }
        created_map = self.create_map(map_data)
        if not created_map:
            return False
        # Optionally link to location
        if location_name:
            try:
                self.create_relationship(
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
                # Relationship creation failures should not abort map creation
                pass
        return True

    # ------------------------------------------------------------------
    # Location creation
    # ------------------------------------------------------------------
    def create_location_with_position(
        self,
        location_data: Dict[str, Any],
        map_id: str,
        position: Optional[Tuple[float, float, float]] = None,
    ) -> Dict[str, Any]:
        """Create a location with optional position on a map.

        The location will be linked to the map via an ``ON_MAP``
        relationship.  If ``position`` is provided, the location's
        ``x``, ``y`` and ``z`` coordinates are set accordingly.  Returns
        the node properties when created, or an empty dict if the
        operation fails.
        """
        with self.driver.session() as session:
            # Add map reference and position to the location data
            location_data = dict(location_data)
            location_data["map_id"] = map_id
            if position:
                location_data["x"] = position[0]
                location_data["y"] = position[1]
                location_data["z"] = position[2] if len(position) > 2 else 0.0
            query = """
            MATCH (m:Map {map_id: $map_id})
            WITH m LIMIT 1
            CREATE (l:Location $location_data)
            CREATE (l)-[:ON_MAP]->(m)
            RETURN l
            """
            result = session.run(query, map_id=map_id, location_data=location_data)
            if not result.peek():
                return {}
            record = result.single()
            return dict(record["l"]) if record else {}

    # ------------------------------------------------------------------
    # Entity positioning
    # ------------------------------------------------------------------
    def set_entity_position(
        self,
        entity_type: str,
        entity_name: str,
        x: float,
        y: float,
        z: float = 0.0,
        map_id: Optional[str] = None,
        location_name: Optional[str] = None,
    ) -> bool:
        """Set the position of an entity (Character, NPC, Monster, etc.).

        The entity's coordinates (``x``, ``y``, ``z``) are updated, and
        the ``current_map_id`` property is set to the supplied map or to
        the map inferred from ``location_name``.  If ``location_name``
        is provided, a ``LOCATED_IN`` relationship is created.
        Returns ``True`` if the update succeeds.
        """
        with self.driver.session() as session:
            position_data = {"x": x, "y": y, "z": z}
            # Determine map from arguments
            if map_id:
                position_data["current_map_id"] = map_id
            elif location_name:
                map_query = (
                    "MATCH (l:Location {name: $location_name}) RETURN l.map_id as map_id LIMIT 1"
                )
                map_result = session.run(map_query, location_name=location_name)
                map_record = map_result.single()
                if map_record and map_record["map_id"]:
                    position_data["current_map_id"] = map_record["map_id"]
            # Update entity coordinates and map reference
            query = """
            MATCH (e {name: $entity_name})
            WHERE $entity_type IN labels(e)
            SET e += $position_data
            WITH e
            LIMIT 1
            """
            # If location specified, create a LOCATED_IN relationship
            if location_name:
                query += """
                MATCH (l:Location {name: $location_name})
                MERGE (e)-[r:LOCATED_IN]->(l)
                SET r.since = timestamp()
                """
            query += " RETURN e"
            result = session.run(
                query,
                entity_name=entity_name,
                entity_type=entity_type,
                position_data=position_data,
                location_name=location_name,
            )
            if result.peek():
                result.consume()
                return True
            return False

    # ------------------------------------------------------------------
    # Distance and range queries
    # ------------------------------------------------------------------
    def calculate_distance(
        self, entity1_type: str, entity1_name: str, entity2_type: str, entity2_name: str
    ) -> Optional[float]:
        """Calculate the 3D distance between two entities in feet."""
        with self.driver.session() as session:
            query = """
            MATCH (e1 {name: $entity1_name})
            WHERE $entity1_type IN labels(e1)
            WITH e1 LIMIT 1
            MATCH (e2 {name: $entity2_name})
            WHERE $entity2_type IN labels(e2)
            WITH e1, e2 LIMIT 1
            WHERE e1.x IS NOT NULL AND e1.y IS NOT NULL
              AND e2.x IS NOT NULL AND e2.y IS NOT NULL
              AND coalesce(e1.current_map_id, '') = coalesce(e2.current_map_id, '')
            RETURN e1.x as x1, e1.y as y1, coalesce(e1.z, 0) as z1,
                   e2.x as x2, e2.y as y2, coalesce(e2.z, 0) as z2
            """
            result = session.run(
                query,
                entity1_name=entity1_name,
                entity1_type=entity1_type,
                entity2_name=entity2_name,
                entity2_type=entity2_type,
            )
            if not result.peek():
                return None
            record = result.single()
            if record:
                dx = record["x2"] - record["x1"]
                dy = record["y2"] - record["y1"]
                dz = record["z2"] - record["z1"]
                distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                return round(distance, 2)
            return None

    def get_entities_in_range(
        self,
        entity_type: str,
        entity_name: str,
        range_feet: float,
        target_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get all entities within a certain range of a given entity."""
        with self.driver.session() as session:
            if target_types:
                type_conditions = " OR ".join([f"'{t}' IN labels(e2)" for t in target_types])
                target_filter = f"AND ({type_conditions})"
            else:
                target_filter = ""
            query = f"""
            MATCH (e1 {{name: $entity_name}})
            WHERE $entity_type IN labels(e1)
              AND e1.x IS NOT NULL AND e1.y IS NOT NULL
            MATCH (e2)
            WHERE e2.name <> $entity_name
              AND e2.x IS NOT NULL AND e2.y IS NOT NULL
              AND coalesce(e1.current_map_id, '') = coalesce(e2.current_map_id, '')
              {target_filter}
            WITH e1, e2,
                 sqrt((e2.x - e1.x)^2 + (e2.y - e1.y)^2 + (coalesce(e2.z, 0) - coalesce(e1.z, 0))^2) as distance
            WHERE distance <= $range_feet
            RETURN e2, labels(e2) as entity_type, distance
            ORDER BY distance
            """
            result = session.run(
                query, entity_name=entity_name, entity_type=entity_type, range_feet=range_feet
            )
            entities: List[Dict[str, Any]] = []
            for record in result:
                entity_dict = dict(record["e2"])
                entity_dict["entity_type"] = record["entity_type"][0] if record["entity_type"] else "Unknown"
                entity_dict["distance"] = round(record["distance"], 2)
                entities.append(entity_dict)
            return entities

    # ------------------------------------------------------------------
    # Scene queries and graph operations
    # ------------------------------------------------------------------
    def get_location_scene(self, location_name: str) -> Dict[str, Any]:
        """Return detailed scene information for a location."""
        with self.driver.session() as session:
            query = """
            MATCH (l:Location {name: $location_name})
            WITH l LIMIT 1
            OPTIONAL MATCH (l)<-[:LOCATED_IN]-(entity)
            OPTIONAL MATCH (l)-[:CONNECTED_TO]->(connected:Location)
            OPTIONAL MATCH (l)-[:ON_MAP]->(m:Map)
            RETURN l,
                   collect(DISTINCT {
                       name: entity.name,
                       type: labels(entity)[0],
                       x: entity.x,
                       y: entity.y,
                       z: entity.z
                   }) as entities,
                   collect(DISTINCT connected.name) as connected_locations,
                   m.name as map_name
            """
            result = session.run(query, location_name=location_name)
            if not result.peek():
                return {}
            record = result.single()
            if record:
                location = dict(record["l"])
                return {
                    "location": location,
                    "entities": [e for e in record["entities"] if e["name"]],
                    "connected_locations": record["connected_locations"],
                    "map_name": record["map_name"],
                }
            return {}

    def create_location_connection(
        self,
        location1: str,
        location2: str,
        connection_type: str = "CONNECTED_TO",
        distance: Optional[float] = None,
        description: Optional[str] = None,
    ) -> bool:
        """Create a connection between two locations."""
        with self.driver.session() as session:
            properties: Dict[str, Any] = {}
            if distance is not None:
                properties["distance"] = distance
            if description:
                properties["description"] = description
            query = f"""
            MATCH (l1:Location {{name: $location1}})
            WITH l1 LIMIT 1
            MATCH (l2:Location {{name: $location2}})
            WITH l1, l2 LIMIT 1
            MERGE (l1)-[r:{connection_type} $properties]->(l2)
            RETURN r
            """
            result = session.run(
                query, location1=location1, location2=location2, properties=properties
            )
            if result.peek():
                result.consume()
                return True
            return False

    def get_path_between_locations(
        self, start_location: str, end_location: str
    ) -> Optional[List[str]]:
        """Find the shortest path between two locations."""
        with self.driver.session() as session:
            query = """
            MATCH path = shortestPath(
                (start:Location {name: $start})-[:CONNECTED_TO*]-(end:Location {name: $end})
            )
            RETURN [node in nodes(path) | node.name] as path
            LIMIT 1
            """
            result = session.run(query, start=start_location, end=end_location)
            if not result.peek():
                return None
            record = result.single()
            return record["path"] if record else None

    # ------------------------------------------------------------------
    # Generic operations
    # ------------------------------------------------------------------
    def create_node(self, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Create a node with the given label and properties."""
        with self.driver.session() as session:
            query = f"""
            CREATE (n:{label} $properties)
            RETURN n
            LIMIT 1
            """
            result = session.run(query, properties=properties)
            if not result.peek():
                return {}
            record = result.single()
            return dict(record["n"]) if record else {}

    def create_relationship(
        self,
        from_node_label: str,
        from_node_property: str,
        from_node_value: Any,
        to_node_label: str,
        to_node_property: str,
        to_node_value: Any,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create a relationship between two nodes."""
        with self.driver.session() as session:
            props = properties or {}
            query = f"""
            MATCH (a:{from_node_label} {{{from_node_property}: $from_value}})
            WITH a LIMIT 1
            MATCH (b:{to_node_label} {{{to_node_property}: $to_value}})
            WITH a, b LIMIT 1
            CREATE (a)-[r:{relationship_type} $properties]->(b)
            RETURN r
            """
            result = session.run(
                query,
                from_value=from_node_value,
                to_value=to_node_value,
                properties=props,
            )
            if result.peek():
                result.consume()
                return True
            return False

    def query_graph(self, cypher_query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute an arbitrary Cypher query and return the results."""
        with self.driver.session() as session:
            result = session.run(cypher_query, parameters or {})
            return [dict(record) for record in result]

    def entity_exists(self, label: str, name: str) -> bool:
        """Check whether an entity with the given label and name exists."""
        with self.driver.session() as session:
            query = f"""
            MATCH (n:{label} {{name: $name}})
            RETURN COUNT(n) > 0 as exists
            LIMIT 1
            """
            result = session.run(query, name=name)
            record = result.single()
            return record["exists"] if record else False

    def get_database_summary(self) -> Dict[str, Any]:
        """Return basic statistics about the database contents."""
        with self.driver.session() as session:
            total_nodes_result = session.run("MATCH (n) RETURN COUNT(n) as count")
            total_nodes = total_nodes_result.single()["count"]
            total_rels_result = session.run("MATCH ()-[r]->() RETURN COUNT(r) as count")
            total_rels = total_rels_result.single()["count"]
            node_counts_result = session.run(
                """
                MATCH (n)
                UNWIND labels(n) as label
                RETURN label, COUNT(*) as count
                ORDER BY count DESC
                """
            )
            node_counts = {record["label"]: record["count"] for record in node_counts_result}
            rel_counts_result = session.run(
                """
                MATCH ()-[r]->()
                RETURN type(r) as type, COUNT(*) as count
                ORDER BY count DESC
                """
            )
            rel_counts = {record["type"]: record["count"] for record in rel_counts_result}
            return {
                "total_nodes": total_nodes,
                "total_relationships": total_rels,
                "node_counts": node_counts,
                "relationship_counts": rel_counts,
            }

    def get_all_entities_by_type(self, entity_type: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Return all entities of a given type up to a specified limit."""
        with self.driver.session() as session:
            query = f"""
            MATCH (n:{entity_type})
            RETURN n
            ORDER BY n.name
            LIMIT $limit
            """
            result = session.run(query, limit=limit)
            return [dict(record["n"]) for record in result]