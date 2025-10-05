"""
Neo4j Spatial manager for DnD Agent
=================================

All methods related to Neo4j graph database operations with spatial awareness.

NEO4J WARNING REFERENCE GUIDE
=============================

If you see warnings like these, they are SAFE TO IGNORE:

1. "warn: unknown label" 
   - Examples: NPC, Monster, Character, Location, Map
   - Cause: Label doesn't exist in database yet
   - Fix: Create entities with that label (automatic)
   
2. "warn: unknown property key"
   - Examples: current_map_id, x, y, z, name
   - Cause: Property hasn't been set on any node yet
   - Fix: Create nodes with those properties (automatic)
   
3. "warn: unknown relationship type"
   - Examples: LOCATED_IN, CONNECTED_TO, ON_MAP, KNOWS
   - Cause: Relationship type doesn't exist yet
   - Fix: Create relationships of that type (automatic)

WHY DO THESE WARNINGS APPEAR?
Neo4j uses a schema-less approach. When you query for something that doesn't 
exist yet, Neo4j can't optimize the query and warns you. Once data exists, 
the warnings stop. They DO NOT indicate errors or problems.

HOW TO VERIFY EVERYTHING IS WORKING:
Run the script twice. On the second run, you'll see FAR FEWER or NO warnings
because the schema now exists with actual data.

TO SUPPRESS WARNINGS (Optional):
Add to your Neo4j connection:
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
However, this will suppress ALL warnings, not just Neo4j's schema warnings.
We recommend keeping warnings visible during development.
"""
from typing import Any, Dict, List, Optional, Tuple
import math
import warnings

from neo4j import GraphDatabase

__all__ = ["Neo4jSpatialManager", "suppress_neo4j_warnings"]

def suppress_neo4j_warnings():
    """
    Optional: Suppress Neo4j schema warnings about unknown labels/properties/relationships.
    
    These warnings are informational and safe to ignore. They appear when:
    - Database is empty or newly created
    - Querying for labels/properties/relationships that don't exist yet
    
    Call this function at the start of your script if you want a cleaner output.
    
    Note: This suppresses ALL UserWarnings from the neo4j module, not just schema warnings.
    """
    warnings.filterwarnings('ignore', category=UserWarning, module='neo4j')
    print("ℹ️  Neo4j schema warnings suppressed for cleaner output.\n")

class Neo4jSpatialManager:
    """Manages Neo4j graph database operations with spatial awareness"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._create_spatial_indexes()
    
    def _create_spatial_indexes(self):
        """Create indexes for spatial queries"""
        with self.driver.session() as session:
            # Create indexes for common queries
            session.run("CREATE INDEX location_name IF NOT EXISTS FOR (n:Location) ON (n.name)")
            session.run("CREATE INDEX character_name IF NOT EXISTS FOR (n:Character) ON (n.name)")
            session.run("CREATE INDEX map_id IF NOT EXISTS FOR (n:Map) ON (n.map_id)")
    
    def close(self):
        self.driver.close()
    
    def create_map(self, map_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a map node"""
        with self.driver.session() as session:
            query = """
            CREATE (m:Map $properties)
            RETURN m
            LIMIT 1
            """
            result = session.run(query, properties=map_data)
            
            if not result.peek():
                return {}
            
            record = result.single()
            return dict(record["m"]) if record else {}
    
    def create_location_with_position(
        self, 
        location_data: Dict[str, Any],
        map_id: str,
        position: Optional[Tuple[float, float, float]] = None
    ) -> Dict[str, Any]:
        """Create a location with optional position on a map"""
        with self.driver.session() as session:
            # Add position and map reference
            location_data['map_id'] = map_id
            if position:
                location_data['x'] = position[0]
                location_data['y'] = position[1]
                location_data['z'] = position[2] if len(position) > 2 else 0.0
            
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
    
    def set_entity_position(
        self,
        entity_type: str,
        entity_name: str,
        x: float,
        y: float,
        z: float = 0.0,
        map_id: str = None,
        location_name: str = None
    ) -> bool:
        """Set the position of an entity (Character, NPC, Monster, etc.)"""
        with self.driver.session() as session:
            # Build position data - always include map_id if provided
            position_data = {
                'x': x, 
                'y': y, 
                'z': z
            }
            
            # If map_id is provided, set it; if location_name is provided, get map_id from location
            if map_id:
                position_data['current_map_id'] = map_id
            elif location_name:
                # Get map_id from location
                map_query = "MATCH (l:Location {name: $location_name}) RETURN l.map_id as map_id LIMIT 1"
                map_result = session.run(map_query, location_name=location_name)
                map_record = map_result.single()
                if map_record and map_record['map_id']:
                    position_data['current_map_id'] = map_record['map_id']
            
            # Update entity with position using label matching
            # Add LIMIT 1 to ensure single result
            query = """
            MATCH (e {name: $entity_name})
            WHERE $entity_type IN labels(e)
            SET e += $position_data
            WITH e
            LIMIT 1
            """
            
            # If location specified, create relationship
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
                location_name=location_name
            )
            
            # Use peek() to check if result exists without consuming it
            if result.peek():
                result.consume()
                return True
            return False
    
    def calculate_distance(
        self,
        entity1_type: str,
        entity1_name: str,
        entity2_type: str,
        entity2_name: str
    ) -> Optional[float]:
        """Calculate distance between two entities in feet"""
        with self.driver.session() as session:
            # Use dynamic label matching to handle any entity type
            # Add LIMIT 1 to ensure single result per entity
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
                entity2_type=entity2_type
            )
            
            # Use peek to check if result exists
            if not result.peek():
                return None
                
            record = result.single()
            
            if record:
                # Calculate 3D Euclidean distance
                dx = record['x2'] - record['x1']
                dy = record['y2'] - record['y1']
                dz = record['z2'] - record['z1']
                distance = math.sqrt(dx**2 + dy**2 + dz**2)
                return round(distance, 2)
            return None
    
    def get_entities_in_range(
        self,
        entity_type: str,
        entity_name: str,
        range_feet: float,
        target_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all entities within range of a given entity"""
        with self.driver.session() as session:
            # Build query with dynamic label matching
            if target_types:
                # Match nodes with any of the target types
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
                 sqrt((e2.x - e1.x)^2 + (e2.y - e1.y)^2 + 
                      (coalesce(e2.z, 0) - coalesce(e1.z, 0))^2) as distance
            WHERE distance <= $range_feet
            RETURN e2, labels(e2) as entity_type, distance
            ORDER BY distance
            """
            result = session.run(
                query,
                entity_name=entity_name,
                entity_type=entity_type,
                range_feet=range_feet
            )
            
            entities = []
            for record in result:
                entity_dict = dict(record["e2"])
                entity_dict['entity_type'] = record['entity_type'][0] if record['entity_type'] else 'Unknown'
                entity_dict['distance'] = round(record['distance'], 2)
                entities.append(entity_dict)
            
            return entities
    
    def get_location_scene(self, location_name: str) -> Dict[str, Any]:
        """Get detailed scene information for a location"""
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
            
            # Check if result exists
            if not result.peek():
                return {}
            
            record = result.single()
            
            if record:
                location = dict(record["l"])
                return {
                    "location": location,
                    "entities": [e for e in record["entities"] if e['name']],
                    "connected_locations": record["connected_locations"],
                    "map_name": record["map_name"]
                }
            return {}
    
    def create_location_connection(
        self,
        location1: str,
        location2: str,
        connection_type: str = "CONNECTED_TO",
        distance: float = None,
        description: str = None
    ) -> bool:
        """Create a connection between two locations"""
        with self.driver.session() as session:
            properties = {}
            if distance:
                properties['distance'] = distance
            if description:
                properties['description'] = description
            
            query = f"""
            MATCH (l1:Location {{name: $location1}})
            WITH l1 LIMIT 1
            MATCH (l2:Location {{name: $location2}})
            WITH l1, l2 LIMIT 1
            MERGE (l1)-[r:{connection_type} $properties]->(l2)
            RETURN r
            """
            result = session.run(
                query,
                location1=location1,
                location2=location2,
                properties=properties
            )
            
            if result.peek():
                result.consume()
                return True
            return False
    
    def get_path_between_locations(
        self,
        start_location: str,
        end_location: str
    ) -> Optional[List[str]]:
        """Find shortest path between two locations"""
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
    
    
    def create_node(self, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Create a node in the graph"""
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
        properties: Dict[str, Any] = None
    ) -> bool:
        """Create a relationship between two nodes"""
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
                properties=props
            )
            
            # Check if result exists
            if result.peek():
                result.consume()
                return True
            return False
    
    def query_graph(self, cypher_query: str, parameters: Dict[str, Any] = None) -> List[Dict]:
        """Execute a custom Cypher query"""
        with self.driver.session() as session:
            result = session.run(cypher_query, parameters or {})
            return [dict(record) for record in result]
    
    

    def entity_exists(self, label: str, name: str) -> bool:
        """Check if an entity with the given label and name already exists"""
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
        """Get a summary of the database contents"""
        with self.driver.session() as session:
            # Count total nodes
            total_nodes_result = session.run("MATCH (n) RETURN COUNT(n) as count")
            total_nodes = total_nodes_result.single()["count"]
            
            # Count total relationships  
            total_rels_result = session.run("MATCH ()-[r]->() RETURN COUNT(r) as count")
            total_rels = total_rels_result.single()["count"]
            
            # Count nodes by label
            node_counts_result = session.run("""
                MATCH (n)
                UNWIND labels(n) as label
                RETURN label, COUNT(*) as count
                ORDER BY count DESC
            """)
            node_counts = {record["label"]: record["count"] 
                        for record in node_counts_result}
            
            # Count relationships by type
            rel_counts_result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, COUNT(*) as count
                ORDER BY count DESC
            """)
            rel_counts = {record["type"]: record["count"] 
                        for record in rel_counts_result}
            
            return {
                "total_nodes": total_nodes,
                "total_relationships": total_rels,
                "node_counts": node_counts,
                "relationship_counts": rel_counts
            }

    def get_all_entities_by_type(self, entity_type: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get all entities of a specific type"""
        with self.driver.session() as session:
            query = f"""
            MATCH (n:{entity_type})
            RETURN n
            ORDER BY n.name
            LIMIT $limit
            """
            result = session.run(query, limit=limit)
            return [dict(record["n"]) for record in result]