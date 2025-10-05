"""
Multi-Agent D&D System - Usage Examples and Migration Guide
==========================================================

This shows how to use the multi-agent system and migrate from the monolithic agent.
"""

import asyncio
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Import the multi-agent system components
from dnd_agent.multiagent import (
    MultiAgentDnDSystem,
    AgentType
)

# ============================================================================
# Configuration and Setup
# ============================================================================

@dataclass
class CampaignDeps:
    """Dependencies for the D&D campaign agents"""
    neo4j_driver: Any
    postgres_conn: Optional[Any] = None
    dnd_api_base: str = "https://www.dnd5eapi.co"
    current_context: Optional[Dict[str, Any]] = None
    campaign_id: Optional[str] = None
    session_id: Optional[str] = None


def create_multiagent_system(
    model_provider: str = "openai",
    model_name: str = "gpt-4o",
    ollama_base_url: str = "http://localhost:11434"
) -> MultiAgentDnDSystem:
    """
    Create and configure the multi-agent D&D system
    """
    # Set up environment variables
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USER", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "password")
    
    # Create Neo4j connection
    from dnd_agent.database.neo4j_manager import Neo4jSpatialManager
    neo4j_manager = Neo4jSpatialManager(
        uri=os.environ["NEO4J_URI"],
        user=os.environ["NEO4J_USER"],
        password=os.environ["NEO4J_PASSWORD"]
    )
    
    # Create PostgreSQL connection (optional)
    postgres_conn = None
    try:
        import psycopg2
        postgres_conn = psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=os.environ.get("POSTGRES_PORT", "5432"),
            database=os.environ.get("POSTGRES_DB", "dnd_campaign"),
            user=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", "password")
        )
    except:
        print("PostgreSQL not available - memory features disabled")
    
    # Create dependencies
    deps = CampaignDeps(
        neo4j_driver=neo4j_manager.driver,
        postgres_conn=postgres_conn,
        dnd_api_base="https://www.dnd5eapi.co",
        current_context={}
    )
    
    # Model configuration
    model_config = {
        "provider": model_provider,
        "model_name": model_name,
        "ollama_base_url": ollama_base_url
    }
    
    # Create the multi-agent system
    return MultiAgentDnDSystem(deps, model_config)


# ============================================================================
# Example Usage Scenarios
# ============================================================================

async def example_campaign_start():
    """Example: Starting a new campaign with the multi-agent system"""
    
    system = create_multiagent_system()
    
    # The orchestrator will automatically delegate to appropriate agents
    result = await system.process_user_input("""
        Start a new campaign called 'The Lost Mines'.
        Create a tavern called 'The Sleeping Giant' in the town of Phandalin.
        Place a human fighter named Sildar at position (10, 10) in the tavern.
        Describe the scene.
    """)
    
    print(result)
    
    # Behind the scenes, this will:
    # 1. Entity Agent: Creates the campaign and entities
    # 2. Spatial Agent: Creates the map and positions
    # 3. Narrative Agent: Generates the scene description
    # 4. Memory Agent: Stores the campaign start


async def example_player_turn():
    """Example: Processing a player's combat turn"""
    
    system = create_multiagent_system()
    
    # Define the player's declared actions
    player_actions = [
        {
            "type": "move",
            "distance": 20,
            "target_position": {"x": 30, "y": 15}
        },
        {
            "type": "attack",
            "target": {"name": "Goblin Scout", "type": "Monster"},
            "attack_type": "melee",
            "weapon": "longsword"
        }
    ]
    
    # Process the turn using multiple agents
    result = await system.process_player_turn("Valeros", player_actions)
    
    print(result)
    
    # This coordinates:
    # 1. Rules Agent: Validates the actions
    # 2. Spatial Agent: Checks movement and range
    # 3. Combat Agent: Processes attack rolls and damage
    # 4. Narrative Agent: Describes the action dramatically


async def example_rules_check():
    """Example: Quick rules lookup"""
    
    system = create_multiagent_system()
    
    # Direct rules query
    result = await system.check_rule(
        "What are the rules for opportunity attacks in D&D 5e?"
    )
    
    print(result)


async def example_complex_scenario():
    """Example: Complex multi-agent scenario - Fireball spell"""
    
    system = create_multiagent_system()
    
    # Complex action requiring multiple agents
    result = await system.process_user_input("""
        The wizard casts Fireball at a point 30 feet away.
        There are 3 goblins and 1 ally within the 20-foot radius.
        Roll damage and apply it, considering the ally has fire resistance.
        Describe the explosion dramatically.
    """)
    
    print(result)
    
    # This coordinates:
    # 1. Rules Agent: Looks up Fireball spell rules
    # 2. Spatial Agent: Determines who's in the area
    # 3. Combat Agent: Rolls damage and saves
    # 4. Entity Agent: Updates HP for all affected
    # 5. Narrative Agent: Describes the fiery explosion


async def example_npc_interaction():
    """Example: NPC dialogue and interaction"""
    
    system = create_multiagent_system()
    
    result = await system.process_user_input("""
        The party approaches the mysterious hooded figure in the tavern.
        The bard tries to persuade them to share information about the 
        missing caravan. Roll persuasion with advantage (the bard used 
        bardic inspiration).
    """)
    
    print(result)
    
    # This uses:
    # 1. Entity Agent: Retrieves NPC information
    # 2. Rules Agent: Handles persuasion check
    # 3. Narrative Agent: Generates appropriate dialogue
    # 4. Memory Agent: Stores the interaction


# ============================================================================
# Advanced Patterns
# ============================================================================

class CampaignSession:
    """Manages a complete D&D session with the multi-agent system"""
    
    def __init__(self):
        self.system = create_multiagent_system()
        self.session_log = []
        self.initiative_order = []
        self.current_turn_index = 0
        
    async def start_session(self, recap: bool = True):
        """Start a new session, optionally with recap"""
        if recap:
            # Memory Agent recalls previous session
            result = await self.system.process_user_input(
                "Provide a recap of our last session"
            )
            self.log_event("recap", result)
            return result
        
    async def setup_encounter(self, enemies: list[Dict[str, Any]]):
        """Setup a combat encounter"""
        # Create enemies using Entity Agent
        for enemy in enemies:
            await self.system.process_user_input(
                f"Create a {enemy['type']} called {enemy['name']} "
                f"at position {enemy['position']}"
            )
        
        # Roll initiative using Combat Agent
        result = await self.system.process_user_input(
            "Roll initiative for all combatants"
        )
        
        self.log_event("encounter_start", result)
        return result
    
    async def next_turn(self):
        """Process the next turn in combat"""
        if not self.initiative_order:
            return "No combat in progress"
        
        current_entity = self.initiative_order[self.current_turn_index]
        
        # Get entity status
        result = await self.system.process_user_input(
            f"It's {current_entity}'s turn. What is their current status?"
        )
        
        self.current_turn_index = (self.current_turn_index + 1) % len(self.initiative_order)
        
        return result
    
    async def exploration_mode(self, location: str):
        """Handle exploration of a new area"""
        # Spatial Agent creates/loads the location
        # Narrative Agent describes it
        # Entity Agent populates with NPCs/items
        
        result = await self.system.process_user_input(
            f"The party enters {location}. Set up the area and describe what they see."
        )
        
        self.log_event("exploration", {"location": location, "description": result})
        return result
    
    async def end_session(self):
        """End the session and save progress"""
        # Memory Agent saves session summary
        summary = await self.system.process_user_input(
            "Create a summary of this session including key events, "
            "NPCs met, items found, and plot developments"
        )
        
        self.log_event("session_end", summary)
        return summary
    
    def log_event(self, event_type: str, data: Any):
        """Log session events"""
        self.session_log.append({
            "type": event_type,
            "data": data,
            "timestamp": asyncio.get_event_loop().time()
        })


# ============================================================================
# Migration Helper
# ============================================================================

class MonolithicToMultiAgentMigrator:
    """Helper to migrate from monolithic to multi-agent system"""
    
    @staticmethod
    async def migrate_command(old_command: str) -> str:
        """
        Convert a command for the old monolithic agent 
        to work with the multi-agent system
        """
        system = create_multiagent_system()
        
        # The orchestrator handles the routing automatically
        # so most commands work as-is
        return await system.process_user_input(old_command)
    
    @staticmethod
    def map_tool_to_agent(tool_name: str) -> AgentType:
        """Map old tools to their responsible agents"""
        
        tool_mapping = {
            # Entity tools
            "store_campaign_entity": AgentType.ENTITY,
            "create_campaign_relationship": AgentType.ENTITY,
            "check_database_status": AgentType.ENTITY,
            "list_entities_of_type": AgentType.ENTITY,
            
            # Spatial tools
            "create_map_location": AgentType.SPATIAL,
            "create_detailed_location": AgentType.SPATIAL,
            "set_entity_position": AgentType.SPATIAL,
            "calculate_distance": AgentType.SPATIAL,
            "get_entities_in_range": AgentType.SPATIAL,
            "connect_locations": AgentType.SPATIAL,
            
            # Rules tools
            "lookup_dnd_resource": AgentType.RULES,
            
            # Narrative tools
            "generate_scene_description": AgentType.NARRATIVE,
            
            # Memory tools
            "save_campaign_info": AgentType.MEMORY,
            "search_campaign_info": AgentType.MEMORY,
            "recall_chat_history": AgentType.MEMORY,
        }
        
        return tool_mapping.get(tool_name, AgentType.ORCHESTRATOR)


# ============================================================================
# Main Example
# ============================================================================

async def main():
    """Complete example showing the multi-agent system in action"""
    
    print("🎲 D&D Multi-Agent System Demo")
    print("=" * 50)
    
    # Create a campaign session
    session = CampaignSession()
    
    # Start with recap
    print("\n📖 Starting Session with Recap...")
    recap = await session.start_session(recap=True)
    print(recap)
    
    # Exploration
    print("\n🗺️ Exploring New Area...")
    exploration = await session.exploration_mode("The Goblin Cave")
    print(exploration)
    
    # Setup encounter
    print("\n⚔️ Setting Up Combat...")
    enemies = [
        {"type": "Goblin", "name": "Goblin Scout 1", "position": {"x": 20, "y": 20}},
        {"type": "Goblin", "name": "Goblin Scout 2", "position": {"x": 25, "y": 20}},
        {"type": "Bugbear", "name": "Bugbear Chief", "position": {"x": 30, "y": 25}}
    ]
    encounter = await session.setup_encounter(enemies)
    print(encounter)
    
    # Process a player turn
    print("\n🎯 Processing Player Turn...")
    turn_result = await session.system.process_player_turn(
        "Valeros",
        [
            {"type": "move", "distance": 25},
            {"type": "attack", "target": {"name": "Goblin Scout 1", "type": "Monster"}}
        ]
    )
    print(turn_result)
    
    # Rules check
    print("\n📚 Checking Rules...")
    rules = await session.system.check_rule("Can you sneak attack with a spell?")
    print(rules)
    
    # End session
    print("\n📝 Ending Session...")
    summary = await session.end_session()
    print(summary)
    
    print("\n✨ Session Complete!")


# ============================================================================
# Performance Comparison
# ============================================================================

async def performance_comparison():
    """Compare performance between monolithic and multi-agent approaches"""
    
    import time
    
    # Test scenario: Complex turn with multiple rule checks
    test_scenario = """
    The wizard wants to:
    1. Move 15 feet toward the dragon
    2. Cast Shield as a reaction if attacked
    3. Cast Fireball at the dragon and nearby enemies
    4. Use Misty Step as bonus action to teleport away
    Check all rules, calculate areas of effect, and describe dramatically.
    """
    
    # Multi-agent system (parallel processing potential)
    system = create_multiagent_system()
    
    start = time.time()
    result = await system.process_user_input(test_scenario)
    multi_agent_time = time.time() - start
    
    print(f"Multi-Agent System: {multi_agent_time:.2f} seconds")
    print("Agents involved: Rules, Spatial, Combat, Narrative")
    print("\nAdvantages:")
    print("- Parallel processing potential")
    print("- Focused, optimized prompts per agent")
    print("- Better error isolation")
    print("- Easier to debug specific issues")


if __name__ == "__main__":
    asyncio.run(main())