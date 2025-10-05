"""
Multi-Agent D&D Campaign System Architecture
============================================

This refactored architecture splits the monolithic agent into specialized agents,
each with focused responsibilities and tools.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import json

from rich.console import Console
from rich.markdown import Markdown
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
console = Console()
print = lambda c: console.print(Markdown(c))
# ============================================================================
# Agent Types and Orchestration
# ============================================================================

class AgentType(Enum):
    """Types of specialized agents in the system"""
    ORCHESTRATOR = "orchestrator"
    RULES = "rules"
    SPATIAL = "spatial"
    ENTITY = "entity"
    COMBAT = "combat"
    NARRATIVE = "narrative"
    MEMORY = "memory"


@dataclass
class AgentRequest:
    """Request to be processed by an agent"""
    agent_type: AgentType
    action: str
    parameters: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


@dataclass
class AgentResponse:
    """Response from an agent"""
    agent_type: AgentType
    success: bool
    data: Any
    message: str
    metadata: Optional[Dict[str, Any]] = None


# ============================================================================
# Base Agent Class
# ============================================================================

class BaseAgent:
    """Base class for all specialized agents"""
    
    def __init__(self, name: str, deps: Any, model_config: Dict[str, Any]):
        self.name = name
        self.deps = deps
        self.model_config = model_config
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the pydantic-ai agent - to be overridden by subclasses"""
        raise NotImplementedError
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process a request - to be overridden by subclasses"""
        result = await self.agent.run(
            request.action,
            deps=self.deps
            )
        return AgentResponse(
            agent_type=request.agent_type,
            success=True,
            data=result,
            message=result.output
        )


# ============================================================================
# 1. DM Orchestrator Agent
# ============================================================================

class DMOrchestratorAgent(BaseAgent):
    """
    Main orchestrator that delegates to specialized agents.
    Responsible for:
    - Understanding user intent
    - Breaking down complex requests
    - Routing to appropriate agents
    - Combining responses
    - Managing game flow
    """
    
    def __init__(self, deps, model_config, sub_agents: Dict[AgentType, BaseAgent]):
        super().__init__("DM Orchestrator", deps, model_config)
        self.sub_agents = sub_agents
    
    def _create_agent(self) -> Agent:
        model = OpenAIChatModel(self.model_config['model_name'])
        
        agent = Agent(
            model,
            deps_type=type(self.deps),
            system_prompt="""You are the master orchestrator for a D&D 5e campaign.
            
Your responsibilities:
1. Understand player intentions and DM requests
2. Break down complex actions into sub-tasks
3. Delegate to specialized agents:
   - Rules Agent: For D&D 5e rules questions and validation
   - Spatial Agent: For positioning, movement, and distance calculations
   - Entity Agent: For creating/modifying characters, NPCs, items
   - Combat Agent: For turn order, attacks, damage, conditions
   - Narrative Agent: For scene descriptions and storytelling
   - Memory Agent: For campaign history and lore
4. Combine responses into coherent game flow
5. Maintain overall campaign consistency

When processing requests:
- Identify which agents are needed
- Determine the order of operations
- Pass context between agents as needed
- Synthesize results into a unified response
""",
            retries=2
        )
        
        # Register orchestration tools
        agent.tool(self.delegate_to_agent, retries=10)
        agent.tool(self.combine_responses, retries=10)
        agent.tool(self.get_game_state, retries=10)
        
        return agent
    
    async def delegate_to_agent(
        self,
        ctx: RunContext,
        agent_type: AgentType,
        action: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Delegate a task to a specialized agent"""
        agent = self.sub_agents.get(agent_type)
        if not agent:
            return {"error": f"Unknown agent type: {agent_type}"}
        
        request = AgentRequest(
            agent_type=agent_type,
            action=action,
            parameters=parameters,
            context=ctx.deps.current_context
        )
        
        response = await agent.process(request)
        return {
            "success": response.success,
            "data": response.data,
            "message": response.message
        }
    
    async def combine_responses(
        self,
        ctx: RunContext,
        responses: List[Dict[str, Any]]
    ) -> str:
        """Combine multiple agent responses into a coherent narrative"""
        combined = []
        for resp in responses:
            if resp.get("success"):
                combined.append(resp.get("message", ""))
        
        return "\n\n".join(combined)
    
    async def get_game_state(self, ctx: RunContext) -> Dict[str, Any]:
        """Get current game state from all agents"""
        state = {}
        for agent_type, agent in self.sub_agents.items():
            if hasattr(agent, 'get_state'):
                state[agent_type.value] = await agent.get_state()
        return state


# ============================================================================
# 2. Rules Agent
# ============================================================================

class RulesAgent(BaseAgent):
    """
    Handles D&D 5e rules lookups and validation.
    Responsible for:
    - Looking up spells, items, monsters
    - Validating actions against rules
    - Calculating modifiers and DCs
    - Checking prerequisites
    """
    
    def _create_agent(self) -> Agent:
        model = OpenAIChatModel(self.model_config['model_name'])
        
        agent = Agent(
            model,
            deps_type=type(self.deps),
            system_prompt="""You are a D&D 5e rules expert.
            
Your responsibilities:
1. Look up official D&D 5e rules, spells, and monsters
2. Validate player actions against rules
3. Calculate modifiers, DCs, and bonuses
4. Check prerequisites and requirements
5. Resolve rules disputes

Always cite specific rules when making determinations.
Be precise about game mechanics and calculations.
""",
            retries=2
        )
        
        # Import the lookup tool from original code
        from dnd_agent.tools import lookup_dnd_resource
        agent.tool(lookup_dnd_resource)
        agent.tool(self.validate_action)
        agent.tool(self.calculate_modifier)
        
        return agent
    
    async def validate_action(
        self,
        ctx: RunContext,
        action: str,
        entity: Dict[str, Any],
        target: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate if an action is legal according to D&D 5e rules"""
        # Implementation would check rules database
        # For now, returning example structure
        return {
            "valid": True,
            "reason": "Action is valid",
            "requirements": [],
            "modifiers": []
        }
    
    async def calculate_modifier(
        self,
        ctx: RunContext,
        ability_score: int,
        proficiency: bool = False,
        proficiency_bonus: int = 2
    ) -> int:
        """Calculate ability modifier with optional proficiency"""
        modifier = (ability_score - 10) // 2
        if proficiency:
            modifier += proficiency_bonus
        return modifier


# ============================================================================
# 3. Spatial Agent
# ============================================================================

class SpatialAgent(BaseAgent):
    """
    Manages positioning, movement, and spatial relationships.
    Responsible for:
    - Tracking entity positions
    - Calculating distances and ranges
    - Managing maps and locations
    - Determining line of sight
    - Handling area effects
    """
    
    def _create_agent(self) -> Agent:
        model = OpenAIChatModel(self.model_config['model_name'])
        
        agent = Agent(
            model,
            deps_type=type(self.deps),
            system_prompt="""You are a spatial awareness specialist for D&D combat and exploration.
            
Your responsibilities:
1. Track positions of all entities on maps
2. Calculate distances for movement and attacks
3. Determine line of sight and cover
4. Manage area of effect calculations
5. Handle movement through difficult terrain
6. Track elevation and 3D positioning

Always consider:
- D&D uses 5-foot squares
- Diagonal movement alternates 5-10-5-10 feet
- Cover provides +2 (half) or +5 (three-quarters) AC
- Difficult terrain costs double movement
""",
            retries=2
        )
        
        # Import spatial tools from original code
        from dnd_agent.tools import (
            create_map_location,
            create_detailed_location,
            set_entity_position,
            calculate_distance,
            get_entities_in_range,
            connect_locations
        )
        
        agent.tool(create_map_location)
        agent.tool(create_detailed_location)
        agent.tool(set_entity_position)
        agent.tool(calculate_distance)
        agent.tool(get_entities_in_range)
        agent.tool(connect_locations)
        agent.tool(self.check_line_of_sight)
        agent.tool(self.calculate_cover)
        
        return agent
    
    async def check_line_of_sight(
        self,
        ctx: RunContext,
        source: Dict[str, Any],
        target: Dict[str, Any]
    ) -> bool:
        """Check if there's line of sight between two entities"""
        # Would check for obstacles between positions
        # Simplified for example
        return True
    
    async def calculate_cover(
        self,
        ctx: RunContext,
        attacker: Dict[str, Any],
        target: Dict[str, Any]
    ) -> str:
        """Calculate cover bonus for target"""
        # Would analyze terrain and obstacles
        # Simplified for example
        return "none"  # Could be "none", "half", "three-quarters", "full"


# ============================================================================
# 4. Entity Agent  
# ============================================================================

class EntityAgent(BaseAgent):
    """
    Manages all game entities (characters, NPCs, monsters, items).
    Responsible for:
    - Creating and modifying entities
    - Managing inventories
    - Tracking relationships
    - Handling entity states
    """
    
    def _create_agent(self) -> Agent:
        model = OpenAIChatModel(self.model_config['model_name'])
        
        agent = Agent(
            model,
            deps_type=type(self.deps),
            system_prompt="""You are the entity manager for the D&D campaign.
            
Your responsibilities:
1. Create characters, NPCs, monsters, and items
2. Track entity properties and statistics
3. Manage inventories and equipment
4. Handle entity relationships
5. Update entity states and conditions

Ensure all entities follow D&D 5e rules for:
- Ability scores (3-20 normally, up to 30 for monsters)
- Hit points and hit dice
- Armor class calculations
- Equipment restrictions
""",
            retries=2
        )
        
        # Import entity tools from original code
        from dnd_agent.tools import (
            store_campaign_entity,
            create_campaign_relationship,
            check_database_status,
            list_entities_of_type
        )
        
        agent.tool(store_campaign_entity)
        agent.tool(create_campaign_relationship)
        agent.tool(check_database_status)
        agent.tool(list_entities_of_type)
        agent.tool(self.update_entity_stats)
        agent.tool(self.manage_inventory)
        
        return agent
    
    async def update_entity_stats(
        self,
        ctx: RunContext,
        entity_name: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update entity statistics (HP, conditions, etc.)"""
        # Would update entity in Neo4j
        return {
            "entity": entity_name,
            "updates": updates,
            "success": True
        }
    
    async def manage_inventory(
        self,
        ctx: RunContext,
        entity_name: str,
        action: str,  # "add", "remove", "equip", "unequip"
        item: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Manage entity inventory"""
        # Would handle inventory operations
        return {
            "entity": entity_name,
            "action": action,
            "item": item,
            "success": True
        }


# ============================================================================
# 5. Combat Agent
# ============================================================================

class CombatAgent(BaseAgent):
    """
    Handles combat mechanics and turn resolution.
    Responsible for:
    - Managing initiative and turn order
    - Processing attacks and damage
    - Applying conditions and effects
    - Tracking combat state
    """
    
    def _create_agent(self) -> Agent:
        model = OpenAIChatModel(self.model_config['model_name'])
        
        agent = Agent(
            model,
            deps_type=type(self.deps),
            system_prompt="""You are the combat manager for D&D 5e encounters.
            
Your responsibilities:
1. Manage initiative order
2. Process attack rolls and damage
3. Apply conditions and status effects
4. Track concentration and reactions
5. Handle death saves and unconsciousness
6. Manage action economy (action, bonus action, reaction, movement)

Combat rules to enforce:
- Attack rolls: d20 + modifiers vs AC
- Advantage/disadvantage mechanics
- Critical hits on natural 20
- Automatic misses on natural 1
- Damage resistance/vulnerability/immunity
- Concentration saves (DC 10 or half damage, whichever is higher)
""",
            retries=2
        )
        
        agent.tool(self.roll_initiative)
        agent.tool(self.process_attack)
        agent.tool(self.apply_damage)
        agent.tool(self.apply_condition)
        agent.tool(self.process_turn)
        
        return agent
    
    async def roll_initiative(
        self,
        ctx: RunContext,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Roll initiative for combat participants"""
        import random
        
        initiative_order = []
        for entity in entities:
            dex_mod = (entity.get('dexterity', 10) - 10) // 2
            roll = random.randint(1, 20) + dex_mod
            initiative_order.append({
                "entity": entity['name'],
                "initiative": roll,
                "dex_mod": dex_mod
            })
        
        return sorted(initiative_order, key=lambda x: x['initiative'], reverse=True)
    
    async def process_attack(
        self,
        ctx: RunContext,
        attacker: Dict[str, Any],
        target: Dict[str, Any],
        attack_type: str,
        advantage: bool = False,
        disadvantage: bool = False
    ) -> Dict[str, Any]:
        """Process an attack roll"""
        import random
        
        # Roll d20
        if advantage and not disadvantage:
            roll = max(random.randint(1, 20), random.randint(1, 20))
        elif disadvantage and not advantage:
            roll = min(random.randint(1, 20), random.randint(1, 20))
        else:
            roll = random.randint(1, 20)
        
        # Calculate modifiers (simplified)
        attack_bonus = attacker.get('proficiency_bonus', 2) + \
                      ((attacker.get('strength', 10) - 10) // 2)
        
        total = roll + attack_bonus
        target_ac = target.get('armor_class', 10)
        
        return {
            "roll": roll,
            "modifier": attack_bonus,
            "total": total,
            "target_ac": target_ac,
            "hit": total >= target_ac,
            "critical": roll == 20,
            "fumble": roll == 1
        }
    
    async def apply_damage(
        self,
        ctx: RunContext,
        target: Dict[str, Any],
        damage: int,
        damage_type: str,
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """Apply damage to a target"""
        current_hp = target.get('hp', 0)
        new_hp = max(0, current_hp - damage)
        
        return {
            "target": target['name'],
            "damage": damage,
            "damage_type": damage_type,
            "previous_hp": current_hp,
            "current_hp": new_hp,
            "unconscious": new_hp == 0,
            "source": source
        }
    
    async def apply_condition(
        self,
        ctx: RunContext,
        target: Dict[str, Any],
        condition: str,
        duration: Optional[int] = None,
        save_dc: Optional[int] = None
    ) -> Dict[str, Any]:
        """Apply a condition to a target"""
        return {
            "target": target['name'],
            "condition": condition,
            "duration": duration,
            "save_dc": save_dc,
            "applied": True
        }
    
    async def process_turn(
        self,
        ctx: RunContext,
        entity: Dict[str, Any],
        actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process a complete turn for an entity"""
        results = []
        
        for action in actions:
            if action['type'] == 'move':
                # Delegate to spatial agent
                results.append({"type": "move", "distance": action['distance']})
            elif action['type'] == 'attack':
                # Process attack
                attack_result = await self.process_attack(
                    ctx,
                    entity,
                    action['target'],
                    action.get('attack_type', 'melee')
                )
                results.append({"type": "attack", "result": attack_result})
            elif action['type'] == 'spell':
                # Delegate to rules agent for spell validation
                results.append({"type": "spell", "spell": action['spell']})
        
        return {
            "entity": entity['name'],
            "actions": results,
            "turn_complete": True
        }


# ============================================================================
# 6. Narrative Agent
# ============================================================================

class NarrativeAgent(BaseAgent):
    """
    Handles storytelling and scene descriptions.
    Responsible for:
    - Generating scene descriptions
    - Creating NPC dialogue
    - Describing combat actions narratively
    - Setting atmosphere and mood
    """
    
    def _create_agent(self) -> Agent:
        model = OpenAIChatModel(self.model_config['model_name'])
        
        agent = Agent(
            model,
            deps_type=type(self.deps),
            system_prompt="""You are the narrative voice of the D&D campaign.
            
Your responsibilities:
1. Create vivid, immersive scene descriptions
2. Generate engaging NPC dialogue
3. Describe combat actions cinematically
4. Set atmosphere and mood
5. Provide sensory details (sights, sounds, smells)
6. Make dice rolls feel dramatic

Writing style:
- Use rich, evocative language
- Include sensory details
- Vary sentence structure
- Match tone to scene (tense, mysterious, lighthearted, etc.)
- Make players feel like heroes in an epic story
""",
            retries=2
        )
        
        # Import narrative tool from original code
        from dnd_agent.tools import generate_scene_description
        
        agent.tool(generate_scene_description)
        agent.tool(self.describe_action)
        agent.tool(self.generate_npc_dialogue)
        
        return agent
    
    async def describe_action(
        self,
        ctx: RunContext,
        action: Dict[str, Any],
        style: str = "dramatic"
    ) -> str:
        """Generate narrative description of an action"""
        # Would create rich description based on action details
        if action.get('type') == 'attack':
            if action.get('hit'):
                return f"The blow strikes true, dealing {action.get('damage', 0)} damage!"
            else:
                return "The attack whistles past, missing by inches!"
        return "The action unfolds..."
    
    async def generate_npc_dialogue(
        self,
        ctx: RunContext,
        npc_name: str,
        personality: str,
        context: str,
        emotion: str = "neutral"
    ) -> str:
        """Generate appropriate NPC dialogue"""
        # Would generate dialogue based on NPC personality and context
        return f'"{npc_name} speaks with a {emotion} tone about {context}"'


# ============================================================================
# 7. Memory Agent
# ============================================================================

class MemoryAgent(BaseAgent):
    """
    Manages campaign history and persistent information.
    Responsible for:
    - Storing campaign events
    - Tracking plot developments
    - Managing NPC backgrounds
    - Recalling past interactions
    - Maintaining world lore
    """
    
    def _create_agent(self) -> Agent:
        model = OpenAIChatModel(self.model_config['model_name'])
        
        agent = Agent(
            model,
            deps_type=type(self.deps),
            system_prompt="""You are the memory keeper for the D&D campaign.
            
Your responsibilities:
1. Store important campaign events
2. Track plot developments and quest progress
3. Remember NPC personalities and histories
4. Recall past player decisions
5. Maintain world lore and history
6. Provide context from previous sessions

Focus on:
- What would be important to remember
- Character development and relationships
- Unresolved plot threads
- Player preferences and play style
""",
            retries=2
        )
        
        # Import memory tools from original code
        from dnd_agent.tools import (
            save_campaign_info,
            search_campaign_info,
            recall_chat_history
        )
        
        agent.tool(save_campaign_info)
        agent.tool(search_campaign_info)
        agent.tool(recall_chat_history)
        agent.tool(self.summarize_session)
        
        return agent
    
    async def summarize_session(
        self,
        ctx: RunContext,
        events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create a session summary for future reference"""
        key_events = []
        npcs_met = []
        items_found = []
        plot_developments = []
        
        for event in events:
            if event.get('type') == 'combat':
                key_events.append(f"Combat with {event.get('enemies', 'unknown')}")
            elif event.get('type') == 'npc_interaction':
                npcs_met.append(event.get('npc_name'))
            elif event.get('type') == 'item_acquired':
                items_found.append(event.get('item_name'))
            elif event.get('type') == 'plot':
                plot_developments.append(event.get('description'))
        
        return {
            "key_events": key_events,
            "npcs_met": npcs_met,
            "items_found": items_found,
            "plot_developments": plot_developments,
            "timestamp": "current_session"
        }


# ============================================================================
# System Orchestration
# ============================================================================

class MultiAgentDnDSystem:
    """Main system that creates and orchestrates all agents"""
    
    def __init__(self, deps, model_config: Dict[str, Any]):
        self.deps = deps
        self.model_config = model_config
        self.agents = self._initialize_agents()
        
    def _initialize_agents(self) -> Dict[AgentType, BaseAgent]:
        """Initialize all specialized agents"""
        agents = {
            AgentType.RULES: RulesAgent("Rules", self.deps, self.model_config),
            AgentType.SPATIAL: SpatialAgent("Spatial", self.deps, self.model_config),
            AgentType.ENTITY: EntityAgent("Entity", self.deps, self.model_config),
            AgentType.COMBAT: CombatAgent("Combat", self.deps, self.model_config),
            AgentType.NARRATIVE: NarrativeAgent("Narrative", self.deps, self.model_config),
            AgentType.MEMORY: MemoryAgent("Memory", self.deps, self.model_config)
        }
        
        # Create orchestrator with references to all agents
        agents[AgentType.ORCHESTRATOR] = DMOrchestratorAgent(
            self.deps, 
            self.model_config, 
            agents
        )
        
        return agents
    
    async def process_user_input(self, user_input: str) -> str:
        """Main entry point for processing user input"""
        orchestrator = self.agents[AgentType.ORCHESTRATOR]
        
        # The orchestrator will analyze the input and delegate appropriately
        result = await orchestrator.agent.run(
            user_input,
            deps=self.deps
        )
        
        return result.output
    
    async def process_player_turn(
        self,
        player_name: str,
        declared_actions: List[Dict[str, Any]]
    ) -> str:
        """Process a player's turn using multiple agents"""
        
        responses = []
        
        # 1. Validate actions with Rules Agent
        rules_agent = self.agents[AgentType.RULES]
        for action in declared_actions:
            validation = await rules_agent.validate_action(
                None,  # ctx
                action['type'],
                {"name": player_name},
                action.get('target')
            )
            if not validation['valid']:
                responses.append(f"Invalid action: {validation['reason']}")
                return "\n".join(responses)
        
        # 2. Check positioning with Spatial Agent
        spatial_agent = self.agents[AgentType.SPATIAL]
        if any(a['type'] in ['attack', 'spell'] for a in declared_actions):
            # Check ranges, line of sight, etc.
            pass
        
        # 3. Process combat actions with Combat Agent
        combat_agent = self.agents[AgentType.COMBAT]
        turn_result = await combat_agent.process_turn(
            None,  # ctx
            {"name": player_name},
            declared_actions
        )
        
        # 4. Generate narrative with Narrative Agent
        narrative_agent = self.agents[AgentType.NARRATIVE]
        for action in turn_result['actions']:
            description = await narrative_agent.describe_action(
                None,  # ctx
                action,
                "dramatic"
            )
            responses.append(description)
        
        # 5. Store important events with Memory Agent
        memory_agent = self.agents[AgentType.MEMORY]
        # Store turn results for future reference
        
        return "\n\n".join(responses)
    
    async def check_rule(self, rule_question: str) -> str:
        """Quick rule check using the Rules Agent"""
        rules_agent = self.agents[AgentType.RULES]
        result = await rules_agent.agent.run(
            rule_question,
            deps=self.deps
        )
        return result.output
    
    async def describe_scene(self, location: str) -> str:
        """Generate a scene description using Narrative and Spatial agents"""
        # Get spatial information
        spatial_agent = self.agents[AgentType.SPATIAL]
        # Get entities in location
        
        # Generate narrative description
        narrative_agent = self.agents[AgentType.NARRATIVE]
        from dnd_agent.tools import generate_scene_description
        description = await generate_scene_description(
            None,  # ctx will be provided by agent
            location
        )
        
        return description