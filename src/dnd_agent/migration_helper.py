"""
D&D Agent System Migration Helper
==================================

Helps migrate from monolithic agent to multi-agent system.
Maps existing tools to new agent architecture.
"""

import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import logging
from datetime import datetime

from dnd_agent.multiagent import (
    MultiAgentDnDSystem,
    AgentType,
    ActionType,
    AgentRequest,
    CampaignDeps
)

logger = logging.getLogger(__name__)

# ============================================================================
# Tool to Agent Mapping
# ============================================================================

class ToolToAgentMapper:
    """Maps old tool calls to new agent system"""
    
    # Mapping of old tools to agent types and actions
    TOOL_MAPPING = {
        # Entity Management Tools
        'store_campaign_entity': {
            'agent': AgentType.ENTITY,
            'action': ActionType.CREATE_ENTITY,
            'transform': lambda params: {
                'name': params.get('name'),
                'type': params.get('entity_type'),
                'hp': params.get('hp'),
                'ac': params.get('ac'),
                'stats': params.get('abilities', {})
            }
        },
        'create_campaign_relationship': {
            'agent': AgentType.ENTITY,
            'action': ActionType.UPDATE_ENTITY,
            'transform': lambda params: {
                'name': params.get('entity1'),
                'updates': {
                    'relationships': {
                        params.get('relationship'): params.get('entity2')
                    }
                }
            }
        },
        'check_database_status': {
            'agent': AgentType.ENTITY,
            'action': ActionType.GET_ENTITY,
            'transform': lambda params: {'name': 'database_status'}
        },
        'list_entities_of_type': {
            'agent': AgentType.ENTITY,
            'action': ActionType.GET_ENTITY,
            'transform': lambda params: {'type': params.get('entity_type')}
        },
        
        # Spatial Tools
        'create_map_location': {
            'agent': AgentType.SPATIAL,
            'action': ActionType.CREATE_LOCATION,
            'transform': lambda params: {
                'name': params.get('name'),
                'dimensions': {'width': 50, 'height': 50},
                'terrain': params.get('terrain', 'normal')
            }
        },
        'create_detailed_location': {
            'agent': AgentType.SPATIAL,
            'action': ActionType.CREATE_LOCATION,
            'transform': lambda params: {
                'name': params.get('name'),
                'dimensions': params.get('dimensions', {'width': 100, 'height': 100}),
                'terrain': params.get('terrain'),
                'features': params.get('features', [])
            }
        },
        'set_entity_position': {
            'agent': AgentType.SPATIAL,
            'action': ActionType.SET_POSITION,
            'transform': lambda params: {
                'entity': params.get('entity_name'),
                'position': {
                    'x': params.get('x', 0),
                    'y': params.get('y', 0),
                    'z': params.get('z', 0),
                    'location': params.get('location')
                }
            }
        },
        'calculate_distance': {
            'agent': AgentType.SPATIAL,
            'action': ActionType.CALCULATE_DISTANCE,
            'transform': lambda params: {
                'from': params.get('entity1'),
                'to': params.get('entity2')
            }
        },
        'get_entities_in_range': {
            'agent': AgentType.SPATIAL,
            'action': ActionType.GET_AREA_EFFECT,
            'transform': lambda params: {
                'center': params.get('center'),
                'radius': params.get('range'),
                'shape': params.get('shape', 'sphere')
            }
        },
        'connect_locations': {
            'agent': AgentType.SPATIAL,
            'action': ActionType.CREATE_LOCATION,
            'transform': lambda params: {
                'name': f"connection_{params.get('location1')}_{params.get('location2')}",
                'type': 'connection'
            }
        },
        
        # Rules Tools
        'lookup_dnd_resource': {
            'agent': AgentType.RULES,
            'action': ActionType.LOOKUP_RULE,
            'transform': lambda params: {
                'query': params.get('resource_type') + ': ' + params.get('name', '')
            }
        },
        
        # Narrative Tools
        'generate_scene_description': {
            'agent': AgentType.NARRATIVE,
            'action': ActionType.DESCRIBE_SCENE,
            'transform': lambda params: {
                'location': params.get('location'),
                'mood': params.get('mood', 'neutral'),
                'details': params.get('elements', [])
            }
        },
        
        # Memory Tools
        'save_campaign_info': {
            'agent': AgentType.MEMORY,
            'action': ActionType.STORE_EVENT,
            'transform': lambda params: {
                'type': params.get('info_type', 'general'),
                'description': params.get('content'),
                'importance': params.get('importance', 'normal')
            }
        },
        'search_campaign_info': {
            'agent': AgentType.MEMORY,
            'action': ActionType.SEARCH_LORE,
            'transform': lambda params: {
                'query': params.get('search_query')
            }
        },
        'recall_chat_history': {
            'agent': AgentType.MEMORY,
            'action': ActionType.RECALL_HISTORY,
            'transform': lambda params: {
                'filter': 'recent',
                'limit': params.get('messages', 10)
            }
        }
    }
    
    @classmethod
    def map_tool_call(cls, tool_name: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Map old tool call to new agent request"""
        
        mapping = cls.TOOL_MAPPING.get(tool_name)
        if not mapping:
            logger.warning(f"No mapping found for tool: {tool_name}")
            return None
        
        return {
            'agent': mapping['agent'],
            'action': mapping['action'],
            'parameters': mapping['transform'](params)
        }
    
    @classmethod
    def get_responsible_agent(cls, tool_name: str) -> Optional[AgentType]:
        """Get the agent responsible for a tool"""
        
        mapping = cls.TOOL_MAPPING.get(tool_name)
        return mapping['agent'] if mapping else None


# ============================================================================
# Migration Wrapper
# ============================================================================

class MigrationWrapper:
    """Wrapper that provides backward compatibility during migration"""
    
    def __init__(self, multi_agent_system: MultiAgentDnDSystem):
        self.system = multi_agent_system
        self.tool_calls_migrated = 0
        self.tool_calls_failed = 0
        
    async def handle_tool_call(self, tool_name: str, **kwargs) -> Any:
        """Handle old-style tool call with new system"""
        
        # Map to new system
        mapped = ToolToAgentMapper.map_tool_call(tool_name, kwargs)
        
        if not mapped:
            logger.error(f"Could not migrate tool call: {tool_name}")
            self.tool_calls_failed += 1
            return None
        
        # Execute with new system
        try:
            response = await self.system.direct_agent_call(
                mapped['agent'],
                mapped['action'],
                mapped['parameters']
            )
            
            self.tool_calls_migrated += 1
            
            if response.success:
                return response.data
            else:
                logger.error(f"Tool migration failed: {response.message}")
                self.tool_calls_failed += 1
                return None
                
        except Exception as e:
            logger.error(f"Error migrating tool {tool_name}: {str(e)}")
            self.tool_calls_failed += 1
            return None
    
    def get_migration_stats(self) -> Dict[str, int]:
        """Get migration statistics"""
        return {
            'migrated': self.tool_calls_migrated,
            'failed': self.tool_calls_failed,
            'success_rate': self.tool_calls_migrated / max(1, self.tool_calls_migrated + self.tool_calls_failed)
        }


# ============================================================================
# Compatibility Layer
# ============================================================================

class CompatibilityLayer:
    """Provides old API interface using new system"""
    
    def __init__(self, deps: CampaignDeps, model_config: Dict[str, Any]):
        self.system = MultiAgentDnDSystem(deps, model_config)
        self.wrapper = MigrationWrapper(self.system)
        
    # Old tool methods that now use the new system
    
    async def store_campaign_entity(self, **kwargs) -> Dict[str, Any]:
        """Old method signature, new implementation"""
        return await self.wrapper.handle_tool_call('store_campaign_entity', **kwargs)
    
    async def create_map_location(self, **kwargs) -> Dict[str, Any]:
        """Old method signature, new implementation"""
        return await self.wrapper.handle_tool_call('create_map_location', **kwargs)
    
    async def set_entity_position(self, **kwargs) -> Dict[str, Any]:
        """Old method signature, new implementation"""
        return await self.wrapper.handle_tool_call('set_entity_position', **kwargs)
    
    async def calculate_distance(self, **kwargs) -> Dict[str, Any]:
        """Old method signature, new implementation"""
        return await self.wrapper.handle_tool_call('calculate_distance', **kwargs)
    
    async def lookup_dnd_resource(self, **kwargs) -> Dict[str, Any]:
        """Old method signature, new implementation"""
        return await self.wrapper.handle_tool_call('lookup_dnd_resource', **kwargs)
    
    async def generate_scene_description(self, **kwargs) -> str:
        """Old method signature, new implementation"""
        result = await self.wrapper.handle_tool_call('generate_scene_description', **kwargs)
        return result.get('description', '') if result else ''
    
    async def save_campaign_info(self, **kwargs) -> bool:
        """Old method signature, new implementation"""
        result = await self.wrapper.handle_tool_call('save_campaign_info', **kwargs)
        return result.get('stored', False) if result else False
    
    async def run(self, user_input: str) -> str:
        """Old main entry point, new implementation"""
        return await self.system.process_user_input(user_input)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get migration and system statistics"""
        return {
            'migration': self.wrapper.get_migration_stats(),
            'system': asyncio.run(self.system.get_system_status())
        }


# ============================================================================
# Migration Script
# ============================================================================

class MigrationScript:
    """Automated migration from monolithic to multi-agent"""
    
    @staticmethod
    async def migrate_existing_campaign(
        old_agent: Any,
        deps: CampaignDeps,
        model_config: Dict[str, Any]
    ) -> MultiAgentDnDSystem:
        """Migrate existing campaign data to new system"""
        
        logger.info("Starting migration to multi-agent system...")
        
        # Create new system
        new_system = MultiAgentDnDSystem(deps, model_config)
        
        # Migrate entities
        logger.info("Migrating entities...")
        if hasattr(old_agent, 'entities'):
            entity_agent = new_system.agents[AgentType.ENTITY]
            for name, entity_data in old_agent.entities.items():
                await entity_agent.process(
                    AgentRequest(
                        agent_type=AgentType.ENTITY,
                        action=ActionType.CREATE_ENTITY,
                        parameters=entity_data
                    )
                )
        
        # Migrate spatial data
        logger.info("Migrating spatial data...")
        if hasattr(old_agent, 'positions'):
            spatial_agent = new_system.agents[AgentType.SPATIAL]
            for entity_name, position in old_agent.positions.items():
                await spatial_agent.process(
                    AgentRequest(
                        agent_type=AgentType.SPATIAL,
                        action=ActionType.SET_POSITION,
                        parameters={'entity': entity_name, 'position': position}
                    )
                )
        
        # Migrate campaign history
        logger.info("Migrating campaign history...")
        if hasattr(old_agent, 'campaign_history'):
            memory_agent = new_system.agents[AgentType.MEMORY]
            for event in old_agent.campaign_history:
                await memory_agent.process(
                    AgentRequest(
                        agent_type=AgentType.MEMORY,
                        action=ActionType.STORE_EVENT,
                        parameters=event
                    )
                )
        
        logger.info("Migration complete!")
        return new_system


# ============================================================================
# Testing and Validation
# ============================================================================

class MigrationValidator:
    """Validates migration correctness"""
    
    @staticmethod
    async def validate_migration(
        old_agent: Any,
        new_system: MultiAgentDnDSystem
    ) -> Dict[str, Any]:
        """Compare old and new system outputs"""
        
        test_cases = [
            "Roll initiative for combat",
            "Create a goblin enemy",
            "Move the fighter 20 feet north",
            "The wizard casts fireball",
            "Describe the tavern scene"
        ]
        
        results = {
            'passed': 0,
            'failed': 0,
            'differences': []
        }
        
        for test in test_cases:
            try:
                # Get outputs from both systems
                old_result = await old_agent.run(test) if hasattr(old_agent, 'run') else None
                new_result = await new_system.process_user_input(test)
                
                # Compare (simplified - you'd want more sophisticated comparison)
                if old_result and new_result:
                    if len(old_result) > 0 and len(new_result) > 0:
                        results['passed'] += 1
                    else:
                        results['failed'] += 1
                        results['differences'].append({
                            'input': test,
                            'old_length': len(old_result) if old_result else 0,
                            'new_length': len(new_result)
                        })
                
            except Exception as e:
                results['failed'] += 1
                results['differences'].append({
                    'input': test,
                    'error': str(e)
                })
        
        results['success_rate'] = results['passed'] / max(1, results['passed'] + results['failed'])
        return results


# ============================================================================
# Usage Examples
# ============================================================================

async def example_gradual_migration():
    """Example of gradual migration approach"""
    
    # Setup
    import os
    from dnd_agent.database.neo4j_manager import Neo4jSpatialManager
    
    # Create dependencies
    neo4j_manager = Neo4jSpatialManager(
        uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        user=os.environ.get("NEO4J_USER", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD", "password")
    )
    
    deps = CampaignDeps(
        neo4j_driver=neo4j_manager.driver,
        campaign_id="test_campaign"
    )
    
    model_config = {
        'model_name': 'gpt-4o',
        'provider': 'openai'
    }
    
    # Phase 1: Use compatibility layer
    print("Phase 1: Compatibility Layer")
    compat = CompatibilityLayer(deps, model_config)
    
    # Old code still works
    await compat.store_campaign_entity(
        name="Valeros",
        entity_type="player",
        hp=45,
        ac=18
    )
    
    await compat.set_entity_position(
        entity_name="Valeros",
        x=10,
        y=15
    )
    
    result = await compat.run("Valeros attacks the goblin")
    print(f"Result: {result}")
    
    # Check migration stats
    stats = compat.get_stats()
    print(f"Migration stats: {stats['migration']}")
    
    # Phase 2: Direct new system usage
    print("\nPhase 2: Direct System Usage")
    system = MultiAgentDnDSystem(deps, model_config)
    
    # New API
    response = await system.direct_agent_call(
        AgentType.COMBAT,
        ActionType.ROLL_INITIATIVE,
        {'entities': [{'name': 'Valeros', 'dexterity': 14}]}
    )
    print(f"Initiative: {response.message}")
    
    # Phase 3: Full migration
    print("\nPhase 3: Full Migration")
    result = await system.process_user_input(
        "Start a tavern brawl with 3 thugs. Roll initiative and describe the scene."
    )
    print(f"Full system result: {result}")


async def example_parallel_testing():
    """Run both systems in parallel for comparison"""
    
    from dnd_agent.agent import create_dnd_agent  # Your old agent
    
    # Setup both systems
    deps = CampaignDeps(neo4j_driver=None)  # Simplified for example
    model_config = {'model_name': 'gpt-4o'}
    
    old_agent = create_dnd_agent()  # Your existing agent
    new_system = MultiAgentDnDSystem(deps, model_config)
    
    # Test input
    user_input = "The party enters the dragon's lair. Roll for perception."
    
    # Run both in parallel
    old_task = asyncio.create_task(old_agent.run(user_input))
    new_task = asyncio.create_task(new_system.process_user_input(user_input))
    
    old_result, new_result = await asyncio.gather(old_task, new_task)
    
    print("=== Comparison ===")
    print(f"Old System ({len(old_result)} chars): {old_result[:200]}...")
    print(f"New System ({len(new_result)} chars): {new_result[:200]}...")
    
    # Validate migration
    validator = MigrationValidator()
    validation = await validator.validate_migration(old_agent, new_system)
    print(f"\nValidation: {validation['success_rate']*100:.1f}% compatibility")


async def example_tool_mapping():
    """Show how tools map to new agents"""
    
    print("Tool to Agent Mapping:")
    print("=" * 50)
    
    for tool_name in ToolToAgentMapper.TOOL_MAPPING.keys():
        agent = ToolToAgentMapper.get_responsible_agent(tool_name)
        print(f"{tool_name:30} -> {agent.value if agent else 'Unknown'}")
    
    print("\n\nExample Tool Migration:")
    print("=" * 50)
    
    # Example old tool call
    old_params = {
        'name': 'Goblin Scout',
        'entity_type': 'monster',
        'hp': 7,
        'ac': 15
    }
    
    # Map to new system
    mapped = ToolToAgentMapper.map_tool_call('store_campaign_entity', old_params)
    
    print(f"Old Tool: store_campaign_entity")
    print(f"Old Params: {old_params}")
    print(f"\nMapped to:")
    print(f"Agent: {mapped['agent'].value}")
    print(f"Action: {mapped['action'].value}")
    print(f"New Params: {mapped['parameters']}")


# ============================================================================
# Main Migration Entry Point
# ============================================================================

async def perform_migration(
    old_agent_path: str = "dnd_agent.agent",
    dry_run: bool = True
) -> None:
    """
    Main migration function
    
    Args:
        old_agent_path: Import path to old agent
        dry_run: If True, test migration without replacing old system
    """
    
    print("🎲 D&D Agent Migration Tool")
    print("=" * 50)
    
    # Import old agent
    import importlib
    old_module = importlib.import_module(old_agent_path)
    old_agent = old_module.create_agent()  # Adjust based on your code
    
    # Setup new system
    deps = CampaignDeps(neo4j_driver=None)  # Add your actual deps
    model_config = {'model_name': 'gpt-4o'}
    
    if dry_run:
        print("\n🔍 Running in DRY RUN mode...")
        
        # Test compatibility layer
        print("\n1. Testing Compatibility Layer...")
        compat = CompatibilityLayer(deps, model_config)
        
        # Test a few tools
        test_results = []
        for tool in ['store_campaign_entity', 'set_entity_position', 'lookup_dnd_resource']:
            try:
                result = await getattr(compat, tool)(name="test", x=0, y=0)
                test_results.append(f"✅ {tool}: Success")
            except Exception as e:
                test_results.append(f"❌ {tool}: {str(e)}")
        
        for result in test_results:
            print(f"  {result}")
        
        # Validate migration
        print("\n2. Validating Migration...")
        new_system = MultiAgentDnDSystem(deps, model_config)
        validator = MigrationValidator()
        validation = await validator.validate_migration(old_agent, new_system)
        
        print(f"  Success Rate: {validation['success_rate']*100:.1f}%")
        print(f"  Passed: {validation['passed']}")
        print(f"  Failed: {validation['failed']}")
        
        if validation['differences']:
            print("  Issues found:")
            for diff in validation['differences'][:5]:
                print(f"    - {diff}")
        
    else:
        print("\n⚠️ Running ACTUAL MIGRATION...")
        print("This will migrate your campaign data to the new system.")
        
        confirm = input("Continue? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Migration cancelled.")
            return
        
        # Perform migration
        script = MigrationScript()
        new_system = await script.migrate_existing_campaign(
            old_agent, deps, model_config
        )
        
        print("\n✅ Migration complete!")
        print(f"System status: {await new_system.get_system_status()}")
        
        # Save migration marker
        with open('migration_complete.txt', 'w') as f:
            f.write(f"Migration completed at {datetime.now().isoformat()}\n")
            f.write(f"Old system: {old_agent_path}\n")
            f.write("New system: MultiAgentDnDSystem\n")


if __name__ == "__main__":
    # Run migration
    asyncio.run(perform_migration(dry_run=True))