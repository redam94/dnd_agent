"""
Multi-Agent D&D Campaign System with Logfire Observability
==========================================================

Complete implementation with Logfire tracing, metrics, and structured logging.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
from datetime import datetime
import random
import time
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel

import logfire
from logfire import instrument, log_slow_async_callbacks, span, metric_counter, metric_histogram

# Initialize Logfire
logfire.configure(
    service_name="dnd-multiagent",
    environment="production",
    send_to_logfire=True
)

# Instrument pydantic-ai for automatic tracing
logfire.instrument_pydantic()

# ============================================================================
# Core Data Models with Logfire
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


class ActionType(Enum):
    """Types of actions agents can process"""
    # Rules actions
    LOOKUP_RULE = "lookup_rule"
    VALIDATE_ACTION = "validate_action"
    CALCULATE_DC = "calculate_dc"
    
    # Spatial actions
    CREATE_LOCATION = "create_location"
    SET_POSITION = "set_position"
    CALCULATE_DISTANCE = "calculate_distance"
    GET_AREA_EFFECT = "get_area_effect"
    
    # Entity actions
    CREATE_ENTITY = "create_entity"
    UPDATE_ENTITY = "update_entity"
    GET_ENTITY = "get_entity"
    MANAGE_INVENTORY = "manage_inventory"
    
    # Combat actions
    ROLL_INITIATIVE = "roll_initiative"
    PROCESS_ATTACK = "process_attack"
    APPLY_DAMAGE = "apply_damage"
    APPLY_CONDITION = "apply_condition"
    PROCESS_TURN = "process_turn"
    
    # Narrative actions
    DESCRIBE_SCENE = "describe_scene"
    DESCRIBE_ACTION = "describe_action"
    GENERATE_DIALOGUE = "generate_dialogue"
    
    # Memory actions
    STORE_EVENT = "store_event"
    RECALL_HISTORY = "recall_history"
    SEARCH_LORE = "search_lore"
    SUMMARIZE_SESSION = "summarize_session"
    
    # Orchestrator actions
    PROCESS_USER_INPUT = "process_user_input"
    COORDINATE_AGENTS = "coordinate_agents"


@dataclass
class AgentRequest:
    """Request to be processed by an agent"""
    agent_type: AgentType
    action: ActionType
    parameters: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None


@dataclass
class AgentResponse:
    """Response from an agent"""
    agent_type: AgentType
    action: ActionType
    success: bool
    data: Any
    message: str
    metadata: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    processing_time_ms: Optional[float] = None
    trace_id: Optional[str] = None


@dataclass
class CampaignDeps:
    """Dependencies for the campaign"""
    neo4j_driver: Any
    postgres_conn: Optional[Any] = None
    dnd_api_base: str = "https://www.dnd5eapi.co"
    current_context: Dict[str, Any] = field(default_factory=dict)
    campaign_id: Optional[str] = None
    session_id: Optional[str] = None


# ============================================================================
# Metrics Collectors
# ============================================================================

class AgentMetrics:
    """Centralized metrics for all agents"""
    
    # Counters
    request_counter = metric_counter(
        'agent.requests.total',
        description='Total number of agent requests'
    )
    
    error_counter = metric_counter(
        'agent.errors.total',
        description='Total number of agent errors'
    )
    
    cache_hit_counter = metric_counter(
        'agent.cache.hits',
        description='Number of cache hits'
    )
    
    cache_miss_counter = metric_counter(
        'agent.cache.misses',
        description='Number of cache misses'
    )
    
    # Histograms
    request_duration = metric_histogram(
        'agent.request.duration_ms',
        description='Request processing duration in milliseconds',
        unit='milliseconds'
    )
    
    llm_token_usage = metric_histogram(
        'agent.llm.tokens',
        description='LLM token usage per request',
        unit='tokens'
    )
    
    @classmethod
    def record_request(cls, agent_type: str, action: str, success: bool, duration_ms: float):
        """Record metrics for a request"""
        labels = {'agent': agent_type, 'action': action, 'success': str(success)}
        cls.request_counter.add(1, labels)
        cls.request_duration.record(duration_ms, labels)
        
        if not success:
            cls.error_counter.add(1, {'agent': agent_type, 'action': action})
    
    @classmethod
    def record_cache(cls, agent_type: str, hit: bool):
        """Record cache hit/miss"""
        if hit:
            cls.cache_hit_counter.add(1, {'agent': agent_type})
        else:
            cls.cache_miss_counter.add(1, {'agent': agent_type})


# ============================================================================
# Base Agent with Logfire Integration
# ============================================================================

class BaseAgent:
    """Base class for all specialized agents with Logfire integration"""
    
    def __init__(self, name: str, deps: CampaignDeps, model_config: Dict[str, Any]):
        self.name = name
        self.deps = deps
        self.model_config = model_config
        self.agent = self._create_agent()
        self._cache: Dict[str, Any] = {}
        
        # Log agent initialization
        logfire.info(
            f"Agent {name} initialized",
            agent_type=self.agent_type.value if hasattr(self, 'agent_type') else 'unknown',
            model=model_config.get('model_name', 'unknown')
        )
    
    def _create_agent(self) -> Agent:
        """Create the pydantic-ai agent - to be overridden by subclasses"""
        raise NotImplementedError
    
    @asynccontextmanager
    async def _trace_span(self, operation: str, **kwargs):
        """Create a trace span for an operation"""
        with logfire.span(
            f"{self.name}.{operation}",
            agent=self.name,
            **kwargs
        ):
            yield
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process a request with full tracing"""
        
        start_time = time.time()
        trace_id = request.trace_id or None
        
        with logfire.span(
            f"agent.{self.name}.process",
            agent_type=self.agent_type.value if hasattr(self, 'agent_type') else 'unknown',
            action=request.action.value,
            correlation_id=request.correlation_id
        ) as span:
            
            try:
                # Log request
                logfire.debug(
                    f"Processing request",
                    agent=self.name,
                    action=request.action.value,
                    parameters=request.parameters
                )
                
                # Check cache
                cache_key = self._get_cache_key(request)
                if cache_key and cache_key in self._cache:
                    logfire.debug(f"Cache hit", agent=self.name, cache_key=cache_key)
                    AgentMetrics.record_cache(self.agent_type.value, hit=True)
                    
                    cached_response = self._cache[cache_key]
                    cached_response.trace_id = trace_id
                    
                    # Record metrics
                    processing_time = (time.time() - start_time) * 1000
                    AgentMetrics.record_request(
                        self.agent_type.value,
                        request.action.value,
                        cached_response.success,
                        processing_time
                    )
                    
                    span.set_attribute('cache_hit', True)
                    return cached_response
                else:
                    if cache_key:
                        AgentMetrics.record_cache(self.agent_type.value, hit=False)
                
                # Process request
                with logfire.span(f"{self.name}.process_internal"):
                    response = await self._process_internal(request)
                
                # Add trace ID and timing
                response.trace_id = trace_id
                response.processing_time_ms = (time.time() - start_time) * 1000
                
                # Cache if applicable
                if cache_key and response.success:
                    self._cache[cache_key] = response
                    logfire.debug(f"Cached response", agent=self.name, cache_key=cache_key)
                
                # Log response
                logfire.info(
                    f"Request processed",
                    agent=self.name,
                    action=request.action.value,
                    success=response.success,
                    duration_ms=response.processing_time_ms
                )
                
                # Record metrics
                AgentMetrics.record_request(
                    self.agent_type.value,
                    request.action.value,
                    response.success,
                    response.processing_time_ms
                )
                
                span.set_attribute('success', response.success)
                span.set_attribute('duration_ms', response.processing_time_ms)
                
                return response
                
            except Exception as e:
                # Log error
                logfire.error(
                    f"Error processing request",
                    agent=self.name,
                    action=request.action.value,
                    error=str(e),
                    exc_info=True
                )
                
                processing_time = (time.time() - start_time) * 1000
                
                # Record error metrics
                AgentMetrics.record_request(
                    self.agent_type.value if hasattr(self, 'agent_type') else 'unknown',
                    request.action.value,
                    False,
                    processing_time
                )
                
                span.set_attribute('error', True)
                span.set_attribute('error_message', str(e))
                
                return AgentResponse(
                    agent_type=self.agent_type if hasattr(self, 'agent_type') else AgentType.ORCHESTRATOR,
                    action=request.action,
                    success=False,
                    data=None,
                    message=f"Error: {str(e)}",
                    errors=[str(e)],
                    processing_time_ms=processing_time,
                    trace_id=trace_id
                )
    
    async def _process_internal(self, request: AgentRequest) -> AgentResponse:
        """Internal processing - to be overridden by subclasses"""
        raise NotImplementedError
    
    def _get_cache_key(self, request: AgentRequest) -> Optional[str]:
        """Generate cache key for request if cacheable"""
        return None
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current agent state"""
        return {
            "name": self.name,
            "cache_size": len(self._cache),
            "status": "active"
        }


# ============================================================================
# 1. DM Orchestrator Agent with Tracing
# ============================================================================

class DMOrchestratorAgent(BaseAgent):
    """Main orchestrator with comprehensive tracing"""
    
    def __init__(self, deps: CampaignDeps, model_config: Dict[str, Any], sub_agents: Dict[AgentType, BaseAgent]):
        self.agent_type = AgentType.ORCHESTRATOR
        self.sub_agents = sub_agents
        super().__init__("DM Orchestrator", deps, model_config)
        
        logfire.info(
            "Orchestrator initialized with sub-agents",
            sub_agents=list(sub_agents.keys())
        )
    
    def _create_agent(self) -> Agent:
        model = OpenAIChatModel(self.model_config.get('model_name', 'gpt-4o'))
        
        # Instrument the model for token tracking
        with logfire.span("create_orchestrator_agent"):
            agent = Agent(
                model,
                deps_type=CampaignDeps,
                system_prompt="""You are the master orchestrator for a D&D 5e campaign.
                
Analyze user requests and delegate to specialized agents.""",
                retries=2
            )
        
        return agent
    
    async def _process_internal(self, request: AgentRequest) -> AgentResponse:
        """Process orchestrator requests with tracing"""
        
        with logfire.span(
            "orchestrator.process",
            action=request.action.value
        ) as span:
            
            if request.action == ActionType.PROCESS_USER_INPUT:
                return await self._handle_user_input(request.parameters.get('input', ''))
            elif request.action == ActionType.COORDINATE_AGENTS:
                return await self._coordinate_agents(request.parameters.get('plan', []))
            else:
                return await self._smart_orchestration(request)
    
    async def _handle_user_input(self, user_input: str) -> AgentResponse:
        """Parse user input and create execution plan with tracing"""
        
        with logfire.span(
            "orchestrator.handle_user_input",
            user_input_length=len(user_input)
        ) as span:
            
            # Analyze with AI
            with logfire.span("orchestrator.analyze_input"):
                result = await self.agent.run(
                    f"Analyze this D&D game request and create an execution plan: {user_input}",
                    deps=self.deps
                )
                
                # Track token usage if available
                if hasattr(result, 'usage'):
                    AgentMetrics.llm_token_usage.record(
                        result.usage().total_tokens,
                        {'agent': 'orchestrator', 'operation': 'analyze_input'}
                    )
            
            # Parse execution plan
            with logfire.span("orchestrator.parse_plan"):
                plan = self._parse_execution_plan(result.output)
                span.set_attribute('plan_steps', len(plan))
                
                logfire.info(
                    "Execution plan created",
                    steps=len(plan),
                    agents_involved=[step['agent'].value for step in plan]
                )
            
            # Execute plan
            return await self._coordinate_agents(plan)
    
    def _parse_execution_plan(self, ai_response: str) -> List[Dict[str, Any]]:
        """Parse AI response into executable plan with logging"""
        
        with logfire.span("orchestrator.parse_execution_plan"):
            plan = []
            lower_input = ai_response.lower()
            
            # Log pattern matching
            patterns_matched = []
            
            # Check each pattern category
            if any(word in lower_input for word in ['attack', 'fight', 'combat']):
                patterns_matched.append('combat')
                plan.append({'agent': AgentType.COMBAT, 'actions': ['process_combat']})
                plan.append({'agent': AgentType.NARRATIVE, 'actions': ['describe_action']})
            
            if any(word in lower_input for word in ['rule', 'spell', 'ability']):
                patterns_matched.append('rules')
                plan.append({'agent': AgentType.RULES, 'actions': ['lookup_rule']})
            
            if any(word in lower_input for word in ['move', 'position', 'distance']):
                patterns_matched.append('spatial')
                plan.append({'agent': AgentType.SPATIAL, 'actions': ['handle_movement']})
            
            if any(word in lower_input for word in ['create', 'character', 'npc']):
                patterns_matched.append('entity')
                plan.append({'agent': AgentType.ENTITY, 'actions': ['create_or_modify']})
            
            if any(word in lower_input for word in ['describe', 'look', 'scene']):
                patterns_matched.append('narrative')
                plan.append({'agent': AgentType.NARRATIVE, 'actions': ['describe_scene']})
            
            if any(word in lower_input for word in ['remember', 'history', 'recap']):
                patterns_matched.append('memory')
                plan.append({'agent': AgentType.MEMORY, 'actions': ['recall_history']})
            
            logfire.debug(
                "Pattern matching complete",
                patterns_matched=patterns_matched,
                plan_length=len(plan)
            )
            
            if not plan:
                plan.append({'agent': AgentType.ORCHESTRATOR, 'actions': ['general_response']})
            
            return plan
    
    async def _coordinate_agents(self, plan: List[Dict[str, Any]]) -> AgentResponse:
        """Execute multi-agent plan with distributed tracing"""
        
        trace_id = None
        
        with logfire.span(
            "orchestrator.coordinate",
            plan_steps=len(plan),
            trace_id=trace_id
        ) as span:
            
            results = []
            context = {}
            
            for i, step in enumerate(plan):
                agent_type = step['agent']
                
                with logfire.span(
                    f"orchestrator.step_{i}",
                    agent=agent_type.value,
                    step_index=i
                ):
                    
                    if agent_type == AgentType.ORCHESTRATOR:
                        continue
                    
                    agent = self.sub_agents.get(agent_type)
                    if not agent:
                        logfire.warning(f"Agent not found", agent_type=agent_type.value)
                        continue
                    
                    # Create request with trace context
                    request = AgentRequest(
                        agent_type=agent_type,
                        action=ActionType.PROCESS_USER_INPUT,
                        parameters=step.get('parameters', {}),
                        context=context,
                        trace_id=trace_id,
                        correlation_id=f"step_{i}"
                    )
                    
                    # Execute with timing
                    start = time.time()
                    response = await agent.process(request)
                    duration = (time.time() - start) * 1000
                    
                    results.append(response)
                    
                    # Update context
                    if response.success:
                        context[agent_type.value] = response.data
                    
                    logfire.debug(
                        f"Step {i} complete",
                        agent=agent_type.value,
                        success=response.success,
                        duration_ms=duration
                    )
            
            # Combine results
            combined = self._combine_results(results)
            
            span.set_attribute('successful_steps', len([r for r in results if r.success]))
            span.set_attribute('failed_steps', len([r for r in results if not r.success]))
            
            return combined
    
    def _combine_results(self, results: List[AgentResponse]) -> AgentResponse:
        """Combine multiple agent responses with metrics"""
        
        with logfire.span("orchestrator.combine_results", result_count=len(results)):
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]
            
            logfire.debug(
                "Combining results",
                total=len(results),
                successful=len(successful),
                failed=len(failed)
            )
            
            if not successful and failed:
                return AgentResponse(
                    agent_type=AgentType.ORCHESTRATOR,
                    action=ActionType.COORDINATE_AGENTS,
                    success=False,
                    data=None,
                    message="All operations failed",
                    errors=[r.message for r in failed]
                )
            
            combined_data = {}
            combined_messages = []
            
            for response in successful:
                combined_data[response.agent_type.value] = response.data
                combined_messages.append(response.message)
            
            return AgentResponse(
                agent_type=AgentType.ORCHESTRATOR,
                action=ActionType.COORDINATE_AGENTS,
                success=True,
                data=combined_data,
                message="\n\n".join(combined_messages),
                metadata={'agents_used': [r.agent_type.value for r in successful]}
            )
    
    async def _smart_orchestration(self, request: AgentRequest) -> AgentResponse:
        """Smart orchestration with pattern detection logging"""
        
        with logfire.span("orchestrator.smart_orchestration"):
            params = request.parameters
            
            # Log detected needs
            needs = {
                'rules': 'spell' in params or 'rule' in params,
                'spatial': 'position' in params or 'location' in params,
                'combat': 'attack' in params or 'damage' in params,
                'narrative': 'describe' in params
            }
            
            logfire.debug("Smart orchestration analysis", needs=needs)
            
            # Build plan based on needs
            plan = []
            for need, required in needs.items():
                if required:
                    agent_type = AgentType[need.upper()]
                    plan.append({'agent': agent_type, 'actions': [need]})
            
            return await self._coordinate_agents(plan)


# ============================================================================
# 2. Rules Agent with Metrics
# ============================================================================

class RulesAgent(BaseAgent):
    """Rules agent with caching metrics"""
    
    def __init__(self, deps: CampaignDeps, model_config: Dict[str, Any]):
        self.agent_type = AgentType.RULES
        super().__init__("Rules Expert", deps, model_config)
        self.rules_cache = {}
        
    def _create_agent(self) -> Agent:
        model = OpenAIChatModel(self.model_config.get('model_name', 'gpt-4o-mini'))
        
        return Agent(
            model,
            deps_type=CampaignDeps,
            system_prompt="""You are a D&D 5e rules expert.""",
            retries=1
        )
    
    async def _process_internal(self, request: AgentRequest) -> AgentResponse:
        """Process rules requests with metrics"""
        
        action = request.action
        params = request.parameters
        
        with logfire.span(f"rules.{action.value}", action=action.value):
            try:
                if action == ActionType.LOOKUP_RULE:
                    result = await self._lookup_rule(params.get('query', ''))
                elif action == ActionType.VALIDATE_ACTION:
                    result = await self._validate_action(
                        params.get('action'),
                        params.get('entity'),
                        params.get('target')
                    )
                elif action == ActionType.CALCULATE_DC:
                    result = await self._calculate_dc(
                        params.get('ability'),
                        params.get('entity')
                    )
                else:
                    result = await self._generic_rules_query(params)
                
                logfire.info(
                    "Rules processed",
                    action=action.value,
                    success=True
                )
                
                return AgentResponse(
                    agent_type=self.agent_type,
                    action=action,
                    success=True,
                    data=result,
                    message=f"Rules: {result.get('summary', 'Success')}"
                )
                
            except Exception as e:
                logfire.error(
                    "Rules processing failed",
                    action=action.value,
                    error=str(e)
                )
                
                return AgentResponse(
                    agent_type=self.agent_type,
                    action=action,
                    success=False,
                    data=None,
                    message=f"Rules error: {str(e)}",
                    errors=[str(e)]
                )
    
    async def _lookup_rule(self, query: str) -> Dict[str, Any]:
        """Look up rule with cache tracking"""
        
        with logfire.span("rules.lookup", query=query):
            # Check cache
            if query in self.rules_cache:
                logfire.debug("Rules cache hit", query=query)
                return self.rules_cache[query]
            
            logfire.debug("Rules cache miss", query=query)
            
            # Simulate lookup
            result = {
                'query': query,
                'rule': f'Rule information for {query}',
                'source': 'PHB',
                'summary': f"Rule: {query}"
            }
            
            self.rules_cache[query] = result
            return result
    
    async def _validate_action(self, action: str, entity: Dict, target: Optional[Dict]) -> Dict[str, Any]:
        """Validate action with detailed logging"""
        
        with logfire.span(
            "rules.validate_action",
            action=action,
            entity_name=entity.get('name')
        ):
            validation = {
                'action': action,
                'entity': entity.get('name'),
                'valid': True,
                'requirements': [],
                'issues': []
            }
            
            # Validation logic with logging
            if action == 'attack':
                if entity.get('conditions', {}).get('paralyzed'):
                    validation['valid'] = False
                    validation['issues'].append('Entity is paralyzed')
                    logfire.debug("Action invalid: paralyzed", entity=entity.get('name'))
            
            elif action == 'cast_spell':
                spell_level = entity.get('spell_level', 1)
                available_slots = entity.get('spell_slots', {}).get(str(spell_level), 0)
                if available_slots <= 0:
                    validation['valid'] = False
                    validation['issues'].append(f'No level {spell_level} spell slots')
                    logfire.debug("Action invalid: no spell slots", entity=entity.get('name'))
            
            validation['summary'] = 'Valid' if validation['valid'] else 'Invalid'
            
            logfire.info(
                "Action validated",
                action=action,
                valid=validation['valid'],
                issues=validation['issues']
            )
            
            return validation
    
    async def _calculate_dc(self, ability: str, entity: Dict) -> Dict[str, Any]:
        """Calculate DC with metrics"""
        
        with logfire.span("rules.calculate_dc", ability=ability):
            ability_score = entity.get(ability, 10)
            proficiency = entity.get('proficiency_bonus', 2)
            
            ability_mod = (ability_score - 10) // 2
            dc = 8 + proficiency + ability_mod
            
            logfire.debug(
                "DC calculated",
                ability=ability,
                score=ability_score,
                modifier=ability_mod,
                dc=dc
            )
            
            return {
                'ability': ability,
                'ability_score': ability_score,
                'ability_modifier': ability_mod,
                'proficiency_bonus': proficiency,
                'dc': dc,
                'summary': f"DC {dc}"
            }
    
    async def _generic_rules_query(self, params: Dict) -> Dict[str, Any]:
        """Generic query with LLM token tracking"""
        
        with logfire.span("rules.generic_query"):
            result = await self.agent.run(
                f"D&D 5e rules query: {json.dumps(params)}",
                deps=self.deps
            )
            
            # Track tokens if available
            if hasattr(result, 'usage'):
                AgentMetrics.llm_token_usage.record(
                    result.usage().total_tokens,
                    {'agent': 'rules', 'operation': 'generic_query'}
                )
            
            return {
                'query': params,
                'response': result.output,
                'summary': 'Processed'
            }
    
    def _get_cache_key(self, request: AgentRequest) -> Optional[str]:
        """Cache key for rules lookups"""
        if request.action == ActionType.LOOKUP_RULE:
            return f"rule_{request.parameters.get('query', '')}"
        return None


# ============================================================================
# 3. Spatial Agent with Performance Tracking
# ============================================================================

class SpatialAgent(BaseAgent):
    """Spatial agent with distance calculation metrics"""
    
    def __init__(self, deps: CampaignDeps, model_config: Dict[str, Any]):
        self.agent_type = AgentType.SPATIAL
        super().__init__("Spatial Manager", deps, model_config)
        self.positions = {}
        self.maps = {}
        
    def _create_agent(self) -> Agent:
        model = OpenAIChatModel(self.model_config.get('model_name', 'gpt-4o-mini'))
        
        return Agent(
            model,
            deps_type=CampaignDeps,
            system_prompt="""You are a D&D spatial specialist.""",
            retries=1
        )
    
    async def _process_internal(self, request: AgentRequest) -> AgentResponse:
        """Process spatial requests with performance tracking"""
        
        action = request.action
        params = request.parameters
        
        with logfire.span(f"spatial.{action.value}"):
            try:
                if action == ActionType.CREATE_LOCATION:
                    result = await self._create_location(params)
                elif action == ActionType.SET_POSITION:
                    result = await self._set_position(
                        params.get('entity'),
                        params.get('position')
                    )
                elif action == ActionType.CALCULATE_DISTANCE:
                    result = await self._calculate_distance(
                        params.get('from'),
                        params.get('to')
                    )
                elif action == ActionType.GET_AREA_EFFECT:
                    result = await self._get_area_effect(
                        params.get('center'),
                        params.get('radius'),
                        params.get('shape')
                    )
                else:
                    result = await self._generic_spatial_query(params)
                
                return AgentResponse(
                    agent_type=self.agent_type,
                    action=action,
                    success=True,
                    data=result,
                    message=f"Spatial: {result.get('summary', 'Processed')}"
                )
                
            except Exception as e:
                logfire.error(f"Spatial error", error=str(e), action=action.value)
                return AgentResponse(
                    agent_type=self.agent_type,
                    action=action,
                    success=False,
                    data=None,
                    message=f"Spatial error: {str(e)}",
                    errors=[str(e)]
                )
    
    async def _create_location(self, params: Dict) -> Dict[str, Any]:
        """Create location with metrics"""
        
        location_name = params.get('name', 'Unknown')
        
        with logfire.span("spatial.create_location", location=location_name):
            dimensions = params.get('dimensions', {'width': 50, 'height': 50})
            
            self.maps[location_name] = {
                'name': location_name,
                'dimensions': dimensions,
                'terrain': params.get('terrain', 'normal'),
                'entities': [],
                'features': params.get('features', [])
            }
            
            logfire.info(
                "Location created",
                name=location_name,
                width=dimensions['width'],
                height=dimensions['height']
            )
            
            # Track map count metric
            metric_counter('spatial.maps.total').add(1)
            
            return {
                'location': location_name,
                'created': True,
                'dimensions': dimensions,
                'summary': f"Created: {location_name}"
            }
    
    async def _set_position(self, entity_name: str, position: Dict) -> Dict[str, Any]:
        """Set position with tracking"""
        
        with logfire.span(
            "spatial.set_position",
            entity=entity_name,
            x=position.get('x'),
            y=position.get('y')
        ):
            self.positions[entity_name] = {
                'x': position.get('x', 0),
                'y': position.get('y', 0),
                'z': position.get('z', 0),
                'location': position.get('location', 'current_map')
            }
            
            logfire.debug(
                f"Position set",
                entity=entity_name,
                position=self.positions[entity_name]
            )
            
            # Track entity positions
            metric_counter('spatial.positioned_entities').add(1)
            
            return {
                'entity': entity_name,
                'position': self.positions[entity_name],
                'summary': f"{entity_name} at ({position.get('x')}, {position.get('y')})"
            }
    
    async def _calculate_distance(self, from_pos: Union[str, Dict], to_pos: Union[str, Dict]) -> Dict[str, Any]:
        """Calculate distance with performance metrics"""
        
        start_calc = time.time()
        
        with logfire.span("spatial.calculate_distance"):
            # Get positions
            if isinstance(from_pos, str):
                from_pos = self.positions.get(from_pos, {'x': 0, 'y': 0, 'z': 0})
            if isinstance(to_pos, str):
                to_pos = self.positions.get(to_pos, {'x': 0, 'y': 0, 'z': 0})
            
            # Calculate
            dx = to_pos.get('x', 0) - from_pos.get('x', 0)
            dy = to_pos.get('y', 0) - from_pos.get('y', 0)
            dz = to_pos.get('z', 0) - from_pos.get('z', 0)
            
            distance = (dx**2 + dy**2 + dz**2) ** 0.5
            squares = distance / 5
            
            calc_time = (time.time() - start_calc) * 1000
            
            # Track calculation time
            metric_histogram('spatial.distance_calc_ms').record(calc_time)
            
            logfire.debug(
                "Distance calculated",
                distance_feet=distance,
                squares=int(squares),
                calc_time_ms=calc_time
            )
            
            return {
                'from': from_pos,
                'to': to_pos,
                'distance_feet': distance,
                'squares': int(squares),
                'summary': f"Distance: {int(distance)}ft ({int(squares)} squares)"
            }
    
    async def _get_area_effect(self, center: Dict, radius: float, shape: str = 'sphere') -> Dict[str, Any]:
        """Get area effect with entity count metrics"""
        
        with logfire.span(
            "spatial.area_effect",
            shape=shape,
            radius=radius
        ) as span:
            
            affected = []
            
            for entity_name, pos in self.positions.items():
                dist_result = await self._calculate_distance(center, pos)
                if dist_result['distance_feet'] <= radius:
                    affected.append({
                        'entity': entity_name,
                        'distance': dist_result['distance_feet'],
                        'position': pos
                    })
            
            span.set_attribute('affected_count', len(affected))
            
            # Track area effect sizes
            metric_histogram('spatial.area_effect_entities').record(len(affected))
            
            logfire.info(
                "Area effect calculated",
                shape=shape,
                radius=radius,
                affected_count=len(affected)
            )
            
            return {
                'center': center,
                'radius': radius,
                'shape': shape,
                'affected_entities': affected,
                'count': len(affected),
                'summary': f"{len(affected)} entities in {radius}ft {shape}"
            }
    
    async def _generic_spatial_query(self, params: Dict) -> Dict[str, Any]:
        """Generic query with logging"""
        
        with logfire.span("spatial.generic_query"):
            return {
                'query': params,
                'positions': dict(list(self.positions.items())[:5]),
                'summary': 'Processed'
            }

class EntityAgent(BaseAgent):
    """Manages all game entities"""
    
    def __init__(self, deps: CampaignDeps, model_config: Dict[str, Any]):
        self.agent_type = AgentType.ENTITY
        super().__init__("Entity Manager", deps, model_config)
        self.entities = {}  # name -> entity data
        
    def _create_agent(self) -> Agent:
        model = OpenAIChatModel(self.model_config.get('model_name', 'gpt-4o-mini'))
        
        return Agent(
            model,
            deps_type=CampaignDeps,
            system_prompt="""You are the D&D entity manager. Create and manage characters, NPCs, and items.""",
            retries=1
        )
    
    async def _process_internal(self, request: AgentRequest) -> AgentResponse:
        """Process entity requests"""
        
        action = request.action
        params = request.parameters
        
        try:
            if action == ActionType.CREATE_ENTITY:
                result = await self._create_entity(params)
            elif action == ActionType.UPDATE_ENTITY:
                result = await self._update_entity(
                    params.get('name'),
                    params.get('updates')
                )
            elif action == ActionType.GET_ENTITY:
                result = await self._get_entity(params.get('name'))
            elif action == ActionType.MANAGE_INVENTORY:
                result = await self._manage_inventory(
                    params.get('entity'),
                    params.get('operation'),
                    params.get('item')
                )
            else:
                result = await self._generic_entity_query(params)
            
            return AgentResponse(
                agent_type=self.agent_type,
                action=action,
                success=True,
                data=result,
                message=f"Entity: {result.get('summary', 'Processed')}"
            )
            
        except Exception as e:
            return AgentResponse(
                agent_type=self.agent_type,
                action=action,
                success=False,
                data=None,
                message=f"Entity error: {str(e)}",
                errors=[str(e)]
            )
    
    async def _create_entity(self, params: Dict) -> Dict[str, Any]:
        """Create new entity"""
        
        entity_type = params.get('type', 'npc')
        name = params.get('name', f'Entity_{len(self.entities)}')
        
        # Base entity structure
        entity = {
            'name': name,
            'type': entity_type,
            'created_at': datetime.now().isoformat(),
            'hp': params.get('hp', 10),
            'max_hp': params.get('max_hp', 10),
            'ac': params.get('ac', 10),
            'stats': params.get('stats', {
                'strength': 10,
                'dexterity': 10,
                'constitution': 10,
                'intelligence': 10,
                'wisdom': 10,
                'charisma': 10
            }),
            'inventory': [],
            'conditions': [],
            'proficiency_bonus': params.get('proficiency_bonus', 2)
        }
        
        # Type-specific attributes
        if entity_type == 'player':
            entity['class'] = params.get('class', 'fighter')
            entity['level'] = params.get('level', 1)
            entity['spell_slots'] = params.get('spell_slots', {})
        elif entity_type == 'monster':
            entity['cr'] = params.get('cr', '1/4')
            entity['abilities'] = params.get('abilities', [])
        
        self.entities[name] = entity
        
        return {
            'entity': entity,
            'created': True,
            'summary': f"Created {entity_type}: {name}"
        }
    
    async def _update_entity(self, name: str, updates: Dict) -> Dict[str, Any]:
        """Update entity attributes"""
        
        if name not in self.entities:
            return {
                'entity': name,
                'updated': False,
                'summary': f"Entity {name} not found"
            }
        
        entity = self.entities[name]
        
        for key, value in updates.items():
            if key == 'hp':
                entity['hp'] = min(value, entity['max_hp'])
            elif key == 'conditions':
                if isinstance(value, list):
                    entity['conditions'] = value
                else:
                    entity['conditions'].append(value)
            else:
                entity[key] = value
        
        return {
            'entity': name,
            'updates': updates,
            'updated': True,
            'current_state': entity,
            'summary': f"Updated {name}"
        }
    
    async def _get_entity(self, name: str) -> Dict[str, Any]:
        """Get entity data"""
        
        entity = self.entities.get(name)
        
        if entity:
            return {
                'found': True,
                'entity': entity,
                'summary': f"Retrieved {name}"
            }
        else:
            return {
                'found': False,
                'entity': None,
                'summary': f"Entity {name} not found"
            }
    
    async def _manage_inventory(self, entity_name: str, operation: str, item: Dict) -> Dict[str, Any]:
        """Manage entity inventory"""
        
        if entity_name not in self.entities:
            return {
                'success': False,
                'summary': f"Entity {entity_name} not found"
            }
        
        entity = self.entities[entity_name]
        inventory = entity.get('inventory', [])
        
        if operation == 'add':
            inventory.append(item)
            result = f"Added {item.get('name')} to {entity_name}'s inventory"
        elif operation == 'remove':
            inventory = [i for i in inventory if i.get('name') != item.get('name')]
            entity['inventory'] = inventory
            result = f"Removed {item.get('name')} from {entity_name}'s inventory"
        elif operation == 'equip':
            item['equipped'] = True
            result = f"{entity_name} equipped {item.get('name')}"
        else:
            result = f"Unknown operation: {operation}"
        
        return {
            'entity': entity_name,
            'operation': operation,
            'item': item,
            'inventory': inventory,
            'summary': result
        }
    
    async def _generic_entity_query(self, params: Dict) -> Dict[str, Any]:
        """Handle generic entity queries"""
        
        return {
            'query': params,
            'entity_count': len(self.entities),
            'entities': list(self.entities.keys()),
            'summary': f"Managing {len(self.entities)} entities"
        }



# I'll implement one more complete example (Combat) and provide the pattern for others

class CombatAgent(BaseAgent):
    """Combat agent with detailed combat metrics"""
    
    def __init__(self, deps: CampaignDeps, model_config: Dict[str, Any]):
        self.agent_type = AgentType.COMBAT
        super().__init__("Combat Manager", deps, model_config)
        self.combat_state = {
            'active': False,
            'round': 0,
            'turn_order': [],
            'current_turn': 0
        }
        
    def _create_agent(self) -> Agent:
        model = OpenAIChatModel(self.model_config.get('model_name', 'gpt-4o-mini'))
        return Agent(
            model,
            deps_type=CampaignDeps,
            system_prompt="""You are the D&D combat manager.""",
            retries=1
        )
    
    async def _process_internal(self, request: AgentRequest) -> AgentResponse:
        """Process combat with detailed metrics"""
        
        action = request.action
        params = request.parameters
        
        with logfire.span(f"combat.{action.value}"):
            try:
                if action == ActionType.ROLL_INITIATIVE:
                    result = await self._roll_initiative(params.get('entities', []))
                elif action == ActionType.PROCESS_ATTACK:
                    result = await self._process_attack(
                        params.get('attacker'),
                        params.get('target'),
                        params.get('attack_type'),
                        params.get('modifiers', {})
                    )
                elif action == ActionType.APPLY_DAMAGE:
                    result = await self._apply_damage(
                        params.get('target'),
                        params.get('damage'),
                        params.get('damage_type')
                    )
                else:
                    result = {'summary': 'Combat action processed'}
                
                return AgentResponse(
                    agent_type=self.agent_type,
                    action=action,
                    success=True,
                    data=result,
                    message=f"Combat: {result.get('summary', 'Processed')}"
                )
                
            except Exception as e:
                logfire.error("Combat error", error=str(e), action=action.value)
                return AgentResponse(
                    agent_type=self.agent_type,
                    action=action,
                    success=False,
                    data=None,
                    message=f"Combat error: {str(e)}",
                    errors=[str(e)]
                )
    
    async def _roll_initiative(self, entities: List[Dict]) -> Dict[str, Any]:
        """Roll initiative with metrics"""
        
        with logfire.span("combat.roll_initiative", entity_count=len(entities)):
            initiative_order = []
            
            for entity in entities:
                dex_mod = (entity.get('dexterity', 10) - 10) // 2
                roll = random.randint(1, 20) + dex_mod
                
                initiative_order.append({
                    'entity': entity.get('name'),
                    'initiative': roll,
                    'dex_modifier': dex_mod
                })
                
                logfire.debug(
                    "Initiative rolled",
                    entity=entity.get('name'),
                    roll=roll
                )
            
            initiative_order.sort(key=lambda x: x['initiative'], reverse=True)
            
            self.combat_state['active'] = True
            self.combat_state['round'] = 1
            self.combat_state['turn_order'] = [x['entity'] for x in initiative_order]
            
            # Metrics
            metric_counter('combat.encounters').add(1)
            metric_histogram('combat.encounter_size').record(len(entities))
            
            logfire.info(
                "Combat started",
                round=1,
                entity_count=len(entities),
                first_turn=initiative_order[0]['entity'] if initiative_order else None
            )
            
            return {
                'initiative_order': initiative_order,
                'round': 1,
                'summary': f"Combat! {initiative_order[0]['entity']} goes first"
            }
    
    async def _process_attack(self, attacker: Dict, target: Dict, attack_type: str, modifiers: Dict) -> Dict[str, Any]:
        """Process attack with detailed metrics"""
        
        with logfire.span(
            "combat.process_attack",
            attacker=attacker.get('name'),
            target=target.get('name'),
            type=attack_type
        ) as span:
            
            # Roll mechanics
            advantage = modifiers.get('advantage', False)
            disadvantage = modifiers.get('disadvantage', False)
            
            if advantage and not disadvantage:
                roll = max(random.randint(1, 20), random.randint(1, 20))
                logfire.debug("Rolling with advantage")
            elif disadvantage and not advantage:
                roll = min(random.randint(1, 20), random.randint(1, 20))
                logfire.debug("Rolling with disadvantage")
            else:
                roll = random.randint(1, 20)
            
            # Calculate hit
            ability_mod = (attacker.get('strength', 10) - 10) // 2
            attack_bonus = ability_mod + attacker.get('proficiency_bonus', 2)
            total = roll + attack_bonus
            target_ac = target.get('ac', 10)
            
            hit = total >= target_ac or roll == 20
            critical = roll == 20
            fumble = roll == 1
            
            # Damage calculation
            damage = 0
            if hit and not fumble:
                damage_dice = modifiers.get('damage_dice', '1d8')
                base_damage = self._roll_damage(damage_dice)
                if critical:
                    base_damage *= 2
                damage = base_damage + ability_mod
            
            # Metrics
            metric_counter('combat.attacks').add(1, {'hit': str(hit), 'critical': str(critical)})
            if damage > 0:
                metric_histogram('combat.damage_dealt').record(damage)
            
            span.set_attribute('hit', hit)
            span.set_attribute('damage', damage)
            span.set_attribute('roll', roll)
            
            logfire.info(
                "Attack processed",
                attacker=attacker.get('name'),
                target=target.get('name'),
                hit=hit,
                damage=damage,
                critical=critical
            )
            
            return {
                'roll': roll,
                'total': total,
                'target_ac': target_ac,
                'hit': hit,
                'critical': critical,
                'damage': damage,
                'summary': f"{'CRIT! ' if critical else ''}{'Hit! ' if hit else 'Miss!'}{damage} damage" if hit else "Miss!"
            }
    
    def _roll_damage(self, damage_str: str) -> int:
        """Roll damage with logging"""
        if 'd' not in damage_str:
            return int(damage_str)
        
        num_dice, die_size = damage_str.split('d')
        num_dice = int(num_dice) if num_dice else 1
        die_size = int(die_size)
        
        result = sum(random.randint(1, die_size) for _ in range(num_dice))
        
        logfire.debug(f"Damage rolled", dice=damage_str, result=result)
        return result
    
    async def _apply_damage(self, target: Dict, damage: int, damage_type: str) -> Dict[str, Any]:
        """Apply damage with health tracking"""
        
        with logfire.span(
            "combat.apply_damage",
            target=target.get('name'),
            damage=damage,
            type=damage_type
        ):
            current_hp = target.get('hp', 0)
            max_hp = target.get('max_hp', current_hp)
            
            # Resistance/immunity checks
            if damage_type in target.get('resistances', []):
                damage = damage // 2
                logfire.debug("Damage reduced by resistance")
            elif damage_type in target.get('immunities', []):
                damage = 0
                logfire.debug("Damage negated by immunity")
            
            new_hp = max(0, current_hp - damage)
            unconscious = new_hp == 0
            
            # Metrics
            metric_histogram('combat.damage_taken').record(damage)
            if unconscious:
                metric_counter('combat.knockouts').add(1)
            
            logfire.info(
                "Damage applied",
                target=target.get('name'),
                damage=damage,
                remaining_hp=new_hp,
                unconscious=unconscious
            )
            
            return {
                'target': target.get('name'),
                'damage': damage,
                'current_hp': new_hp,
                'unconscious': unconscious,
                'summary': f"{damage} damage! {'Unconscious!' if unconscious else f'{new_hp} HP left'}"
            }

class NarrativeAgent(BaseAgent):
    """Handles storytelling and scene descriptions"""
    
    def __init__(self, deps: CampaignDeps, model_config: Dict[str, Any]):
        self.agent_type = AgentType.NARRATIVE
        super().__init__("Narrative Voice", deps, model_config)
        self.scene_cache = {}
        
    def _create_agent(self) -> Agent:
        model = OpenAIChatModel(self.model_config.get('model_name', 'gpt-4o'))
        
        return Agent(
            model,
            deps_type=CampaignDeps,
            system_prompt="""You are the narrative voice of the D&D campaign. Create vivid descriptions.""",
            retries=2
        )
    
    async def _process_internal(self, request: AgentRequest) -> AgentResponse:
        """Process narrative requests"""
        
        action = request.action
        params = request.parameters
        
        try:
            if action == ActionType.DESCRIBE_SCENE:
                result = await self._describe_scene(params)
            elif action == ActionType.DESCRIBE_ACTION:
                result = await self._describe_action(params)
            elif action == ActionType.GENERATE_DIALOGUE:
                result = await self._generate_dialogue(params)
            else:
                result = await self._generic_narrative_query(params)
            
            return AgentResponse(
                agent_type=self.agent_type,
                action=action,
                success=True,
                data=result,
                message=result.get('description', result.get('summary', 'Narrated'))
            )
            
        except Exception as e:
            return AgentResponse(
                agent_type=self.agent_type,
                action=action,
                success=False,
                data=None,
                message=f"Narrative error: {str(e)}",
                errors=[str(e)]
            )
    
    async def _describe_scene(self, params: Dict) -> Dict[str, Any]:
        """Generate scene description"""
        
        location = params.get('location', 'unknown location')
        mood = params.get('mood', 'neutral')
        details = params.get('details', [])
        
        # Check cache
        cache_key = f"{location}_{mood}"
        if cache_key in self.scene_cache:
            return self.scene_cache[cache_key]
        
        # Generate description
        prompt = f"Describe a {mood} scene in {location}. Include: {', '.join(details)}"
        
        result = await self.agent.run(prompt, deps=self.deps)
        
        description_data = {
            'location': location,
            'mood': mood,
            'description': result.output,
            'summary': f"Scene: {location}"
        }
        
        self.scene_cache[cache_key] = description_data
        return description_data
    
    async def _describe_action(self, params: Dict) -> Dict[str, Any]:
        """Describe an action dramatically"""
        
        action_type = params.get('type')
        actor = params.get('actor', 'someone')
        target = params.get('target')
        result = params.get('result', {})
        
        # Generate appropriate description based on action
        if action_type == 'attack':
            if result.get('critical'):
                description = f"{actor} lands a devastating critical hit on {target}!"
            elif result.get('hit'):
                description = f"{actor}'s attack strikes {target} for {result.get('damage', 0)} damage!"
            else:
                description = f"{actor}'s attack whistles past {target}, missing by inches!"
        elif action_type == 'spell':
            spell = params.get('spell', 'magic')
            description = f"{actor} weaves arcane energy, casting {spell}!"
        elif action_type == 'move':
            distance = params.get('distance', 0)
            description = f"{actor} swiftly moves {distance} feet across the battlefield!"
        else:
            description = f"{actor} performs an action!"
        
        return {
            'action': action_type,
            'description': description,
            'dramatic': True,
            'summary': description[:50] + '...'
        }
    
    async def _generate_dialogue(self, params: Dict) -> Dict[str, Any]:
        """Generate NPC dialogue"""
        
        npc_name = params.get('npc', 'the stranger')
        personality = params.get('personality', 'neutral')
        topic = params.get('topic', 'general conversation')
        emotion = params.get('emotion', 'calm')
        
        prompt = f"Generate dialogue for {npc_name} ({personality} personality) discussing {topic} in a {emotion} manner."
        
        result = await self.agent.run(prompt, deps=self.deps)
        
        return {
            'npc': npc_name,
            'dialogue': result.output,
            'emotion': emotion,
            'summary': f"{npc_name} speaks"
        }
    
    async def _generic_narrative_query(self, params: Dict) -> Dict[str, Any]:
        """Handle generic narrative queries"""
        
        result = await self.agent.run(
            f"Narrate: {json.dumps(params)}",
            deps=self.deps
        )
        
        return {
            'narration': result.output,
            'summary': 'Narrated'
        }
    
    def _get_cache_key(self, request: AgentRequest) -> Optional[str]:
        """Generate cache key for scene descriptions"""
        if request.action == ActionType.DESCRIBE_SCENE:
            location = request.parameters.get('location', '')
            mood = request.parameters.get('mood', '')
            return f"scene_{location}_{mood}"
        return None


# ============================================================================
# 7. Memory Agent - Complete Implementation
# ============================================================================

class MemoryAgent(BaseAgent):
    """Manages campaign history and persistent information"""
    
    def __init__(self, deps: CampaignDeps, model_config: Dict[str, Any]):
        self.agent_type = AgentType.MEMORY
        super().__init__("Memory Keeper", deps, model_config)
        self.campaign_history = []
        self.session_notes = {}
        self.lore = {}
        
    def _create_agent(self) -> Agent:
        model = OpenAIChatModel(self.model_config.get('model_name', 'gpt-4o-mini'))
        
        return Agent(
            model,
            deps_type=CampaignDeps,
            system_prompt="""You are the campaign memory keeper. Store and recall important information.""",
            retries=1
        )
    
    async def _process_internal(self, request: AgentRequest) -> AgentResponse:
        """Process memory requests"""
        
        action = request.action
        params = request.parameters
        
        try:
            if action == ActionType.STORE_EVENT:
                result = await self._store_event(params)
            elif action == ActionType.RECALL_HISTORY:
                result = await self._recall_history(params)
            elif action == ActionType.SEARCH_LORE:
                result = await self._search_lore(params.get('query'))
            elif action == ActionType.SUMMARIZE_SESSION:
                result = await self._summarize_session(params)
            else:
                result = await self._generic_memory_query(params)
            
            return AgentResponse(
                agent_type=self.agent_type,
                action=action,
                success=True,
                data=result,
                message=result.get('summary', 'Memory processed')
            )
            
        except Exception as e:
            return AgentResponse(
                agent_type=self.agent_type,
                action=action,
                success=False,
                data=None,
                message=f"Memory error: {str(e)}",
                errors=[str(e)]
            )
    
    async def _store_event(self, params: Dict) -> Dict[str, Any]:
        """Store campaign event"""
        
        event = {
            'timestamp': datetime.now().isoformat(),
            'session': params.get('session', self.deps.session_id),
            'type': params.get('type', 'general'),
            'description': params.get('description'),
            'participants': params.get('participants', []),
            'location': params.get('location'),
            'importance': params.get('importance', 'normal')
        }
        
        self.campaign_history.append(event)
        
        # Store in session notes
        session = params.get('session', 'current')
        if session not in self.session_notes:
            self.session_notes[session] = []
        self.session_notes[session].append(event)
        
        return {
            'stored': True,
            'event': event,
            'total_events': len(self.campaign_history),
            'summary': f"Stored: {event['description'][:50]}..."
        }
    
    async def _recall_history(self, params: Dict) -> Dict[str, Any]:
        """Recall campaign history"""
        
        filter_type = params.get('filter', 'all')
        limit = params.get('limit', 10)
        session = params.get('session')
        
        # Filter events
        if session and session in self.session_notes:
            events = self.session_notes[session]
        elif filter_type == 'recent':
            events = self.campaign_history[-limit:]
        elif filter_type == 'important':
            events = [e for e in self.campaign_history if e.get('importance') == 'high'][:limit]
        else:
            events = self.campaign_history[:limit]
        
        # Create summary
        if events:
            summary = f"Recalled {len(events)} events"
        else:
            summary = "No matching events found"
        
        return {
            'events': events,
            'count': len(events),
            'filter': filter_type,
            'summary': summary
        }
    
    async def _search_lore(self, query: str) -> Dict[str, Any]:
        """Search campaign lore"""
        
        results = []
        query_lower = query.lower()
        
        # Search in lore
        for key, value in self.lore.items():
            if query_lower in key.lower() or query_lower in str(value).lower():
                results.append({
                    'topic': key,
                    'content': value
                })
        
        # Search in history
        for event in self.campaign_history:
            if query_lower in event.get('description', '').lower():
                results.append({
                    'type': 'event',
                    'content': event
                })
        
        return {
            'query': query,
            'results': results[:10],  # Limit results
            'count': len(results),
            'summary': f"Found {len(results)} results for '{query}'"
        }
    
    async def _summarize_session(self, params: Dict) -> Dict[str, Any]:
        """Create session summary"""
        
        session = params.get('session', 'current')
        events = self.session_notes.get(session, self.campaign_history[-20:])
        
        if not events:
            return {
                'session': session,
                'summary': "No events to summarize",
                'highlights': []
            }
        
        # Categorize events
        combat_events = [e for e in events if e.get('type') == 'combat']
        npc_interactions = [e for e in events if e.get('type') == 'npc_interaction']
        discoveries = [e for e in events if e.get('type') == 'discovery']
        plot_events = [e for e in events if e.get('importance') == 'high']
        
        summary = {
            'session': session,
            'event_count': len(events),
            'combat_encounters': len(combat_events),
            'npcs_met': list(set([e.get('participants', [])[0] for e in npc_interactions if e.get('participants')])),
            'key_discoveries': [e.get('description') for e in discoveries],
            'plot_developments': [e.get('description') for e in plot_events],
            'summary': f"Session included {len(combat_events)} combats, met {len(npc_interactions)} NPCs"
        }
        
        return summary
    
    async def _generic_memory_query(self, params: Dict) -> Dict[str, Any]:
        """Handle generic memory queries"""
        
        return {
            'query': params,
            'total_events': len(self.campaign_history),
            'sessions': list(self.session_notes.keys()),
            'lore_topics': list(self.lore.keys())[:10],
            'summary': 'Memory query processed'
        }
# ============================================================================
# Multi-Agent System with Full Logfire Integration
# ============================================================================

class MultiAgentDnDSystem:
    """Main system with comprehensive observability"""
    
    def __init__(self, deps: CampaignDeps, model_config: Dict[str, Any]):
        self.deps = deps
        self.model_config = model_config
        self.request_log = []
        
        with logfire.span("system.initialize"):
            self.agents = self._initialize_agents()
            
            logfire.info(
                "Multi-agent system initialized",
                agent_count=len(self.agents),
                campaign_id=deps.campaign_id,
                model=model_config.get('model_name')
            )
    
    def _initialize_agents(self) -> Dict[AgentType, BaseAgent]:
        """Initialize all agents with instrumentation"""
        
        with logfire.span("system.create_agents"):
            # Create individual agents
            agents = {
                AgentType.RULES: RulesAgent(self.deps, self.model_config),
                AgentType.SPATIAL: SpatialAgent(self.deps, self.model_config),
                # Add other agents here with same pattern
                AgentType.NARRATIVE: NarrativeAgent(self.deps, self.model_config),
                AgentType.MEMORY: MemoryAgent(self.deps, self.model_config),
                AgentType.COMBAT: CombatAgent(self.deps, self.model_config),
                AgentType.ENTITY: EntityAgent(self.deps, self.model_config),
            }
            
            # Create orchestrator
            agents[AgentType.ORCHESTRATOR] = DMOrchestratorAgent(
                self.deps, 
                self.model_config, 
                agents
            )
            
            # Log agent creation
            for agent_type in agents:
                logfire.debug(f"Agent created", type=agent_type.value)
            
            return agents
    
    async def process_user_input(self, user_input: str) -> str:
        """Main entry point with full tracing"""
        
        trace_id = logfire.get_context().get('trace_id', None)
        
        with logfire.span(
            "system.process_user_input",
            trace_id=trace_id,
            input_length=len(user_input)
        ) as span:
            
            start_time = time.time()
            
            logfire.info(
                "Processing user input",
                campaign_id=self.deps.campaign_id,
                session_id=self.deps.session_id,
                input_preview=user_input[:100]
            )
            
            # Create request
            request = AgentRequest(
                agent_type=AgentType.ORCHESTRATOR,
                action=ActionType.PROCESS_USER_INPUT,
                parameters={'input': user_input},
                context=self.deps.current_context,
                trace_id=trace_id
            )
            
            # Process through orchestrator
            orchestrator = self.agents[AgentType.ORCHESTRATOR]
            response = await orchestrator.process(request)
            
            # Calculate metrics
            processing_time = (time.time() - start_time) * 1000
            
            # Log request
            self.request_log.append({
                'timestamp': datetime.now().isoformat(),
                'trace_id': trace_id,
                'input': user_input,
                'response': response.message,
                'success': response.success,
                'processing_time_ms': processing_time
            })
            
            # Record system-level metrics
            metric_counter('system.requests').add(1, {'success': str(response.success)})
            metric_histogram('system.request_duration_ms').record(processing_time)
            
            span.set_attribute('success', response.success)
            span.set_attribute('processing_time_ms', processing_time)
            
            logfire.info(
                "Request complete",
                trace_id=trace_id,
                success=response.success,
                duration_ms=processing_time
            )
            
            return response.message
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        with logfire.span("system.get_status"):
            status = {
                'agents': {},
                'request_count': len(self.request_log),
                'campaign_id': self.deps.campaign_id,
                'session_id': self.deps.session_id,
                'metrics': {
                    'total_requests': len(self.request_log),
                    'avg_response_time_ms': sum(r['processing_time_ms'] for r in self.request_log[-100:]) / min(100, len(self.request_log)) if self.request_log else 0
                }
            }
            
            for agent_type, agent in self.agents.items():
                status['agents'][agent_type.value] = await agent.get_state()
            
            logfire.debug("System status retrieved", status=status)
            
            return status


# ============================================================================
# Usage Examples with Observability
# ============================================================================

@log_slow_async_callbacks(slow_duration=.1)
async def example_with_monitoring():
    """Example usage with full observability"""
    
    # # Configure Logfire for local development
    # logfire.configure(
    #     service_name="dnd-multiagent-dev",
    #     environment="development",
    #     console=True,  # Print to console
    #     send_to_logfire=True  # Send to Logfire dashboard
    # )
    
    with logfire.span("example.session") as session_span:
        # Setup
        deps = CampaignDeps(
            neo4j_driver=None,
            campaign_id="example_campaign",
            session_id="session_001"
        )
        
        model_config = {
            'model_name': 'gpt-4o',
            'provider': 'openai'
        }
        
        # Create system
        with logfire.span("example.create_system"):
            system = MultiAgentDnDSystem(deps, model_config)
        
        # Process some requests
        test_inputs = [
            "Create a goblin enemy with 7 HP",
            "Roll initiative for combat",
            "The fighter attacks the goblin",
            "Describe the battle scene"
        ]
        
        for i, input_text in enumerate(test_inputs):
            with logfire.span(f"example.request_{i}", request=input_text):
                result = await system.process_user_input(input_text)
                
                logfire.info(
                    f"Request {i} processed",
                    input=input_text,
                    response_length=len(result)
                )
                
                print(f"\n{'='*50}")
                print(f"Input: {input_text}")
                print(f"Response: {result[:200]}...")
        
        # Get system metrics
        with logfire.span("example.get_metrics"):
            status = await system.get_system_status()
            
            logfire.info(
                "Session complete",
                total_requests=status['request_count'],
                avg_response_ms=status['metrics']['avg_response_time_ms']
            )
            
            print(f"\n{'='*50}")
            print("System Metrics:")
            print(f"Total Requests: {status['request_count']}")
            print(f"Avg Response Time: {status['metrics']['avg_response_time_ms']:.2f}ms")
            print(f"Active Agents: {list(status['agents'].keys())}")
        
        session_span.set_attribute('total_requests', len(test_inputs))
        session_span.set_attribute('session_id', 'session_001')


# ============================================================================
# Logfire Dashboard Configuration
# ============================================================================

def setup_logfire_dashboards():
    """Configure Logfire dashboards for monitoring"""
    
    # This would be configured in the Logfire UI, but here's the structure:
    dashboards = {
        "system_overview": {
            "panels": [
                {
                    "title": "Request Rate",
                    "query": "rate(system.requests[5m])",
                    "type": "graph"
                },
                {
                    "title": "Response Time P95",
                    "query": "histogram_quantile(0.95, system.request_duration_ms)",
                    "type": "gauge"
                },
                {
                    "title": "Agent Usage",
                    "query": "sum by (agent) (agent.requests.total)",
                    "type": "pie"
                },
                {
                    "title": "Error Rate",
                    "query": "rate(agent.errors.total[5m])",
                    "type": "graph"
                }
            ]
        },
        "agent_performance": {
            "panels": [
                {
                    "title": "Agent Response Times",
                    "query": "avg by (agent) (agent.request.duration_ms)",
                    "type": "bar"
                },
                {
                    "title": "Cache Hit Rate",
                    "query": "rate(agent.cache.hits) / (rate(agent.cache.hits) + rate(agent.cache.misses))",
                    "type": "percentage"
                },
                {
                    "title": "LLM Token Usage",
                    "query": "sum by (agent) (rate(agent.llm.tokens[5m]))",
                    "type": "graph"
                }
            ]
        },
        "combat_metrics": {
            "panels": [
                {
                    "title": "Combat Encounters",
                    "query": "sum(combat.encounters)",
                    "type": "counter"
                },
                {
                    "title": "Average Damage",
                    "query": "avg(combat.damage_dealt)",
                    "type": "gauge"
                },
                {
                    "title": "Hit/Miss Ratio",
                    "query": "sum(combat.attacks{hit='true'}) / sum(combat.attacks)",
                    "type": "percentage"
                }
            ]
        }
    }
    
    return dashboards


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    # Run example with full monitoring
    asyncio.run(example_with_monitoring())
    
    print("\n" + "="*50)
    print("🎲 D&D Multi-Agent System with Logfire")
    print("="*50)
    print("\n✅ Monitoring Features:")
    print("  - Distributed tracing across all agents")
    print("  - Performance metrics and histograms")
    print("  - Cache hit/miss tracking")
    print("  - LLM token usage monitoring")
    print("  - Error tracking and debugging")
    print("\n📊 View metrics at: https://logfire.pydantic.dev")