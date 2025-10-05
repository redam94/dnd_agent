"""
agents
======

This package contains the base classes and concrete implementations for the
various agents used by the multi‑agent D&D system.  Each agent is
responsible for a distinct domain of the game (rules, spatial reasoning,
entities, combat, narrative, memory, orchestration).  The goal of this
package is to separate concerns so that individual agents can evolve
independently, be tested in isolation, and be composed together by the
orchestrator.

The :class:`BaseAgent` provides common behaviour such as model creation
and request handling.  Sub‑classes implement the `_create_agent` method
to instantiate their underlying language model, register tools and set
their system prompt.

The :class:`AgentType` enumeration defines the available agent domains and
is used for routing within the orchestrator.  The dataclasses
:class:`AgentRequest` and :class:`AgentResponse` encapsulate a request
from the orchestrator to a specialised agent and the response back.

Usage example:

    >>> from dnd_agent.agents import DMOrchestratorAgent, RulesAgent, SpatialAgent
    >>> # deps and model_config defined elsewhere
    >>> rules = RulesAgent(deps, model_config)
    >>> spatial = SpatialAgent(deps, model_config)
    >>> orchestrator = DMOrchestratorAgent(deps, model_config, {
    ...     AgentType.RULES: rules,
    ...     AgentType.SPATIAL: spatial,
    ... })
    >>> # process a high level request
    >>> result = await orchestrator.process(AgentRequest(
    ...     agent_type=AgentType.ORCHESTRATOR,
    ...     action="resolve_turn",
    ...     parameters={"player": "Alice", "action": "attack"}
    ... ))
"""

from .base import AgentType, AgentRequest, AgentResponse, BaseAgent
from .orchestrator import DMOrchestratorAgent
from .rules import RulesAgent
from .spatial import SpatialAgent
from .entity import EntityAgent
from .combat import CombatAgent
from .narrative import NarrativeAgent
from .memory import MemoryAgent

__all__ = [
    "AgentType",
    "AgentRequest",
    "AgentResponse",
    "BaseAgent",
    "DMOrchestratorAgent",
    "RulesAgent",
    "SpatialAgent",
    "EntityAgent",
    "CombatAgent",
    "NarrativeAgent",
    "MemoryAgent",
]