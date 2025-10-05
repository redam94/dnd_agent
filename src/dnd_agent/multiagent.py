"""
Multi‑agent campaign system entry point.

This module builds the full agent hierarchy used by the D&D campaign
framework.  It wires together the individual agents (rules, spatial,
entity, combat, narrative, memory) under a DM orchestrator.  Downstream
applications should import and call :func:`create_campaign_system` to
obtain a ready‑made orchestrator.
"""

from __future__ import annotations

from typing import Any, Dict

from .agents import (
    AgentType,
    DMOrchestratorAgent,
    RulesAgent,
    SpatialAgent,
    EntityAgent,
    CombatAgent,
    NarrativeAgent,
    MemoryAgent,
)


def create_campaign_system(deps: Any, model_config: Dict[str, Any]) -> DMOrchestratorAgent:
    """Construct and return the orchestrator with all sub‑agents wired up.

    :param deps: An instance containing campaign dependencies such as database
        connections and API clients.  See :mod:`dnd_agent.models.agent_deps`.
    :param model_config: A dictionary with at least a ``model_name`` key
        specifying which OpenAI model to use.
    :returns: A fully initialised :class:`DMOrchestratorAgent` with all
        sub‑agents registered.
    """
    sub_agents = {
        AgentType.RULES: RulesAgent(deps, model_config),
        AgentType.SPATIAL: SpatialAgent(deps, model_config),
        AgentType.ENTITY: EntityAgent(deps, model_config),
        AgentType.COMBAT: CombatAgent(deps, model_config),
        AgentType.NARRATIVE: NarrativeAgent(deps, model_config),
        AgentType.MEMORY: MemoryAgent(deps, model_config),
    }
    orchestrator = DMOrchestratorAgent(deps, model_config, sub_agents)
    return orchestrator


__all__ = ["create_campaign_system"]