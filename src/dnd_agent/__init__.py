"""
dnd_agent package.

This package exposes the multi‑agent D&D campaign system.  The main
entry point is :func:`create_campaign_system` which constructs an
orchestrator agent wired up with all sub‑agents.  See the
``dnd_agent.agents`` package for lower level agent classes and the
``dnd_agent.tools`` package for individual tool functions.
"""

from .multiagent import create_campaign_system
from .agents import AgentType

__all__ = ["create_campaign_system", "AgentType"]