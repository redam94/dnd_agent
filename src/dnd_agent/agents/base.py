"""
Base definitions for agents.

This module defines the common types and base class used by the
dnd_agent multi‑agent system.  It centralises the agent taxonomy and
request/response objects so that the rest of the code can import from
here rather than redefining these primitives in multiple places.

The :class:`BaseAgent` implements a uniform interface for dispatching
requests to a pydantic‑ai Agent.  Sub‑classes should implement
``_create_agent`` to construct and configure the underlying language
model and register any tools on it.  The :meth:`process` method wraps
the language model execution and returns a structured response.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from pydantic_ai import Agent, RunContext


class AgentType(Enum):
    """Enumeration of the supported agent domains.

    The orchestrator uses this to determine which sub‑agent should
    handle a particular request.  Keep this enum in sync with the
    concrete agent classes defined under :mod:`dnd_agent.agents`.
    """

    ORCHESTRATOR = "orchestrator"
    RULES = "rules"
    SPATIAL = "spatial"
    ENTITY = "entity"
    COMBAT = "combat"
    NARRATIVE = "narrative"
    MEMORY = "memory"


@dataclass
class AgentRequest:
    """A request for an agent to perform an action.

    :param agent_type: the intended recipient agent
    :param action: the tool name or action identifier
    :param parameters: arguments passed to the underlying tool
    :param context: optional additional metadata for the agent
    """

    agent_type: AgentType
    action: str
    parameters: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


@dataclass
class AgentResponse:
    """A response from an agent.

    :param agent_type: which agent produced the response
    :param success: whether the action was successful
    :param data: raw result from the language model or tool
    :param message: summarised message to present to the user
    :param metadata: optional auxiliary information (e.g. debug info)
    """

    agent_type: AgentType
    success: bool
    data: Any
    message: str
    metadata: Optional[Dict[str, Any]] = None


class BaseAgent:
    """Base class for specialised agents.

    Each specialised agent receives a set of dependencies (``deps``)
    and a model configuration.  The dependencies should encapsulate
    external services (databases, API clients, etc.) to allow easy
    substitution in tests.  Sub‑classes must override
    :meth:`_create_agent` to instantiate and configure a pydantic‑ai
    :class:`Agent` (typically registering tools and setting the
    system prompt).
    """

    def __init__(self, name: str, deps: Any, model_config: Dict[str, Any]):
        self.name = name
        self.deps = deps
        self.model_config = model_config
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent:
        """Create the underlying pydantic‑ai agent.

        Sub‑classes must implement this method to return a properly
        configured :class:`Agent` instance.  This base implementation
        raises :class:`NotImplementedError` to ensure derived classes
        override it.
        """
        raise NotImplementedError

    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process a request using the underlying language model.

        This wrapper calls the underlying pydantic‑ai agent's
        :meth:`~pydantic_ai.Agent.run` method with the given action and
        parameters.  Any exceptions raised by the model are captured
        and returned in the response as a failure.
        """
        try:
            result = await self.agent.run(request.action + "\nParameters:\n" + str(request.parameters), deps=self.deps)
            message = getattr(result, "output", str(result))
            return AgentResponse(
                agent_type=request.agent_type,
                success=True,
                data=result,
                message=message,
                metadata=None,
            )
        except Exception as exc:  # noqa: BLE001 broad exception capturing
            # Provide a structured failure response rather than bubbling
            return AgentResponse(
                agent_type=request.agent_type,
                success=False,
                data=None,
                message=str(exc),
                metadata={"error": repr(exc)},
            )