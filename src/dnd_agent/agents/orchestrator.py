"""
Orchestrator agent.

The :class:`DMOrchestratorAgent` delegates high‑level user requests to
specialised agents.  It understands which domain each sub‑task belongs
to and coordinates their execution.  The orchestrator itself is also a
pydantic‑ai agent with its own system prompt and registered tools for
delegation and response aggregation.

Sub‑agents must be supplied via the ``sub_agents`` mapping at
construction time.  Each key should be an :class:`AgentType` and
each value a concrete instance of :class:`BaseAgent` (or subclass).
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel

from .base import AgentType, BaseAgent, AgentRequest, AgentResponse


class DMOrchestratorAgent(BaseAgent):
    """Master orchestrator for the D&D campaign.

    This agent receives high‑level user instructions and uses its
    registered tools to delegate to sub‑agents, combine their
    responses and query the overall game state.  A minimal prompt
    describes its responsibilities and instructs it on how to route
    tasks.
    """

    def __init__(self, deps: Any, model_config: Dict[str, Any], sub_agents: Dict[AgentType, BaseAgent]):
        self.sub_agents = sub_agents
        super().__init__(name="DM Orchestrator", deps=deps, model_config=model_config)

    def _create_agent(self) -> Agent:
        # Initialise the language model
        model = OpenAIChatModel(self.model_config["model_name"])
        agent = Agent(
            model,
            deps_type=type(self.deps),
            system_prompt=(
                "You are the master orchestrator for a D&D 5e campaign.\n"
                "Your responsibilities:\n"
                "1. Understand player intentions and DM requests\n"
                "2. Break down complex actions into sub‑tasks\n"
                "3. Delegate to specialised agents for rules, spatial, entity, combat, narrative and memory\n"
                "4. Combine responses into a cohesive narrative and maintain game state."
            ),
            retries=2,
        )

        # Register orchestration tools.  These wrappers simply forward
        # calls to the corresponding instance methods.
        agent.tool(self.delegate_to_agent, retries=3)
        agent.tool(self.combine_responses, retries=3)
        agent.tool(self.get_game_state, retries=3)
        return agent

    async def delegate_to_agent(self, ctx: RunContext, agent_type: AgentType, action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate a task to a specialised agent.

        The orchestrator looks up the appropriate sub‑agent and passes the
        action and parameters through.  It returns a simplified
        dictionary capturing success, data and message.
        """
        agent = self.sub_agents.get(agent_type)
        if agent is None:
            return {"success": False, "data": None, "message": f"Unknown agent type: {agent_type}"}
        request = AgentRequest(agent_type=agent_type, action=action, parameters=parameters, context=ctx.deps.current_context if hasattr(ctx.deps, "current_context") else None)
        response: AgentResponse = await agent.process(request)
        return {
            "success": response.success,
            "data": response.data,
            "message": response.message,
        }

    async def combine_responses(self, ctx: RunContext, responses: List[Dict[str, Any]]) -> str:
        """Combine multiple responses into a single narrative string."""
        parts: List[str] = []
        for resp in responses:
            if resp.get("success"):
                parts.append(resp.get("message", ""))
        return "\n\n".join(parts)

    async def get_game_state(self, ctx: RunContext) -> Dict[str, Any]:
        """Collect state snapshots from all sub‑agents that expose a ``get_state`` method."""
        state: Dict[str, Any] = {}
        for atype, agent in self.sub_agents.items():
            if hasattr(agent, "get_state"):
                state[atype.value] = await agent.get_state()  # type: ignore[func-returns-value]
        return state