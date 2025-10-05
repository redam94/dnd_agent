"""
Memory agent.

Handles long‑term storage and recall of campaign information.  This
includes saving plot developments, NPC backgrounds, location lore,
quest details and entire chat histories.  It exposes tools backed
by a vector database and semantic search engine.
"""

from __future__ import annotations

from typing import Any, Dict

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

from .base import BaseAgent
from dnd_agent.tools import save_campaign_info, search_campaign_info, recall_chat_history


class MemoryAgent(BaseAgent):
    """Agent responsible for campaign memory and lore storage."""

    def __init__(self, deps: Any, model_config: Dict[str, Any]):
        super().__init__(name="Memory Agent", deps=deps, model_config=model_config)

    def _create_agent(self) -> Agent:
        model = OpenAIChatModel(self.model_config["model_name"])
        agent = Agent(
            model,
            deps_type=type(self.deps),
            system_prompt=(
                "You manage the campaign's long‑term memory.\n"
                "Use the tools to save and retrieve notes, lore and chat history so that events are coherent over time.\n"
            ),
            retries=2,
        )
        agent.tool(save_campaign_info, name="save_info", description="Save campaign information to the vector database.")
        agent.tool(search_campaign_info, name="search_info", description="Search campaign information by semantic query.")
        agent.tool(recall_chat_history, name="recall_chat", description="Retrieve or search chat history for a campaign.")
        return agent