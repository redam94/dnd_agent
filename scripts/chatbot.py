"""
Example script for demonstrating the functionality of the refactored D&D
multi‑agent system.

This script shows how to initialise the campaign system, create a dummy
set of dependencies, and use the orchestrator to perform a couple of
example actions:

1. Look up a spell using the RulesAgent via the orchestrator.
2. Create a new character entity via the EntityAgent.

Note: This example uses a ``DummyDeps`` class to satisfy the expected
dependency interface. In a real application you should supply
``CampaignDeps`` from ``dnd_agent.models.agent_deps`` with proper
database connections and API endpoints configured.
"""

import asyncio
from dnd_agent import create_campaign_system, AgentType
from dnd_agent.agents.base import AgentRequest


class DummyDeps:
    """A minimal dependency object for demonstration purposes."""

    def __init__(self):
        # Base URL for the open DnD 5e API. Adjust as needed.
        self.dnd_api_base: str = "https://www.dnd5eapi.co"
        # No PostgreSQL connection configured in this example.
        self.postgres_conn = None
        # Placeholder for campaign context. Can store session-specific data.
        self.current_context: dict = {}


async def main() -> None:
    # Instantiate dependencies and model config
    from rich.console import Console
    from rich.markdown import Markdown
    console = Console()
    deps = DummyDeps()
    model_config = {"model_name": "gpt-4o"}

    # Build the campaign system
    orchestrator = create_campaign_system(deps, model_config)
    # Create an AgentRequest for the orchestrator
    request = AgentRequest(
        agent_type=AgentType.ORCHESTRATOR,
        action="handle_user_input",
        parameters={},
    )
    while True:
        user_input = input("\nEnter your command (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting the chatbot. Goodbye!")
            break

        # Create an AgentRequest for the orchestrator
        request.parameters["user_input"] = user_input
        request.parameters['previous_responses'] = []
        # Process the request using the orchestrator
        response = await orchestrator.process(request)
        request.parameters['previous_responses'].append({"user_input": user_input, "response": response.message})
        # Print the response message
        console.print("\nResponse from Orchestrator:")
        console.print(Markdown(response.message))


if __name__ == "__main__":
    asyncio.run(main())