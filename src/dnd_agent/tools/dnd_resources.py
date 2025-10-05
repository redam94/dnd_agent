"""
D&D 5e resource lookup tools.

These helpers wrap the `DnD5eAPIClient` to retrieve rules,
monsters, spells, classes, equipment and other data from an
external API.  The client is created on the fly for each call and
closed afterwards to avoid leaking network resources.

If an exception occurs, a user‑friendly error message is returned
instead of raising.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from pydantic_ai import RunContext

from dnd_agent.database.api_client import DnD5eAPIClient
from dnd_agent.models.agent_deps import CampaignDeps


async def lookup_dnd_resource(
    ctx: RunContext[CampaignDeps],
    resource_type: str,
    resource_index: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> str:
    """Look up a D&D 5e resource.

    :param resource_type: The category to query (e.g. "spells", "monsters", "classes")
    :param resource_index: Optional specific index (e.g. "fireball", "goblin").  If omitted
        the full list for the category is returned.  Filters may be provided
        for list queries (e.g. {"level": [1, 2]} for spells).
    :param filters: Optional filtering parameters for list queries.
    :returns: A JSON formatted string describing the requested resource or an error message.
    """
    try:
        
        client = DnD5eAPIClient(base_url=ctx.deps.dnd_api_base)
        if resource_index:
            resource_index = resource_index.lower().replace(" ", "-")
            result = client.get_resource(resource_type, resource_index, params=filters)
        else:
            result = client.get_resource(resource_type, params=filters)
        client.close()
        result = json.dumps(result, indent=2, default=str)
        
        return result
    except Exception as exc:  # noqa: BLE001
        print(f"Error looking up D&D resource: {exc}")
        return f"Error looking up D&D resource: {exc}"