"""
Tools package.

This package contains a collection of asynchronous utility functions
(``tools``) that are registered on the pydantic‑ai agents.  Each module
is responsible for a specific domain of the game (e.g. graph database,
spatial reasoning, D&D resource lookup, campaign storage).  Grouping
tools by domain makes it easier to find and maintain them and
eliminates the monolithic ``tools.py`` file.

Functions defined here are thin wrappers over the underlying database
managers and API clients.  They are designed to be safe to call
concurrently and to be invoked from within the agent context.

Importing from this module will re‑export the public tool functions
from each sub‑module for convenience, e.g.:

    >>> from dnd_agent.tools import store_campaign_entity, calculate_distance
"""

from .graph import (
    store_campaign_entity,
    create_campaign_relationship,
    query_campaign_graph,
    check_database_status,
    list_entities_of_type,
)
from .dnd_resources import lookup_dnd_resource
from .spatial import (
    create_map_location,
    set_entity_position,
    calculate_distance,
    get_entities_in_range,
    generate_scene_description,
    connect_locations,
)
from .campaign_storage import (
    save_campaign_info,
    search_campaign_info,
    recall_chat_history,
)

__all__ = [
    # graph
    "store_campaign_entity",
    "create_campaign_relationship",
    "query_campaign_graph",
    "check_database_status",
    "list_entities_of_type",
    # dnd resources
    "lookup_dnd_resource",
    # spatial
    "create_map_location",
    "set_entity_position",
    "calculate_distance",
    "get_entities_in_range",
    "generate_scene_description",
    "connect_locations",
    # campaign storage
    "save_campaign_info",
    "search_campaign_info",
    "recall_chat_history",
]