"""Agent dependencies dataclasses
=================================

All dataclasses related to dependencies for the D&D campaign agent.
"""
from typing import Any, Optional
from dataclasses import dataclass

@dataclass
class CampaignDeps:
    """Dependencies for the D&D campaign agent"""
    neo4j_driver: Any
    postgres_conn: Optional[Any] = None
    dnd_api_base: str = "https://www.dnd5eapi.co"
