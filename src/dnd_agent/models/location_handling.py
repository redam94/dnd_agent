from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field


class Position(BaseModel):
    """Model for a position on a map"""

    x: float = Field(description="X coordinate")
    y: float = Field(description="Y coordinate")
    z: float = Field(default=0.0, description="Z coordinate (elevation)")
    map_id: str = Field(description="ID of the map this position is on")


class MapLocation(BaseModel):
    """Model for a location with spatial properties"""

    name: str = Field(description="Name of the location")
    description: str = Field(description="Detailed description of the location")
    location_type: str = Field(description="Type (room, outdoor, dungeon, city, etc.)")
    size: Tuple[float, float] = Field(description="Size in feet (width, height)")
    terrain: str = Field(default="normal", description="Terrain type (difficult, normal, etc.)")
    lighting: str = Field(default="bright", description="Lighting conditions")
    features: List[str] = Field(default_factory=list, description="Notable features")
    connections: List[str] = Field(default_factory=list, description="Connected location names")


class EntityPosition(BaseModel):
    """Model for tracking entity position"""

    entity_name: str
    entity_type: str
    position: Position


class BattleMap(BaseModel):
    """Model for a battle/tactical map"""

    map_id: str
    name: str
    description: str
    grid_size: int = Field(default=5, description="Grid size in feet (usually 5 for D&D)")
    width: int = Field(description="Width in grid squares")
    height: int = Field(description="Height in grid squares")
    terrain_features: List[Dict[str, Any]] = Field(default_factory=list)
