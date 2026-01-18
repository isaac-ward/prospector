# src/prospector/blender_integration/types.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple


@dataclass
class AgentTrajectory:
    agent_id: str
    positions: List[Tuple[float, float, float]]   # (x,y,z) per step
    rotations: List[Tuple[float, float, float]]   # (rx,ry,rz) per step (radians)
    actions: List[Any]                            # raw actions (whatever shape)
