# src/prospector/blender_integration/trajectories.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

from .types import AgentTrajectory
from .paths import infer_agent_id
from .rotations import compute_rotations_from_positions, smooth_euler_rotations_quat_window


def load_agent_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Trajectory json not found: {path}")
    with path.open("r") as f:
        return json.load(f)


def parse_states_actions(payload: Dict[str, Any]) -> Tuple[List[List[float]], List[Any]]:
    if "states" not in payload:
        raise KeyError("Trajectory JSON missing key: 'states'")
    states = payload["states"]
    actions = payload.get("actions", [])
    if not isinstance(states, list):
        raise TypeError(f"'states' must be a list, got {type(states)}")
    return states, actions


def load_agent_trajectory(
    json_path: Path,
    *,
    smooth_positions: bool = True,
    smooth_positions_factor: int = 16,
    smooth_rotations: bool = True,
    smooth_rotations_factor: int = 8,
) -> AgentTrajectory:
    payload = load_agent_json(json_path)
    states, actions = parse_states_actions(payload)

    positions: List[Tuple[float, float, float]] = []
    rotations: List[Tuple[float, float, float]] = []

    for s in states:
        if not isinstance(s, (list, tuple)):
            raise TypeError(f"Each state must be a list/tuple, got {type(s)} in {json_path}")
        if len(s) < 3:
            raise ValueError(f"State must have >=3 elems for position, got len={len(s)} in {json_path}")

        x, y, z = float(s[0]), float(s[1]), float(s[2])
        positions.append((x, y, z))

        if len(s) >= 6:
            rx, ry, rz = float(s[3]), float(s[4]), float(s[5])
            rotations.append((rx, ry, rz))

    if smooth_positions:
        smoothed_positions: List[Tuple[float, float, float]] = []
        n = len(positions)
        for i in range(n):
            count = 0
            sx = sy = sz = 0.0
            for j in range(max(0, i - smooth_positions_factor), min(n, i + smooth_positions_factor + 1)):
                px, py, pz = positions[j]
                sx += px
                sy += py
                sz += pz
                count += 1
            smoothed_positions.append((sx / count, sy / count, sz / count))
        positions = smoothed_positions

    if len(rotations) != len(positions):
        rotations = compute_rotations_from_positions(positions)

    if smooth_rotations:
        rotations = smooth_euler_rotations_quat_window(rotations, window=int(smooth_rotations_factor))

    return AgentTrajectory(
        agent_id=infer_agent_id(json_path),
        positions=positions,
        rotations=rotations,
        actions=actions,
    )
