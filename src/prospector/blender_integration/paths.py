# src/prospector/blender_integration/paths.py

from __future__ import annotations

from pathlib import Path
from typing import List

import bpy


def find_repo_root_from_blend() -> Path:
    """
    Find repo root by walking upward from the directory containing the open .blend.
    Marker: src/prospector exists.
    """
    if not bpy.data.filepath:
        raise RuntimeError("Blend file must be saved to locate repo root relative to it.")

    blend_dir = Path(bpy.data.filepath).resolve().parent
    for p in (blend_dir,) + tuple(blend_dir.parents):
        if (p / "src" / "prospector").exists():
            return p
    raise RuntimeError(f"Could not find repo root (expected src/prospector) above: {blend_dir}")


def resolve_log_folder(repo_root: Path, log_folder: str) -> Path:
    """
    Resolve a log folder path.

    Supports:
      - absolute path: returned as-is (must exist)
      - relative run name: we search in:
          1) <repo_root>/saved/<log_folder>
          2) <repo_root>/logs/<log_folder>

    Returns the first existing match.
    """
    lf = Path(log_folder)

    if lf.is_absolute():
        if not lf.exists():
            raise FileNotFoundError(f"log_folder absolute path does not exist: {lf}")
        return lf

    candidates = [
        repo_root / "saved" / log_folder,
        repo_root / "logs" / log_folder,
    ]

    for c in candidates:
        if c.exists():
            return c

    tried = "\n  - " + "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not resolve log_folder='{log_folder}'. Tried:{tried}")


def find_agent_jsons(log_folder: Path) -> List[Path]:
    traj_dir = log_folder / "trajectories"
    if not traj_dir.exists():
        raise FileNotFoundError(f"Expected trajectories dir not found: {traj_dir}")

    jsons = sorted(traj_dir.glob("agent_*.json"))
    if not jsons:
        raise FileNotFoundError(f"No agent_*.json found in: {traj_dir}")
    return jsons


def infer_agent_id(json_path: Path) -> str:
    name = json_path.stem
    if name.startswith("agent_"):
        return name[len("agent_") :]
    return name
