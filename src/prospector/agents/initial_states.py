# src/prospector/agents/initial_states.py

from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np


def sample_initial_states(
    *,
    cave_map,
    num_agents: int,
    state_dim: int,
    position_dims: Sequence[int],
    agent_radius: float,
    rng: np.random.Generator,
    min_distance_multiplier: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample initial (x,y[,z]) positions in free space from the cave_map and
    construct the full initial state matrix for the dynamics model.

    Parameters
    ----------
    cave_map : CaveMap2D or CaveMap3D
        Must provide sample_free_points(num_points, min_distance, rng).

    num_agents : int
        Number of agents.

    state_dim : int
        Dimensionality of the full state for the chosen dynamics model.

    position_dims : list[int] or tuple[int]
        Which state indices represent spatial position (len 2 or 3).

    agent_radius : float
        Used to enforce min spacing between sampled positions.

    rng : np.random.Generator
        Random number generator.

    min_distance_multiplier : float, optional
        Default 4.0 → enforces min_distance = agent_radius * 4.

    Returns
    -------
    initial_state : np.ndarray, shape (num_agents, state_dim)
    start_points  : np.ndarray, shape (num_agents, 2 or 3)
    """
    # ------------------------------------------------------ #
    # Sample free-space world positions
    # ------------------------------------------------------ #
    min_distance = agent_radius * float(min_distance_multiplier)

    start_points = cave_map.sample_free_points(
        num_points=num_agents,
        min_distance=min_distance,
        rng=rng,
    )  # (N, 2) or (N, 3)

    # ------------------------------------------------------ #
    # Build the full initial state array
    # ------------------------------------------------------ #
    initial_state = np.zeros((num_agents, state_dim), dtype=np.float32)

    for i in range(num_agents):
        initial_state[i, position_dims] = start_points[i]

    return initial_state, start_points
