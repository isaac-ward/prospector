from __future__ import annotations

from collections import deque
from typing import Callable, Iterable, List, Sequence, Tuple, Optional

import heapq

import numpy as np
from tqdm import tqdm


Index = Tuple[int, ...]

def flood_fill_reachable(
    occupancy: np.ndarray,
    start_idx: Sequence[int],
    neighbors: Sequence[Sequence[int]],
    *,
    free_value: int = 0,
    tqdm_desc: str = "Flood fill (progress wrt upper bound)",
) -> np.ndarray:
    """
    Generic N-D BFS flood-fill of reachable free cells.
    """
    if occupancy.ndim != len(start_idx):
        raise ValueError(
            f"start_idx length {len(start_idx)} does not match "
            f"occupancy.ndim={occupancy.ndim}"
        )

    start = tuple(int(i) for i in start_idx)
    reachable = np.zeros_like(occupancy, dtype=bool)

    shape = occupancy.shape

    def in_bounds(idx: Index) -> bool:
        return all(0 <= i < s for i, s in zip(idx, shape))

    if not in_bounds(start):
        raise ValueError(
            f"start_idx={start} is out of bounds for grid shape={shape}"
        )

    if occupancy[start] != free_value:
        raise ValueError(
            f"start_idx={start} is not free (occupancy={occupancy[start]})."
        )

    # Upper bound on how many voxels we might ever visit
    total_free = int(np.count_nonzero(occupancy == free_value))

    q: deque[Index] = deque()
    q.append(start)
    reachable[start] = True

    pbar = tqdm(total=total_free, desc=tqdm_desc, unit="vox")

    visited_count = 0

    while q:
        current = q.popleft()
        visited_count += 1
        pbar.update(1)

        for offset in neighbors:
            nxt = tuple(c + dc for c, dc in zip(current, offset))
            if not in_bounds(nxt):
                continue
            if reachable[nxt]:
                continue
            if occupancy[nxt] != free_value:
                continue
            reachable[nxt] = True
            q.append(nxt)

    # If reachable < total_free, the bar won't be full, which is fine.
    # Optionally, you could pbar.n = visited_count; pbar.refresh()
    pbar.close()
    return reachable


def inflate_obstacles_nd(
    occupancy: np.ndarray,
    voxel_size: float,
    keep_out_radius: float,
    *,
    obstacle_value: int = 1,
    tqdm_desc: str = "Inflating obstacles",
) -> np.ndarray:
    """
    Inflate obstacles in an N-D occupancy grid by a keep-out radius.

    Parameters
    ----------
    occupancy : np.ndarray
        Integer occupancy grid. Cells with value == obstacle_value are treated
        as obstacles.
    voxel_size : float
        Size of a voxel in world units (assumed isotropic across dimensions).
    keep_out_radius : float
        Keep-out radius in world units.
    obstacle_value : int, optional
        Occupancy value considered as obstacle (default: 1).
    tqdm_desc : str, optional
        Description for tqdm progress bar.

    Returns
    -------
    inflated : np.ndarray of bool
        True where inflated obstacles occupy space.
    """
    if keep_out_radius <= 0.0:
        return np.zeros_like(occupancy, dtype=bool)

    r_vox = int(np.ceil(keep_out_radius / float(voxel_size)))
    if r_vox <= 0:
        return np.zeros_like(occupancy, dtype=bool)

    ndim = occupancy.ndim
    shape = occupancy.shape

    obstacle_mask = occupancy == obstacle_value
    inflated = np.zeros_like(obstacle_mask, dtype=bool)

    # Precompute N-D offsets inside a radius r_vox (Euclidean sphere in index space).
    ranges = [range(-r_vox, r_vox + 1) for _ in range(ndim)]
    offsets: List[Index] = []
    for idx_offset in np.ndindex(*([2 * r_vox + 1] * ndim)):
        offset = tuple(o - r_vox for o in idx_offset)
        if sum(o * o for o in offset) <= r_vox * r_vox:
            offsets.append(offset)

    obs_indices = np.argwhere(obstacle_mask)

    print(
        f"[grid_ops] Inflating obstacles: keep_out_radius={keep_out_radius:.3f}, "
        f"r_vox={r_vox}, ndim={ndim}, num_obstacles={len(obs_indices)}"
    )

    def in_bounds(idx: Index) -> bool:
        return all(0 <= i < s for i, s in zip(idx, shape))

    for base in tqdm(obs_indices, desc=tqdm_desc, unit="obs"):
        base_idx = tuple(int(i) for i in base)
        for off in offsets:
            nbr = tuple(b + o for b, o in zip(base_idx, off))
            if in_bounds(nbr):
                inflated[nbr] = True

    return inflated


def grid_astar(
    start_idx: Sequence[int],
    goal_idx: Sequence[int],
    in_bounds_fn: Callable[[Index], bool],
    is_free_fn: Callable[[Index], bool],
    neighbors: Sequence[Sequence[int]],
    *,
    tqdm_desc: str = "A* search",
) -> Optional[List[Index]]:
    """
    Generic N-D A* search on a grid.

    Cost:
        - Unit cost per neighbor step.
    Heuristic:
        - Manhattan distance in index space.

    Parameters
    ----------
    start_idx : sequence of ints
    goal_idx : sequence of ints
    in_bounds_fn : callable
        in_bounds_fn(idx: Index) -> bool
    is_free_fn : callable
        is_free_fn(idx: Index) -> bool
    neighbors : sequence of integer offset sequences
    tqdm_desc : str, optional
        Description for tqdm progress bar.

    Returns
    -------
    path : list of Index or None
        Path from start to goal (inclusive) as list of index tuples, or None
        if no path is found.
    """
    start = tuple(int(i) for i in start_idx)
    goal = tuple(int(i) for i in goal_idx)

    if not in_bounds_fn(start):
        raise ValueError(f"start_idx={start} out of bounds.")
    if not in_bounds_fn(goal):
        raise ValueError(f"goal_idx={goal} out of bounds.")
    if not is_free_fn(start):
        raise ValueError(f"start_idx={start} is not free in planning grid.")
    if not is_free_fn(goal):
        raise ValueError(f"goal_idx={goal} is not free in planning grid.")

    def heuristic(a: Index, b: Index) -> float:
        return float(sum(abs(ai - bi) for ai, bi in zip(a, b)))

    open_set: List[Tuple[float, Index]] = []
    heapq.heappush(open_set, (0.0, start))

    came_from: dict[Index, Index] = {}
    g_score: dict[Index, float] = {start: 0.0}

    pbar = tqdm(total=None, desc=tqdm_desc, unit="exp")

    while open_set:
        _, current = heapq.heappop(open_set)
        pbar.update(1)

        if current == goal:
            pbar.close()
            # Reconstruct path
            path: List[Index] = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        current_g = g_score[current]

        for off in neighbors:
            nbr = tuple(c + dc for c, dc in zip(current, off))
            if not in_bounds_fn(nbr):
                continue
            if not is_free_fn(nbr):
                continue

            tentative_g = current_g + 1.0
            if tentative_g < g_score.get(nbr, float("inf")):
                came_from[nbr] = current
                g_score[nbr] = tentative_g
                f_score = tentative_g + heuristic(nbr, goal)
                heapq.heappush(open_set, (f_score, nbr))

    pbar.close()
    return None
