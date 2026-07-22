from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, List, Union

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless / non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from prospector.caves.cave_map_2d import CaveMap2D
from prospector.caves.cave_map_3d import CaveMap3D


CaveMapType = Union[CaveMap2D, CaveMap3D]


@dataclass
class CaveGraph:
    """
    Lightweight container for a cave graph.

    Attributes
    ----------
    node_positions : (N, D) array
        World coordinates of each node (D = 2 or 3).
    edges : (M, 2) int array
        Undirected edges as pairs of node indices (i, j) with i < j.
    A : (N, N) int array
        Symmetric communication matrix (1 on diag; off-diagonal 1 if LoS, 0 otherwise).
    min_adjacent_distance : float
        Smallest edge-length threshold such that the adjacency graph is connected,
        under the constraint that edges require line-of-sight.
    """
    node_positions: np.ndarray
    edges: np.ndarray
    A: np.ndarray
    min_adjacent_distance: float


# ---------------------------------------------------------------------- #
# Node placement: "maximally filling" the free space, LOS-connected      #
# ---------------------------------------------------------------------- #

def _get_free_voxel_centers(
    cave_map: CaveMapType,
    *,
    max_candidates: int = 100_000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Collect voxel centers for free-navigable cells (label 0).

    To keep things tractable on large maps, we subsample at most `max_candidates`
    voxel centers uniformly at random.
    """
    if rng is None:
        rng = np.random.default_rng()

    free_mask = cave_map.occupancy == 0
    free_indices = np.argwhere(free_mask)

    if free_indices.shape[0] == 0:
        raise ValueError("No free-navigable voxels (label == 0) found in cave map.")

    if free_indices.shape[0] > max_candidates:
        sel = rng.choice(free_indices.shape[0], size=max_candidates, replace=False)
        free_indices = free_indices[sel]

    centers = np.stack(
        [cave_map.index_to_world_center(idx) for idx in free_indices],
        axis=0,
    )
    return centers


def _farthest_point_sampling_los_connected(
    cave_map: CaveMapType,
    candidates: np.ndarray,
    num_points: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Greedy farthest-point sampling on a discrete candidate set, with the
    additional constraint that the resulting node set is LOS-connected.

    Implementation
    --------------
    - Start from a roughly central seed (geometric center of free space).
    - Maintain a set of selected nodes S.
    - Maintain a visibility mask: for each candidate c, whether there is
      line-of-sight to at least one node in S.
    - At each step:
        * restrict to candidates that are visible (LoS to S) and not selected
        * among those, choose the one that maximizes the minimum distance to S
    - If at any step no visible, unselected candidates exist, we cannot
      place more nodes while preserving LOS-connectivity, and we raise.

    This guarantees that the induced LOS graph over the sampled nodes is connected.

    Progress
    --------
    - Uses verbose line-of-sight checks (tqdm from the CaveMap LoS implementation).
    - Prints a short status line after each node is placed.
    """
    if rng is None:
        rng = np.random.default_rng()

    candidates = np.asarray(candidates, dtype=np.float64)
    M, D = candidates.shape

    if num_points > M:
        raise ValueError(
            f"Requested {num_points} nodes but only {M} free candidates available."
        )

    # ------------------------------------------------------------------ #
    # Seed: candidate closest to geometric center                        #
    # ------------------------------------------------------------------ #
    center = candidates.mean(axis=0)
    d2_center = np.sum((candidates - center[None, :]) ** 2, axis=1)
    first = int(np.argmin(d2_center))

    selected_indices: List[int] = [first]
    selected_mask = np.zeros(M, dtype=bool)
    selected_mask[first] = True

    # Distances to nearest selected node: initialize with seed.
    diff0 = candidates - candidates[first][None, :]
    min_d2 = np.sum(diff0 ** 2, axis=1)

    # Initial visibility: LOS from seed to all candidates.
    print("[cave_to_graph] LOS pre-pass from seed to all candidates ...")
    p_seed = np.broadcast_to(candidates[first][None, :], candidates.shape)
    p_all = candidates
    los_to_seed = cave_map.batch_has_line_of_sight(p_seed, p_all, verbose=True)
    visible_mask = los_to_seed.copy()
    visible_mask[first] = True  # the seed is trivially in the connected component

    print(
        f"[cave_to_graph] Initial seed selected (idx={first}), "
        f"visible candidates: {visible_mask.sum()} / {M}"
    )

    # ------------------------------------------------------------------ #
    # Iteratively add farthest visible nodes                             #
    # ------------------------------------------------------------------ #
    while len(selected_indices) < num_points:
        k = len(selected_indices)

        candidate_mask = (~selected_mask) & visible_mask

        if not np.any(candidate_mask):
            # No more nodes can be added while preserving LOS-connectivity.
            raise ValueError(
                f"Cannot place {num_points} LOS-connected nodes; "
                f"only {len(selected_indices)} nodes are mutually visible "
                f"from the current seed and its descendants."
            )

        # Among currently visible, unselected candidates, choose the one with
        # largest min distance to the selected set.
        idx_candidates = np.where(candidate_mask)[0]
        best_local = idx_candidates[np.argmax(min_d2[idx_candidates])]

        selected_indices.append(best_local)
        selected_mask[best_local] = True

        # Update min_d2 with this new node.
        diff_new = candidates - candidates[best_local][None, :]
        d2_new = np.sum(diff_new ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2_new)

        # Update visibility: any candidate that has LOS to this new node
        # becomes visible (connected). Only check currently non-visible ones
        # to avoid redundant LoS work.
        not_visible_mask = ~visible_mask
        if np.any(not_visible_mask):
            idx_not_vis = np.where(not_visible_mask)[0]
            print(
                f"[cave_to_graph] Node {k+1}/{num_points}: LOS from new node "
                f"(idx={best_local}) to {idx_not_vis.size} not-yet-visible candidates ..."
            )
            p_new = np.broadcast_to(
                candidates[best_local][None, :], (idx_not_vis.size, D)
            )
            p_targets = candidates[idx_not_vis]
            los_to_new_sub = cave_map.batch_has_line_of_sight(
                p_new,
                p_targets,
                verbose=True,
            )
            visible_mask[idx_not_vis] |= los_to_new_sub

        print(
            f"[cave_to_graph] Placed node {k+1}/{num_points} "
            f"(idx={best_local}); visible candidates: {visible_mask.sum()} / {M}"
        )

    return candidates[selected_indices, :]


def compute_node_positions(
    cave_map: CaveMapType,
    num_nodes: int,
    *,
    rng: Optional[np.random.Generator] = None,
    max_candidates: int = 100_000,
) -> np.ndarray:
    """
    Compute 'maximally filling' node positions in the free-navigable region,
    subject to LOS-connectivity.

    Implementation
    --------------
    - Sample voxel centers for all free-navigable cells (label 0), subsampling
      if needed.
    - Run a LOS-constrained greedy farthest-point sampler to select `num_nodes`
      points that:
        * spread over the free region, and
        * form a connected graph under the line-of-sight relation.

    If the requested `num_nodes` cannot be placed while preserving LOS-connectivity
    (given the map geometry and voxelization), this raises a ValueError.

    Returns
    -------
    node_positions : (num_nodes, D) array
        World coordinates (D=2 or 3).
    """
    centers = _get_free_voxel_centers(
        cave_map,
        max_candidates=max_candidates,
        rng=rng,
    )
    nodes = _farthest_point_sampling_los_connected(
        cave_map,
        centers,
        num_points=num_nodes,
        rng=rng,
    )
    return nodes


# ---------------------------------------------------------------------- #
# Line-of-sight and adjacency / communication matrix                     #
# ---------------------------------------------------------------------- #

def _compute_pairwise_los_and_dist(
    cave_map: CaveMapType,
    node_positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise line-of-sight and distances between nodes.

    Returns
    -------
    los_mat : (N, N) bool array
        los_mat[i, j] is True iff LoS exists (i != j). Diagonal is False.
    dist_mat : (N, N) float array
        Symmetric Euclidean distances; diagonal is 0.
    """
    nodes = np.asarray(node_positions, dtype=np.float64)
    N, D = nodes.shape

    # Upper-triangular pairs
    iu, ju = np.triu_indices(N, k=1)
    p0 = nodes[iu]
    p1 = nodes[ju]

    # LoS using the cave map
    print("[cave_to_graph] Pairwise LoS between all node pairs ...")
    los_pairs = cave_map.batch_has_line_of_sight(p0, p1, verbose=True)

    # Distances
    dist_pairs = np.linalg.norm(p1 - p0, axis=1)

    los_mat = np.zeros((N, N), dtype=bool)
    dist_mat = np.zeros((N, N), dtype=np.float64)

    los_mat[iu, ju] = los_pairs
    los_mat[ju, iu] = los_pairs

    dist_mat[iu, ju] = dist_pairs
    dist_mat[ju, iu] = dist_pairs

    # Diagonal: no self-LoS, zero distance
    np.fill_diagonal(los_mat, False)
    np.fill_diagonal(dist_mat, 0.0)

    return los_mat, dist_mat


def _is_connected(num_nodes: int, edges: np.ndarray) -> bool:
    """
    Check graph connectivity via DFS/BFS on an undirected graph.
    """
    if num_nodes == 0:
        return False
    if num_nodes == 1:
        return True

    adjacency = [[] for _ in range(num_nodes)]
    for i, j in edges:
        adjacency[i].append(j)
        adjacency[j].append(i)

    visited = np.zeros(num_nodes, dtype=bool)
    stack = [0]
    visited[0] = True

    while stack:
        v = stack.pop()
        for u in adjacency[v]:
            if not visited[u]:
                visited[u] = True
                stack.append(u)

    return bool(visited.all())


def _compute_min_adjacent_distance(
    los_mat: np.ndarray,
    dist_mat: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Find the smallest distance threshold such that the adjacency graph
    (edges require LOS AND distance <= threshold) is connected.

    Assumes that the LOS graph itself is connected, i.e. there exists at least
    one spanning tree using only line-of-sight edges. Under that assumption,
    this procedure must find some threshold at or below the maximum LOS edge
    length that yields a connected graph.

    Returns
    -------
    min_adjacent_distance : float
        Chosen radius threshold.
    edges : (M, 2) int array
        Final edges under this threshold.
    """
    N = los_mat.shape[0]
    if N <= 1:
        return 0.0, np.zeros((0, 2), dtype=np.int32)

    # Consider only pairs with LoS.
    iu, ju = np.triu_indices(N, k=1)
    mask_los = los_mat[iu, ju]

    if not np.any(mask_los):
        raise RuntimeError(
            "No candidate edges with line-of-sight exist between any node pairs; "
            "this should not happen if nodes were sampled with LOS-connectivity."
        )

    dists = dist_mat[iu, ju]
    dists_los = dists[mask_los]
    unique_dists = np.unique(dists_los)

    best_threshold: Optional[float] = None
    best_edges: Optional[np.ndarray] = None

    # Monotone search over thresholds: smallest that yields connectivity.
    print("[cave_to_graph] Searching minimal adjacency distance for connectivity ...")
    for thr in unique_dists:
        mask_edges = mask_los & (dists <= thr)
        if not np.any(mask_edges):
            continue

        edge_i = iu[mask_edges]
        edge_j = ju[mask_edges]
        edges = np.stack([edge_i, edge_j], axis=1).astype(np.int32)

        if _is_connected(N, edges):
            best_threshold = float(thr)
            best_edges = edges
            break

    if best_threshold is None or best_edges is None:
        # This means even using all LOS edges does not form a connected graph.
        # That contradicts the LOS-connected sampling assumption, so treat as hard error.
        raise RuntimeError(
            "Even with all line-of-sight edges, the node graph remains disconnected. "
            "This should not happen if nodes were sampled with LOS-connectivity."
        )

    print(
        f"[cave_to_graph] Found min_adjacent_distance={best_threshold:.3f} "
        f"with {best_edges.shape[0]} edges."
    )
    return best_threshold, best_edges


def _augment_edges_with_extra_los(
    los_mat: np.ndarray,
    base_edges: np.ndarray,
    num_nodes: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Final step augmentation:

    If num_edges < num_nodes * 2, attempt to add up to
    (num_nodes * 2 - num_edges) extra edges between nodes that have
    a line of sight, with 50% probability per candidate extra edge.

    - Existing edges are preserved.
    - Only edges with LOS are considered.
    - Extra edges are NOT constrained by min_adjacent_distance; they can be
      longer-range LoS links.
    """
    E0 = base_edges.shape[0]
    target_edges = num_nodes * 2

    if E0 >= target_edges:
        print(
            f"[cave_to_graph] Edge augmentation skipped: "
            f"{E0} >= target target_edges={target_edges}."
        )
        return base_edges

    max_extra = target_edges - E0
    print(
        f"[cave_to_graph] Attempting to add up to {max_extra} extra LOS edges "
        f"(current_edges={E0}, target_edges={target_edges}) ..."
    )

    # Build a fast membership set of existing edges (i<j).
    existing = set()
    for i, j in base_edges:
        a, b = (int(i), int(j))
        if a > b:
            a, b = b, a
        existing.add((a, b))

    N = num_nodes
    iu, ju = np.triu_indices(N, k=1)
    mask_los = los_mat[iu, ju]

    # Candidate edges = LOS pairs not already in base_edges.
    cand_i = iu[mask_los]
    cand_j = ju[mask_los]

    candidate_pairs: List[Tuple[int, int]] = []
    for a, b in zip(cand_i, cand_j):
        aa, bb = int(a), int(b)
        if aa > bb:
            aa, bb = bb, aa
        if (aa, bb) not in existing:
            candidate_pairs.append((aa, bb))

    if not candidate_pairs:
        print("[cave_to_graph] No additional LOS pairs available for edge augmentation.")
        return base_edges

    candidate_pairs = np.array(candidate_pairs, dtype=np.int32)
    # Shuffle candidate order so we don't bias by index.
    perm = rng.permutation(candidate_pairs.shape[0])
    candidate_pairs = candidate_pairs[perm]

    extra_edges: List[Tuple[int, int]] = []
    for a, b in candidate_pairs:
        if len(extra_edges) >= max_extra:
            break
        # Add with probability 0.5
        if rng.random() < 0.75:
            extra_edges.append((int(a), int(b)))

    if not extra_edges:
        print("[cave_to_graph] No extra edges added (probabilistic skip).")
        return base_edges

    extra_edges_arr = np.array(extra_edges, dtype=np.int32)
    all_edges = np.concatenate([base_edges, extra_edges_arr], axis=0)

    print(
        f"[cave_to_graph] Extra edges added: {extra_edges_arr.shape[0]} "
        f"(final_edges={all_edges.shape[0]})."
    )
    return all_edges


# Edge-selection tuning for the shortest-biased graph (see _shortest_biased_edges).
#   EDGE_DISTANCE_QUANTILE : only LoS pairs shorter than this quantile of all LoS
#       pairwise distances are eligible as (non-MST) edges -> favours short links.
#   EDGE_TARGET_PER_NODE   : cap on total edges as a multiple of N (incl. the MST).
EDGE_DISTANCE_QUANTILE = 0.25
EDGE_TARGET_PER_NODE = 1.5


def _shortest_biased_edges(
    los_mat: np.ndarray,
    dist_mat: np.ndarray,
    num_nodes: int,
    *,
    quantile: float = EDGE_DISTANCE_QUANTILE,
    target_per_node: float = EDGE_TARGET_PER_NODE,
    max_edge_length: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Build a sparse, short-edge-biased traversibility graph.

    1. Minimum spanning tree over LoS edges (weighted by Euclidean distance) via
       Kruskal + union-find. This connects the graph with the shortest total edge
       length -- chaining short hops rather than a few long ones.
    2. Augment with additional LoS edges, shortest-first, but ONLY those below the
       `quantile` of all LoS pairwise distances, up to ceil(target_per_node * N)
       edges total. Gives local redundancy without reintroducing long links.

    A long edge appears only where the MST genuinely has no shorter way to reach a
    node (e.g. an isolated pocket).

    max_edge_length : optional
        If set, NO edge longer than this is ever added. Useful on porous/sparse
        maps where line-of-sight can leak through wall gaps and produce phantom
        long-range links. The result may then be a spanning FOREST rather than a
        tree: any node with no in-range neighbour is left (weakly) disconnected --
        an honest signal of a coverage gap -- instead of being bridged by a fake
        long edge.

    Returns
    -------
    edges : (M, 2) int array
    min_adjacent_distance : float
        Longest edge length in the returned graph (kept for API compatibility).
    """
    N = int(num_nodes)
    if N <= 1:
        return np.zeros((0, 2), dtype=np.int32), 0.0

    iu, ju = np.triu_indices(N, k=1)
    mask_los = los_mat[iu, ju]
    if not np.any(mask_los):
        raise RuntimeError(
            "No candidate edges with line-of-sight exist between any node pairs; "
            "this should not happen if nodes were sampled with LOS-connectivity."
        )

    ei = iu[mask_los]
    ej = ju[mask_los]
    ed = dist_mat[iu, ju][mask_los]

    # Optional hard cap on edge length (drop phantom long-range LoS on porous maps).
    if max_edge_length is not None:
        keep = ed <= float(max_edge_length)
        ei, ej, ed = ei[keep], ej[keep], ed[keep]
        if ed.size == 0:
            raise RuntimeError(
                f"No line-of-sight edges within max_edge_length={max_edge_length}."
            )

    # Sort candidate LoS edges by ascending distance (short edges first).
    order = np.argsort(ed, kind="stable")
    ei, ej, ed = ei[order], ej[order], ed[order]

    # --- Kruskal MST over LoS edges --------------------------------------- #
    parent = list(range(N))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    selected: List[Tuple[int, int]] = []
    selected_set = set()
    n_mst = 0
    for a, b in zip(ei, ej):
        a, b = int(a), int(b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            key = (a, b) if a < b else (b, a)
            selected.append(key)
            selected_set.add(key)
            n_mst += 1
            if n_mst == N - 1:
                break

    n_components = N - n_mst
    if n_mst < N - 1:
        if max_edge_length is None:
            raise RuntimeError(
                "Even with all line-of-sight edges, the node graph remains disconnected. "
                "This should not happen if nodes were sampled with LOS-connectivity."
            )
        # With a length cap a forest is expected: some nodes have no in-range
        # neighbour. Report it rather than fabricating a long bridge.
        print(
            f"[cave_to_graph] Length cap {max_edge_length} -> spanning FOREST with "
            f"{n_components} component(s) ({n_mst} tree edges); "
            f"{n_components - 1} coverage gap(s) left unbridged."
        )

    # --- Short-edge augmentation under a distance quantile ----------------- #
    threshold = float(np.quantile(ed, quantile)) if ed.size else 0.0
    target = int(np.ceil(target_per_node * N))
    for a, b, d in zip(ei, ej, ed):
        if len(selected) >= target:
            break
        if d > threshold:
            break  # ed is ascending: nothing shorter remains
        key = (int(a), int(b)) if a < b else (int(b), int(a))
        if key in selected_set:
            continue
        selected.append(key)
        selected_set.add(key)

    edges = np.array(selected, dtype=np.int32)
    edge_lengths = dist_mat[edges[:, 0], edges[:, 1]]
    min_adj = float(edge_lengths.max()) if edge_lengths.size else 0.0

    print(
        f"[cave_to_graph] Shortest-biased edges: {edges.shape[0]} total "
        f"(MST={n_mst}, quantile={quantile} -> thr={threshold:.3f}, "
        f"longest edge={min_adj:.3f})."
    )
    return edges, min_adj


def compute_edges_and_communication_matrix(
    cave_map: CaveMapType,
    node_positions: np.ndarray,
    *,
    rng: Optional[np.random.Generator] = None,
    max_edge_length: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute adjacency edges and communication matrix for the given node positions.

    - Edges:
        * Undirected edges between nodes that
          (a) have line-of-sight, and
          (b) are within a distance <= min_adjacent_distance,
        where min_adjacent_distance is chosen as the smallest distance threshold
        under which the resulting graph is connected.
        * As a final step, if num_edges < num_nodes * 2, attempt to add up to
          (num_nodes * 2 - num_edges) extra edges between nodes that have LoS,
          with 50% probability per candidate extra edge. These optional edges
          are not constrained by min_adjacent_distance (they can be long-range).

    - Communication matrix A:
        * A is symmetric, A[i,i] = 1.
        * For i != j, A[i,j] = 1 if there exists line-of-sight between i and j,
          else 0.

    Returns
    -------
    edges : (M, 2) int array
    A : (N, N) int array
    min_adjacent_distance : float
    """
    if rng is None:
        rng = np.random.default_rng()

    nodes = np.asarray(node_positions, dtype=np.float64)
    N = nodes.shape[0]

    los_mat, dist_mat = _compute_pairwise_los_and_dist(cave_map, nodes)

    # Communication matrix
    A = np.zeros((N, N), dtype=np.int32)
    A[los_mat] = 1
    np.fill_diagonal(A, 1)

    # Adjacency edges: shortest-biased graph (MST backbone + short-quantile extras).
    # Favours chaining short hops over a few long edges; see _shortest_biased_edges.
    edges_final, min_adj_dist = _shortest_biased_edges(
        los_mat, dist_mat, N, max_edge_length=max_edge_length
    )

    print(
        f"[cave_to_graph] Final edges: {edges_final.shape[0]} (nodes={N})."
    )

    return edges_final, A, min_adj_dist


# ---------------------------------------------------------------------- #
# Visualization                                                          #
# ---------------------------------------------------------------------- #

def _plot_graph_2d(
    node_positions: np.ndarray,
    edges: np.ndarray,
    out_path: Path,
    *,
    title: str = "",
) -> None:
    """
    Simple 2D visualization of nodes and edges.
    """
    nodes = np.asarray(node_positions, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 8))

    # Edges
    for i, j in edges:
        x = [nodes[i, 0], nodes[j, 0]]
        y = [nodes[i, 1], nodes[j, 1]]
        ax.plot(x, y, linewidth=1.5, alpha=0.8)

    # Nodes
    ax.scatter(nodes[:, 0], nodes[:, 1], s=50, zorder=3)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", "box")
    if title:
        ax.set_title(title)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_graph_3d(
    node_positions: np.ndarray,
    edges: np.ndarray,
    out_path: Path,
    *,
    title: str = "",
) -> None:
    """
    3D visualization with a big perspective view and three axial projections.

    Layout (GridSpec 4x3)
    ----------------------
    - Top: rows 0:3, cols 0:3 -> big 3D view
    - Bottom row:
        * (3,0) : x-y view
        * (3,1) : x-z view
        * (3,2) : y-z view
    """
    nodes = np.asarray(node_positions, dtype=np.float64)
    xs, ys, zs = nodes[:, 0], nodes[:, 1], nodes[:, 2]

    fig = plt.figure(figsize=(10, 12))
    gs = GridSpec(4, 3, figure=fig)

    # Big 3D view
    ax3d = fig.add_subplot(gs[0:3, 0:3], projection="3d")
    for i, j in edges:
        ax3d.plot(
            [xs[i], xs[j]],
            [ys[i], ys[j]],
            [zs[i], zs[j]],
            linewidth=1.5,
            alpha=0.7,
        )
    ax3d.scatter(xs, ys, zs, s=40, depthshade=True)

    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    if title:
        ax3d.set_title(title)

    # Try to make aspect roughly equal
    x_range = xs.max() - xs.min()
    y_range = ys.max() - ys.min()
    z_range = zs.max() - zs.min()
    max_range = max(x_range, y_range, z_range, 1e-6)
    x_mid = 0.5 * (xs.max() + xs.min())
    y_mid = 0.5 * (ys.max() + ys.min())
    z_mid = 0.5 * (zs.max() + zs.min())
    ax3d.set_xlim(x_mid - 0.5 * max_range, x_mid + 0.5 * max_range)
    ax3d.set_ylim(y_mid - 0.5 * max_range, y_mid + 0.5 * max_range)
    ax3d.set_zlim(z_mid - 0.5 * max_range, z_mid + 0.5 * max_range)

    # Axial views
    ax_xy = fig.add_subplot(gs[3, 0])
    for i, j in edges:
        ax_xy.plot(
            [xs[i], xs[j]],
            [ys[i], ys[j]],
            linewidth=1.0,
            alpha=0.8,
        )
    ax_xy.scatter(xs, ys, s=40)
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.set_aspect("equal", "box")
    ax_xy.set_title("x-y")

    ax_xz = fig.add_subplot(gs[3, 1])
    for i, j in edges:
        ax_xz.plot(
            [xs[i], xs[j]],
            [zs[i], zs[j]],
            linewidth=1.0,
            alpha=0.8,
        )
    ax_xz.scatter(xs, zs, s=40)
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")
    ax_xz.set_aspect("equal", "box")
    ax_xz.set_title("x-z")

    ax_yz = fig.add_subplot(gs[3, 2])
    for i, j in edges:
        ax_yz.plot(
            [ys[i], ys[j]],
            [zs[i], zs[j]],
            linewidth=1.0,
            alpha=0.8,
        )
    ax_yz.scatter(ys, zs, s=40)
    ax_yz.set_xlabel("y")
    ax_yz.set_ylabel("z")
    ax_yz.set_aspect("equal", "box")
    ax_yz.set_title("y-z")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_cave_graph(
    graph: CaveGraph,
    out_path: Path,
    *,
    title: str = "",
) -> None:
    """
    Dispatch to 2D or 3D plotting depending on node dimensionality.
    """
    nodes = graph.node_positions
    dim = nodes.shape[1]

    if dim == 2:
        _plot_graph_2d(nodes, graph.edges, out_path, title=title)
    elif dim == 3:
        _plot_graph_3d(nodes, graph.edges, out_path, title=title)
    else:
        raise ValueError(f"Unsupported node dimensionality: {dim}")


# ---------------------------------------------------------------------- #
# Export / main entrypoint                                               #
# ---------------------------------------------------------------------- #

def save_cave_graph(
    export_folder: Path,
    cave_name: str,
    graph: CaveGraph,
) -> Tuple[Path, Path]:
    """
    Save nodes, edges, A, and metadata as a single .npz,
    and save a PNG visualization.

    Also prints a summary of the .npz contents for inspection.

    Returns
    -------
    data_path : Path
        Path to the saved .npz file.
    fig_path : Path
        Path to the saved PNG file.
    """
    export_folder = Path(export_folder)
    export_folder.mkdir(parents=True, exist_ok=True)

    nodes = graph.node_positions
    dim = nodes.shape[1]
    N = nodes.shape[0]

    base_name = f"{cave_name}_graph_{N:03d}_nodes"
    data_path = export_folder / f"{base_name}.npz"
    fig_path = export_folder / f"{base_name}.png"

    np.savez(
        data_path,
        nodes=graph.node_positions,
        edges=graph.edges,
        A=graph.A,
        min_adjacent_distance=np.array(graph.min_adjacent_distance, dtype=np.float64),
        num_nodes=np.array(N, dtype=np.int32),
        dim=np.array(dim, dtype=np.int32),
    )

    # Print NPZ contents for inspection
    print(f"[cave_to_graph] Saved graph npz: {data_path}")
    with np.load(data_path) as data:
        print("[cave_to_graph] npz contents:")
        for key in data.files:
            arr = data[key]
            if arr.ndim == 0:
                print(f"  - {key}: scalar {arr.dtype} -> {arr}")
            else:
                print(f"  - {key}: shape={arr.shape}, dtype={arr.dtype}")
                print(f"{arr}")

    plot_cave_graph(
        graph=graph,
        out_path=fig_path,
        title=f"{cave_name} (N={N})",
    )

    print(f"[cave_to_graph] Saved graph figure: {fig_path}")

    return data_path, fig_path


def build_cave_graph(
    cave_map: CaveMapType,
    num_nodes: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> CaveGraph:
    """
    Convenience wrapper: node placement + edges + communication matrix.

    Returns
    -------
    CaveGraph
    """
    if rng is None:
        rng = np.random.default_rng()

    nodes = compute_node_positions(
        cave_map,
        num_nodes=num_nodes,
        rng=rng,
    )
    edges, A, min_adj_dist = compute_edges_and_communication_matrix(
        cave_map,
        nodes,
        rng=rng,
    )
    return CaveGraph(
        node_positions=nodes,
        edges=edges,
        A=A,
        min_adjacent_distance=min_adj_dist,
    )


def build_and_export_cave_graph(
    cave_map: CaveMapType,
    num_nodes: int,
    export_folder: Path,
    cave_name: str,
    *,
    rng: Optional[np.random.Generator] = None,
) -> CaveGraph:
    """
    Full pipeline:

        cave_map + num_nodes
        -> CaveGraph (nodes, edges, comm matrix)
        -> .npz + .png in export_folder

    Returns
    -------
    graph : CaveGraph
        The constructed graph (also saved to disk).
    """
    if rng is None:
        rng = np.random.default_rng()

    graph = build_cave_graph(
        cave_map,
        num_nodes=num_nodes,
        rng=rng,
    )
    save_cave_graph(
        export_folder=export_folder,
        cave_name=cave_name,
        graph=graph,
    )
    return graph
