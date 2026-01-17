# src/prospector/caves/cave_map_3d.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from plyfile import PlyData

from tqdm import tqdm

from .utils.grid_ops import (
    flood_fill_reachable,
    inflate_obstacles_nd,
    grid_astar,
    sample_free_indices_with_min_distance,
)


@dataclass
class CaveMap3D:
    """
    3D voxel cave map built from a PLY point cloud.

    Labels
    ------
    occupancy[x, y, z] is:
        0 : free & reachable from known_internal_xyz (free-navigable)
        1 : obstacle (occupied)
        2 : free but unreachable from known_internal_xyz (free-inaccessible)

    World <-> grid mapping
    ----------------------
    world -> index:
        rel = (p_world - grid_min) / voxel_size
        idx = floor(rel)

    index -> world (voxel center):
        p_world = grid_min + (idx + 0.5) * voxel_size
    """

    voxel_size: float
    grid_min: np.ndarray        # (3,)
    grid_max: np.ndarray        # (3,)
    occupancy: np.ndarray       # (nx, ny, nz) uint8, labels {0,1,2}
    ply_path: Path

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_ply(
        cls,
        ply_path: str | Path,
        voxel_size: float,
    ) -> "CaveMap3D":
        """
        Build a 3D cave map from a PLY point cloud.
        """
        ply_path = Path(ply_path)
        if not ply_path.is_file():
            raise FileNotFoundError(f"PLY file not found: {ply_path}")

        print(f"[CaveMap3D] Loading point cloud from {ply_path} ...")
        # Load from the path
        ply = PlyData.read(str(ply_path))
        # Expect standard vertex fields
        vertex = ply["vertex"]
        points = np.stack(
            [vertex["x"], vertex["y"], vertex["z"]],
            axis=1,
        ).astype(np.float64)

        if points.size == 0:
            raise ValueError(f"Point cloud is empty: {ply_path}")

        # Compute bounds and grid shape
        grid_min = points.min(axis=0)
        grid_max = points.max(axis=0)

        padding = 1e-6
        grid_min = grid_min - padding
        grid_max = grid_max + padding

        extent = grid_max - grid_min
        grid_shape = np.ceil(extent / voxel_size).astype(int)
        nx, ny, nz = grid_shape.tolist()

        print(
            f"[CaveMap3D] Voxel grid: shape={grid_shape.tolist()}, "
            f"voxel_size={voxel_size:.3f}"
        )

        occupancy = np.zeros((nx, ny, nz), dtype=np.uint8)

        # Map points to voxel indices
        rel = (points - grid_min[None, :]) / voxel_size
        idx = np.floor(rel).astype(int)

        idx[:, 0] = np.clip(idx[:, 0], 0, nx - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, ny - 1)
        idx[:, 2] = np.clip(idx[:, 2], 0, nz - 1)

        print("[CaveMap3D] Marking obstacle voxels from point cloud ...")
        flat = idx[:, 0] * (ny * nz) + idx[:, 1] * nz + idx[:, 2]
        unique_flat = np.unique(flat)
        ix = unique_flat // (ny * nz)
        iy = (unique_flat // nz) % ny
        iz = unique_flat % nz
        occupancy[ix, iy, iz] = 1

        return cls(
            voxel_size=float(voxel_size),
            grid_min=grid_min,
            grid_max=grid_max,
            occupancy=occupancy,
            ply_path=ply_path,
        )

    # ------------------------------------------------------------------ #
    # World <-> index utilities                                         #
    # ------------------------------------------------------------------ #
    def batch_world_to_index(
        self,
        points_world: np.ndarray,
        *,
        return_in_bounds_mask: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Batch map world points to voxel indices.
        """
        pts = np.asarray(points_world, dtype=np.float64)
        pts = np.atleast_2d(pts)

        rel = (pts - self.grid_min[None, :]) / self.voxel_size
        idx_f = np.floor(rel)
        idx = idx_f.astype(int)

        nx, ny, nz = self.occupancy.shape
        in_bounds = (
            (idx[:, 0] >= 0)
            & (idx[:, 0] < nx)
            & (idx[:, 1] >= 0)
            & (idx[:, 1] < ny)
            & (idx[:, 2] >= 0)
            & (idx[:, 2] < nz)
        )

        idx[:, 0] = np.clip(idx[:, 0], 0, nx - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, ny - 1)
        idx[:, 2] = np.clip(idx[:, 2], 0, nz - 1)

        if return_in_bounds_mask:
            return idx, in_bounds
        else:
            return idx, None

    def world_to_index(
        self,
        point_world: Sequence[float],
        *,
        strict: bool = False,
    ) -> np.ndarray:
        """
        Single-point world -> voxel index, implemented via batch_world_to_index.
        """
        pts = np.asarray(point_world, dtype=np.float64)[None, :]
        idx, in_bounds = self.batch_world_to_index(
            pts, return_in_bounds_mask=True
        )
        idx0 = idx[0]
        if strict and not in_bounds[0]:
            raise ValueError(
                f"Point {point_world} maps to index {idx0.tolist()} "
                f"outside grid shape {self.occupancy.shape}"
            )
        return idx0

    def index_to_world_center(self, idx: Sequence[int]) -> np.ndarray:
        """
        Convert voxel indices [ix, iy, iz] to world-space voxel center.
        """
        idx = np.asarray(idx, dtype=np.float64)
        return self.grid_min + (idx + 0.5) * self.voxel_size

    # ------------------------------------------------------------------ #
    # Accessible-space labeling                                          #
    # ------------------------------------------------------------------ #
    def label_accessible_space(self, known_internal_xyz: Sequence[float]) -> None:
        """
        Label reachable and unreachable free space via 3D flood fill.

        - Obstacles (1) remain 1.
        - Free reachable cells remain 0.
        - Free unreachable cells become 2.
        """
        start_idx = self.world_to_index(known_internal_xyz, strict=True)
        if self.occupancy[tuple(start_idx)] == 1:
            raise ValueError(
                f"known_internal_xyz {known_internal_xyz} lies in obstacle voxel."
            )

        neighbors_6 = [
            (+1, 0, 0),
            (-1, 0, 0),
            (0, +1, 0),
            (0, -1, 0),
            (0, 0, +1),
            (0, 0, -1),
        ]

        print("[CaveMap3D] Flood-filling reachable free space ...")
        reachable = flood_fill_reachable(
            self.occupancy,
            start_idx,
            neighbors_6,
            free_value=0,
            tqdm_desc="Flood fill (3D reachable)",
        )

        free_mask = self.occupancy == 0
        unreachable_free = np.logical_and(free_mask, ~reachable)
        self.occupancy[unreachable_free] = 2

        num_reachable = int(reachable.sum())
        num_unreachable = int(unreachable_free.sum())
        num_obstacles = int((self.occupancy == 1).sum())

        print(
            "[CaveMap3D] Accessible space labeling complete: "
            f"reachable={num_reachable}, inaccessible_free={num_unreachable}, "
            f"obstacles={num_obstacles}"
        )

    # ------------------------------------------------------------------ #
    # Occupancy queries                                                  #
    # ------------------------------------------------------------------ #
    def batch_query_label(self, points_world: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Batch occupancy query in world coordinates.

        Returns
        -------
        labels : (N,) int array
            0 : free-navigable
            1 : obstacle
            2 : free-inaccessible or out-of-bounds
        """
        pts = np.asarray(points_world, dtype=np.float64)
        pts = np.atleast_2d(pts)

        idx, in_bounds = self.batch_world_to_index(
            pts, return_in_bounds_mask=True
        )

        labels = np.full((pts.shape[0],), 2, dtype=np.int32)  # default to 2
        labels[in_bounds] = self.occupancy[
            idx[in_bounds, 0],
            idx[in_bounds, 1],
            idx[in_bounds, 2],
        ]
        return labels

    def query_label(self, point_world: Sequence[float]) -> int:
        """
        Single-point occupancy query, via batch_query_label.
        """
        labels = self.batch_query_label([point_world])
        return int(labels[0])

    # ------------------------------------------------------------------ #
    # A* planning                                                        #
    # ------------------------------------------------------------------ #
    def astar_plan(
        self,
        start_world: Sequence[float],
        goal_world: Sequence[float],
        keep_out_radius: float = 0.0,
    ) -> np.ndarray:
        """
        A* path between two free-navigable world-space points.
        """
        start_label = self.query_label(start_world)
        goal_label = self.query_label(goal_world)

        if start_label != 0:
            raise ValueError(
                f"Start {start_world} not free-navigable (label={start_label})."
            )
        if goal_label != 0:
            raise ValueError(
                f"Goal {goal_world} not free-navigable (label={goal_label})."
            )

        start_idx = tuple(self.world_to_index(start_world, strict=True))
        goal_idx = tuple(self.world_to_index(goal_world, strict=True))

        inflated = inflate_obstacles_nd(
            self.occupancy,
            voxel_size=self.voxel_size,
            keep_out_radius=keep_out_radius,
            obstacle_value=1,
            tqdm_desc="Inflating obstacles (3D)",
        )

        free_planning = np.logical_and(self.occupancy == 0, ~inflated)
        nx, ny, nz = self.occupancy.shape

        def in_bounds(idx: Tuple[int, int, int]) -> bool:
            x, y, z = idx
            return (0 <= x < nx) and (0 <= y < ny) and (0 <= z < nz)

        def is_free(idx: Tuple[int, int, int]) -> bool:
            x, y, z = idx
            return bool(free_planning[x, y, z])

        neighbors_6 = [
            (+1, 0, 0),
            (-1, 0, 0),
            (0, +1, 0),
            (0, -1, 0),
            (0, 0, +1),
            (0, 0, -1),
        ]

        print("[CaveMap3D] Running A* search ...")
        path_idx = grid_astar(
            start_idx,
            goal_idx,
            in_bounds_fn=in_bounds,
            is_free_fn=is_free,
            neighbors=neighbors_6,
            tqdm_desc="A* (3D)",
        )

        if path_idx is None:
            raise RuntimeError("A* search failed: no path found (3D).")

        path_world = np.stack(
            [self.index_to_world_center(idx) for idx in path_idx],
            axis=0,
        )
        print(
            f"[CaveMap3D] A* complete, path length={len(path_world)} "
            f"(keep_out_radius={keep_out_radius:.3f})"
        )
        return path_world

    # ------------------------------------------------------------------ #
    # Line-of-sight                                                      #
    # ------------------------------------------------------------------ #
    def batch_has_line_of_sight(
        self,
        points0_world: Sequence[Sequence[float]],
        points1_world: Sequence[Sequence[float]],
        verbose: bool = False,
    ) -> np.ndarray:
        """
        Check unobstructed line of sight between pairs of world-space points.

        - Obstacles (label == 1) block LoS.
        - Free-inaccessible (2) does NOT block LoS.
        """
        p0 = np.asarray(points0_world, dtype=np.float64)
        p1 = np.asarray(points1_world, dtype=np.float64)
        p0 = np.atleast_2d(p0)
        p1 = np.atleast_2d(p1)

        if p0.shape != p1.shape or p0.shape[1] != 3:
            raise ValueError("points0_world and points1_world must have shape (N,3).")

        n = p0.shape[0]
        los = np.zeros((n,), dtype=bool)

        if verbose: print("[CaveMap3D] Checking line-of-sight for point pairs ...")
        for i in tqdm(range(n), desc="Line-of-sight (3D)", unit="pair", disable=not verbose):
            a = p0[i]
            b = p1[i]

            # Endpoints must be in-bounds.
            _, mask_a = self.batch_world_to_index(
                a[None, :], return_in_bounds_mask=True
            )
            _, mask_b = self.batch_world_to_index(
                b[None, :], return_in_bounds_mask=True
            )
            if not (mask_a[0] and mask_b[0]):
                los[i] = False
                continue

            diff = b - a
            length = np.linalg.norm(diff)
            if length == 0.0:
                label = self.query_label(a)
                los[i] = label != 1
                continue

            step = self.voxel_size * 0.5
            n_steps = int(np.ceil(length / step))

            t_vals = np.linspace(0.0, 1.0, n_steps + 1)
            pts = a[None, :] + t_vals[:, None] * diff[None, :]

            idx, seg_in_bounds = self.batch_world_to_index(
                pts, return_in_bounds_mask=True
            )
            if not np.all(seg_in_bounds):
                los[i] = False
                continue

            occ = self.occupancy[
                idx[:, 0].astype(int),
                idx[:, 1].astype(int),
                idx[:, 2].astype(int),
            ]
            los[i] = not np.any(occ == 1)

        return los

    def has_line_of_sight(
        self,
        p0_world: Sequence[float],
        p1_world: Sequence[float],
    ) -> bool:
        """
        Single-pair LoS check, via batch_has_line_of_sight.
        """
        result = self.batch_has_line_of_sight([p0_world], [p1_world])
        return bool(result[0])

    # ------------------------------------------------------------------ #
    # Random free-point sampling (3D)                                    #
    # ------------------------------------------------------------------ #
    def sample_free_points(
        self,
        num_points: int,
        min_distance: float,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Sample random world-space points in free-navigable 3D cells.

        - Only cells with occupancy == 0 are considered.
        - Points are returned at voxel centers in world coordinates.
        - Enforces a minimum Euclidean separation of `min_distance` between
          any pair of sampled points (in world units).

        Typically you will call this after `label_accessible_space`, so that
        inaccessible free cells (label 2) are excluded. If the sampler cannot
        find a configuration that satisfies the constraints, a ValueError is
        raised.
        """
        free_mask = self.occupancy == 0

        idxs = sample_free_indices_with_min_distance(
            free_mask=free_mask,
            voxel_size=self.voxel_size,
            num_points=num_points,
            min_distance=min_distance,
            rng=rng,
        )

        pts_world = np.stack(
            [self.index_to_world_center(idx) for idx in idxs],
            axis=0,
        )
        return pts_world

    # ------------------------------------------------------------------ #
    # Collision check within a radius (3D)                               #
    # ------------------------------------------------------------------ #
    def is_collision_within_radius(
        self,
        world_coordinate: Sequence[float],
        radius: float,
    ) -> bool:
        """
        Treat the agent as a sphere of radius `radius` and check for collisions.

        Implementation
        --------------
        - Build a dense 3D grid of sample points around `world_coordinate`
          with step ~= voxel_size in each dimension.
        - Keep only those points whose Euclidean distance <= radius.
        - Call `batch_query_label` once on this set.
        - If any label != 0 (obstacle, unreachable, or out-of-bounds), we
          treat it as a collision and return True.

        Returns
        -------
        collision : bool
            True if ANY sampled point within radius is not label 0.
        """
        radius = float(radius)
        if radius <= 0.0:
            label = self.query_label(world_coordinate)
            return label != 0

        center = np.asarray(world_coordinate, dtype=np.float64)
        if center.shape[0] != 3:
            raise ValueError(
                f"world_coordinate must have length 3, got shape {center.shape}"
            )

        cx, cy, cz = center
        step = float(self.voxel_size)

        # Grid extents in each direction
        num = int(np.ceil(2.0 * radius / step)) + 1
        xs = np.linspace(cx - radius, cx + radius, num=num)
        ys = np.linspace(cy - radius, cy + radius, num=num)
        zs = np.linspace(cz - radius, cz + radius, num=num)

        XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="xy")

        dx = XX - cx
        dy = YY - cy
        dz = ZZ - cz
        dist_sq = dx * dx + dy * dy + dz * dz
        mask = dist_sq <= radius * radius

        if not np.any(mask):
            # Extremely small radius or numerical issues; fall back to single-point check.
            label = self.query_label(center)
            return label != 0

        pts_world = np.column_stack(
            [
                XX[mask].ravel(),
                YY[mask].ravel(),
                ZZ[mask].ravel(),
            ]
        )

        labels = self.batch_query_label(pts_world)
        # 0 : free & reachable
        # 1 : obstacle
        # 2 : unreachable or out-of-bounds
        return bool(np.any(labels != 0))

