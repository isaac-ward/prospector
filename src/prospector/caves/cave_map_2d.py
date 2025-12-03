# src/prospector/caves/cave_map_2d.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
from tqdm import tqdm

from .utils.grid_ops import (
    flood_fill_reachable,
    inflate_obstacles_nd,
    grid_astar,
    sample_free_indices_with_min_distance,
)

@dataclass
class CaveMap2D:
    """
    2D cave map built from a thick z-slice of a PLY point cloud.

    Conceptually:
        - We take all points whose z is within [slice_z - thickness/2, slice_z + thickness/2].
        - We ignore z and voxelize in (x, y) only.
        - Occupancy is a 2D grid: (nx, ny).

    Labels
    ------
        0 : free & reachable from known_internal_xyz (projected to XY)
        1 : obstacle
        2 : free but unreachable (free-inaccessible)

    World <-> grid mapping (2D)
    ---------------------------
    Only x, y are used for index computation; z is ignored for occupancy:

        rel_xy = (p_xy - grid_min_xy) / voxel_size
        idx_xy = floor(rel_xy)

    For index -> world, we return:

        x, y from center of voxel;
        z = slice_z            (mid-plane of the slice)
    """

    voxel_size: float
    grid_min_xy: np.ndarray     # (2,)
    grid_max_xy: np.ndarray     # (2,)
    slice_z: float              # central z of the slice
    slice_thickness: float      # thickness used when constructing map
    occupancy: np.ndarray       # (nx, ny) uint8, labels {0,1,2}
    ply_path: Path

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_ply(
        cls,
        ply_path: str | Path,
        voxel_size: float,
        *,
        slice_for_2d_z: float,
        slice_for_2d_thickness: float,
    ) -> "CaveMap2D":
        """
        Build a 2D cave map from a thick z-slice of the PLY point cloud.

        Parameters
        ----------
        ply_path : str or Path
            Path to PLY obstacle point cloud.
        voxel_size : float
        slice_for_2d_z : float
            Center z of the slice.
        slice_for_2d_thickness : float
            Slice thickness in z. Points with
                |z - slice_for_2d_z| <= slice_for_2d_thickness / 2
            are included.

        Returns
        -------
        CaveMap2D
        """
        ply_path = Path(ply_path)
        if not ply_path.is_file():
            raise FileNotFoundError(f"PLY file not found: {ply_path}")

        print(f"[CaveMap2D] Loading point cloud from {ply_path} ...")
        pcd = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(pcd.points, dtype=np.float64)

        if points.size == 0:
            raise ValueError(f"Point cloud is empty: {ply_path}")

        z = points[:, 2]
        half_thick = slice_for_2d_thickness * 0.5
        mask = np.abs(z - slice_for_2d_z) <= half_thick

        slice_points = points[mask]
        if slice_points.size == 0:
            raise ValueError(
                f"No points found in z-slice around z={slice_for_2d_z} "
                f"with thickness={slice_for_2d_thickness}"
            )

        # Work in XY only
        xy = slice_points[:, :2]

        grid_min_xy = xy.min(axis=0)
        grid_max_xy = xy.max(axis=0)

        padding = 1e-6
        grid_min_xy = grid_min_xy - padding
        grid_max_xy = grid_max_xy + padding

        extent_xy = grid_max_xy - grid_min_xy
        grid_shape_xy = np.ceil(extent_xy / voxel_size).astype(int)
        nx, ny = grid_shape_xy.tolist()

        print(
            f"[CaveMap2D] Voxel grid (2D): shape={grid_shape_xy.tolist()}, "
            f"voxel_size={voxel_size:.3f}, slice_z={slice_for_2d_z:.3f}, "
            f"thickness={slice_for_2d_thickness:.3f}"
        )

        occupancy = np.zeros((nx, ny), dtype=np.uint8)

        # Map slice points into XY voxels
        rel_xy = (xy - grid_min_xy[None, :]) / voxel_size
        idx_xy = np.floor(rel_xy).astype(int)

        idx_xy[:, 0] = np.clip(idx_xy[:, 0], 0, nx - 1)
        idx_xy[:, 1] = np.clip(idx_xy[:, 1], 0, ny - 1)

        print("[CaveMap2D] Marking obstacle voxels from slice points ...")
        flat = idx_xy[:, 0] * ny + idx_xy[:, 1]
        unique_flat = np.unique(flat)
        ix = unique_flat // ny
        iy = unique_flat % ny
        occupancy[ix, iy] = 1

        return cls(
            voxel_size=float(voxel_size),
            grid_min_xy=grid_min_xy,
            grid_max_xy=grid_max_xy,
            slice_z=float(slice_for_2d_z),
            slice_thickness=float(slice_for_2d_thickness),
            occupancy=occupancy,
            ply_path=ply_path,
        )

    # ------------------------------------------------------------------ #
    # World <-> index utilities (2D)                                     #
    # ------------------------------------------------------------------ #
    def batch_world_to_index(
        self,
        points_world: np.ndarray,
        *,
        return_in_bounds_mask: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Batch map world points to 2D voxel indices (ix, iy), ignoring z.
        """
        pts = np.asarray(points_world, dtype=np.float64)
        pts = np.atleast_2d(pts)
        xy = pts[:, :2]

        rel = (xy - self.grid_min_xy[None, :]) / self.voxel_size
        idx_f = np.floor(rel)
        idx = idx_f.astype(int)

        nx, ny = self.occupancy.shape
        in_bounds = (
            (idx[:, 0] >= 0)
            & (idx[:, 0] < nx)
            & (idx[:, 1] >= 0)
            & (idx[:, 1] < ny)
        )

        idx[:, 0] = np.clip(idx[:, 0], 0, nx - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, ny - 1)

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
        Single-point world -> 2D voxel index, via batch_world_to_index.
        """
        pts = np.asarray(point_world, dtype=np.float64)[None, :]
        idx, in_bounds = self.batch_world_to_index(
            pts, return_in_bounds_mask=True
        )
        idx0 = idx[0]
        if strict and not in_bounds[0]:
            raise ValueError(
                f"Point {point_world} maps to index {idx0.tolist()} "
                f"outside 2D grid shape {self.occupancy.shape}"
            )
        return idx0

    def index_to_world_center(self, idx: Sequence[int]) -> np.ndarray:
        """
        Convert 2D voxel indices to world coordinates at voxel center in XY,
        and z = slice_z.
        """
        idx = np.asarray(idx, dtype=np.float64)
        xy = self.grid_min_xy + (idx + 0.5) * self.voxel_size
        return np.array([xy[0], xy[1], self.slice_z], dtype=np.float64)

    # ------------------------------------------------------------------ #
    # Accessible-space labeling (2D flood fill)                          #
    # ------------------------------------------------------------------ #
    def label_accessible_space(self, known_internal_xyz: Sequence[float]) -> None:
        """
        Label reachable/unreachable free space in the 2D map.

        known_internal_xyz is projected to XY and used as the flood-fill start.
        """
        start_idx = self.world_to_index(known_internal_xyz, strict=True)
        if self.occupancy[tuple(start_idx)] == 1:
            raise ValueError(
                f"known_internal_xyz {known_internal_xyz} lies in obstacle voxel (2D)."
            )

        neighbors_4 = [
            (+1, 0),
            (-1, 0),
            (0, +1),
            (0, -1),
        ]

        print("[CaveMap2D] Flood-filling reachable free space (2D) ...")
        reachable = flood_fill_reachable(
            self.occupancy,
            start_idx,
            neighbors_4,
            free_value=0,
            tqdm_desc="Flood fill (2D reachable)",
        )

        free_mask = self.occupancy == 0
        unreachable_free = np.logical_and(free_mask, ~reachable)
        self.occupancy[unreachable_free] = 2

        num_reachable = int(reachable.sum())
        num_unreachable = int(unreachable_free.sum())
        num_obstacles = int((self.occupancy == 1).sum())

        print(
            "[CaveMap2D] Accessible space labeling complete: "
            f"reachable={num_reachable}, inaccessible_free={num_unreachable}, "
            f"obstacles={num_obstacles}"
        )

    # ------------------------------------------------------------------ #
    # Occupancy queries                                                  #
    # ------------------------------------------------------------------ #
    def batch_query_label(self, points_world: Sequence[Sequence[float]]) -> np.ndarray:
        """
        Batch occupancy query in world coordinates (XY projected, z ignored).
        """
        pts = np.asarray(points_world, dtype=np.float64)
        pts = np.atleast_2d(pts)

        idx, in_bounds = self.batch_world_to_index(
            pts, return_in_bounds_mask=True
        )

        labels = np.full((pts.shape[0],), 2, dtype=np.int32)  # default 2
        labels[in_bounds] = self.occupancy[
            idx[in_bounds, 0],
            idx[in_bounds, 1],
        ]
        return labels

    def query_label(self, point_world: Sequence[float]) -> int:
        """
        Single-point occupancy query, via batch_query_label.
        """
        labels = self.batch_query_label([point_world])
        return int(labels[0])

    # ------------------------------------------------------------------ #
    # A* planning in 2D                                                  #
    # ------------------------------------------------------------------ #
    def astar_plan(
        self,
        start_world: Sequence[float],
        goal_world: Sequence[float],
        keep_out_radius: float = 0.0,
    ) -> np.ndarray:
        """
        A* path between two free-navigable world-space points projected to 2D.

        Raises if start/goal are not free-navigable (label != 0),
        or if no path is found.
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
            tqdm_desc="Inflating obstacles (2D)",
        )

        free_planning = np.logical_and(self.occupancy == 0, ~inflated)
        nx, ny = self.occupancy.shape

        def in_bounds(idx: Tuple[int, int]) -> bool:
            x, y = idx
            return (0 <= x < nx) and (0 <= y < ny)

        def is_free(idx: Tuple[int, int]) -> bool:
            x, y = idx
            return bool(free_planning[x, y])

        neighbors_4 = [
            (+1, 0),
            (-1, 0),
            (0, +1),
            (0, -1),
        ]

        print("[CaveMap2D] Running A* search (2D) ...")
        path_idx = grid_astar(
            start_idx,
            goal_idx,
            in_bounds_fn=in_bounds,
            is_free_fn=is_free,
            neighbors=neighbors_4,
            tqdm_desc="A* (2D)",
        )

        if path_idx is None:
            raise RuntimeError("A* search failed: no path found (2D).")

        path_world = np.stack(
            [self.index_to_world_center(idx) for idx in path_idx],
            axis=0,
        )
        print(
            f"[CaveMap2D] A* complete, path length={len(path_world)} "
            f"(keep_out_radius={keep_out_radius:.3f})"
        )
        return path_world

    # ------------------------------------------------------------------ #
    # Line-of-sight in 2D                                                #
    # ------------------------------------------------------------------ #
    def batch_has_line_of_sight(
        self,
        points0_world: Sequence[Sequence[float]],
        points1_world: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """
        2D line-of-sight: project endpoints to XY, ignore z.

        Obstacles (label == 1) block LoS. Free-inaccessible (2) does not.

        Returns
        -------
        los : (N,) bool array
        """
        p0 = np.asarray(points0_world, dtype=np.float64)
        p1 = np.asarray(points1_world, dtype=np.float64)
        p0 = np.atleast_2d(p0)
        p1 = np.atleast_2d(p1)

        if p0.shape != p1.shape or p0.shape[1] != 3:
            raise ValueError("points0_world and points1_world must have shape (N,3).")

        n = p0.shape[0]
        los = np.zeros((n,), dtype=bool)

        print("[CaveMap2D] Checking line-of-sight for point pairs (2D) ...")
        for i in tqdm(range(n), desc="Line-of-sight (2D)", unit="pair"):
            a = p0[i]
            b = p1[i]

            # Project endpoints to XY and check bounds
            _, mask_a = self.batch_world_to_index(
                a[None, :], return_in_bounds_mask=True
            )
            _, mask_b = self.batch_world_to_index(
                b[None, :], return_in_bounds_mask=True
            )
            if not (mask_a[0] and mask_b[0]):
                los[i] = False
                continue

            a_xy = a[:2]
            b_xy = b[:2]
            diff_xy = b_xy - a_xy
            length = float(np.linalg.norm(diff_xy))
            if length == 0.0:
                label = self.query_label(a)
                los[i] = label != 1
                continue

            step = self.voxel_size * 0.5
            n_steps = int(np.ceil(length / step))

            t_vals = np.linspace(0.0, 1.0, n_steps + 1)
            # Build 3D points with z irrelevant but required by signature
            pts_xy = a_xy[None, :] + t_vals[:, None] * diff_xy[None, :]
            pts = np.column_stack(
                [
                    pts_xy[:, 0],
                    pts_xy[:, 1],
                    np.full((pts_xy.shape[0],), self.slice_z, dtype=np.float64),
                ]
            )

            idx, seg_in_bounds = self.batch_world_to_index(
                pts, return_in_bounds_mask=True
            )
            if not np.all(seg_in_bounds):
                los[i] = False
                continue

            occ = self.occupancy[
                idx[:, 0].astype(int),
                idx[:, 1].astype(int),
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
    # Random free-point sampling (2D)                                    #
    # ------------------------------------------------------------------ #
    def sample_free_points(
        self,
        num_points: int,
        min_distance: float,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Sample random world-space points in free-navigable 2D cells.

        - Only cells with occupancy == 0 are considered.
        - Points are returned at voxel centers in world coordinates.
        - Enforces a minimum Euclidean separation of `min_distance` between
          any pair of sampled points (in world units).

        This will typically be used after `label_accessible_space`, so that
        unreachable free cells have label 2 and are automatically excluded.
        If the request cannot be satisfied given the map and constraints,
        a ValueError is raised.
        """
        free_mask = self.occupancy == 0

        idxs = sample_free_indices_with_min_distance(
            free_mask=free_mask,
            voxel_size=self.voxel_size,
            num_points=num_points,
            min_distance=min_distance,
            rng=rng,
        )

        # Map voxel indices to world-space voxel centers (with z = slice_z).
        pts_world = np.stack(
            [self.index_to_world_center(idx) for idx in idxs],
            axis=0,
        )
        return pts_world

