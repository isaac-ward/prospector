from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import matplotlib.axes

from ..cave_map_3d import CaveMap3D


@dataclass
class CaveMap3DPlotter:
    """
    Visualizer for CaveMap3D.

    Features
    --------
    - 3D occupancy plot:
        * Obstacle voxels (label == 1) as black, transparent points.
        * Optionally subsampled to at most `max_obstacle_points`.
        * Fixed view: elev=30°, azim=45°.

    - 2D slices through agent position:
        * XY slice at agent's z index
        * XZ slice at agent's y index
        * YZ slice at agent's x index

      In all 2D slices:
        * Free-navigable (label == 0) is white.
        * Everything else (1, 2) is black.
        * Rendered via imshow in world coordinates.
    """

    cave_map: CaveMap3D
    # New: limit number of obstacle points drawn in 3D for speed
    max_obstacle_points: Optional[int] = 50_000
    random_seed: int = 0

    # Cached centers for obstacle voxels (possibly subsampled)
    _cached_obstacle_centers: Optional[np.ndarray] = field(
        default=None, init=False, repr=False
    )

    # ------------------------------------------------------------------ #
    # Internal helper: get cached / subsampled obstacle centers          #
    # ------------------------------------------------------------------ #
    def _get_obstacle_centers(self) -> np.ndarray:
        """
        Return an (M, 3) array of obstacle voxel centers (possibly subsampled).

        Caches the result on first call to avoid recomputing across frames.
        """
        if self._cached_obstacle_centers is not None:
            return self._cached_obstacle_centers

        occ = self.cave_map.occupancy  # (nx, ny, nz)
        obstacle_idx = np.argwhere(occ == 1)  # (K, 3)
        num_obs = obstacle_idx.shape[0]

        if num_obs == 0:
            self._cached_obstacle_centers = np.zeros((0, 3), dtype=float)
            return self._cached_obstacle_centers

        if self.max_obstacle_points is not None and num_obs > self.max_obstacle_points:
            rng = np.random.default_rng(self.random_seed)
            sel = rng.choice(num_obs, size=self.max_obstacle_points, replace=False)
            obstacle_idx = obstacle_idx[sel]

        centers = np.stack(
            [self.cave_map.index_to_world_center(idx) for idx in obstacle_idx],
            axis=0,
        )  # (M, 3)

        self._cached_obstacle_centers = centers
        return centers

    # ------------------------------------------------------------------ #
    # 3D occupancy                                                       #
    # ------------------------------------------------------------------ #
    def plot_occupancy_3d(
        self,
        ax: matplotlib.axes.Axes,
        *,
        clear: bool = True,
        alpha: float = 0.01,
        size: float = 0.5,
    ):
        """
        Render a 3D scatter of obstacle voxels.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            3D axes (projection='3d').
        clear : bool, optional
            If True, clears the axis before drawing.
        alpha : float, optional
            Transparency for obstacle points.
        size : float, optional
            Marker size for scatter (passed to ax.scatter).

        Returns
        -------
        scatter : matplotlib.collections.PathCollection or None
        """
        centers = self._get_obstacle_centers()  # (M, 3)
        if centers.shape[0] == 0:
            if clear:
                ax.clear()
            ax.set_title("No obstacle voxels")
            return None

        xs = centers[:, 0]
        ys = centers[:, 1]
        zs = centers[:, 2]

        if clear:
            ax.clear()

        scatter = ax.scatter(xs, ys, zs, c="black", alpha=alpha, s=size)

        # Set viewing angle
        ax.view_init(elev=30.0, azim=45.0)

        # Axis labels and limits from grid bounds
        x_min, y_min, z_min = self.cave_map.grid_min
        x_max, y_max, z_max = self.cave_map.grid_max

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        ax.set_xlabel("x [world]")
        ax.set_ylabel("y [world]")
        ax.set_zlabel("z [world]")

        # Try to keep aspect approximately equal
        max_range = max(
            x_max - x_min,
            y_max - y_min,
            z_max - z_min,
        )
        mid_x = 0.5 * (x_min + x_max)
        mid_y = 0.5 * (y_min + y_max)
        mid_z = 0.5 * (z_min + z_max)

        ax.set_xlim(mid_x - 0.5 * max_range, mid_x + 0.5 * max_range)
        ax.set_ylim(mid_y - 0.5 * max_range, mid_y + 0.5 * max_range)
        ax.set_zlim(mid_z - 0.5 * max_range, mid_z + 0.5 * max_range)

        return scatter

    # ------------------------------------------------------------------ #
    # 2D slice helpers                                                   #
    # ------------------------------------------------------------------ #
    def _slice_xy_at_index(self, iz: int) -> np.ndarray:
        """
        Return a 2D occupancy slice in XY at fixed z index.
        """
        occ = self.cave_map.occupancy  # (nx, ny, nz)
        return occ[:, :, iz]  # (nx, ny)

    def _slice_xz_at_index(self, iy: int) -> np.ndarray:
        """
        Return a 2D occupancy slice in XZ at fixed y index.
        """
        occ = self.cave_map.occupancy  # (nx, ny, nz)
        # shape (nx, nz), axes: x, z
        slice_xz = occ[:, iy, :]
        return slice_xz  # (nx, nz)

    def _slice_yz_at_index(self, ix: int) -> np.ndarray:
        """
        Return a 2D occupancy slice in YZ at fixed x index.
        """
        occ = self.cave_map.occupancy  # (nx, ny, nz)
        # shape (ny, nz), axes: y, z
        slice_yz = occ[ix, :, :]
        return slice_yz  # (ny, nz)

    # ------------------------------------------------------------------ #
    # 2D slice plotting                                                  #
    # ------------------------------------------------------------------ #
    def plot_xy_slice_through_point(
        self,
        ax: matplotlib.axes.Axes,
        agent_world_pos: Sequence[float],
        *,
        clear: bool = True,
        **imshow_kwargs,
    ):
        """
        Plot an XY slice (z fixed) that passes through the agent's current z.

        - Free-navigable cells (0) are white.
        - Everything else (1, 2) is black.
        """
        idx = self.cave_map.world_to_index(agent_world_pos, strict=True)
        iz = int(idx[2])

        slice_xy = self._slice_xy_at_index(iz)  # (nx, ny)
        free_mask = (slice_xy == 0).astype(float)

        # Transpose for imshow: (ny, nx) so rows=y, cols=x
        img_data = free_mask.T

        x_min, y_min, _ = self.cave_map.grid_min
        x_max, y_max, _ = self.cave_map.grid_max

        if clear:
            ax.clear()

        img = ax.imshow(
            img_data,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            origin="lower",
            extent=(x_min, x_max, y_min, y_max),
            interpolation="nearest",
            **imshow_kwargs,
        )

        ax.set_xlabel("x [world]")
        ax.set_ylabel("y [world]")
        ax.set_aspect("equal")
        ax.set_title(f"XY slice @ z-index {iz}\nabout agent=0")

        return img

    def plot_xz_slice_through_point(
        self,
        ax: matplotlib.axes.Axes,
        agent_world_pos: Sequence[float],
        *,
        clear: bool = True,
        **imshow_kwargs,
    ):
        """
        Plot an XZ slice (y fixed) that passes through the agent's current y.

        - x axis: world X
        - y axis: world Z
        """
        idx = self.cave_map.world_to_index(agent_world_pos, strict=True)
        iy = int(idx[1])

        slice_xz = self._slice_xz_at_index(iy)  # (nx, nz) axes: x, z
        free_mask = (slice_xz == 0).astype(float)

        # We want rows=z, cols=x for imshow, so transpose: (nz, nx)
        img_data = free_mask.T

        x_min, y_min, z_min = self.cave_map.grid_min
        x_max, y_max, z_max = self.cave_map.grid_max

        if clear:
            ax.clear()

        img = ax.imshow(
            img_data,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            origin="lower",
            extent=(x_min, x_max, z_min, z_max),
            interpolation="nearest",
            **imshow_kwargs,
        )

        ax.set_xlabel("x [world]")
        ax.set_ylabel("z [world]")
        ax.set_aspect("equal")
        ax.set_title(f"XZ slice @ y-index {iy}\nabout agent=0")

        return img

    def plot_yz_slice_through_point(
        self,
        ax: matplotlib.axes.Axes,
        agent_world_pos: Sequence[float],
        *,
        clear: bool = True,
        **imshow_kwargs,
    ):
        """
        Plot a YZ slice (x fixed) that passes through the agent's current x.

        - x axis: world Y
        - y axis: world Z
        """
        idx = self.cave_map.world_to_index(agent_world_pos, strict=True)
        ix = int(idx[0])

        slice_yz = self._slice_yz_at_index(ix)  # (ny, nz) axes: y, z
        free_mask = (slice_yz == 0).astype(float)

        # We want rows=z, cols=y, so transpose: (nz, ny)
        img_data = free_mask.T

        x_min, y_min, z_min = self.cave_map.grid_min
        x_max, y_max, z_max = self.cave_map.grid_max

        if clear:
            ax.clear()

        img = ax.imshow(
            img_data,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            origin="lower",
            extent=(y_min, y_max, z_min, z_max),
            interpolation="nearest",
            **imshow_kwargs,
        )

        ax.set_xlabel("y [world]")
        ax.set_ylabel("z [world]")
        ax.set_aspect("equal")
        ax.set_title(f"YZ slice @ x-index {ix}\nabout agent=0")

        return img
