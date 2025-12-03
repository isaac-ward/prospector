from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import matplotlib.axes

from ..cave_map_2d import CaveMap2D


@dataclass
class CaveMap2DPlotter:
    """
    Simple 2D visualizer for CaveMap2D.

    - x axis: world X
    - y axis: world Y
    - Free-navigable cells (label == 0) are white
    - All other cells (1 = obstacle, 2 = free-inaccessible) are black

    Rendering uses imshow with a grayscale colormap.
    """

    cave_map: CaveMap2D

    def plot_occupancy(
        self,
        ax: matplotlib.axes.Axes,
        *,
        clear: bool = True,
        **imshow_kwargs,
    ):
        """
        Render the 2D cave occupancy on the given Axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            2D axes to draw on.
        agent_world_pos : optional (3,) sequence
            If provided, scatter the agent's position on top (projected to XY).
        clear : bool, optional
            If True, clears the axis before drawing.
        **imshow_kwargs :
            Additional keyword args passed to ax.imshow().

        Returns
        -------
        img : matplotlib.image.AxesImage
        """
        occ = self.cave_map.occupancy  # shape (nx, ny)
        nx, ny = occ.shape

        # Mask for free-navigable cells
        free_mask = (occ == 0).astype(float)  # 1.0 where free, 0.0 elsewhere

        # imshow expects array[row, col] => (y, x).
        # Our first index is x, second is y, so transpose.
        img_data = free_mask.T  # shape (ny, nx)

        x_min, y_min = self.cave_map.grid_min_xy
        x_max, y_max = self.cave_map.grid_max_xy

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
        
        return img
