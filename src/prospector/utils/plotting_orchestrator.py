# src/prospector/utils/plotting_orchestrator.py

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as path_effects

from prospector.caves.cave_map_2d import CaveMap2D
from prospector.caves.cave_map_3d import CaveMap3D
from prospector.caves.plotting.cave_map_2d_plotting import CaveMap2DPlotter
from prospector.caves.plotting.cave_map_3d_plotting import CaveMap3DPlotter
from prospector.utils.video import image_folder_to_video


CaveMapType = Union[CaveMap2D, CaveMap3D]


class PlottingOrchestrator:
    """
    Unified, stateful plotting orchestrator for cave maps (2D + 3D).

    Responsibilities
    ----------------
    - Create and own a matplotlib Figure + Axes layout at construction time.
    - On every call to `render_frame(...)`, draw the cave and optional agent
      and graph information into the existing axes and save a PNG frame.
    - Optionally convert accumulated frames into an MP4 via `finalize_video`.

    Modes
    -----
    mode == "2d":
        - Single Axes showing the 2D occupancy (CaveMap2D).

    mode == "3d":
        - Figure with a 4x3 GridSpec:
            * Top 3 rows (all 3 columns): 3D occupancy.
            * Bottom row: XY, XZ, YZ slices through a point:
                - If agents given: slice through the first agent's position.
                - If no agents: slice through the center of the cave extents.
    """

    def __init__(
        self,
        mode: Literal["2d", "3d"],
        *,
        cave_map: CaveMapType,
        cave_name: str,
        log_dir: Path,
        fps: int = 24,
        max_obstacle_points_3d: Optional[int] = None,
        alpha_obstacles_3d: float = 0.1,
        dpi: int = 150,
        figure_size_multiplier: float = 1.0,
    ) -> None:
        self.mode = mode
        self.cave_name = cave_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self.dpi = int(dpi)

        # Generic reference to cave map (2D or 3D)
        self.cave_map: CaveMapType = cave_map

        self.frames_dir = self.log_dir / f"{cave_name}_{mode}_frames"
        print(f"[PlottingOrchestrator] Creating frames directory: {self.frames_dir}")
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self._frame_counter: int = 0

        if mode == "2d":
            if not isinstance(cave_map, CaveMap2D):
                raise TypeError(
                    "mode='2d' requires cave_map to be a CaveMap2D instance, "
                    f"got {type(cave_map)}"
                )
            self.cave_map_2d: CaveMap2D = cave_map
            self.cave_map_3d: Optional[CaveMap3D] = None

            self.plotter_2d = CaveMap2DPlotter(self.cave_map_2d)

            # Create 2D figure + axes once
            self.fig, self.ax2d = plt.subplots(
                figsize=(
                    6 * figure_size_multiplier,
                    6 * figure_size_multiplier,
                )
            )
            self.fig.suptitle(f"{cave_name} – 2D occupancy")
            self.fig.tight_layout()

        elif mode == "3d":
            if not isinstance(cave_map, CaveMap3D):
                raise TypeError(
                    "mode='3d' requires cave_map to be a CaveMap3D instance, "
                    f"got {type(cave_map)}"
                )
            self.cave_map_3d: CaveMap3D = cave_map
            self.cave_map_2d: Optional[CaveMap2D] = None

            print(f"[PlottingOrchestrator]   max_obstacle_points_3d: {max_obstacle_points_3d}")
            self.plotter_3d = CaveMap3DPlotter(
                self.cave_map_3d,
                max_obstacle_points=max_obstacle_points_3d,
            )
            self.alpha_obstacles_3d = float(alpha_obstacles_3d)

            # Create 3D figure + GridSpec + axes once
            self.fig = plt.figure(
                figsize=(
                    10 * figure_size_multiplier,
                    12 * figure_size_multiplier,
                )
            )
            gs = GridSpec(4, 3, figure=self.fig)

            # Top 3 rows: 3D
            self.ax3d = self.fig.add_subplot(gs[0:3, 0:3], projection="3d")

            # Bottom row: XY, XZ, YZ slices
            self.ax_xy = self.fig.add_subplot(gs[3, 0])
            self.ax_xz = self.fig.add_subplot(gs[3, 1])
            self.ax_yz = self.fig.add_subplot(gs[3, 2])

            self.fig.suptitle(f"{cave_name} – 3D occupancy + slices")
            self.fig.tight_layout()
        else:
            raise ValueError(f"Unknown mode {mode!r}; expected '2d' or '3d'.")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_agent_inputs(
        agent_positions: np.ndarray,
        agent_colors: np.ndarray,
        agent_alives: np.ndarray,
        agent_communications_matrix: Optional[np.ndarray],
    ):
        """
        Validate shapes and types of agent inputs and return normalized forms.

        All of positions/colors/alives are required and must agree on N.
        Communications matrix is optional.
        """
        positions = np.asarray(agent_positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] not in (2, 3):
            raise ValueError(
                "agent_positions must have shape (N, 2) or (N, 3); "
                f"got shape {positions.shape}"
            )
        N = positions.shape[0]
        if N < 1:
            raise ValueError("agent_positions must contain at least one agent.")

        colors = np.asarray(agent_colors, dtype=object)
        if colors.shape[0] < N:
            raise ValueError(
                f"agent_colors must have length at least {N}, got shape {colors.shape}"
            )

        alives = np.asarray(agent_alives, dtype=bool)
        if alives.shape[0] != N:
            raise ValueError(
                f"agent_alives must have length {N}, got shape {alives.shape}"
            )

        if agent_communications_matrix is not None:
            comm = np.asarray(agent_communications_matrix, dtype=bool)
            if comm.shape != (N, N):
                raise ValueError(
                    f"agent_communications_matrix must have shape ({N}, {N}), "
                    f"got {comm.shape}"
                )
        else:
            comm = None

        return positions, colors, alives, comm

    @staticmethod
    def _draw_bicolor_dashed_line_2d(ax, p0, p1, color0, color1, num_segments: int = 20):
        """
        Draw a dashed line between p0 and p1 in 2D with alternating colors.
        """
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        ts = np.linspace(0.0, 1.0, num_segments + 1)
        pts = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
        for k in range(num_segments):
            c = color0 if k % 2 == 0 else color1
            segment = pts[k : k + 2]
            ax.plot(segment[:, 0], segment[:, 1], color=c, linewidth=1.0)

    @staticmethod
    def _draw_bicolor_dashed_line_3d(
        ax3d, p0, p1, color0, color1, num_segments: int = 20
    ):
        """
        Draw a dashed line between p0 and p1 in 3D with alternating colors.
        """
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        ts = np.linspace(0.0, 1.0, num_segments + 1)
        pts = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
        for k in range(num_segments):
            c = color0 if k % 2 == 0 else color1
            segment = pts[k : k + 2]
            ax3d.plot(
                segment[:, 0],
                segment[:, 1],
                segment[:, 2],
                color=c,
                linewidth=1.0,
            )

    # ------------------------------------------------------------------ #
    # Graph plotting helpers                                             #
    # ------------------------------------------------------------------ #
    def _plot_graph_nodes_2d(
        self,
        ax,
        graph_nodes: np.ndarray,
    ) -> None:
        """
        Plot graph nodes + index labels in 2D (orange).
        Node index is implied by position in the list.
        """
        nodes = np.asarray(graph_nodes, dtype=float)
        if nodes.ndim != 2 or nodes.shape[1] not in (2, 3):
            raise ValueError(
                "graph_nodes must have shape (num_nodes, 2) or (num_nodes, 3); "
                f"got {nodes.shape}"
            )

        xs = nodes[:, 0]
        ys = nodes[:, 1]

        # Plot nodes as orange circles with black outline
        ax.scatter(
            xs,
            ys,
            c="orange",
            s=40,
            marker="o",
            edgecolors="k",
            linewidths=1.0,
            zorder=5,
        )

        # Index labels
        pe = [path_effects.withStroke(linewidth=1.5, foreground="k")]
        for idx, (x, y) in enumerate(zip(xs, ys)):
            ax.text(
                x,
                y,
                str(idx),
                va="center",
                ha="center",
                color="orange",
                fontsize=8,
                weight="bold",
                path_effects=pe,
                zorder=6,
            )

    def _plot_graph_edges_2d(
        self,
        ax,
        graph_nodes: np.ndarray,
        graph_edges: np.ndarray,
    ) -> None:
        """
        Plot graph edges as orange lines in 2D.
        """
        nodes = np.asarray(graph_nodes, dtype=float)
        edges = np.asarray(graph_edges, dtype=int)

        if nodes.ndim != 2 or nodes.shape[1] not in (2, 3):
            raise ValueError(
                "graph_nodes must have shape (num_nodes, 2) or (num_nodes, 3); "
                f"got {nodes.shape}"
            )
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(
                "graph_edges must have shape (num_edges, 2); "
                f"got {edges.shape}"
            )

        xs = nodes[:, 0]
        ys = nodes[:, 1]
        num_nodes = nodes.shape[0]

        for i, j in edges:
            if not (0 <= i < num_nodes and 0 <= j < num_nodes):
                continue
            x0, y0 = xs[i], ys[i]
            x1, y1 = xs[j], ys[j]
            ax.plot(
                [x0, x1],
                [y0, y1],
                color="orange",
                linewidth=1.5,
                zorder=4,
            )

    def _plot_graph_nodes_3d(
        self,
        ax3d,
        ax_xy,
        ax_xz,
        ax_yz,
        graph_nodes: np.ndarray,
    ) -> None:
        """
        Plot graph nodes + index labels in 3D + projections (orange).
        Node index is implied by position in the list.
        """
        nodes = np.asarray(graph_nodes, dtype=float)
        if nodes.ndim != 2 or nodes.shape[1] not in (2, 3):
            raise ValueError(
                "graph_nodes must have shape (num_nodes, 2) or (num_nodes, 3); "
                f"got {nodes.shape}"
            )

        if nodes.shape[1] == 3:
            nodes3d = nodes
        else:
            # Lift 2D into 3D using grid_min[2]
            z_default = float(self.cave_map.grid_min[2])
            nodes3d = np.column_stack(
                [nodes[:, 0], nodes[:, 1], np.full(nodes.shape[0], z_default)]
            )

        xs = nodes3d[:, 0]
        ys = nodes3d[:, 1]
        zs = nodes3d[:, 2]

        # Plot nodes as orange circles with black outline
        s = 40 * 3  # reuse marker_size_multiplier ~ 3
        ax3d.scatter(
            xs,
            ys,
            zs,
            c="orange",
            s=s,
            marker="o",
            edgecolors="k",
            linewidths=1.0,
            depthshade=False,
            zorder=5,
        )
        ax_xy.scatter(
            xs,
            ys,
            c="orange",
            s=s,
            marker="o",
            edgecolors="k",
            linewidths=1.0,
            zorder=5,
        )
        ax_xz.scatter(
            xs,
            zs,
            c="orange",
            s=s,
            marker="o",
            edgecolors="k",
            linewidths=1.0,
            zorder=5,
        )
        ax_yz.scatter(
            ys,
            zs,
            c="orange",
            s=s,
            marker="o",
            edgecolors="k",
            linewidths=1.0,
            zorder=5,
        )

        # Index labels
        pe = [path_effects.withStroke(linewidth=1.5, foreground="k")]
        for idx, (x, y, z) in enumerate(nodes3d):
            text = str(idx)
            ax3d.text(
                x,
                y,
                z,
                text,
                va="center",
                ha="center",
                color="orange",
                fontsize=8,
                weight="bold",
                path_effects=pe,
                zorder=6,
            )
            ax_xy.text(
                x,
                y,
                text,
                va="center",
                ha="center",
                color="orange",
                fontsize=8,
                weight="bold",
                path_effects=pe,
                zorder=6,
            )
            ax_xz.text(
                x,
                z,
                text,
                va="center",
                ha="center",
                color="orange",
                fontsize=8,
                weight="bold",
                path_effects=pe,
                zorder=6,
            )
            ax_yz.text(
                y,
                z,
                text,
                va="center",
                ha="center",
                color="orange",
                fontsize=8,
                weight="bold",
                path_effects=pe,
                zorder=6,
            )

    def _plot_graph_edges_3d(
        self,
        ax3d,
        ax_xy,
        ax_xz,
        ax_yz,
        graph_nodes: np.ndarray,
        graph_edges: np.ndarray,
    ) -> None:
        """
        Plot graph edges as orange lines in 3D + projections.
        """
        nodes = np.asarray(graph_nodes, dtype=float)
        edges = np.asarray(graph_edges, dtype=int)

        if nodes.ndim != 2 or nodes.shape[1] not in (2, 3):
            raise ValueError(
                "graph_nodes must have shape (num_nodes, 2) or (num_nodes, 3); "
                f"got {nodes.shape}"
            )
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError(
                "graph_edges must have shape (num_edges, 2); "
                f"got {edges.shape}"
            )

        if nodes.shape[1] == 3:
            nodes3d = nodes
        else:
            z_default = float(self.cave_map.grid_min[2])
            nodes3d = np.column_stack(
                [nodes[:, 0], nodes[:, 1], np.full(nodes.shape[0], z_default)]
            )

        xs = nodes3d[:, 0]
        ys = nodes3d[:, 1]
        zs = nodes3d[:, 2]
        num_nodes = nodes3d.shape[0]

        for i, j in edges:
            if not (0 <= i < num_nodes and 0 <= j < num_nodes):
                continue
            x0, y0, z0 = xs[i], ys[i], zs[i]
            x1, y1, z1 = xs[j], ys[j], zs[j]

            # 3D line
            ax3d.plot(
                [x0, x1],
                [y0, y1],
                [z0, z1],
                color="orange",
                linewidth=1.5,
                zorder=4,
            )

            # XY projection
            ax_xy.plot(
                [x0, x1],
                [y0, y1],
                color="orange",
                linewidth=1.5,
                zorder=4,
            )

            # XZ projection
            ax_xz.plot(
                [x0, x1],
                [z0, z1],
                color="orange",
                linewidth=1.5,
                zorder=4,
            )

            # YZ projection
            ax_yz.plot(
                [y0, y1],
                [z0, z1],
                color="orange",
                linewidth=1.5,
                zorder=4,
            )

    # ------------------------------------------------------------------ #
    # Frame rendering                                                    #
    # ------------------------------------------------------------------ #
    def render_frame(
        self,
        frame_idx: int,
        *,
        agent_positions: Optional[np.ndarray] = None,
        agent_colors: Optional[np.ndarray] = None,
        agent_alives: Optional[np.ndarray] = None,
        agent_communications_matrix: Optional[np.ndarray] = None,
        waypoint_positions: Optional[np.ndarray] = None,
        graph_nodes: Optional[np.ndarray] = None,
        graph_edges: Optional[np.ndarray] = None,
    ) -> Path:
        """
        Render a single frame.

        Agent inputs are OPTIONAL:
        - If agent_positions is None:
            * No agents, waypoints, or communications are drawn.
            * In 3D mode, slice is taken through the center of the cave extents.
        - If agent_positions is provided:
            * agent_colors and agent_alives must also be provided.
            * Full agent, waypoint, and communications behavior is used.

        Graph inputs are OPTIONAL:
        - graph_nodes: (num_nodes, 2 or 3)
        - graph_edges: (num_edges, 2) with integer node indices
        - If provided:
            * Nodes are drawn in orange with index labels.
            * Edges are drawn as orange lines between nodes.
        """
        # Normalize agent inputs (if provided)
        if agent_positions is not None:
            if agent_colors is None or agent_alives is None:
                raise ValueError(
                    "If agent_positions is provided, agent_colors and "
                    "agent_alives must also be provided."
                )

            positions, colors, alives, comm = self._validate_agent_inputs(
                agent_positions=agent_positions,
                agent_colors=agent_colors,
                agent_alives=agent_alives,
                agent_communications_matrix=agent_communications_matrix,
            )
        else:
            positions = None
            colors = None
            alives = None
            # Without positions, we cannot meaningfully render communications
            comm = None

        # Normalize waypoint_positions if provided
        if waypoint_positions is not None:
            wp = np.asarray(waypoint_positions, dtype=float)
            if wp.ndim != 2:
                raise ValueError(
                    "waypoint_positions must have shape (N, D); "
                    f"got shape {wp.shape}"
                )
            if wp.shape[1] not in (2, 3):
                raise ValueError(
                    "waypoint_positions must be 2D or 3D points; "
                    f"got D={wp.shape[1]}"
                )
            # If agents are present, enforce matching N
            if positions is not None and wp.shape[0] != positions.shape[0]:
                raise ValueError(
                    "waypoint_positions must have shape (N, D) with same N as "
                    f"agent_positions; got {wp.shape}, expected "
                    f"({positions.shape[0]}, D)"
                )
        else:
            wp = None

        # Normalize graph inputs if provided
        graph_nodes_arr = None
        graph_edges_arr = None
        if graph_nodes is not None:
            graph_nodes_arr = np.asarray(graph_nodes, dtype=float)
            if graph_nodes_arr.ndim != 2 or graph_nodes_arr.shape[1] not in (2, 3):
                raise ValueError(
                    "graph_nodes must have shape (num_nodes, 2) or (num_nodes, 3); "
                    f"got {graph_nodes_arr.shape}"
                )
        if graph_edges is not None:
            graph_edges_arr = np.asarray(graph_edges, dtype=int)
            # it can be empty
            if graph_edges_arr.size == 0:
                graph_edges_arr = graph_edges_arr.reshape((0, 2))
            if graph_edges_arr.ndim != 2 or graph_edges_arr.shape[1] != 2:
                raise ValueError(
                    "graph_edges must have shape (num_edges, 2); "
                    f"got {graph_edges_arr.shape}"
                )

        # ------------------------------------------------------------------ #
        # 2D mode                                                            #
        # ------------------------------------------------------------------ #
        if self.mode == "2d":
            ax = self.ax2d
            ax.clear()

            # Base occupancy plot
            self.plotter_2d.plot_occupancy(ax)

            # Agents (if any)
            if positions is not None:
                # Use XY only (drop Z if present)
                xs = positions[:, 0]
                ys = positions[:, 1]

                alive_mask = alives
                dead_mask = ~alives

                # Alive (circle with black outline)
                if np.any(alive_mask):
                    ax.scatter(
                        xs[alive_mask],
                        ys[alive_mask],
                        c=colors[alive_mask],
                        s=40,
                        marker="o",
                        edgecolors="k",  # black outline
                        linewidths=1.0,
                        zorder=3,
                    )

                # Dead agents: X with black outline AND color fill
                if np.any(dead_mask):
                    # 1) Draw thick black X underneath
                    ax.scatter(
                        xs[dead_mask],
                        ys[dead_mask],
                        c="black",
                        s=60,
                        marker="x",
                        linewidths=3.0,  # thick: becomes the outline
                        zorder=3,
                    )
                    # 2) Draw thinner colored X on top
                    ax.scatter(
                        xs[dead_mask],
                        ys[dead_mask],
                        c=colors[dead_mask],
                        s=60,
                        marker="x",
                        linewidths=2.0,  # thin colored X
                        zorder=4,  # ensure drawn on top
                    )

                # Waypoint markers: big hollow circles ("O") in agent color
                if wp is not None:
                    # Use XY only
                    wx = wp[:, 0]
                    wy = wp[:, 1]

                    # mask invalid waypoints (NaNs)
                    valid_mask = np.isfinite(wx) & np.isfinite(wy)

                    if np.any(valid_mask):
                        # Use per-agent colors if available; otherwise a default
                        if colors is not None and colors.shape[0] >= wp.shape[0]:
                            edgecols = colors[valid_mask]
                        else:
                            edgecols = "C0"

                        ax.scatter(
                            wx[valid_mask],
                            wy[valid_mask],
                            s=200,
                            facecolors="none",
                            edgecolors=edgecols,
                            linewidths=2.0,
                            marker="o",
                            zorder=2.5,
                        )

                # Communications: dashed bicolor lines between communicating ALIVE pairs
                if comm is not None:
                    N = positions.shape[0]
                    for i in range(N):
                        for j in range(i + 1, N):
                            if not (alive_mask[i] and alive_mask[j]):
                                continue
                            if not (comm[i, j] or comm[j, i]):
                                continue
                            p0 = np.array([xs[i], ys[i]])
                            p1 = np.array([xs[j], ys[j]])
                            c0 = colors[i]
                            c1 = colors[j]
                            self._draw_bicolor_dashed_line_2d(
                                ax, p0, p1, c0, c1, num_segments=20
                            )

                # Agent index labels (same style as before)
                pe = [path_effects.withStroke(linewidth=1.5, foreground="k")]
                for agent_idx, (x, y) in enumerate(zip(xs, ys)):
                    c = colors[agent_idx]
                    ax.text(
                        x,
                        y,
                        str(agent_idx),
                        va="center",
                        ha="center",
                        color=c,
                        fontsize=8,
                        weight="bold",
                        path_effects=pe,
                        zorder=5,
                    )

            # Graph nodes/edges (if provided)
            if graph_nodes_arr is not None:
                self._plot_graph_nodes_2d(ax, graph_nodes_arr)
            if graph_nodes_arr is not None and graph_edges_arr is not None:
                self._plot_graph_edges_2d(ax, graph_nodes_arr, graph_edges_arr)

            self.fig.suptitle(self.cave_name)
            self.fig.tight_layout()

            frame_path = self.frames_dir / f"{frame_idx:04d}.png"
            self.fig.savefig(frame_path, dpi=self.dpi)
            return frame_path

        # ------------------------------------------------------------------ #
        # 3D mode                                                            #
        # ------------------------------------------------------------------ #
        ax3d = self.ax3d
        ax_xy = self.ax_xy
        ax_xz = self.ax_xz
        ax_yz = self.ax_yz

        # Clear all axes
        ax3d.clear()
        ax_xy.clear()
        ax_xz.clear()
        ax_yz.clear()

        # 3D occupancy
        self.plotter_3d.plot_occupancy_3d(
            ax3d,
            clear=False,
            alpha=self.alpha_obstacles_3d,
        )

        # Ensure we have 3D positions for plotting/slicing
        if positions is not None:
            if positions.shape[1] == 3:
                positions3d = positions
            else:
                # If only XY provided, lift into 3D using grid_min[2]
                z_default = float(self.cave_map.grid_min[2])
                positions3d = np.column_stack(
                    [
                        positions[:, 0],
                        positions[:, 1],
                        np.full(len(positions), z_default),
                    ]
                )

            xs = positions3d[:, 0]
            ys = positions3d[:, 1]
            zs = positions3d[:, 2]

            alive_mask = alives
            dead_mask = ~alives
        else:
            # No agents: use empty arrays
            positions3d = np.empty((0, 3), dtype=float)
            xs = positions3d[:, 0]
            ys = positions3d[:, 1]
            zs = positions3d[:, 2]
            alive_mask = np.zeros(0, dtype=bool)
            dead_mask = np.zeros(0, dtype=bool)

        # Handle waypoint positions in 3D (if provided AND agents exist)
        if wp is not None and positions3d.shape[0] > 0:
            if wp.shape[1] == 3:
                wp3d = wp
            else:
                # Lift 2D waypoints into 3D using same z_default strategy
                z_default = float(self.cave_map.grid_min[2])
                wp3d = np.column_stack(
                    [wp[:, 0], wp[:, 1], np.full(wp.shape[0], z_default)]
                )
            wx = wp3d[:, 0]
            wy = wp3d[:, 1]
            wz = wp3d[:, 2]
            wp_valid_mask = np.isfinite(wx) & np.isfinite(wy) & np.isfinite(wz)
        else:
            wp3d = None
            wp_valid_mask = None
            wx = wy = wz = None  # for type checkers

        # Choose slice point:
        # - If there is at least one agent, use the first agent's position.
        # - Otherwise, use the center of the cave extents.
        if positions3d.shape[0] > 0:
            slice_point = positions3d[0]
        else:
            grid_min = np.asarray(self.cave_map.grid_min, dtype=float)
            grid_max = np.asarray(self.cave_map.grid_max, dtype=float)
            slice_point = 0.5 * (grid_min + grid_max)

        # XY / XZ / YZ slices through slice_point
        self.plotter_3d.plot_xy_slice_through_point(
            ax_xy,
            slice_point,
            clear=False,
            agent_0_color=agent_colors[0] if agent_colors is not None else None,
        )
        self.plotter_3d.plot_xz_slice_through_point(
            ax_xz,
            slice_point,
            clear=False,
            agent_0_color=agent_colors[0] if agent_colors is not None else None,
        )
        self.plotter_3d.plot_yz_slice_through_point(
            ax_yz,
            slice_point,
            clear=False,
            agent_0_color=agent_colors[0] if agent_colors is not None else None,
        )

        # This controls how much larger or smaller to make the markers
        marker_size_multiplier = 3

        # Alive agents: circles with black outline (3D + projections)
        if np.any(alive_mask):
            xs_alive = xs[alive_mask]
            ys_alive = ys[alive_mask]
            zs_alive = zs[alive_mask]
            colors_alive = colors[alive_mask]

            # 3D
            ax3d.scatter(
                xs_alive,
                ys_alive,
                zs_alive,
                c=colors_alive,
                s=40 * marker_size_multiplier,
                marker="o",
                edgecolors="k",
                linewidths=1,
                depthshade=False,
            )
            # Projections
            ax_xy.scatter(
                xs_alive,
                ys_alive,
                c=colors_alive,
                s=40 * marker_size_multiplier,
                marker="o",
                edgecolors="k",
                linewidths=1,
            )
            ax_xz.scatter(
                xs_alive,
                zs_alive,
                c=colors_alive,
                s=40 * marker_size_multiplier,
                marker="o",
                edgecolors="k",
                linewidths=1,
            )
            ax_yz.scatter(
                ys_alive,
                zs_alive,
                c=colors_alive,
                s=40 * marker_size_multiplier,
                marker="o",
                edgecolors="k",
                linewidths=1,
            )

        # Dead agents: X markers with black outline hack (3D + projections)
        if np.any(dead_mask):
            xs_dead = xs[dead_mask]
            ys_dead = ys[dead_mask]
            zs_dead = zs[dead_mask]
            colors_dead = colors[dead_mask]

            # 3D: black X under, colored X on top
            ax3d.scatter(
                xs_dead,
                ys_dead,
                zs_dead,
                c="black",
                s=70 * marker_size_multiplier,
                marker="x",
                linewidths=3.0,
            )
            ax3d.scatter(
                xs_dead,
                ys_dead,
                zs_dead,
                c=colors_dead,
                s=70 * marker_size_multiplier,
                marker="x",
                linewidths=2.0,
            )

            # XY projection
            ax_xy.scatter(
                xs_dead,
                ys_dead,
                c="black",
                s=70 * marker_size_multiplier,
                marker="x",
                linewidths=3.0,
            )
            ax_xy.scatter(
                xs_dead,
                ys_dead,
                c=colors_dead,
                s=70 * marker_size_multiplier,
                marker="x",
                linewidths=2.0,
            )

            # XZ projection
            ax_xz.scatter(
                xs_dead,
                zs_dead,
                c="black",
                s=70 * marker_size_multiplier,
                marker="x",
                linewidths=3.0,
            )
            ax_xz.scatter(
                xs_dead,
                zs_dead,
                c=colors_dead,
                s=70 * marker_size_multiplier,
                marker="x",
                linewidths=2.0,
            )

            # YZ projection
            ax_yz.scatter(
                ys_dead,
                zs_dead,
                c="black",
                s=70 * marker_size_multiplier,
                marker="x",
                linewidths=3.0,
            )
            ax_yz.scatter(
                ys_dead,
                zs_dead,
                c=colors_dead,
                s=70 * marker_size_multiplier,
                marker="x",
                linewidths=2.0,
            )

        # Regardless of alive or dead, put the agent's index
        # as a small number with the agent's fill color and a black outline.
        if positions3d.shape[0] > 0:
            pe = [path_effects.withStroke(linewidth=1.5, foreground="k")]
            for agent_idx, pos in enumerate(positions3d):
                x, y, z = pos
                c = colors[agent_idx]

                ax3d.text(
                    x,
                    y,
                    z,
                    str(agent_idx),
                    va="center",
                    ha="center",
                    color=c,
                    fontsize=8,
                    weight="bold",
                    path_effects=pe,
                    zorder=5,
                )
                ax_xy.text(
                    x,
                    y,
                    str(agent_idx),
                    va="center",
                    ha="center",
                    color=c,
                    fontsize=8,
                    weight="bold",
                    path_effects=pe,
                    zorder=5,
                )
                ax_xz.text(
                    x,
                    z,
                    str(agent_idx),
                    va="center",
                    ha="center",
                    color=c,
                    fontsize=8,
                    weight="bold",
                    path_effects=pe,
                    zorder=5,
                )
                ax_yz.text(
                    y,
                    z,
                    str(agent_idx),
                    va="center",
                    ha="center",
                    color=c,
                    fontsize=8,
                    weight="bold",
                    path_effects=pe,
                    zorder=5,
                )

        # Waypoints in 3D: big hollow circles ("O") in agent color
        if (
            positions3d.shape[0] > 0
            and wp3d is not None
            and wp_valid_mask is not None
            and np.any(wp_valid_mask)
        ):
            wx_valid = wx[wp_valid_mask]
            wy_valid = wy[wp_valid_mask]
            wz_valid = wz[wp_valid_mask]
            colors_valid = colors[wp_valid_mask]

            # 3D hollow circles
            ax3d.scatter(
                wx_valid,
                wy_valid,
                wz_valid,
                s=150,
                facecolors="none",
                edgecolors=colors_valid,
                linewidths=1.5,
                marker="o",
                depthshade=False,
            )

            # TODO: tiny little number indicating which agent / waypoint index

            # Projections
            ax_xy.scatter(
                wx_valid,
                wy_valid,
                s=120,
                facecolors="none",
                edgecolors=colors_valid,
                linewidths=1.5,
                marker="o",
            )
            ax_xz.scatter(
                wx_valid,
                wz_valid,
                s=120,
                facecolors="none",
                edgecolors=colors_valid,
                linewidths=1.5,
                marker="o",
            )
            ax_yz.scatter(
                wy_valid,
                wz_valid,
                s=120,
                facecolors="none",
                edgecolors=colors_valid,
                linewidths=1.5,
                marker="o",
            )

        # Communications: dashed bicolor lines between communicating ALIVE pairs
        if comm is not None and positions3d.shape[0] > 0:
            N = positions3d.shape[0]
            for i in range(N):
                for j in range(i + 1, N):
                    if not (alive_mask[i] and alive_mask[j]):
                        continue
                    if not (comm[i, j] or comm[j, i]):
                        continue
                    if i == j:
                        continue

                    p0 = positions3d[i]
                    p1 = positions3d[j]
                    c0 = colors[i]
                    c1 = colors[j]

                    # 3D dashed line
                    self._draw_bicolor_dashed_line_3d(
                        ax3d, p0, p1, c0, c1, num_segments=20
                    )
                    # Projected dashed lines
                    self._draw_bicolor_dashed_line_2d(
                        ax_xy,
                        p0[[0, 1]],
                        p1[[0, 1]],
                        c0,
                        c1,
                        num_segments=20,
                    )
                    self._draw_bicolor_dashed_line_2d(
                        ax_xz,
                        p0[[0, 2]],
                        p1[[0, 2]],
                        c0,
                        c1,
                        num_segments=20,
                    )
                    self._draw_bicolor_dashed_line_2d(
                        ax_yz,
                        p0[[1, 2]],
                        p1[[1, 2]],
                        c0,
                        c1,
                        num_segments=20,
                    )

        # Graph nodes/edges (if provided)
        if graph_nodes_arr is not None:
            self._plot_graph_nodes_3d(
                ax3d,
                ax_xy,
                ax_xz,
                ax_yz,
                graph_nodes_arr,
            )
        if graph_nodes_arr is not None and graph_edges_arr is not None:
            self._plot_graph_edges_3d(
                ax3d,
                ax_xy,
                ax_xz,
                ax_yz,
                graph_nodes_arr,
                graph_edges_arr,
            )

        self.fig.suptitle(f"{self.cave_name} – frame {frame_idx:03d}")
        self.fig.tight_layout()

        frame_path = self.frames_dir / f"{frame_idx:04d}.png"
        self.fig.savefig(frame_path, dpi=self.dpi)
        return frame_path

    # ------------------------------------------------------------------ #
    # Video export                                                       #
    # ------------------------------------------------------------------ #
    def finalize_video(
        self,
        *,
        output_name: Optional[str] = None,
    ) -> Path:
        """
        Convert all PNG frames in `frames_dir` into an MP4 video.

        Parameters
        ----------
        output_name : optional str
            Base name (without extension) for the output MP4. If None, we
            default to f"{self.cave_name}_{self.mode}.mp4".

        Returns
        -------
        video_output_path : Path
            Path to the generated MP4 file.
        """
        if output_name is None:
            output_name = f"{self.cave_name}_{self.mode}"

        video_output_path = self.log_dir / f"{output_name}.mp4"

        print(
            f"[PlottingOrchestrator] Converting frames in {self.frames_dir} "
            f"to video: {video_output_path}"
        )
        image_folder_to_video(
            filepath_image_folder=self.frames_dir,
            filepath_output_video=video_output_path,
            fps=self.fps,
        )
        print(f"[PlottingOrchestrator] Saved video to: {video_output_path}")
        return video_output_path
