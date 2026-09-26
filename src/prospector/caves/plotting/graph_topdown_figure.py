# src/prospector/caves/plotting/graph_topdown_figure.py
"""
Traversability / coordination graph figures (black = rock, white = free,
orange graph), drawn with the PlottingOrchestrator's 2D graph rendering, with
no axes or title (paper figures).

    uv run python -m prospector.caves.plotting.graph_topdown_figure \
        <cave> <graph.json> <out_dir> [slice|footprint]

- slice     : the cave's configured 2D slice map (as for tunnels / chamber)
- footprint : top-down footprint of the 3D navigable volume (a cell is free if
              there is reachable free space at any height). For strongly 3D
              caves (e.g. grapevine) a single slice cuts through most nodes.

Writes <n>_graph_traversibility.png and <n>_graph_communications.png.
Coordination edges connect node pairs with line of sight on the cave map
(3D map for footprint mode).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from prospector.caves.cave_map_2d import CaveMap2D
from prospector.caves.utils.build_cavemap import build_cavemap
from prospector.utils.custom_logging import get_repo_root_dir
from prospector.utils.plotting_orchestrator import PlottingOrchestrator


def footprint_map(cave3d) -> CaveMap2D:
    free = (cave3d.occupancy == 0).any(axis=2)
    return CaveMap2D(
        voxel_size=cave3d.voxel_size,
        grid_min_xy=np.asarray(cave3d.grid_min[:2]),
        grid_max_xy=np.asarray(cave3d.grid_max[:2]),
        slice_z=float("nan"),
        slice_thickness=float("nan"),
        occupancy=np.where(free, 0, 1).astype(np.uint8),
        ply_path=cave3d.ply_path,
    )


def main() -> None:
    cave_name, graph_file, out_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    mode = sys.argv[4] if len(sys.argv) > 4 else "slice"
    repo = Path(get_repo_root_dir())
    cfg = OmegaConf.load(repo / "src/prospector/conf/config.yaml")
    cave_cfg = cfg.caves[cave_name]

    g = json.loads((graph_file if graph_file.is_absolute() else repo / graph_file).read_text())
    nodes = np.asarray(g["nodes"], dtype=float)
    trav = np.asarray(g["edges"], dtype=int).reshape(-1, 2)

    if mode == "footprint":
        los_map, _ = build_cavemap(cave_name, cave_cfg, repo_root=repo, use_3d=True)
        plot_map = footprint_map(los_map)
    else:
        plot_map, _ = build_cavemap(cave_name, cave_cfg, repo_root=repo, use_3d=False)
        los_map = plot_map
    los_nodes = nodes[:, : 3 if mode == "footprint" else 2]
    comm = np.asarray(
        [(i, j) for i in range(len(nodes)) for j in range(i + 1, len(nodes))
         if los_map.has_line_of_sight(los_nodes[i], los_nodes[j])],
        dtype=int,
    ).reshape(-1, 2)

    out_dir.mkdir(parents=True, exist_ok=True)
    orch = PlottingOrchestrator("2d", cave_map=plot_map, cave_name=cave_name, log_dir=out_dir)
    for kind, edges in (("traversibility", trav), ("communications", comm)):
        orch.render_frame(frame_idx=0, graph_nodes=nodes[:, :2], graph_edges=edges)
        # Paper figure: no axes, labels or title, just the map and the graph
        ax = orch.ax2d
        ax.axis("off")
        orch.fig.suptitle("")
        # Keep nodes on the map edge (and their labels) in frame; the margin
        # beyond the map is drawn black, like rock
        pad = 1.0
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_xlim(min(x0, nodes[:, 0].min() - pad), max(x1, nodes[:, 0].max() + pad))
        ax.set_ylim(min(y0, nodes[:, 1].min() - pad), max(y1, nodes[:, 1].max() + pad))
        orch.fig.savefig(
            out_dir / f"{len(nodes)}_graph_{kind}.png", dpi=150,
            bbox_inches="tight", pad_inches=0.0, facecolor="black",
        )
    print(f"[graph_topdown_figure] {cave_name} ({mode}): {len(nodes)} nodes, {len(trav)} "
          f"traversability edges, {len(comm)} coordination edges -> {out_dir}")


if __name__ == "__main__":
    main()
