from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import hydra

from prospector.utils.custom_logging import make_log_dir, get_repo_root_dir, get_assets_dir
from prospector.utils.plotting_orchestrator import PlottingOrchestrator
from prospector.caves.utils.build_cavemap import build_cavemap
from prospector.caves.cave_map_3d import CaveMap3D


# ---------------------------------------------------------------------- #
# Which caves + sizes to generate 3D docs / graph.npz for.               #
#                                                                        #
#  - grapevine-exped-3: normal path -- line-of-sight communications are  #
#    computed on the cave map, traversibility edges come from the JSON.  #
#  - combined: HAND-SPECIFIED. combined.ply's tunnels<->connector throat  #
#    has no straight line-of-sight and the cloud leaks under flood-fill,  #
#    so the graph (nodes, traversibility edges, and communication matrix  #
#    A) is taken verbatim from src/assets/graphs/n=*/combined_3d_graph.npz #
#    (built to the rules in graphs/docs/hand_specification.txt). No LoS   #
#    is recomputed for combined; the map is loaded only as a plot backdrop #
#    (no flood-fill).                                                     #
# ---------------------------------------------------------------------- #
CAVE_NODES = {
    "combined": [35, 40],
    "grapevine-exped-3": [20, 25, 30],
}
DIM = "3d"


def comms_single(cave: CaveMap3D, nodes: np.ndarray) -> np.ndarray:
    """All-pairs line-of-sight communications matrix on one cave map."""
    n = len(nodes)
    A = np.eye(n, dtype=int)
    pbar = tqdm(total=n * (n - 1) // 2, desc="    comms", unit="pair", leave=False)
    for i in range(n):
        for j in range(i + 1, n):
            pbar.update(1)
            if cave.has_line_of_sight(nodes[i], nodes[j]):
                A[i, j] = A[j, i] = 1
    pbar.close()
    return A


def render_pair(orchestrator, nodes: np.ndarray, edges, out_path: Path) -> None:
    """Render one frame with the given edges and move it to out_path."""
    frame_path = orchestrator.render_frame(frame_idx=0, graph_nodes=nodes, graph_edges=edges)
    Path(frame_path).rename(out_path)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(args: DictConfig):
    yaml_str = OmegaConf.to_yaml(args)
    print(f"[{inspect.stack()[0][3]}] configuration:\n{yaml_str}")

    log_dir = Path(make_log_dir(prefix="test_cave_graphs_manual"))
    print(f"[test_cave_graphs_manual] Logging to: {log_dir}")

    repo_root = Path(get_repo_root_dir())
    assets_dir = Path(get_assets_dir())
    docs_root = assets_dir / "graphs" / "docs"

    for cave_name, sizes in CAVE_NODES.items():
        print(f"\n[test_cave_graphs_manual] === {cave_name} ({DIM}) ===")
        is_combined = cave_name == "combined"

        # ---- cave map --------------------------------------------------- #
        if is_combined:
            # backdrop only -- do NOT flood-fill (combined.ply leaks; the graph
            # is hand-specified, not derived from this map's line-of-sight).
            rel = str(args.caves[cave_name].relative_path).lstrip("/")
            cave = CaveMap3D.from_ply(
                ply_path=repo_root / rel, voxel_size=float(args.caves[cave_name].voxel_size)
            )
        else:
            cave, _ = build_cavemap(
                cave_cfg=args.caves[cave_name], cave_name=cave_name, repo_root=repo_root, use_3d=True
            )
        plot_cfg = args.caves[cave_name].plotting

        docs_dir = docs_root / cave_name / DIM
        docs_dir.mkdir(parents=True, exist_ok=True)

        for n in tqdm(sizes, desc=f"{cave_name}: graphs", unit="graph"):
            if is_combined:
                # hand-specified nodes / traversibility edges / communication matrix
                d = np.load(assets_dir / "graphs" / f"n={n}" / "combined_3d_graph.npz")
                nodes, edges, comms_matrix = d["nodes"], d["edges"], d["A"]
            else:
                gd = json.loads(
                    (assets_dir / "graphs" / f"n={n}" / f"{cave_name}_{DIM}.json").read_text()
                )
                nodes = np.array(gd["nodes"])
                edges = np.array(gd["edges"])
                comms_matrix = comms_single(cave, nodes)

            out_dir = log_dir / cave_name / DIM / f"{n}"
            out_dir.mkdir(parents=True, exist_ok=True)
            frames_dir = out_dir / f"{cave_name}_{DIM}_frames"

            orchestrator = PlottingOrchestrator(
                mode=DIM,
                cave_map=cave,
                cave_name=cave_name,
                log_dir=out_dir,
                fps=24,
                max_obstacle_points_3d=plot_cfg.obstacle_points.max_num,
                alpha_obstacles_3d=plot_cfg.obstacle_points.alpha,
            )

            trav_png = out_dir / "graph_traversibility.png"
            render_pair(orchestrator, nodes, edges, trav_png)

            comm_edges = [
                (i, j)
                for i in range(len(nodes))
                for j in range(len(nodes))
                if comms_matrix[i, j] == 1 and i != j
            ]
            comm_png = out_dir / "graph_communications.png"
            render_pair(orchestrator, nodes, comm_edges, comm_png)

            shutil.rmtree(frames_dir)

            # ---- npz alongside the run (nodes, traversibility edges, comm A) ---- #
            np.savez_compressed(
                out_dir / "graph.npz",
                nodes=nodes,
                edges=edges,
                A=comms_matrix,
                comms=comms_matrix,
            )

            shutil.copy(trav_png, docs_dir / f"{n}_graph_traversibility.png")
            shutil.copy(comm_png, docs_dir / f"{n}_graph_communications.png")
            print(f"[test_cave_graphs_manual]   {cave_name} n={n}: "
                  f"docs -> {docs_dir}; graph.npz -> {out_dir}")

    print("\n[test_cave_graphs_manual] Done.")


if __name__ == "__main__":
    main()
