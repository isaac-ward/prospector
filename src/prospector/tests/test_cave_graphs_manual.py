from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import hydra
import json

from prospector.utils.custom_logging import make_log_dir, get_repo_root_dir, get_assets_dir, get_assets_dir
from prospector.utils.plotting_orchestrator import PlottingOrchestrator
from prospector.caves.utils.build_cavemap import build_cavemap
from prospector.caves.utils.cave_to_graph import build_and_export_cave_graph


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(args: DictConfig):
    # ------------------------------------------------------------------ #
    # Print full config                                                  #
    # ------------------------------------------------------------------ #
    yaml_str = OmegaConf.to_yaml(args)
    print(f"[{inspect.stack()[0][3]}] configuration:\n{yaml_str}")

    # ------------------------------------------------------------------ #
    # Setup logging directory                                            #
    # ------------------------------------------------------------------ #
    log_dir = make_log_dir(prefix="test_cave_graphs_manual")
    print(f"[test_cave_graphs_manual] Logging to: {log_dir}")

    repo_root = Path(get_repo_root_dir())
    assets_dir = Path(get_assets_dir())
    log_dir = Path(log_dir)

    cave_names = ["chamber", "tunnels"]
    num_nodes_list = [15]

    rng = np.random.default_rng(seed=args.get("seed", 42))

    # ------------------------------------------------------------------ #
    # Iterate through each cave and load the graph representation        #
    # ------------------------------------------------------------------ #
    for cave_name in cave_names:
        print(f"\n[test_cave_graphs_manual] Building maps for cave='{cave_name}' ...")


        for dim in ["2d", "3d"]:
            print(f"[test_cave_graphs_manual]   Processing {dim} graphs...")
            
            # Get the cave information
            cave, _ = build_cavemap(cave_cfg=args.caves[cave_name], cave_name=cave_name, repo_root=repo_root, use_3d=(dim=="3d"))

            cave_log_root = log_dir / cave_name / dim
            cave_log_root.mkdir(parents=True, exist_ok=True)

            for n in tqdm(num_nodes_list, desc=f"{cave_name}: graphs", unit="graph"):
                # Find the file in assets   
                graph_path_input = assets_dir / f"graphs" / f"n={n}" / f"{cave_name}_{dim}.json"

                # And where will we export to
                # Note that we will export an image, and the 
                # npz file (nodes, edges, comms matrix)
                graph_path_output = cave_log_root / f"{n}"
                graph_path_output.mkdir(parents=True, exist_ok=True)

                # Load the data
                with open(graph_path_input, "r") as f:
                    graph_data = json.load(f)
                nodes = np.array(graph_data["nodes"])
                edges = np.array(graph_data["edges"])

                # Compute the communications matrix
                comms_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)
                # The communications matrix is 1s down the
                # diagonal and symmetric, so we can add the
                # transpose and it's fine
                pbar = tqdm(total=len(nodes)**2, desc="    Computing comms matrix", unit="pair", leave=True)
                for i in range(len(nodes)):
                    for j in range(len(nodes)):
                        pbar.update(1)
                        if i == j:
                            comms_matrix[i, j] = 1
                            continue
                        # Use the cave 'has_line_of_sight' method
                        if cave.has_line_of_sight(nodes[i], nodes[j]):
                            comms_matrix[i, j] = 1
                            comms_matrix[j, i] = 1
                pbar.close()

                # We will use a plot orchestrator to 
                # plot the cave and the graph representation,
                # one where the edges are the traversibility
                # edges, and one where they're the 
                # communications edges
                orchestrator = PlottingOrchestrator(
                    mode=dim,
                    cave_map=cave,
                    cave_name=cave_name,
                    log_dir=graph_path_output,
                    fps=24,
                    max_obstacle_points_3d=args.caves[cave_name].plotting.obstacle_points.max_num,
                    alpha_obstacles_3d=args.caves[cave_name].plotting.obstacle_points.alpha,
                )
                orchestrator.render_frame(
                    frame_idx=0,
                    graph_nodes=nodes,
                    graph_edges=edges,
                )
                # Rename 0000.png to graph_traversibility.png
                (graph_path_output / f"{cave_name}_{dim}_frames" / "0000.png").rename(graph_path_output / "graph_traversibility.png")
                orchestrator.render_frame(
                    frame_idx=0,
                    graph_nodes=nodes,
                    graph_edges=[ (i, j) for i in range(len(nodes)) for j in range(len(nodes)) if comms_matrix[i, j] == 1 and i != j ],
                )
                # Rename 0000.png to graph_communications.png
                (graph_path_output / f"{cave_name}_{dim}_frames" / "0000.png").rename(graph_path_output / "graph_communications.png")

                # Save the npz file with keys nodes, edges, and comms 
                np.savez_compressed(
                    graph_path_output / "graph.npz",
                    nodes=nodes,
                    edges=edges,
                    comms=comms_matrix,
                )


    print("\n[test_cave_graphs_manual] Done.")


if __name__ == "__main__":
    main()
