from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import hydra

from prospector.utils.custom_logging import make_log_dir, get_repo_root_dir
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
    log_dir = make_log_dir(prefix="test_cave_graphs_automatic")
    print(f"[test_cave_graphs_automatic] Logging to: {log_dir}")

    repo_root = Path(get_repo_root_dir())
    log_dir = Path(log_dir)

    cave_names = ["chamber", "tunnels"]
    num_nodes_list = [5, 10, 15, 20, 25, 30]

    rng = np.random.default_rng(seed=args.get("seed", 42))

    # ------------------------------------------------------------------ #
    # Iterate through each cave and compute the graph representation     #
    # ------------------------------------------------------------------ #
    for cave_name in cave_names:
        print(f"\n[test_cave_graphs_automatic] Building maps for cave='{cave_name}' ...")

        # build_cavemap is expected to return (CaveMap2D, CaveMap3D)
        cave2d, _ = build_cavemap(cave_cfg=args.caves[cave_name], cave_name=cave_name, repo_root=repo_root, use_3d=False)
        cave3d, _ = build_cavemap(cave_cfg=args.caves[cave_name], cave_name=cave_name, repo_root=repo_root, use_3d=True)

        cave_log_root = log_dir / cave_name
        cave_log_root.mkdir(parents=True, exist_ok=True)

        # 2D graphs
        export_2d = cave_log_root / "2d_graphs"
        export_2d.mkdir(parents=True, exist_ok=True)

        # 3D graphs
        export_3d = cave_log_root / "3d_graphs"
        export_3d.mkdir(parents=True, exist_ok=True)

        print(f"[test_cave_graphs_automatic]   2D and 3D export dirs:")
        print(f"    2D: {export_2d}")
        print(f"    3D: {export_3d}")

        for n in tqdm(num_nodes_list, desc=f"{cave_name}: graphs", unit="graph"):
            # 2D graph
            print(f"[test_cave_graphs_automatic]   Cave='{cave_name}' 2D, N={n}")
            build_and_export_cave_graph(
                cave_map=cave2d,
                num_nodes=n,
                export_folder=export_2d,
                cave_name=f"{cave_name}_2d",
                rng=rng,
            )

            # 3D graph
            print(f"[test_cave_graphs_automatic]   Cave='{cave_name}' 3D, N={n}")
            build_and_export_cave_graph(
                cave_map=cave3d,
                num_nodes=n,
                export_folder=export_3d,
                cave_name=f"{cave_name}_3d",
                rng=rng,
            )

    print("\n[test_cave_graphs_automatic] Done.")


if __name__ == "__main__":
    main()
