# src/prospector/tests/test_caves_matplotlib.py

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
    log_dir = make_log_dir(suffix="test_caves_matplotlib")
    print(f"[test_caves_matplotlib] Logging to: {log_dir}")

    repo_root = Path(get_repo_root_dir())

    # Only use these caves
    cave_names = ["chamber", "tunnels"]

    for cave_name in cave_names:
        print(f"[test_caves_matplotlib] Processing cave: {cave_name}")
        cave_cfg = args.caves[cave_name]

        # ------------------------------------------------------------------
        # 2D cave map + 2D orchestrator
        # ------------------------------------------------------------------
        cave_map_2d, ply_path = build_cavemap(
            cave_name=cave_name,
            cave_cfg=cave_cfg,
            repo_root=repo_root,
            use_3d=False,
        )
        print(f"  - PLY path: {ply_path}")

        orch_2d = PlottingOrchestrator(
            mode="2d",
            cave_map=cave_map_2d,
            cave_name=cave_name,
            log_dir=log_dir,
        )
        frame2d_path = orch_2d.render_frame(
            frame_idx=0
        )
        print(f"  - Saved 2D frame for {cave_name} to: {frame2d_path}")

        # ------------------------------------------------------------------
        # 3D cave map + 3D orchestrator / animation
        # ------------------------------------------------------------------
        cave_map_3d, _ = build_cavemap(
            cave_name=cave_name,
            cave_cfg=cave_cfg,
            repo_root=repo_root,
            use_3d=True,
        )

        max_obstacle_points = cave_cfg.plotting.obstacle_points.max_num
        alpha_obstacles = cave_cfg.plotting.obstacle_points.alpha

        orch_3d = PlottingOrchestrator(
            mode="3d",
            cave_map=cave_map_3d,
            cave_name=cave_name,
            log_dir=log_dir,
            fps=24,
            max_obstacle_points_3d=max_obstacle_points,
            alpha_obstacles_3d=alpha_obstacles,
        )

        # Agent positions: linearly from grid_min to grid_max
        min_xyz = cave_map_3d.grid_min
        max_xyz = cave_map_3d.grid_max
        num_frames = 100
        agent_positions = np.linspace(min_xyz, max_xyz, num=num_frames)
        
        orch_3d.render_frame(
            frame_idx=0, 
        )

    print("[test_caves_matplotlib] Done.")


if __name__ == "__main__":
    main()
