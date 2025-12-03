# src/prospector/tests/test_caves_matplotlib.py

from __future__ import annotations

import inspect
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import hydra

from prospector.utils.custom_logging import make_log_dir, get_repo_root_dir
from prospector.utils.video import image_folder_to_video
from prospector.utils.cacher import Cacher

from prospector.caves.cave_map_2d import CaveMap2D
from prospector.caves.cave_map_3d import CaveMap3D
from prospector.caves.plotting.cave_map_2d_plotting import CaveMap2DPlotter
from prospector.caves.plotting.cave_map_3d_plotting import CaveMap3DPlotter


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
    log_dir = make_log_dir(prefix="test_caves_matplotlib")
    print(f"[test_caves_matplotlib] Logging to: {log_dir}")

    repo_root = Path(get_repo_root_dir())

    # Only use these caves
    cave_names = ["chamber", "tunnels"]

    for cave_name in cave_names:
        print(f"[test_caves_matplotlib] Processing cave: {cave_name}")
        cave_cfg = args.caves[cave_name]

        # Resolve PLY path relative to repo root
        rel_path = str(cave_cfg.relative_path)
        ply_path = repo_root / rel_path.lstrip("/")
        print(f"  - PLY path: {ply_path}")

        voxel_size = float(cave_cfg.voxel_size)
        known_internal_xyz = list(cave_cfg.known_internal_xyz)

        # ------------------------------------------------------------------
        # 2D map: slice around cave_cfg.slice_for_2d.z with slice thickness
        # ------------------------------------------------------------------
        slice_z = float(cave_cfg.slice_for_2d.z)
        slice_thickness = float(cave_cfg.slice_for_2d.thickness)

        print(
            f"  - Building 2D map (z={slice_z}, thickness={slice_thickness}) "
            f"with caching..."
        )

        cacher_2d = Cacher(
            computation_inputs=(
                "CaveMap2D_v1",      # version tag for cache invalidation
                cave_name,
                str(ply_path),
                voxel_size,
                slice_z,
                slice_thickness,
                tuple(known_internal_xyz),
            ),
            tag="cavemap2d",
        )

        if cacher_2d.exists():
            cache_2d = cacher_2d.load()
            cave_map_2d: CaveMap2D = cache_2d["cave_map_2d"]
        else:
            cave_map_2d = CaveMap2D.from_ply(
                ply_path=ply_path,
                voxel_size=voxel_size,
                slice_for_2d_z=slice_z,
                slice_for_2d_thickness=slice_thickness,
            )
            cave_map_2d.label_accessible_space(known_internal_xyz)

            cacher_2d.save({"cave_map_2d": cave_map_2d})

        plotter_2d = CaveMap2DPlotter(cave_map_2d)

        fig2d, ax2d = plt.subplots(figsize=(6, 6))
        plotter_2d.plot_occupancy(ax2d)
        fig2d.suptitle(f"{cave_name} – 2D occupancy")
        fig2d.tight_layout()

        out_2d_path = log_dir / f"{cave_name}_2d.png"
        fig2d.savefig(out_2d_path, dpi=200, bbox_inches="tight")
        plt.close(fig2d)
        print(f"  - Saved 2D occupancy image to: {out_2d_path}")

        # ------------------------------------------------------------------
        # 3D map: full voxelization
        # ------------------------------------------------------------------
        print("  - Building 3D map with caching...")

        cacher_3d = Cacher(
            computation_inputs=(
                "CaveMap3D_v1",      # version tag
                cave_name,
                str(ply_path),
                voxel_size,
                tuple(known_internal_xyz),
            ),
            tag="cavemap3d",
        )

        if cacher_3d.exists():
            cache_3d = cacher_3d.load()
            cave_map_3d: CaveMap3D = cache_3d["cave_map_3d"]
        else:
            cave_map_3d = CaveMap3D.from_ply(
                ply_path=ply_path,
                voxel_size=voxel_size,
            )
            cave_map_3d.label_accessible_space(known_internal_xyz)

            cacher_3d.save({"cave_map_3d": cave_map_3d})

        plotter_3d = CaveMap3DPlotter(
            cave_map_3d,
            max_obstacle_points=cave_cfg.plotting.obstacle_points.max_num,
        )

        # Agent positions: linearly from grid_min to grid_max
        min_xyz = cave_map_3d.grid_min
        max_xyz = cave_map_3d.grid_max
        num_frames = 100

        agent_positions = np.linspace(min_xyz, max_xyz, num=num_frames)

        # Frame folder and video output path
        frames_dir = log_dir / f"{cave_name}_3d_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        video_output_path = log_dir / f"{cave_name}_3d.mp4"

        print(
            f"  - Rendering 3D + slice animation "
            f"({num_frames} frames) to: {frames_dir}"
        )

        # ------------------------------------------------------------------
        # Render frames: 4x3 GridSpec
        #   - Top 3x3: 3D plot
        #   - Bottom row: XY, XZ, YZ slices
        # ------------------------------------------------------------------
        for frame_idx, agent_pos in enumerate(
            tqdm(agent_positions, desc=f"Rendering {cave_name} animation", unit="frame")
        ):
            fig = plt.figure(figsize=(10, 12))
            gs = GridSpec(4, 3, figure=fig)

            # 3D plot occupies top 3 rows, all 3 columns
            ax3d = fig.add_subplot(gs[0:3, 0:3], projection="3d")

            # Bottom row: XY, XZ, YZ slices
            ax_xy = fig.add_subplot(gs[3, 0])
            ax_xz = fig.add_subplot(gs[3, 1])
            ax_yz = fig.add_subplot(gs[3, 2])

            # Draw current frame
            plotter_3d.plot_occupancy_3d(
                ax3d, 
                clear=True,
                alpha=cave_cfg.plotting.obstacle_points.alpha
            )
            plotter_3d.plot_xy_slice_through_point(ax_xy, agent_pos, clear=True)
            plotter_3d.plot_xz_slice_through_point(ax_xz, agent_pos, clear=True)
            plotter_3d.plot_yz_slice_through_point(ax_yz, agent_pos, clear=True)

            fig.suptitle(f"{cave_name} – frame {frame_idx:03d}")
            fig.tight_layout()

            frame_path = frames_dir / f"{frame_idx:04d}.png"
            fig.savefig(frame_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        # ------------------------------------------------------------------
        # Convert image sequence to MP4
        # ------------------------------------------------------------------
        print(f"  - Converting frames to video: {video_output_path}")
        image_folder_to_video(
            filepath_image_folder=frames_dir,
            filepath_output_video=video_output_path,
            fps=24,
        )
        print(f"  - Saved video for {cave_name} to: {video_output_path}")

    print("[test_caves_matplotlib] Done.")


if __name__ == "__main__":
    main()
