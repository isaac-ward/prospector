# src/prospector/blender_integration/api.py

from __future__ import annotations

from typing import Optional

from .blender_script import main


def run(
    *,
    log_folder: str,
    render: bool = False,
    render_first_n_frames: Optional[int] = None,
    skip_existing_renders: bool = True,
    video_fps: Optional[int] = None,
    sim_dt: Optional[float] = None,
    views: Optional[list] = None,
    render_last_frame_only: bool = False,
) -> None:
    main(
        log_folder=log_folder,
        render=render,
        render_first_n_frames=render_first_n_frames,
        skip_existing_renders=skip_existing_renders,
        video_fps=video_fps,
        sim_dt=sim_dt,
        views=views,
        render_last_frame_only=render_last_frame_only,
    )
