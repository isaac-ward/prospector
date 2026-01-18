# src/prospector/blender_integration/api.py

from __future__ import annotations

from typing import Optional

from .blender_script import main


def run(
    *,
    log_folder: str,
    render: bool = False,
    render_first_n_frames: Optional[int] = None,
) -> None:
    main(
        log_folder=log_folder,
        render=render,
        render_first_n_frames=render_first_n_frames,
    )
