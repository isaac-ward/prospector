# utils/image_folder_to_video.py

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _collect_images(folder: Path) -> List[Path]:
    """
    Collect frame files from a directory, sorted numerically when possible.
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")

    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not files:
        raise FileNotFoundError(f"No image files found in: {folder}")

    def sort_key(path: Path):
        stem = path.stem
        try:
            return (0, int(stem))  # numeric sort
        except ValueError:
            return (1, stem)       # fallback lexicographic

    files.sort(key=sort_key)
    return files


def image_folder_to_video(
    filepath_image_folder: str | os.PathLike,
    filepath_output_video: str | os.PathLike,
    *,
    fps: int = 24,
) -> Path:
    """
    Generic converter: turn a folder of images into an MP4 video.

    Parameters
    ----------
    filepath_image_folder : str | Path
        Directory containing image frames.
    filepath_output_video : str | Path
        Desired output video path. If missing '.mp4', a warning is printed.
    fps : int, default=24
        Frames per second.

    Returns
    -------
    Path
        Path to the written MP4 file.
    """
    folder = Path(filepath_image_folder)
    output = Path(filepath_output_video)

    # Warn if user forgot extension
    if output.suffix.lower() != ".mp4":
        print(f"[warning] Output filename does not end with '.mp4': {output}")

    image_files = _collect_images(folder)
    clip = ImageSequenceClip([str(p) for p in image_files], fps=fps)

    # Don't want all the obnoxious logging from MoviePy here
    clip.write_videofile(
        str(output),
        fps=fps,
        codec="libx264",
        audio=False,
        logger=None,
    )
    clip.close()
    return output
