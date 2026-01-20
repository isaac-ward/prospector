from __future__ import annotations

import os
from pathlib import Path
from typing import List

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from tqdm import tqdm
from proglog import ProgressBarLogger

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


class _TqdmMoviePyLogger(ProgressBarLogger):
    """
    MoviePy progress logger that updates a tqdm bar.

    MoviePy/Proglog may report progress either as:
      - bar='t', attr='index'  (integer step counter)
      - bar='t', attr='progress' (float 0..1)
      - other variants depending on MoviePy version
    We handle both.
    """
    def __init__(self, *, total_frames: int, desc: str) -> None:
        super().__init__()
        self._total = int(total_frames)
        self._pbar = tqdm(total=self._total, desc=desc, unit="frame")
        self._last_n = 0  # last frame count reflected in the bar

    def _set_to(self, n: int) -> None:
        n = max(0, min(self._total, int(n)))
        delta = n - self._last_n
        if delta > 0:
            self._pbar.update(delta)
            self._last_n = n

    def bars_callback(self, bar, attr, value, old_value=None):
        # Most common: frames index
        if attr == "index":
            try:
                self._set_to(int(value))
            except Exception:
                return

        # Some versions report progress as 0..1
        if attr == "progress":
            try:
                p = float(value)
                self._set_to(int(round(p * self._total)))
            except Exception:
                return

    def close(self) -> None:
        try:
            self._pbar.close()
        except Exception:
            pass


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

    if output.suffix.lower() != ".mp4":
        print(f"[warning] Output filename does not end with '.mp4': {output}")

    image_files = _collect_images(folder)

    # Helpful sanity: show which ffmpeg moviepy will try to use
    try:
        from moviepy.config import get_setting
        print(f"[image_folder_to_video] MoviePy FFMPEG_BINARY: {get_setting('FFMPEG_BINARY')}")
    except Exception:
        pass

    print(f"[image_folder_to_video] Image collected, writing video to: {output} ...")

    clip = ImageSequenceClip([str(p) for p in image_files], fps=fps)

    logger = _TqdmMoviePyLogger(total_frames=len(image_files), desc=f"Writing {output.name}")

    try:
        clip.write_videofile(
            str(output),
            fps=fps,
            codec="libx264",
            audio=False,
            logger=logger,  # keep it minimal, but real
        )
    finally:
        try:
            clip.close()
        finally:
            logger.close()

    if not output.exists():
        raise RuntimeError(f"MoviePy finished but output file does not exist: {output}")

    return output
