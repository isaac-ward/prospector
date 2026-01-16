# src/prospector/utils/custom_logging.py

import os 
import time 
import datetime
import pathlib
import tempfile
import shutil
import wandb
import numpy as np
import matplotlib.pyplot as plt

def get_repo_root_dir():
    # We're in repo/src/project_name/utils/logging.py
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def get_timestamp():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def make_log_dir(prefix="run", suffix=""):
    log_dir = os.path.join(get_repo_root_dir(), "logs")

    # Make a timestamped subdirectory
    suffix = f"_{suffix}" if suffix else ""
    prefix = f"{prefix}_" if prefix else ""
    foldername = f"{prefix}{get_timestamp()}{suffix}"
    full_path = os.path.join(log_dir, foldername)
    os.makedirs(full_path, exist_ok=True)
    # Return as a pathlib.Path object
    return pathlib.Path(full_path)

def get_saved_dir():
    return os.path.join(get_repo_root_dir(), "saved")

def get_cache_dir():
    return os.path.join(get_saved_dir(), "cache")

def get_weights_dir():
    return os.path.join(get_saved_dir(), "weights")

def get_assets_dir():
    return os.path.join(get_repo_root_dir(), "src", "assets")

def log_figure_to_wandb(figure, key: str):
    """
    Save a Matplotlib figure to a temporary PNG (dpi=600), log it to W&B, then close the figure.
    """
    # Create a temp file path
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp_path = tmp.name
    tmp.close()

    try:
        # Save figure at high resolution
        figure.savefig(tmp_path, dpi=300, bbox_inches="tight")
        # Log to wandb with global step
        wandb.log({key: wandb.Image(tmp_path)})
    finally:
        # Close and clean up
        plt.close(figure)
        try:
            os.remove(tmp_path)
        except OSError:
            pass

def log_video_to_wandb(clip, key: str):
    """
    Save a MoviePy clip to a temporary MP4, log it to W&B, then clean up.
    Uses the clip's own fps/size.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_path = tmp.name
    tmp.close()

    try:
        clip.write_videofile(
            tmp_path,
            codec="libx264",
            audio=False,
            logger=None
        )
        wandb.log({key: wandb.Video(tmp_path, format="mp4")})
    finally:
        try:
            clip.close()   # make sure resources (ffmpeg, readers) are freed
        except Exception:
            pass
        try:
            os.remove(tmp_path)
        except OSError:
            pass
