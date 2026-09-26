# src/prospector/tests/test_blender_outputs_to_video.py

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, List

from omegaconf import OmegaConf, DictConfig
import hydra

from prospector.utils.video import image_folder_to_video
from prospector.utils.custom_logging import get_repo_root_dir


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def resolve_log_folder(repo_root: Path, folder_name: str) -> Path:
    """
    Resolve <folder_name> under:
      - repo_root/saved/<folder_name>
      - repo_root/logs/<folder_name>
    """
    candidates = [
        repo_root / "saved" / folder_name,
        repo_root / "logs" / folder_name,
    ]

    for c in candidates:
        if c.exists():
            return c

    tried = "\n  - " + "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find log folder '{folder_name}'. Tried:{tried}"
    )


def list_image_files(folder: Path) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def _require_video(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Expected video not found: {path}")
    return path

def composite_blender_renders(
    blender_renders_dir: Path,
    *,
    fps: int,
    out_name: str = "composite.mp4",
) -> Path:
    """
    Build a composite video:

      [ overhead (SxS) ] [ a0_top (S/2) | a0_pov (S/2) ]
      [      left      ] [ a1_top (S/2) | a1_pov (S/2) ]

    Output: (2S) x S
    """
    # MoviePy v2: import from moviepy (moviepy.editor removed)
    from moviepy import VideoFileClip, CompositeVideoClip, ColorClip  # :contentReference[oaicite:3]{index=3}

    def _require_video(path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError(f"Expected video not found: {path}")
        return path

    overhead_path = _require_video(blender_renders_dir / "overhead.mp4")
    a0_top_path = _require_video(blender_renders_dir / "agent_0_top_down.mp4")
    a1_top_path = _require_video(blender_renders_dir / "agent_1_top_down.mp4")
    a0_pov_path = _require_video(blender_renders_dir / "agent_0_pov.mp4")
    a1_pov_path = _require_video(blender_renders_dir / "agent_1_pov.mp4")

    # Load
    overhead = VideoFileClip(str(overhead_path))
    a0_top = VideoFileClip(str(a0_top_path))
    a1_top = VideoFileClip(str(a1_top_path))
    a0_pov = VideoFileClip(str(a0_pov_path))
    a1_pov = VideoFileClip(str(a1_pov_path))

    # Canonical square size from overhead
    S = int(overhead.w)
    if overhead.w != overhead.h:
        raise ValueError(f"Expected overhead square, got {overhead.w}x{overhead.h}")

    tile = S // 2
    out_w, out_h = 2 * S, S

    # Ensure all inputs are SxS (v2: resize -> resized)
    def force_square(clip):
        if clip.w != S or clip.h != S:
            return clip.resized((S, S))  # :contentReference[oaicite:4]{index=4}
        return clip

    overhead = force_square(overhead)
    a0_top = force_square(a0_top)
    a1_top = force_square(a1_top)
    a0_pov = force_square(a0_pov)
    a1_pov = force_square(a1_pov)

    # Tile down to half-size
    a0_top_t = a0_top.resized((tile, tile))  # :contentReference[oaicite:5]{index=5}
    a1_top_t = a1_top.resized((tile, tile))
    a0_pov_t = a0_pov.resized((tile, tile))
    a1_pov_t = a1_pov.resized((tile, tile))

    # Cut everything to shortest duration
    duration = min(
        overhead.duration,
        a0_top_t.duration,
        a1_top_t.duration,
        a0_pov_t.duration,
        a1_pov_t.duration,
    )
    overhead = overhead.subclipped(0, duration) if hasattr(overhead, "subclipped") else overhead.subclip(0, duration)
    a0_top_t = a0_top_t.subclipped(0, duration) if hasattr(a0_top_t, "subclipped") else a0_top_t.subclip(0, duration)
    a1_top_t = a1_top_t.subclipped(0, duration) if hasattr(a1_top_t, "subclipped") else a1_top_t.subclip(0, duration)
    a0_pov_t = a0_pov_t.subclipped(0, duration) if hasattr(a0_pov_t, "subclipped") else a0_pov_t.subclip(0, duration)
    a1_pov_t = a1_pov_t.subclipped(0, duration) if hasattr(a1_pov_t, "subclipped") else a1_pov_t.subclip(0, duration)

    # Background
    bg = ColorClip(size=(out_w, out_h), color=(0, 0, 0)).with_duration(duration)

    # Layout (v2 prefers with_position / with_duration style)
    x0 = S
    y0 = 0

    overhead = overhead.with_position((0, 0))

    a0_top_t = a0_top_t.with_position((x0 + 0,     y0 + 0))
    a1_top_t = a1_top_t.with_position((x0 + 0,     y0 + tile))
    a0_pov_t = a0_pov_t.with_position((x0 + tile,  y0 + 0))
    a1_pov_t = a1_pov_t.with_position((x0 + tile,  y0 + tile))

    comp = CompositeVideoClip(
        [bg, overhead, a0_top_t, a1_top_t, a0_pov_t, a1_pov_t],
        size=(out_w, out_h),
    ).with_duration(duration)

    out_path = blender_renders_dir / out_name
    comp.write_videofile(
        str(out_path),
        fps=int(fps),
        codec="libx264",
        audio=False,
        preset="medium",
        threads=0,
    )

    # Close (important on Windows)
    comp.close()
    overhead.close()
    a0_top.close()
    a1_top.close()
    a0_pov.close()
    a1_pov.close()

    return out_path



# ---------------------------------------------------------------------
# Hydra entrypoint
# ---------------------------------------------------------------------

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(args: DictConfig):
    # ------------------------------------------------------------------
    # Print full config
    # ------------------------------------------------------------------
    yaml_str = OmegaConf.to_yaml(args)
    print(f"[{inspect.stack()[0][3]}] configuration:\n{yaml_str}")

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    folder_name = args.simulation.post_processing.render_blender_outputs_folder
    # Must match the fps the frames were keyed for (blender.video_fps) to play
    # back in real time
    fps = int(args.blender.get("video_fps", None) or args.simulation.render.fps)

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    repo_root = Path(get_repo_root_dir())
    log_folder = resolve_log_folder(repo_root, folder_name)

    blender_renders_dir = log_folder / "blender_renders"
    if not blender_renders_dir.exists():
        raise FileNotFoundError(
            f"Expected blender_renders folder not found: {blender_renders_dir}"
        )

    print(f"[test_blender_outputs_to_video] Using blender_renders at: {blender_renders_dir}")

    # ------------------------------------------------------------------
    # Process each view folder -> mp4
    # ------------------------------------------------------------------
    view_dirs = sorted(d for d in blender_renders_dir.iterdir() if d.is_dir())
    if not view_dirs:
        print("[test_blender_outputs_to_video] No view folders found.")
        return

    for view_dir in view_dirs:
        view_name = view_dir.name
        image_files = list_image_files(view_dir)

        # If the file already exists, skip
        skip_if_already_exists = True
        out_video_path = blender_renders_dir / f"{view_name}.mp4"
        if skip_if_already_exists and out_video_path.exists():
            print(
                f"[test_blender_outputs_to_video] Skipping '{view_name}' "
                f"(output video already exists at: {out_video_path})"
            )
            continue

        print(
            f"[test_blender_outputs_to_video] View '{view_name}': "
            f"{len(image_files)} image files"
        )

        if not image_files:
            print(f"  - Skipping '{view_name}' (no images)")
            continue

        image_folder_to_video(
            filepath_image_folder=view_dir,
            filepath_output_video=out_video_path,
            fps=fps,
        )

    # ------------------------------------------------------------------
    # Composite together into a multiview video
    # ------------------------------------------------------------------
    try:
        out_comp = composite_blender_renders(blender_renders_dir, fps=fps, out_name="composite.mp4")
        print(f"[test_blender_outputs_to_video] Wrote composite: {out_comp}")
    except Exception as e:
        print(f"[test_blender_outputs_to_video] Composite skipped/failed: {e}")

    print("[test_blender_outputs_to_video] Done.")


if __name__ == "__main__":
    main()
