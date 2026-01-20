# src/prospector/tests/test_blender_render_headless.py

from __future__ import annotations

import inspect
import os
import subprocess
from pathlib import Path

from omegaconf import OmegaConf, DictConfig
import hydra

from prospector.utils.custom_logging import get_repo_root_dir


def _resolve_blender_executable(path_str: str) -> Path:
    """
    Resolve blender executable path. Works for:
      - absolute paths
      - 'blender' on PATH (Windows/Linux/macOS)
    """
    p = Path(path_str)

    # Absolute or relative direct path
    if p.exists():
        return p.resolve()

    # Try PATH lookup
    which = shutil.which(path_str)  # type: ignore[name-defined]
    if which is not None:
        return Path(which).resolve()

    raise FileNotFoundError(
        f"Could not find blender executable.\n"
        f"  config blender.blender_executable_path = {path_str}\n"
        f"  Tried as path: {p}\n"
        f"  Tried PATH lookup for: {path_str}\n"
        f"Tips:\n"
        f"  - Windows: set to ...\\blender.exe\n"
        f"  - cmd: `where blender`\n"
        f"  - In Blender python: `import sys; print(sys.executable)`\n"
    )


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(args: DictConfig):
    # ------------------------------------------------------------------ #
    # Print full config                                                  #
    # ------------------------------------------------------------------ #
    yaml_str = OmegaConf.to_yaml(args)
    print(f"[{inspect.stack()[0][3]}] configuration:\n{yaml_str}")

    repo_root = Path(get_repo_root_dir())
    print(f"[test_blender_render_headless] repo_root: {repo_root}")

    # ------------------------------------------------------------------ #
    # Read blender config                                                #
    # ------------------------------------------------------------------ #
    if "blender" not in args:
        raise KeyError("Expected args.blender section in config")

    blender_cfg = args.blender

    blender_exe = Path(str(blender_cfg.blender_executable_path))
    blend_rel = Path(str(blender_cfg.relative_to_repo_blend_file_path))
    log_folder = str(blender_cfg.log_folder_to_process)

    render = bool(blender_cfg.render)
    first_n = blender_cfg.render_first_n_frames
    skip_existing = bool(blender_cfg.skip_existing_renders)

    # Resolve paths
    blender_exe = _resolve_blender_executable(str(blender_exe))
    blend_file = (repo_root / blend_rel).resolve()
    if blend_file.suffix.lower() != ".blend":
        raise ValueError(f"Expected a .blend file, got: {blend_file}")
    if not blend_file.exists():
        raise FileNotFoundError(f"Blend file not found: {blend_file}")

    print(f"[test_blender_render_headless] blender_exe: {blender_exe}")
    print(f"[test_blender_render_headless] blend_file:  {blend_file}")
    print(f"[test_blender_render_headless] log_folder:  {log_folder}")
    print(f"[test_blender_render_headless] render: {render}")
    print(f"[test_blender_render_headless] render_first_n_frames: {first_n}")
    print(f"[test_blender_render_headless] skip_existing_renders: {skip_existing}")

    # ------------------------------------------------------------------ #
    # Build headless blender command                                     #
    # ------------------------------------------------------------------ #
    # We run the Prospector blender entrypoint inside Blender.
    #
    # Important: use `--python-expr` so you don't need an external runner file.
    # We insert <repo_root>/src to sys.path and then call:
    #   from prospector.blender_integration.api import run
    #   run(log_folder=..., render=..., render_first_n_frames=..., skip_existing_renders=...)
    #
    # Your api.run(...) must accept these args. If it currently doesn't, tell me
    # what signature you have and I’ll match it.
    #
    py_expr = (
        "import sys; "
        f"sys.path.insert(0, r'{str((repo_root / 'src').resolve())}'); "
        "from prospector.blender_integration.api import run; "
        "run("
        f"log_folder=r'{log_folder}', "
        f"render={str(render)}, "
        f"render_first_n_frames={('None' if first_n is None else int(first_n))}, "
        f"skip_existing_renders={str(skip_existing)}"
        ");"
    )

    cmd = [
        str(blender_exe),
        "-b",
        str(blend_file),
        "--python-expr",
        py_expr,
    ]

    print("[test_blender_render_headless] Running command (note that 'Read blend' may take up to 15 minutes - I recommend going off and making a cuppa):")
    print("  " + " ".join(cmd))

    # ------------------------------------------------------------------ #
    # Execute                                                           #
    # ------------------------------------------------------------------ #
    # Inherit environment so CUDA/OptiX/etc are visible. Also ensures PATH works.
    env = os.environ.copy()

    # Stream output to console so you can see progress live
    proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env)
    ret = proc.wait()

    if ret != 0:
        raise RuntimeError(f"Blender headless render failed with return code: {ret}")

    print("[test_blender_render_headless] Done.")


if __name__ == "__main__":
    # Needed because we used shutil.which in helper; keep it local import-free above
    import shutil
    main()
