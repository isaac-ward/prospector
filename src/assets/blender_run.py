# This script will execute when you click Run Script or press Alt+P in Blender.

import sys
from pathlib import Path
import bpy

log_folder = "run_2026_01_17_17_29_08_test_simulation"
cave_name = None  # e.g., "chamber" or "tunnels"
render = True
render_first_n_frames = None
skip_existing_renders = True

def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for p in (start,) + tuple(start.parents):
        if (p / "src" / "prospector").exists():
            return p
        if (p / "pyproject.toml").exists():
            return p
        if (p / ".git").exists():
            return p
    raise RuntimeError(f"Could not find Prospector repo root walking up from: {start}")

def add_src_to_syspath(repo_root: Path) -> None:
    src = repo_root / "src"
    if not src.exists():
        raise RuntimeError(f"Expected src/ at: {src}")
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

def ensure_deps(repo_root: Path, requirements=("numpy",)) -> None:
    import subprocess
    import ensurepip

    deps_dir = repo_root / ".blender_deps"
    deps_dir.mkdir(exist_ok=True)

    deps_str = str(deps_dir)
    if deps_str not in sys.path:
        sys.path.insert(0, deps_str)

    missing = []
    for pkg in requirements:
        try:
            __import__(pkg)
        except Exception:
            missing.append(pkg)

    if not missing:
        return

    try:
        import pip  # noqa: F401
    except Exception:
        ensurepip.bootstrap()

    cmd = [
        sys.executable, "-m", "pip", "install", "--upgrade",
        "--target", str(deps_dir),
        *missing,
    ]
    print("[Prospector] Installing missing deps into:", deps_dir)
    print("[Prospector] Running:", " ".join(cmd))
    subprocess.check_call(cmd)

def reload_package(pkg_prefix: str) -> None:
    """
    Reload all loaded modules whose name starts with pkg_prefix.
    Skips/purges stale modules that no longer have an import spec (after refactors).
    """
    import importlib

    names = [
        name for name in list(sys.modules.keys())
        if name == pkg_prefix or name.startswith(pkg_prefix + ".")
    ]
    names.sort(key=lambda s: s.count("."), reverse=True)

    for name in names:
        mod = sys.modules.get(name)
        if mod is None:
            continue

        spec = getattr(mod, "__spec__", None)
        if spec is None:
            del sys.modules[name]
            continue

        try:
            importlib.reload(mod)
        except ModuleNotFoundError:
            del sys.modules[name]

DEV_RELOAD = True

if not bpy.data.filepath:
    raise RuntimeError("This .blend has not been saved yet. Save it so we can locate the repo root.")

repo_root = find_repo_root(Path(bpy.data.filepath).resolve().parent)
add_src_to_syspath(repo_root)
ensure_deps(repo_root, requirements=("numpy",))

# Import AFTER sys.path and deps are set
from prospector.blender_integration.api import run

if DEV_RELOAD:
    reload_package("prospector.blender_integration")

# Quick test: render only first 2 frames
run(
    log_folder=log_folder, 
    render=render,
    render_first_n_frames=render_first_n_frames,
    skip_existing_renders=skip_existing_renders
)
