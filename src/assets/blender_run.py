# This script will execute when you click Run Script or press Alt+P in Blender's Text Editor.

import sys
from pathlib import Path
import bpy


# ---------------------------------------------------------------------
# Repo root discovery (portable, no hard-coded paths)
# ---------------------------------------------------------------------

def find_repo_root(start: Path) -> Path:
    """
    Walk upwards from `start` until we find a Prospector repo root marker.
    """
    start = start.resolve()
    for p in (start,) + tuple(start.parents):
        if (p / "pyproject.toml").exists():
            return p
        if (p / ".git").exists():
            return p
        if (p / "src" / "prospector").exists():
            return p
    raise RuntimeError(
        f"Could not find Prospector repo root walking up from: {start}"
    )


def add_src_to_syspath(repo_root: Path) -> Path:
    src = repo_root / "src"
    if not src.exists():
        raise RuntimeError(f"Expected src/ at: {src}")
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    return src


# ---------------------------------------------------------------------
# Local dependency bootstrap (installs into <repo_root>/.blender_deps)
# ---------------------------------------------------------------------

def ensure_deps(repo_root: Path, requirements=("numpy",)) -> Path:
    """
    Install packages into a repo-local folder using Blender's Python, only if missing.
    Needs internet unless you vendor wheels.
    """
    import subprocess
    import ensurepip

    deps_dir = repo_root / ".blender_deps"
    deps_dir.mkdir(exist_ok=True)

    deps_str = str(deps_dir)
    if deps_str not in sys.path:
        sys.path.insert(0, deps_str)

    # Fast-path: already importable
    missing = []
    for pkg in requirements:
        try:
            __import__(pkg)
        except Exception:
            missing.append(pkg)

    if not missing:
        return deps_dir

    # Ensure pip exists in Blender's python
    try:
        import pip  # noqa: F401
    except Exception:
        ensurepip.bootstrap()

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--target",
        str(deps_dir),
        *missing,
    ]
    print("[Prospector] Installing missing deps into:", deps_dir)
    print("[Prospector] Running:", " ".join(cmd))
    subprocess.check_call(cmd)

    return deps_dir


# ---------------------------------------------------------------------
# Hot reload helper (dev only)
# ---------------------------------------------------------------------

def reload_package(pkg_prefix: str) -> None:
    """
    Reload all loaded modules whose name starts with pkg_prefix.
    Reload deeper modules first.
    """
    import importlib

    to_reload = [
        name for name in sys.modules.keys()
        if name == pkg_prefix or name.startswith(pkg_prefix + ".")
    ]
    to_reload.sort(key=lambda s: s.count("."), reverse=True)
    for name in to_reload:
        importlib.reload(sys.modules[name])


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------

DEV_RELOAD = True  # set False for normal users if you like

if not bpy.data.filepath:
    raise RuntimeError("This .blend has not been saved yet. Save it so we can locate the repo root.")

blend_path = Path(bpy.data.filepath)
repo_root = find_repo_root(blend_path.parent)
add_src_to_syspath(repo_root)

# Install deps if needed (json is stdlib; numpy often not)
ensure_deps(repo_root, requirements=("numpy",))

# Import + optional hot reload
from prospector.blender_integration import blender_script

if DEV_RELOAD:
    reload_package("prospector.blender_integration")

# Run your integration entrypoint
blender_script.main()
