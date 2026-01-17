# src/prospector/blender_integration/blender_script.py

from __future__ import annotations

def main() -> None:
    print("[Prospector] blender_script.main() running")

    # import bpy only inside Blender
    import bpy  # noqa: F401

    # Your real logic goes here
    from . import blender_utils
    blender_utils.example()


