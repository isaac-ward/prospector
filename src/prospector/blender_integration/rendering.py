# src/prospector/blender_integration/rendering.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import bpy


# ---------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------

FRAME_FILENAME_DIGITS = 6
IMAGE_FORMAT = "PNG"  # "PNG" or "JPEG"
IMAGE_EXT = ".png"    # keep consistent with IMAGE_FORMAT

# If you want faster test renders:
# set these from your runner / constants later
DEFAULT_RENDER_FIRST_N_FRAMES: Optional[int] = None


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _hierarchy_objects(root: bpy.types.Object) -> List[bpy.types.Object]:
    return [root] + list(root.children_recursive)

def _hierarchy_names(root: bpy.types.Object) -> List[str]:
    return [o.name for o in _hierarchy_objects(root)]

def _find_quad_root_for_agent(agent_id: str) -> bpy.types.Object:
    """
    Find the duplicated quad root for agent_id.
    Handles accidental Blender suffixes like quad_agent_0.001.
    """
    exact = bpy.data.objects.get(f"quad_agent_{agent_id}")
    if exact is not None:
        return exact

    prefix = f"quad_agent_{agent_id}."
    for o in bpy.data.objects:
        if o.name.startswith(prefix):
            return o

    raise KeyError(f"Could not find quad root for agent_id={agent_id} (expected 'quad_agent_{agent_id}'*)")

def _find_all_quad_roots() -> List[bpy.types.Object]:
    """
    Return roots named quad_agent_<id> (with optional .### suffix).
    We assume these are the top-level parent objects for each quad hierarchy.
    """
    roots = []
    for o in bpy.data.objects:
        if o.name.startswith("quad_agent_"):
            # likely a root; we prefer ones that have children
            roots.append(o)

    # Deduplicate by base "quad_agent_X" (ignore .001)
    def base(n: str) -> str:
        return n.split(".")[0]

    uniq: Dict[str, bpy.types.Object] = {}
    for o in roots:
        b = base(o.name)
        # prefer the one that actually has children (more likely root)
        if b not in uniq:
            uniq[b] = o
        else:
            if len(o.children_recursive) > len(uniq[b].children_recursive):
                uniq[b] = o

    return list(uniq.values())


def _find_object(name: str) -> bpy.types.Object:
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise KeyError(f"Object not found in bpy.data.objects: '{name}'")
    return obj


def _find_camera(name: str) -> bpy.types.Object:
    obj = _find_object(name)
    if obj.type != "CAMERA":
        raise TypeError(f"Expected CAMERA for '{name}', got type={obj.type}")
    return obj


def _set_scene_render_settings(*, scene: bpy.types.Scene) -> None:
    scene.render.image_settings.file_format = IMAGE_FORMAT
    # ensure still renders don't add weird suffixes
    # (we set scene.render.filepath per frame)
    scene.render.use_file_extension = True


def _frame_path(out_dir: Path, frame: int) -> Path:
    return out_dir / f"{frame:0{FRAME_FILENAME_DIGITS}d}{IMAGE_EXT}"


def _iter_frames(scene: bpy.types.Scene, *, first_n: Optional[int]) -> List[int]:
    start = int(scene.frame_start)
    end = int(scene.frame_end)
    frames = list(range(start, end + 1))
    if first_n is not None:
        frames = frames[: int(first_n)]
    return frames


def infer_cave_name_from_scene() -> str:
    """
    Infer cave_name from object naming convention:
      <cave_name>_enclosed
      <cave_name>_bounding_box
      <cave_name>_open_top
    """
    suffixes = ("_enclosed", "_bounding_box", "_open_top")
    candidates: Dict[str, int] = {}

    for obj in bpy.data.objects:
        n = obj.name
        for suf in suffixes:
            if n.endswith(suf):
                base = n[: -len(suf)]
                candidates[base] = candidates.get(base, 0) + 1

    if not candidates:
        raise RuntimeError(
            "Could not infer cave_name. Expected objects like '<cave_name>_enclosed' "
            "and/or '<cave_name>_open_top' to exist in the .blend."
        )

    # prefer one that matches most suffixes
    cave_name, score = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)[0]
    if score < 1:
        raise RuntimeError("infer_cave_name_from_scene found no valid cave candidates")
    return cave_name


@dataclass
class VisibilitySnapshot:
    hide_render: Dict[str, bool]
    hide_viewport: Dict[str, bool]


def snapshot_visibility(objs: Iterable[bpy.types.Object]) -> VisibilitySnapshot:
    hr: Dict[str, bool] = {}
    hv: Dict[str, bool] = {}
    for o in objs:
        hr[o.name] = bool(o.hide_render)
        hv[o.name] = bool(o.hide_viewport)
    return VisibilitySnapshot(hide_render=hr, hide_viewport=hv)


def restore_visibility(snapshot: VisibilitySnapshot) -> None:
    for name, v in snapshot.hide_render.items():
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_render = bool(v)
    for name, v in snapshot.hide_viewport.items():
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_viewport = bool(v)


def set_only_these_objects_render_visible(
    *,
    render_visible_names: List[str],
) -> VisibilitySnapshot:
    """
    Makes ONLY the named objects render-visible; hides all other objects for render.
    Returns snapshot so you can restore after the view render is done.

    Important: this affects ALL objects in the file. That’s what you want for clean view-specific renders.
    """
    all_objs = list(bpy.data.objects)
    snap = snapshot_visibility(all_objs)

    visible = set(render_visible_names)

    for o in all_objs:
        # keep cameras render-invisible? Doesn't matter; cameras don't render as geometry.
        o.hide_render = (o.name not in visible)

    return snap


def render_frames_from_camera(
    *,
    scene: bpy.types.Scene,
    camera_obj: bpy.types.Object,
    out_dir: Path,
    first_n_frames: Optional[int] = DEFAULT_RENDER_FIRST_N_FRAMES,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_scene_render_settings(scene=scene)

    # Use this camera
    scene.camera = camera_obj

    frames = _iter_frames(scene, first_n=first_n_frames)
    if not frames:
        return

    print(f"[Prospector] Rendering {len(frames)} frames from camera '{camera_obj.name}' -> {out_dir}")

    for f in frames:
        scene.frame_set(int(f))
        fp = _frame_path(out_dir, int(f))
        scene.render.filepath = str(fp)
        bpy.ops.render.render(write_still=True)

    print(f"[Prospector] Done rendering camera '{camera_obj.name}'")


# ---------------------------------------------------------------------
# View orchestration (your 3 view families)
# ---------------------------------------------------------------------

def render_all_views(
    *,
    log_folder_path: Path,
    cave_name: Optional[str] = None,
    first_n_frames: Optional[int] = None,
) -> None:
    """
    Renders:
      - overhead: camera named 'camera_<cave_name>_angled_overhead'
      - agent_{i}_pov: camera named 'camera_agent_{i}_pov'
      - agent_{i}_top_down: camera named 'camera_agent_{i}_local_overhead'

    Writes frames under:
      <log_folder>/blender_renders/<view_name>/
    """
    scene = bpy.context.scene

    if cave_name is None:
        cave_name = infer_cave_name_from_scene()

    renders_root = Path(log_folder_path) / "blender_renders"
    renders_root.mkdir(parents=True, exist_ok=True)

    # Environment objects per your rules
    obj_enclosed = _find_object(f"{cave_name}_enclosed")
    obj_bbox = _find_object(f"{cave_name}_bounding_box")
    obj_open_top = _find_object(f"{cave_name}_open_top")

    # Find stuff
    quad_names: List[str] = []
    for qroot in _find_all_quad_roots():
        quad_names.extend(_hierarchy_names(qroot))

    curve_names = [o.name for o in bpy.data.objects if o.name.startswith("traj_agent_")]
    agent_cam_names = [o.name for o in bpy.data.objects if o.type == "CAMERA" and o.name.startswith("camera_agent_")]

    # -------------------------
    # 1) Angled overhead view
    # -------------------------
    overhead_cam_name = f"camera_{cave_name}_angled_overhead"
    overhead_cam = _find_camera(overhead_cam_name)

    # overhead uses half mesh
    snap = set_only_these_objects_render_visible(
        render_visible_names=[
            overhead_cam.name,
            obj_open_top.name,
            # keep your agents/curves visible too:
            # (we do NOT hide them; so include them explicitly)
            *quad_names,
            *curve_names,
            *agent_cam_names,
            # Keep the sun visible too
            "sun",
        ]
    )
    try:
        render_frames_from_camera(
            scene=scene,
            camera_obj=overhead_cam,
            out_dir=renders_root / "overhead",
            first_n_frames=first_n_frames,
        )
    finally:
        restore_visibility(snap)

    # -------------------------
    # 2) Agent POV + top-down
    # -------------------------
    # Discover agents by cameras present
    pov_cams = sorted(
        [o for o in bpy.data.objects if o.type == "CAMERA" and o.name.startswith("camera_agent_") and o.name.endswith("_pov")],
        key=lambda o: o.name,
    )
    top_cams = sorted(
        [o for o in bpy.data.objects if o.type == "CAMERA" and o.name.startswith("camera_agent_") and o.name.endswith("_local_overhead")],
        key=lambda o: o.name,
    )

    # Map agent_id from naming
    def _agent_id_from_cam(name: str) -> str:
        # camera_agent_{id}_pov OR camera_agent_{id}_local_overhead
        s = name[len("camera_agent_") :]
        if s.endswith("_pov"):
            return s[: -len("_pov")]
        if s.endswith("_local_overhead"):
            return s[: -len("_local_overhead")]
        return s

    top_by_id = {_agent_id_from_cam(c.name): c for c in top_cams}
    pov_by_id = {_agent_id_from_cam(c.name): c for c in pov_cams}

    agent_ids = sorted(set(top_by_id.keys()) | set(pov_by_id.keys()))
    if not agent_ids:
        print("[Prospector] No agent cameras found (camera_agent_*). Skipping agent view renders.")
        return

    for aid in agent_ids:
        # ---- POV (full mesh + bbox)
        if aid in pov_by_id:
            cam = pov_by_id[aid]
            snap = set_only_these_objects_render_visible(
                render_visible_names=[
                    cam.name,
                    obj_enclosed.name,
                    obj_bbox.name,
                    # agents/curves visibility
                    *quad_names,
                    #*curve_names,
                    *agent_cam_names,
                    # Keep the sun visible too
                    "sun",
                ]
            )
            try:
                render_frames_from_camera(
                    scene=scene,
                    camera_obj=cam,
                    out_dir=renders_root / f"agent_{aid}_pov",
                    first_n_frames=first_n_frames,
                )
            finally:
                restore_visibility(snap)

        # ---- Local overhead (half mesh)
        if aid in top_by_id:
            cam = top_by_id[aid]
            snap = set_only_these_objects_render_visible(
                render_visible_names=[
                    cam.name,
                    obj_open_top.name,
                    # keep your agents/curves visible too:
                    # (we do NOT hide them; so include them explicitly)
                    *quad_names,
                    *curve_names,
                    *agent_cam_names,
                    # Keep the sun visible too
                    "sun",
                ]
            )
            try:
                render_frames_from_camera(
                    scene=scene,
                    camera_obj=cam,
                    out_dir=renders_root / f"agent_{aid}_top_down",
                    first_n_frames=first_n_frames,
                )
            finally:
                restore_visibility(snap)
