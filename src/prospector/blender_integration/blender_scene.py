# src/prospector/blender_integration/blender_scene.py

from __future__ import annotations

from typing import Dict, List, Tuple

import bpy
import math

from .constants import (
    ASSET_ROTATION_OFFSET_EULER,
    CURVE_BEVEL_DEPTH,
    CURVE_RESOLUTION_U,
    EPS_NORM,
    KEYFRAME_START_FRAME,
    KEYFRAME_STRIDE,
    BLADE_OBJECT_NAMES,
    BLADE_RPM,
    BLADE_DIR_SIGNS,
    BLADE_DIR_FLIP,
    ASSETS_COLLECTION_NAME,
    CAMERA_EGO_EXAMPLE_NAME,
    CAMERA_EGO_ROT_OFFSET_EULER,
    LOCAL_OVERHEAD_Z_OFFSET,
    LOCAL_OVERHEAD_USE_AGENT_YAW,
)



def get_or_create_collection(name: str) -> bpy.types.Collection:
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll


def clear_collection(name: str) -> bpy.types.Collection:
    coll = get_or_create_collection(name)

    objs = list(coll.objects)
    for obj in objs:
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        bpy.data.objects.remove(obj, do_unlink=True)

    for curve in list(bpy.data.curves):
        if curve.users == 0:
            bpy.data.curves.remove(curve)

    return coll


def create_bezier_curve_from_points(
    *,
    name: str,
    points,
    collection,
    material=None,
    bevel_depth=CURVE_BEVEL_DEPTH,
    resolution_u=CURVE_RESOLUTION_U,
):
    curve_data = bpy.data.curves.new(name=name + "_curve", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = int(resolution_u)
    curve_data.bevel_depth = float(bevel_depth)
    curve_data.use_fill_caps = True

    spline = curve_data.splines.new(type="BEZIER")
    spline.bezier_points.add(len(points) - 1)

    for i, p in enumerate(points):
        bp = spline.bezier_points[i]
        bp.co = p
        bp.handle_left_type = "AUTO"
        bp.handle_right_type = "AUTO"

    obj = bpy.data.objects.new(name, curve_data)
    collection.objects.link(obj)

    # Apply material
    if material is not None:
        curve_data.materials.clear()
        curve_data.materials.append(material)

    # No shadows (as you requested earlier)
    obj.visible_shadow = False

    return obj


def find_object_in_collection(collection: bpy.types.Collection, object_name: str) -> bpy.types.Object:
    obj = collection.objects.get(object_name)
    if obj is None:
        raise KeyError(f"Object '{object_name}' not found directly in collection '{collection.name}'")
    return obj


def set_scene_frame_range(
    num_steps: int,
    *,
    start_frame: int = KEYFRAME_START_FRAME,
    stride: int = KEYFRAME_STRIDE,
) -> None:
    end_frame = start_frame + max(0, num_steps - 1) * stride
    bpy.context.scene.frame_start = int(start_frame)
    bpy.context.scene.frame_end = int(end_frame)


def keyframe_object_trajectory(
    *,
    obj: bpy.types.Object,
    positions: List[Tuple[float, float, float]],
    rotations: List[Tuple[float, float, float]],
    start_frame: int = KEYFRAME_START_FRAME,
    stride: int = KEYFRAME_STRIDE,
) -> None:
    if len(positions) != len(rotations):
        raise ValueError(f"positions and rotations must match length: {len(positions)} vs {len(rotations)}")

    obj.rotation_mode = "XYZ"

    ox, oy, oz = ASSET_ROTATION_OFFSET_EULER

    for i, (pos, rot) in enumerate(zip(positions, rotations)):
        frame = int(start_frame + i * stride)
        obj.location = pos

        rx, ry, rz = rot
        obj.rotation_euler = (rx + ox, ry + oy, rz + oz)

        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)


def duplicate_object_hierarchy_into_collection(
    *,
    root_obj: bpy.types.Object,
    dst_collection: bpy.types.Collection,
    new_root_name: str | None = None,
    copy_data: bool = True,
) -> bpy.types.Object:
    originals: List[bpy.types.Object] = [root_obj] + list(root_obj.children_recursive)
    obj_map: Dict[bpy.types.Object, bpy.types.Object] = {}

    for src in originals:
        dup = src.copy()
        if copy_data and src.data is not None:
            dup.data = src.data.copy()

        dup.parent = None
        dst_collection.objects.link(dup)
        obj_map[src] = dup

    for src, dup in obj_map.items():
        if src.parent is None:
            continue
        if src.parent not in obj_map:
            continue
        dup.parent = obj_map[src.parent]
        dup.parent_type = src.parent_type
        dup.matrix_parent_inverse = src.matrix_parent_inverse.copy()

    dup_root = obj_map[root_obj]
    if new_root_name is not None:
        dup_root.name = new_root_name

    return dup_root

def _iter_descendants(root: bpy.types.Object) -> List[bpy.types.Object]:
    return [root] + list(root.children_recursive)


def _base_object_name(name: str) -> str:
    """
    Blender duplicate naming:
      'foo' -> 'foo.001' etc.
    For names that already contain dots (like 'default.036'), duplicates become
      'default.036.001'
    We want the first two components for 'default.036' style names,
    otherwise the first component.
    """
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "default":
        # keep 'default.036' as the base
        return ".".join(parts[:2])
    return parts[0]


def _find_descendant_by_base_name(
    root: bpy.types.Object,
    desired_base: str,
) -> bpy.types.Object:
    desired_base = _base_object_name(desired_base)

    for obj in _iter_descendants(root):
        if _base_object_name(obj.name) == desired_base:
            return obj

    # Helpful debug on failure:
    sample = [o.name for o in _iter_descendants(root)[:80]]
    raise KeyError(
        f"Could not find descendant base='{desired_base}' under '{root.name}'. "
        f"First descendants: {sample}"
    )


def _set_fcurve_linear_and_cycles(
    obj: bpy.types.Object,
    data_path: str,
    array_index: int,
) -> None:
    """
    Ensure fcurve interpolation is linear + add Cycles modifier for looping.
    """
    ad = obj.animation_data
    if ad is None or ad.action is None:
        return

    fc = ad.action.fcurves.find(data_path, index=array_index)
    if fc is None:
        return

    # Linear interpolation for constant angular speed
    for kp in fc.keyframe_points:
        kp.interpolation = "LINEAR"

    # Loop forever
    has_cycles = any(m.type == "CYCLES" for m in fc.modifiers)
    if not has_cycles:
        fc.modifiers.new(type="CYCLES")


def animate_quadrotor_blades(
    *,
    quad_root: bpy.types.Object,
    start_frame: int,
    end_frame: int,
    rpm: float = BLADE_RPM,
) -> None:
    """
    Animate blades to spin at constant RPM about their LOCAL Z axis.
    Applies to the specified quad_root (so works per duplicated quad).

    Implementation:
      - keyframe Z rotation at start and end
      - set interpolation to LINEAR
      - add CYCLES modifier so it repeats
    """
    if end_frame <= start_frame:
        return

    scene = bpy.context.scene
    fps = float(scene.render.fps) / float(scene.render.fps_base or 1.0)
    duration_s = float(end_frame - start_frame) / fps

    omega = 2.0 * math.pi * float(rpm) / 60.0  # rad/s
    delta = omega * duration_s                  # total radians over the range

    flip = -1.0 if BLADE_DIR_FLIP else 1.0

    for blade_name, sign in zip(BLADE_OBJECT_NAMES, BLADE_DIR_SIGNS):
        blade = _find_descendant_by_base_name(quad_root, blade_name)

        # Keep whatever initial orientation the blade already has
        blade.rotation_mode = "XYZ"
        base_eul = blade.rotation_euler.copy()

        # Keyframe at start
        blade.rotation_euler = base_eul
        blade.keyframe_insert(data_path="rotation_euler", frame=start_frame, index=1)

        # Keyframe at end: spin about y should do it
        end_eul = base_eul.copy()
        end_eul.y = float(base_eul.y + flip * float(sign) * delta)
        blade.rotation_euler = end_eul
        blade.keyframe_insert(data_path="rotation_euler", frame=end_frame, index=1)

        _set_fcurve_linear_and_cycles(blade, "rotation_euler", 1)


def animate_curve_reveal(
    *,
    curve_obj: bpy.types.Object,
    num_steps: int,
    start_frame: int,
    stride: int,
) -> None:
    """
    Animate curve bevel_factor_end so the curve reveals progressively
    in sync with trajectory time.
    """
    if num_steps <= 1:
        return

    curve_data = curve_obj.data
    curve_data.bevel_factor_start = 0.0

    # Start: nothing visible
    curve_data.bevel_factor_end = 0.0
    curve_data.keyframe_insert(
        data_path="bevel_factor_end",
        frame=start_frame,
    )

    # End: fully visible
    end_frame = start_frame + (num_steps - 1) * stride
    curve_data.bevel_factor_end = 1.0
    curve_data.keyframe_insert(
        data_path="bevel_factor_end",
        frame=end_frame,
    )

    # Make reveal linear
    ad = curve_data.animation_data
    if ad and ad.action:
        fc = ad.action.fcurves.find("bevel_factor_end")
        if fc:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"

# ---------------------------------------------------------------------
# Camera helpers (keyframing only)
# ---------------------------------------------------------------------

def _find_object_under_collection_recursive(
    collection: bpy.types.Collection,
    object_name: str,
) -> bpy.types.Object:
    """
    Find an object by name within a collection, searching all descendant collections too.
    """
    obj = collection.objects.get(object_name)
    if obj is not None:
        return obj
    for child in collection.children_recursive:
        obj = child.objects.get(object_name)
        if obj is not None:
            return obj
    raise KeyError(f"Object '{object_name}' not found under collection '{collection.name}'")


def get_ego_camera_template() -> bpy.types.Object:
    """
    Returns the camera object used as a template for POV/local overhead cameras.
    Expected to exist under ASSETS_COLLECTION_NAME (possibly in a child collection).
    """
    assets = bpy.data.collections.get(ASSETS_COLLECTION_NAME)
    if assets is None:
        raise KeyError(f"Assets collection not found: {ASSETS_COLLECTION_NAME}")

    cam = _find_object_under_collection_recursive(assets, CAMERA_EGO_EXAMPLE_NAME)
    if cam.type != "CAMERA":
        raise TypeError(f"Expected CAMERA for '{CAMERA_EGO_EXAMPLE_NAME}', got type={cam.type}")
    return cam

def get_overhead_camera_template() -> bpy.types.Object:
    """
    Returns the camera object used as a template for local overhead cameras.
    Expected to exist under ASSETS_COLLECTION_NAME (possibly in a child collection).
    """
    assets = bpy.data.collections.get(ASSETS_COLLECTION_NAME)
    if assets is None:
        raise KeyError(f"Assets collection not found: {ASSETS_COLLECTION_NAME}")

    cam = _find_object_under_collection_recursive(assets, "camera_overhead_example")
    if cam.type != "CAMERA":
        raise TypeError(f"Expected CAMERA for 'camera_overhead_example', got type={cam.type}")
    return cam

def duplicate_camera_into_collection(
    *,
    src_cam_obj: bpy.types.Object,
    dst_collection: bpy.types.Collection,
    new_name: str,
) -> bpy.types.Object:
    """
    Duplicate a camera object + its camera datablock into dst_collection.
    """
    cam_copy = src_cam_obj.copy()
    cam_copy.data = src_cam_obj.data.copy()
    cam_copy.name = new_name
    dst_collection.objects.link(cam_copy)
    return cam_copy


def keyframe_camera_pov_from_agent(
    *,
    cam_obj: bpy.types.Object,
    positions: List[Tuple[float, float, float]],
    rotations: List[Tuple[float, float, float]],
    start_frame: int = KEYFRAME_START_FRAME,
    stride: int = KEYFRAME_STRIDE,
) -> None:
    """
    Keyframe a POV camera to exactly match the agent pose each timestep.
    rotations are Euler XYZ radians.
    """
    if len(positions) != len(rotations):
        raise ValueError("positions and rotations must have same length")

    ox, oy, oz = CAMERA_EGO_ROT_OFFSET_EULER

    cam_obj.rotation_mode = "XYZ"
    for i, (pos, rot) in enumerate(zip(positions, rotations)):
        f = int(start_frame + i * stride)
        cam_obj.location = pos
        # Offset it slightly forward in the direction of the next position
        next_pos = positions[min(i + 1, len(positions) - 1)]
        direction = (
            next_pos[0] - pos[0],
            next_pos[1] - pos[1],
            next_pos[2] - pos[2],
        )
        norm = math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
        if norm > EPS_NORM:
            forward_offset = 0.25
            dx = (direction[0] / norm) * forward_offset
            dy = (direction[1] / norm) * forward_offset
            dz = (direction[2] / norm) * forward_offset
            cam_obj.location.x += dx
            cam_obj.location.y += dy
            cam_obj.location.z += dz
        rx, ry, rz = rot
        cam_obj.rotation_euler = (float(rx + ox), float(ry + oy), float(rz + oz))
        cam_obj.keyframe_insert(data_path="location", frame=f)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=f)


def keyframe_camera_local_overhead_from_agent(
    *,
    cam_obj: bpy.types.Object,
    positions: List[Tuple[float, float, float]],
    rotations: List[Tuple[float, float, float]],
    start_frame: int = KEYFRAME_START_FRAME,
    stride: int = KEYFRAME_STRIDE,
    z_offset: float = LOCAL_OVERHEAD_Z_OFFSET,
) -> None:
    """
    Keyframe a local overhead camera: directly above agent by z_offset.
    If LOCAL_OVERHEAD_USE_AGENT_YAW=True, yaw follows agent yaw (rz).
    """
    if len(positions) != len(rotations):
        raise ValueError("positions and rotations must have same length")

    cam_obj.rotation_mode = "XYZ"

    for i, (pos, rot) in enumerate(zip(positions, rotations)):
        f = int(start_frame + i * stride)
        x, y, z = pos
        cam_obj.location = (float(x), float(y), float(z + z_offset))

        if LOCAL_OVERHEAD_USE_AGENT_YAW:
            rz = float(rot[2])
            cam_obj.rotation_euler = (0.0, 0.0, rz)
        else:
            cam_obj.rotation_euler = (0.0, 0.0, 0.0)

        cam_obj.keyframe_insert(data_path="location", frame=f)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=f)
