# src/prospector/blender_integration/blender_script.py

from __future__ import annotations

import bpy

from typing import Optional

from .constants import (
    ASSETS_COLLECTION_NAME,
    ASSET_QUADROTOR_NAME,
    CURVE_BEVEL_DEPTH,
    CURVE_RESOLUTION_U,
    KEYFRAME_START_FRAME,
    KEYFRAME_STRIDE,
    TEMP_COLLECTION_NAME,
)
from .paths import find_repo_root_from_blend, resolve_log_folder, find_agent_jsons
from .trajectories import load_agent_trajectory
from .blender_scene import (
    animate_curve_reveal,
    animate_quadrotor_blades,
    clear_collection,
    create_bezier_curve_from_points,
    find_object_in_collection,
    duplicate_object_hierarchy_into_collection,
    keyframe_object_trajectory,
    set_scene_frame_range,
    get_ego_camera_template,
    get_overhead_camera_template,
    duplicate_camera_into_collection,
    keyframe_camera_pov_from_agent,
    keyframe_camera_local_overhead_from_agent,
)
from .rendering import render_all_views
from .materials import get_or_create_curve_material
from .colors import get_agent_color_list

CURVE_NAME_PREFIX = "traj_agent_"
QUAD_NAME_PREFIX = "quad_agent_"


def main(
    *,
    log_folder: str,
    render: bool = False,
    render_first_n_frames: Optional[int] = None,
    skip_existing_renders: bool = True,
    video_fps: Optional[int] = None,
    sim_dt: Optional[float] = None,
    views: Optional[list] = None,
    render_last_frame_only: bool = False,
) -> None:
    repo_root = find_repo_root_from_blend()
    lf = resolve_log_folder(repo_root, log_folder)

    print(f"[Prospector] repo_root: {repo_root}")
    print(f"[Prospector] log_folder resolved: {lf}")

    # Real-time playback: one simulation step spans (video_fps * sim_dt)
    # frames; Blender interpolates between the keyframes. Without both
    # arguments, one step per frame (the old behaviour).
    stride = KEYFRAME_STRIDE
    if video_fps is not None and sim_dt is not None:
        s = float(video_fps) * float(sim_dt)
        if abs(s - round(s)) > 1e-6 or round(s) < 1:
            raise ValueError(
                f"video_fps * sim_dt must be a positive integer for real-time playback "
                f"(got {video_fps} * {sim_dt} = {s})"
            )
        stride = int(round(s))
        bpy.context.scene.render.fps = int(video_fps)
        bpy.context.scene.render.fps_base = 1.0
        print(f"[Prospector] Real-time playback: {video_fps} fps, {stride} frames per sim step (dt={sim_dt})")

    agent_jsons = find_agent_jsons(lf)
    trajs = [load_agent_trajectory(p, dt=float(sim_dt) if sim_dt else 0.1) for p in agent_jsons]
    print(f"[Prospector] Loaded {len(trajs)} agent trajectories")

    temp_coll = clear_collection(TEMP_COLLECTION_NAME)

    assets_coll = bpy.data.collections.get(ASSETS_COLLECTION_NAME)
    if assets_coll is None:
        raise KeyError(f"Assets collection not found: {ASSETS_COLLECTION_NAME}")

    quad_asset_obj = find_object_in_collection(assets_coll, ASSET_QUADROTOR_NAME)

    colors = get_agent_color_list(len(trajs))

    ego_template = get_ego_camera_template()
    overhead_template = get_overhead_camera_template()

    if trajs:
        set_scene_frame_range(len(trajs[0].positions), stride=stride)

    for i, t in enumerate(trajs):
        rgb = colors[i]  # (r,g,b) in [0,1]

        mat = get_or_create_curve_material(
            name=f"traj_mat_agent_{t.agent_id}",
            rgb=rgb,
            alpha=0.7,
        )
        print(f"[Prospector] Assigned color {rgb} to agent_id={t.agent_id}")

        curve_name = f"{CURVE_NAME_PREFIX}{t.agent_id}"
        curve_obj = create_bezier_curve_from_points(
            name=curve_name,
            points=t.positions,
            collection=temp_coll,
            bevel_depth=CURVE_BEVEL_DEPTH,
            resolution_u=CURVE_RESOLUTION_U,
            material=mat,
        )

        curve_obj.visible_shadow = False

        animate_curve_reveal(
            curve_obj=curve_obj,
            num_steps=len(t.positions),
            start_frame=KEYFRAME_START_FRAME,
            stride=stride,
        )

        quad_name = f"{QUAD_NAME_PREFIX}{t.agent_id}"
        quad_root = duplicate_object_hierarchy_into_collection(
            root_obj=quad_asset_obj,
            dst_collection=temp_coll,
            new_root_name=quad_name,
            copy_data=False,
        )

        keyframe_object_trajectory(
            obj=quad_root,
            positions=t.positions,
            rotations=t.rotations,
            start_frame=KEYFRAME_START_FRAME,
            stride=stride,
        )

        scene = bpy.context.scene
        
        animate_quadrotor_blades(
            quad_root=quad_root,
            start_frame=scene.frame_start,
            end_frame=scene.frame_end,
        )

        print(f"[Prospector] Animated quadrotor for agent={t.agent_id} with {len(t.positions)} frames")

        cam_pov = duplicate_camera_into_collection(
            src_cam_obj=ego_template,
            dst_collection=temp_coll,
            new_name=f"camera_agent_{t.agent_id}_pov",
        )
        keyframe_camera_pov_from_agent(
            cam_obj=cam_pov,
            positions=t.positions,
            rotations=t.rotations,
            start_frame=KEYFRAME_START_FRAME,
            stride=stride,
        )

        print(f"[Prospector] Animated POV camera for agent={t.agent_id} with {len(t.positions)} frames")

        cam_local = duplicate_camera_into_collection(
            src_cam_obj=overhead_template,
            dst_collection=temp_coll,
            new_name=f"camera_agent_{t.agent_id}_local_overhead",
        )
        keyframe_camera_local_overhead_from_agent(
            cam_obj=cam_local,
            positions=t.positions,
            rotations=t.rotations,
            start_frame=KEYFRAME_START_FRAME,
            stride=stride,
        )
        
        print(f"[Prospector] Animated local overhead camera for agent={t.agent_id} with {len(t.positions)} frames")

    if render:
        num_total_frames = bpy.context.scene.frame_end - bpy.context.scene.frame_start + 1
        if render_first_n_frames is None:
            render_first_n_frames = num_total_frames
        print(
            f"[Prospector] Rendering enabled: rendering {render_first_n_frames} / {num_total_frames} frames to: {lf}"
        )
        render_all_views(
            log_folder_path=lf,
            first_n_frames=render_first_n_frames,
            skip_existing_renders=skip_existing_renders,
            views=views,
            last_frame_only=render_last_frame_only,
        )

    print("[Prospector] Done.")
