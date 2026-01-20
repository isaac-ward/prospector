# src/prospector/blender_integration/constants.py

from __future__ import annotations

import math

# ---------------------------------------------------------------------
# Globals / knobs
# ---------------------------------------------------------------------

TEMP_COLLECTION_NAME = "rendering_temporary"
ASSETS_COLLECTION_NAME = "rendering_assets"
ASSET_QUADROTOR_NAME = "quadrotor"

# Asset orientation correction (radians) applied on top of trajectory rotations.
# (rx, ry, rz) in Euler XYZ
ASSET_ROTATION_OFFSET_EULER = (math.pi / 2, 0.0, -math.pi / 2)

KEYFRAME_START_FRAME = 1
KEYFRAME_STRIDE = 1
SCENE_FPS = 60

EPS_NORM = 1e-8

# Curve rendering defaults
CURVE_BEVEL_DEPTH = 0.08
CURVE_RESOLUTION_U = 8

# ---------------------------------------------------------------------
# Blade spin (quadrotor prop animation)
# ---------------------------------------------------------------------

BLADE_OBJECT_NAMES = (
    "blade0",
    "blade1",
    "blade2",
    "blade3",
)

# 1000 ensures that with full motion blur in render, adjacent frames
# will have a full revolution of blur
BLADE_RPM = 1000.0

# Per-blade direction multipliers (+1 CCW, -1 CW) around LOCAL +Z.
# This pattern is typical for quads (alternating).
BLADE_DIR_SIGNS = (+1.0, -1.0, +1.0, -1.0)

# If True, swap the direction for ALL blades (CCW<->CW).
BLADE_DIR_FLIP = False

# ---------------------------------------------------------------------
# Camera keyframing knob
# ---------------------------------------------------------------------

CAMERA_EGO_EXAMPLE_NAME = "camera_ego_example"

LOCAL_OVERHEAD_Z_OFFSET = 10.0  # meters above agent
LOCAL_OVERHEAD_USE_AGENT_YAW = False

# If your ego camera ends up 'looking the wrong way', set an offset here (radians).
CAMERA_EGO_ROT_OFFSET_EULER = (math.pi / 2, 0.0, -math.pi / 2)
