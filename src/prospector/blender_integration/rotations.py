# src/prospector/blender_integration/rotations.py

from __future__ import annotations

from typing import List, Tuple
import math
import mathutils

from .constants import EPS_NORM
from .attitude import quadrotor_attitude_from_positions


def _safe_norm3(v: Tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def direction_to_euler_xyz(direction: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Compute (rx, ry, rz) in radians from a direction vector.

    Convention used:
      - +X is forward
      - yaw (rz) rotates about +Z
      - pitch (ry) rotates about +Y
      - roll (rx) set to 0

    yaw   = atan2(dy, dx)
    pitch = atan2(dz, sqrt(dx^2 + dy^2))
    """
    dx, dy, dz = direction
    horiz = math.sqrt(dx * dx + dy * dy)
    if horiz < EPS_NORM and abs(dz) < EPS_NORM:
        return (0.0, 0.0, 0.0)

    yaw = math.atan2(dy, dx)
    pitch = math.atan2(dz, max(horiz, EPS_NORM))
    roll = 0.0
    return (roll, pitch, yaw)


def compute_rotations_from_positions(
    positions: List[Tuple[float, float, float]],
    *,
    dt: float = 0.1,
) -> List[Tuple[float, float, float]]:
    """
    Quadrotor-style attitude (yaw follows velocity when moving, tilt into
    acceleration, holds steady when hovering). See attitude.py.
    """
    return quadrotor_attitude_from_positions(positions, dt=dt)


def compute_rotations_from_positions_legacy(
    positions: List[Tuple[float, float, float]],
) -> List[Tuple[float, float, float]]:
    """
    Previous behaviour, kept for reference: heading from every per-step
    displacement (jitters wildly when hovering) with a fixed -15 deg offset on
    rx, which is roll about +X forward, not pitch.
    """
    n = len(positions)
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, 0.0)]

    rots: List[Tuple[float, float, float]] = []
    for i in range(n - 1):
        x0, y0, z0 = positions[i]
        x1, y1, z1 = positions[i + 1]
        d = (x1 - x0, y1 - y0, z1 - z0)

        if _safe_norm3(d) < EPS_NORM:
            rots.append(rots[-1] if rots else (0.0, 0.0, 0.0))
        else:
            rots.append(direction_to_euler_xyz(d))

    rots.append(rots[-1])

    # If you're computing rotations for a non rotational system, it can be useful
    # to pitch slightly forward (a bit more realistic)
    for i in range(n):
        rx, ry, rz = rots[i]
        rots[i] = (rx - math.pi / 12, ry, rz)

    return rots


def _quat_from_euler_xyz(rx: float, ry: float, rz: float) -> mathutils.Quaternion:
    return mathutils.Euler((rx, ry, rz), "XYZ").to_quaternion()


def _euler_xyz_from_quat(q: mathutils.Quaternion) -> Tuple[float, float, float]:
    e = q.to_euler("XYZ")
    return (float(e.x), float(e.y), float(e.z))


def smooth_euler_rotations_quat_window(
    rotations: List[Tuple[float, float, float]],
    *,
    window: int,
) -> List[Tuple[float, float, float]]:
    """
    Windowed quaternion mean with sign-alignment, then back to Euler XYZ.
    """
    n = len(rotations)
    if n == 0:
        return []
    if window <= 0:
        return list(rotations)

    qs = [_quat_from_euler_xyz(rx, ry, rz) for (rx, ry, rz) in rotations]

    out: List[Tuple[float, float, float]] = []
    for i in range(n):
        j0 = max(0, i - window)
        j1 = min(n, i + window + 1)

        q_ref = qs[i]

        qw = qx = qy = qz = 0.0
        for j in range(j0, j1):
            q = qs[j]
            if q.dot(q_ref) < 0.0:
                q = mathutils.Quaternion((-q.w, -q.x, -q.y, -q.z))
            qw += q.w
            qx += q.x
            qy += q.y
            qz += q.z

        norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if norm < EPS_NORM:
            out.append(rotations[i])
            continue

        q_mean = mathutils.Quaternion((qw / norm, qx / norm, qy / norm, qz / norm))
        out.append(_euler_xyz_from_quat(q_mean))

    return out
