# src/prospector/blender_integration/attitude.py
"""
Plausible quadrotor attitude from a position-only trajectory.

Pure math (no bpy / mathutils) so it can be tested outside Blender.

Why not "point along the per-step displacement"? For position/velocity-level
dynamics the per-step displacement is noisy (MPPI sampling), and when an agent
hovers or waits it is pure jitter, so heading and pitch jump by up to 180 deg
from frame to frame. A quadrotor also doesn't pitch its nose toward vertical
velocity; it yaws freely and tilts its thrust vector into horizontal
acceleration.

So:
  - yaw   follows the horizontal velocity heading, but only while the agent is
          actually moving (speed > min_speed_for_heading), through a first-order
          filter; otherwise it holds. Yaw is kept unwrapped (continuous).
  - tilt  is the thrust tilt needed for the horizontal acceleration plus a
          small drag term: tan(tilt) = (a_h + drag * v_h) / g, expressed in the
          yawed body frame (pitch forward/back, roll left/right), clamped.

Convention: body +X forward, +Z up; returned (rx, ry, rz) = (roll, pitch, yaw)
for Euler XYZ (R = Rz(yaw) Ry(pitch) Rx(roll)). Positive pitch tips the nose
down (forward acceleration); positive roll tips the right side down.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Vec3 = Tuple[float, float, float]


def _moving_average(xs: List[Vec3], window: int) -> List[Vec3]:
    n = len(xs)
    if window <= 0 or n == 0:
        return list(xs)
    # prefix sums for O(n)
    px, py, pz = [0.0], [0.0], [0.0]
    for x, y, z in xs:
        px.append(px[-1] + x)
        py.append(py[-1] + y)
        pz.append(pz[-1] + z)
    out: List[Vec3] = []
    for i in range(n):
        j0, j1 = max(0, i - window), min(n, i + window + 1)
        c = j1 - j0
        out.append(((px[j1] - px[j0]) / c, (py[j1] - py[j0]) / c, (pz[j1] - pz[j0]) / c))
    return out


def _derivative(xs: List[Vec3], dt: float) -> List[Vec3]:
    n = len(xs)
    if n < 2:
        return [(0.0, 0.0, 0.0)] * n
    out: List[Vec3] = []
    for i in range(n):
        i0, i1 = max(0, i - 1), min(n - 1, i + 1)
        h = (i1 - i0) * dt
        out.append(tuple((xs[i1][k] - xs[i0][k]) / h for k in range(3)))  # type: ignore[misc]
    return out


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def quadrotor_attitude_from_positions(
    positions: Sequence[Vec3],
    *,
    dt: float,
    smoothing_window: int = 8,
    min_speed_for_heading: float = 0.15,
    yaw_filter_gain: float = 0.04,
    drag_coeff: float = 1.0,
    max_tilt_rad: float = math.radians(25.0),
    g: float = 9.81,
) -> List[Vec3]:
    """
    Returns per-step (roll, pitch, yaw) in radians, same length as positions.

    dt                    : seconds between consecutive positions
    smoothing_window      : half-width (steps) of the moving average applied to
                            velocity and acceleration
    min_speed_for_heading : below this horizontal speed [m/s] yaw holds
    yaw_filter_gain       : per-step fraction of the heading error removed
    drag_coeff            : [1/s] extra tilt per m/s (steady cruise tilt)
    """
    pos = [tuple(map(float, p)) for p in positions]
    n = len(pos)
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0, 0.0)]

    vel = _moving_average(_derivative(pos, dt), smoothing_window)  # type: ignore[arg-type]
    acc = _moving_average(_derivative(vel, dt), smoothing_window)

    # Initial yaw: first confident heading, else 0
    yaw = 0.0
    for vx, vy, _ in vel:
        if math.hypot(vx, vy) > min_speed_for_heading:
            yaw = math.atan2(vy, vx)
            break

    tan_max = math.tan(max_tilt_rad)
    out: List[Vec3] = []
    for (vx, vy, _), (ax, ay, _) in zip(vel, acc):
        speed = math.hypot(vx, vy)
        if speed > min_speed_for_heading:
            # Ramp the gain in with speed so heading doesn't snap at the threshold
            ramp = min(1.0, (speed - min_speed_for_heading) / min_speed_for_heading)
            yaw += yaw_filter_gain * ramp * _wrap_pi(math.atan2(vy, vx) - yaw)

        # World-frame horizontal thrust tilt (tan of tilt angle)
        tx = (ax + drag_coeff * vx) / g
        ty = (ay + drag_coeff * vy) / g
        mag = math.hypot(tx, ty)
        if mag > tan_max:
            tx, ty = tx * tan_max / mag, ty * tan_max / mag

        # Into the yawed body frame: forward f, left l
        c, s = math.cos(yaw), math.sin(yaw)
        f = c * tx + s * ty
        l = -s * tx + c * ty
        pitch = math.atan(f)   # accelerate forward -> nose down (+ry)
        roll = -math.atan(l)   # accelerate left -> left side down (-rx)
        out.append((roll, pitch, yaw))
    return out
