# src/prospector/tests/test_dynamics_3d_tuning.py
"""
Sanity + closed-loop tuning checks for the 12D quadrotor (Dynamics3D).

Run (from the repo root):

    uv run python src/prospector/tests/test_dynamics_3d_tuning.py
    uv run python src/prospector/tests/test_dynamics_3d_tuning.py --mode attitude
    uv run python src/prospector/tests/test_dynamics_3d_tuning.py --obstacles

Checks
------
(a) hover equilibrium in every control mode (and residual accelerations),
(b) open-loop sanity with raw rotor speeds: stays upright at hover, climbs
    with more thrust, correct lateral acceleration sign for roll / pitch,
    correct yaw sign; RK4 vs fine-step convergence,
(c) MPPI closed loop in free space (optionally with synthetic spherical
    obstacles) through 5 waypoints 3-8 m apart using the *real*
    TaskWaypointFollowing reward (vectorised with
    TaskWaypointFollowing.batch_reward_for_active_subtask). Reports time to reach every
    waypoint, average speed, max tilt, max body rate, stability and ms/step.

Every check raises AssertionError on failure; the script exits non-zero.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Sequence

import numpy as np
import jax.numpy as jnp

from prospector.dynamics.dynamics_3d import Dynamics3D
from prospector.agents.policies.policy_mppi import PolicyMPPI
from prospector.tasks.task_waypoint_following import TaskWaypointFollowing


# ---------------------------------------------------------------------- #
# Recommended settings (mirror the YAML snippet in the report)           #
# ---------------------------------------------------------------------- #
DT = 0.1
# 10 RK4 sub-steps per 0.1 s -> 100 Hz inner attitude / rate loop. The
# class default (20, i.e. 200 Hz) gives identical closed-loop results at
# ~2x the rollout cost.
RECOMMENDED_DYN_KWARGS: Dict = {"substeps": 10, "integrator": "rk4"}

ACTION_LIMITS: Dict[str, List[List[float]]] = {
    # [vx, vy, vz, yaw_rate]
    "velocity": [[-2.0, 2.0], [-2.0, 2.0], [-1.0, 1.0], [-0.5, 0.5]],
    # [a_z offset, roll, pitch, yaw_rate]
    "attitude": [[-4.0, 4.0], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]],
    # rotor speeds [rad/s]
    "rotor_speeds": [[100.0, 1500.0]] * 4,
}

MPPI_SETTINGS: Dict[str, Dict] = {
    # "median" normalisation: weights exp((r - r_max) / (lambda * (r_max -
    # median r))). Unlike "range"/"std" it is not flattened by the 1e6
    # collision penalty on a few samples.
    "velocity": dict(
        num_samples=256, horizon=20, lambda_=0.2, noise_std=0.5,
        sampling_mode="gaussian", warm_start=True, noise_smoothing=0.5,
        use_batched_rollouts=True, reward_normalization="median",
    ),
    "attitude": dict(
        num_samples=512, horizon=20, lambda_=0.2, noise_std=[1.5, 0.15, 0.15, 0.2],
        sampling_mode="gaussian", warm_start=True, noise_smoothing=0.7,
        use_batched_rollouts=True, reward_normalization="median",
    ),
    "rotor_speeds": dict(
        num_samples=1024, horizon=20, lambda_=0.2, noise_std=40.0,
        sampling_mode="gaussian", warm_start=True, noise_smoothing=0.5,
        use_batched_rollouts=True, reward_normalization="median",
    ),
}

# Reward weights / radii as in conf/config.yaml
DISTANCE_WEIGHT = 1.0
CONTROL_WEIGHT = 0.1
COLLISION_PENALTY = 1e6
AGENT_RADIUS = 0.1
GOAL_MULTIPLIER = 3.0     # -> waypoint reached within 0.3 m

WAYPOINTS = np.array(
    [
        [4.0, 0.0, 1.0],
        [4.0, 5.0, 2.0],
        [-2.0, 6.0, 1.0],
        [-3.0, 0.0, 0.5],
        [0.0, 0.0, 1.5],
    ]
)
START = np.array([0.0, 0.0, 1.0])

OBSTACLES = [  # (center, radius) spheres placed on the straight-line legs
    (np.array([4.0, 2.5, 1.5]), 0.6),
    (np.array([1.0, 5.5, 1.5]), 0.6),
    (np.array([-2.5, 3.0, 0.75]), 0.6),
]


# ---------------------------------------------------------------------- #
# Synthetic free-space "cave" map                                        #
# ---------------------------------------------------------------------- #
class SyntheticWorld:
    """Free space inside an axis-aligned box, minus spherical obstacles."""

    def __init__(self, obstacles=(), lo=(-20, -20, -5), hi=(20, 20, 15)):
        self.obstacles = list(obstacles)
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)

    def batch_is_collision_within_radius(self, pts, radius):
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        hit = np.any(pts - radius < self.lo, axis=-1) | np.any(pts + radius > self.hi, axis=-1)
        for c, r in self.obstacles:
            hit |= np.linalg.norm(pts - c, axis=-1) <= r + radius
        return hit

    def is_collision_within_radius(self, p, radius):
        return bool(self.batch_is_collision_within_radius(np.asarray(p, float)[None], radius)[0])


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #
def tilt_deg(state) -> float:
    s = np.asarray(state)
    return float(np.rad2deg(np.arccos(np.clip(np.cos(s[4]) * np.cos(s[5]), -1, 1))))


def run_open_loop(dyn: Dynamics3D, state, action, steps: int):
    s = jnp.asarray(state, dtype=jnp.float32)
    traj = []
    for _ in range(steps):
        s, _ = dyn.step(s, jnp.asarray(action, dtype=jnp.float32))
        traj.append(np.asarray(s))
    return np.stack(traj)


def hover_state(z=1.0):
    s = np.zeros(12, dtype=np.float32)
    s[2] = z
    return s


# ---------------------------------------------------------------------- #
# (a) Hover equilibrium                                                  #
# ---------------------------------------------------------------------- #
def test_hover_equilibrium() -> None:
    print("\n(a) Hover equilibrium")
    for mode in ("rotor_speeds", "attitude", "velocity"):
        dyn = Dynamics3D(dt=DT, control_mode=mode)
        traj = run_open_loop(dyn, hover_state(), dyn.hover_action(), 100)  # 10 s
        drift = np.max(np.abs(traj[-1] - hover_state()))
        print(f"  {mode:13s}: max |state drift| after 10 s = {drift:.2e}")
        # Open-loop rotor-speed hover is only marginally stable, so float32
        # round-off (~1e-8 N m torque residual under jit) integrates to ~1 cm
        # in 10 s; the closed-loop modes reject it entirely.
        tol = 2e-2 if mode == "rotor_speeds" else 1e-4
        assert drift < tol, f"hover not an equilibrium in mode {mode}"
    dyn = Dynamics3D(dt=DT)
    w_h = dyn.hover_rotor_speed
    sdot = dyn._state_derivative(jnp.asarray(hover_state()), jnp.full(4, w_h**2))
    print(f"  hover rotor speed = {w_h:.1f} rad/s, thrust/weight = {dyn.thrust_to_weight:.2f}, "
          f"|s_dot| at hover = {float(jnp.max(jnp.abs(sdot))):.1e}")
    assert float(jnp.max(jnp.abs(sdot))) < 1e-4
    assert 1.8 <= dyn.thrust_to_weight <= 2.2


# ---------------------------------------------------------------------- #
# (b) Open-loop sanity with raw rotor speeds                             #
# ---------------------------------------------------------------------- #
def test_open_loop_sanity() -> None:
    print("\n(b) Open-loop sanity (rotor_speeds)")
    dyn = Dynamics3D(dt=DT, control_mode="rotor_speeds")
    w_h = dyn.hover_rotor_speed
    s0 = hover_state()

    # Upright at hover
    traj = run_open_loop(dyn, s0, [w_h] * 4, 50)
    print(f"  hover 5 s: max tilt {max(tilt_deg(s) for s in traj):.2e} deg")
    assert max(tilt_deg(s) for s in traj) < 1e-3

    # Climbs with more thrust (+5% rotor speed ~ +10% thrust)
    traj = run_open_loop(dyn, s0, [1.05 * w_h] * 4, 10)
    az = 2 * traj[0, 8] / DT  # ~ initial accel
    print(f"  +5% rotor speed for 1 s: dz = {traj[-1, 2] - 1.0:+.3f} m, vz = {traj[-1, 8]:+.3f} m/s "
          f"(expected a_z ~ {9.81 * (1.05**2 - 1):.2f} m/s^2 minus drag)")
    assert traj[-1, 2] > 1.2 and traj[-1, 8] > 0.5
    traj = run_open_loop(dyn, s0, [0.95 * w_h] * 4, 10)
    assert traj[-1, 2] < 0.8

    # Tilt sign -> lateral acceleration sign (thrust set to hold altitude)
    th = np.deg2rad(10.0)
    for name, idx, ax_idx, sign in (("pitch +10deg", 4, 6, +1), ("roll +10deg", 5, 7, -1)):
        s = s0.copy()
        s[idx] = th
        w = w_h / np.sqrt(np.cos(th))
        traj = run_open_loop(dyn, s, [w] * 4, 5)
        v = traj[-1, ax_idx]
        print(f"  {name}: {'vx' if ax_idx == 6 else 'vy'} after 0.5 s = {v:+.3f} m/s "
              f"(expected sign {sign:+d}, ~{sign * 9.81 * np.tan(th) * 0.5:+.2f}), "
              f"dz = {traj[-1, 2] - 1:+.4f} m")
        assert np.sign(v) == sign and abs(v) > 0.5

    # Differential thrust signs: left (w1) faster -> +roll; rear (w2) faster -> +pitch
    d = 0.01 * w_h
    traj = run_open_loop(dyn, s0, [w_h + d, w_h, w_h - d, w_h], 1)
    print(f"  w1 (left) > w3 (right): roll after 0.1 s = {traj[-1, 5]:+.4f} rad (expect > 0)")
    assert traj[-1, 5] > 0
    traj = run_open_loop(dyn, s0, [w_h, w_h + d, w_h, w_h - d], 1)
    print(f"  w2 (rear) > w4 (front): pitch after 0.1 s = {traj[-1, 4]:+.4f} rad (expect > 0, nose down)")
    assert traj[-1, 4] > 0
    traj = run_open_loop(dyn, s0, [w_h + d, w_h - d, w_h + d, w_h - d], 1)
    print(f"  CW rotors (w1, w3) faster: yaw after 0.1 s = {traj[-1, 3]:+.4f} rad (expect > 0)")
    assert traj[-1, 3] > 0

    # Integrator convergence: RK4 @ 5 ms vs RK4 @ 0.5 ms on an aggressive manoeuvre
    act = [w_h + 5 * d, w_h - 3 * d, w_h - 5 * d, w_h + 3 * d]
    ref = run_open_loop(Dynamics3D(dt=DT, substeps=200), s0, act, 5)[-1]
    for integ, sub in (("rk4", 20), ("euler", 20), ("rk4", 1)):
        out = run_open_loop(Dynamics3D(dt=DT, substeps=sub, integrator=integ), s0, act, 5)[-1]
        print(f"  {integ:5s} x{sub:3d} substeps: max state error vs fine = {np.max(np.abs(out - ref)):.2e}")
    out = run_open_loop(Dynamics3D(dt=DT), s0, act, 5)[-1]
    assert np.max(np.abs(out - ref)) < 1e-3

    # Closed-loop inner controllers
    dyn_v = Dynamics3D(dt=DT, control_mode="velocity")
    traj = run_open_loop(dyn_v, s0, [2.0, -1.0, 0.5, 0.0], 30)
    print(f"  velocity mode, step to v=[2,-1,0.5]: v after 3 s = {np.round(traj[-1, 6:9], 3)}, "
          f"max tilt {max(tilt_deg(s) for s in traj):.1f} deg")
    assert np.allclose(traj[-1, 6:9], [2.0, -1.0, 0.5], atol=0.05)
    dyn_a = Dynamics3D(dt=DT, control_mode="attitude")
    traj = run_open_loop(dyn_a, s0, [0.0, 0.2, -0.3, 0.0], 10)
    print(f"  attitude mode, step to roll 0.2 / pitch -0.3: (roll, pitch) after 1 s = "
          f"({traj[-1, 5]:.3f}, {traj[-1, 4]:.3f}), dz = {traj[-1, 2] - 1:+.4f} m")
    assert abs(traj[-1, 5] - 0.2) < 0.01 and abs(traj[-1, 4] + 0.3) < 0.01
    assert abs(traj[-1, 2] - 1.0) < 0.05


def test_batched_reward_matches_task() -> None:
    """The vectorised reward must equal TaskWaypointFollowing's scalar one."""
    print("\n(b2) Batched reward == task.reward_for_active_subtask; batched rollouts == step()")
    world = SyntheticWorld(OBSTACLES)
    task = TaskWaypointFollowing(
        waypoints=WAYPOINTS, position_dims=[0, 1, 2], cave_map=world,
        agent_radius=AGENT_RADIUS, distance_weight=DISTANCE_WEIGHT,
        control_weight=CONTROL_WEIGHT, collision_penalty=COLLISION_PENALTY,
        multiplier_within_radius_of_goal=GOAL_MULTIPLIER,
    )
    batch_fn = task.batch_reward_for_active_subtask
    rng = np.random.default_rng(0)
    N, T = 16, 12
    states = np.zeros((N, T, 12), dtype=np.float32)
    states[..., :3] = rng.uniform([2, 0, 0], [6, 5, 3], size=(N, T, 3))
    actions = rng.normal(size=(N, T - 1, 4)).astype(np.float32)
    ref = np.array([float(task.reward_for_active_subtask(jnp.asarray(states[k]), jnp.asarray(actions[k])))
                    for k in range(N)])
    out = batch_fn(states, actions)
    n_coll = int(np.sum(ref < -COLLISION_PENALTY / 2))
    print(f"  max |batched - scalar| = {np.max(np.abs(out - ref)):.2e} "
          f"(relative {np.max(np.abs(out - ref) / np.abs(ref)):.1e}); {n_coll}/{N} samples collide")
    assert np.allclose(out, ref, rtol=1e-5)

    dyn = Dynamics3D(dt=DT, control_mode="velocity")
    seqs = rng.uniform(-1, 1, size=(4, 10, 4)).astype(np.float32)
    s0 = jnp.asarray(hover_state())
    batched = np.asarray(dyn.rollout_batch(s0, seqs))
    for k in range(4):
        seq_states = run_open_loop(dyn, s0, seqs[k, 0], 1)  # first step
        s = s0
        for t in range(10):
            s, _ = dyn.step(s, seqs[k, t])
        assert np.allclose(np.asarray(s), batched[k, -1], atol=1e-4)
        assert np.allclose(seq_states[0], batched[k, 0], atol=1e-5)
    print("  rollout_batch matches sequential step() calls")


# ---------------------------------------------------------------------- #
# (c) MPPI closed loop                                                   #
# ---------------------------------------------------------------------- #
def run_closed_loop(
    mode: str = "velocity",
    *,
    obstacles: bool = False,
    max_time: float = 60.0,
    seed: int = 0,
    mppi_overrides: Dict | None = None,
    use_scalar_reward: bool = False,
    verbose: bool = True,
    dyn_kwargs: Dict | None = None,
) -> Dict:
    world = SyntheticWorld(OBSTACLES if obstacles else ())
    dyn = Dynamics3D(
        dt=DT, control_mode=mode, **(RECOMMENDED_DYN_KWARGS if dyn_kwargs is None else dyn_kwargs)
    )
    task =TaskWaypointFollowing(
        waypoints=WAYPOINTS,
        position_dims=[0, 1, 2],
        cave_map=world,
        agent_radius=AGENT_RADIUS,
        distance_weight=DISTANCE_WEIGHT,
        control_weight=CONTROL_WEIGHT,
        collision_penalty=COLLISION_PENALTY,
        multiplier_within_radius_of_goal=GOAL_MULTIPLIER,
    )
    settings = dict(MPPI_SETTINGS[mode])
    settings.update(mppi_overrides or {})
    pol = PolicyMPPI(
        action_dim=4,
        action_limits=ACTION_LIMITS[mode],
        nominal_action=dyn.hover_action(),
        seed=seed,
        **settings,
    )
    pol.set_dynamics(dyn)
    pol.set_reward_function(task.reward_for_active_subtask)
    if not use_scalar_reward:
        pol.set_batch_reward_function(task.batch_reward_for_active_subtask)

    history_len = 16
    s = jnp.asarray(np.concatenate([START, np.zeros(9)]), dtype=jnp.float32)
    states: List[np.ndarray] = [np.asarray(s)]
    actions: List[np.ndarray] = []
    reach_times: List[float] = []
    act_ms: List[float] = []
    collided = False
    n_steps = int(max_time / DT)
    for t in range(n_steps):
        hs = jnp.asarray(np.stack(states[-history_len:]))
        ha = jnp.asarray(np.stack(actions[-history_len:])) if actions else jnp.zeros((0, 4))
        t0 = time.perf_counter()
        a = pol.act(hs, ha)
        a = np.asarray(a)
        act_ms.append((time.perf_counter() - t0) * 1e3)
        s, _ = dyn.step(s, a)
        s_np = np.asarray(s)
        states.append(s_np)
        actions.append(a)
        if not np.all(np.isfinite(s_np)):
            break
        if world.is_collision_within_radius(s_np[:3], AGENT_RADIUS):
            collided = True
            break
        before = task.current_subtask_idx
        task.update_subtask_from_state(s)
        for _ in range(task.current_subtask_idx - before):
            reach_times.append((t + 1) * DT)
            if verbose:
                print(f"    waypoint {len(reach_times)} reached at t = {(t + 1) * DT:5.1f} s "
                      f"(pos {np.round(s_np[:3], 2)}, |v| = {np.linalg.norm(s_np[6:9]):.2f} m/s)")
        if task.is_task_completed():
            break

    S = np.stack(states)
    finite = bool(np.all(np.isfinite(S)))
    path = float(np.sum(np.linalg.norm(np.diff(S[:, :3], axis=0), axis=1))) if finite else float("nan")
    t_total = (len(states) - 1) * DT
    legs = np.linalg.norm(np.diff(np.vstack([START, WAYPOINTS]), axis=0), axis=1)
    return dict(
        mode=mode,
        reached=len(reach_times),
        reach_times=reach_times,
        completed=task.is_task_completed(),
        collided=collided,
        finite=finite,
        t_total=t_total,
        path_length=path,
        avg_speed=path / t_total if t_total > 0 else 0.0,
        straight_line_speed=float(np.sum(legs[: len(reach_times)])) / reach_times[-1] if reach_times else 0.0,
        max_tilt_deg=max(tilt_deg(x) for x in S) if finite else float("nan"),
        max_rate=float(np.max(np.abs(S[:, 9:12]))) if finite else float("nan"),
        max_speed=float(np.max(np.linalg.norm(S[:, 6:9], axis=1))) if finite else float("nan"),
        ms_per_step=float(np.median(act_ms[1:])) if len(act_ms) > 1 else float("nan"),
        first_call_ms=act_ms[0] if act_ms else float("nan"),
        states=S,
    )


def print_result(r: Dict) -> None:
    print(f"  mode={r['mode']}: reached {r['reached']}/{len(WAYPOINTS)} waypoints, "
          f"completed={r['completed']}, collided={r['collided']}, finite={r['finite']}")
    print(f"    reach times [s]: {[round(x, 1) for x in r['reach_times']]}  (total {r['t_total']:.1f} s)")
    print(f"    path {r['path_length']:.1f} m, avg speed {r['avg_speed']:.2f} m/s "
          f"(straight-line {r['straight_line_speed']:.2f} m/s), max speed {r['max_speed']:.2f} m/s")
    print(f"    max tilt {r['max_tilt_deg']:.1f} deg, max body rate {r['max_rate']:.2f} rad/s")
    print(f"    policy compute: {r['ms_per_step']:.1f} ms/step (median; first call incl. jit "
          f"{r['first_call_ms']:.0f} ms)")


def test_mppi_closed_loop(mode: str = "velocity", obstacles: bool = False, seed: int = 0) -> Dict:
    print(f"\n(c) MPPI closed loop, mode={mode}, obstacles={obstacles}, seed={seed}")
    r = run_closed_loop(mode, obstacles=obstacles, seed=seed)
    print_result(r)
    assert r["finite"], "state diverged"
    assert not r["collided"], "collision"
    assert r["completed"], "did not reach all waypoints"
    assert r["avg_speed"] >= 0.5, "too slow"
    assert r["max_tilt_deg"] < 45.0, "excessive tilt"
    return r


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="velocity", choices=["velocity", "attitude", "rotor_speeds"])
    ap.add_argument("--obstacles", action="store_true")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--skip-open-loop", action="store_true")
    args = ap.parse_args(argv)

    failures = 0
    if not args.skip_open_loop:
        for fn in (test_hover_equilibrium, test_open_loop_sanity, test_batched_reward_matches_task):
            try:
                fn()
            except AssertionError as e:
                failures += 1
                print(f"  FAIL: {fn.__name__}: {e}")
    for seed in range(args.seeds):
        try:
            test_mppi_closed_loop(args.mode, args.obstacles, seed)
        except AssertionError as e:
            failures += 1
            print(f"  FAIL: closed loop seed {seed}: {e}")
    print("\nALL PASSED" if failures == 0 else f"\n{failures} FAILURE(S)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
