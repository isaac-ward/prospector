# src/prospector/benchmarks/benchmark_tours.py
"""
Benchmark high-level multi-agent tours with the low-level controller.

A tour file (e.g. src/assets/tours/example_tours.json) gives, per map,
step-aligned per-agent node paths over a cave graph, plus which steps end with
the agents in contact. For each tour we fly the paths with the configured
dynamics + MPPI controller, `benchmark.num_trials` times, and report whether
the controller can execute them.

Semantics
---------
- Agents fly their own node sequence at their own pace.
- At every step flagged `contact`, each agent holds at that step's node until
  all agents have arrived there AND the environment reports line of sight
  between them; then all continue (see CommsRendezvous).
- A trial succeeds if every agent completes its path, every rendezvous
  happens, and no agent collides, within the step budget.

Usage
-----
    uv run --env-file .env -- python -m prospector.benchmarks.benchmark_tours
    uv run --env-file .env -- python -m prospector.benchmarks.benchmark_tours \
        benchmark.num_trials=2 benchmark.run_instances=[tunnel_3d_025]
"""

from __future__ import annotations

import inspect
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from omegaconf import OmegaConf, DictConfig
import hydra


def _limit_threads() -> None:
    # Each worker process is one trial; keep XLA / BLAS single-threaded so
    # parallel workers don't oversubscribe the CPU.
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    )
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def _load_tour(repo_root: Path, tours_file: str, instance: str) -> Dict[str, Any]:
    data = json.loads((repo_root / tours_file).read_text())
    for m in data["maps"]:
        if m["instance"] == instance:
            return m
    raise KeyError(f"Instance '{instance}' not in {tours_file}")


def _agent_paths(tour: Dict[str, Any]) -> List[List[int]]:
    keys = sorted(
        (k for k in tour if k.startswith("agent") and k.endswith("_path")),
        key=lambda k: int(k[len("agent") : -len("_path")]),
    )
    return [list(tour[k]) for k in keys]


def run_trial(
    cfg_dict: Dict[str, Any],
    instance: str,
    trial_idx: int,
    out_dir: str,
    render: bool,
) -> Dict[str, Any]:
    """Run one trial of one tour. Returns a JSON-serializable result dict."""
    _limit_threads()
    import matplotlib

    matplotlib.use("Agg")

    from prospector.utils.custom_logging import get_repo_root_dir
    from prospector.caves.utils.build_cavemap import build_cavemap
    from prospector.agents.policies.mppi_factory import make_mppi_policies, mppi_settings
    from prospector.agents.agent_multi import MultiAgent
    from prospector.agents.colors import get_agent_color_list
    from prospector.environment.environment import ProspectorEnvironment
    from prospector.tasks.task_waypoint_following import TaskWaypointFollowing
    from prospector.tasks.comms_rendezvous import CommsRendezvous
    from prospector.tests.test_simulation import _DYNAMICS_REGISTRY

    args = OmegaConf.create(cfg_dict)
    bcfg = args.benchmark
    repo_root = Path(get_repo_root_dir())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inst_cfg = bcfg.instances[instance]
    cave_name = str(inst_cfg.cave)
    tour = _load_tour(repo_root, bcfg.tours_file, instance)
    paths = _agent_paths(tour)
    num_agents = len(paths)
    nodes = np.asarray(
        json.loads((repo_root / inst_cfg.graph_file).read_text())["nodes"], dtype=float
    )

    # Dynamics
    dynamics_name = str(args.simulation.dynamics_model)
    dyn_cfg = args.dynamics[dynamics_name]
    dyn_class = _DYNAMICS_REGISTRY[dynamics_name]
    # from_config also loads physical / controller parameters (Dynamics3D)
    dynamics = (
        dyn_class.from_config(dyn_cfg)
        if hasattr(dyn_class, "from_config")
        else dyn_class(dt=float(dyn_cfg.dt))
    )
    state_dim = int(dyn_cfg.state_dim)
    action_dim = int(dyn_cfg.action_dim)
    action_limits = np.asarray(dyn_cfg.action_limits, dtype=np.float64)
    position_dims = [int(i) for i in dyn_cfg.position_dimensions]

    cave_map, _ = build_cavemap(
        cave_name=cave_name,
        cave_cfg=args.caves[cave_name],
        repo_root=repo_root,
        use_3d=True,
    )

    # Rendezvous: tour step s (1-based) is waypoint index s-1 (paths[i][0] is
    # the start node, not a waypoint).
    contact_steps = [int(s["step"]) for s in tour["steps"] if s["contact"]]
    rendezvous_indices = [s - 1 for s in contact_steps] if bool(bcfg.wait_for_comms) else []

    agent_radius = float(args.agents.agent_radius)
    rw_cfg = args.rewards.waypoint_following
    initial_state = np.zeros((num_agents, state_dim), dtype=np.float32)
    tasks: List[TaskWaypointFollowing] = []
    for i, path in enumerate(paths):
        initial_state[i, position_dims] = nodes[path[0]]
        tasks.append(
            TaskWaypointFollowing(
                waypoints=nodes[path[1:]],
                position_dims=position_dims,
                cave_map=cave_map,
                agent_radius=agent_radius,
                distance_weight=float(rw_cfg.distance_weight),
                control_weight=float(rw_cfg.control_effort_weight),
                collision_penalty=float(rw_cfg.collision_penalty),
                multiplier_within_radius_of_goal=float(rw_cfg.multiplier_within_radius_of_goal),
                hold_at_indices=rendezvous_indices,
            )
        )
    rendezvous = (
        CommsRendezvous(
            tasks,
            rendezvous_indices,
            timeout_steps=int(bcfg.rendezvous_timeout_steps),
        )
        if rendezvous_indices
        else None
    )

    # Controller (same factory as test_simulation)
    seed = int(args.seed) + 1000 * trial_idx
    pol_cfg = mppi_settings(args, dynamics_name)
    policies = make_mppi_policies(args, dynamics_name, dynamics, tasks, action_limits, seed)

    agents = MultiAgent(
        num_agents=num_agents,
        state_dim=state_dim,
        action_dim=action_dim,
        history_len=int(args.agents.history_length),
        policies=policies,
    )
    # Real-time video at video_fps: each rendered step (render_every_n_steps
    # * dt seconds) is drawn as video_fps * that many frames, interpolated
    video_fps = int(bcfg.video_fps)
    frames = video_fps * float(dyn_cfg.dt) * int(bcfg.render_every_n_steps)
    if abs(frames - round(frames)) > 1e-6 or round(frames) < 1:
        raise ValueError(
            f"benchmark.video_fps * dt * render_every_n_steps must be a positive integer, got {frames}"
        )
    frames_per_step = int(round(frames))

    env = ProspectorEnvironment(
        config=args,
        dynamics=dynamics,
        cave_map=cave_map,
        agents=agents,
        initial_state=initial_state,
        render=render,
        render_fps=video_fps,
        render_frames_per_step=frames_per_step,
        render_every_n_steps=int(bcfg.render_every_n_steps),
        log_dir=out_dir,
        cave_name=cave_name,
        agent_colors=get_agent_color_list(num_agents),
        tasks=tasks,
        rendezvous=rendezvous,
        render_graph_nodes=nodes if bool(bcfg.render_graph) else None,
        render_graph_edges=(
            np.asarray(json.loads((repo_root / inst_cfg.graph_file).read_text())["edges"], dtype=int)
            if bool(bcfg.render_graph)
            else None
        ),
    )

    # Rollout
    max_steps = int(bcfg.max_steps)
    stall_timeout = int(bcfg.stall_timeout_steps)
    infos: List[Dict[str, Any]] = []
    outcome, reason = "timeout", f"hit max_steps={max_steps}"
    last_progress_step = 0
    last_idx = [0] * num_agents
    collision = None
    t0 = time.perf_counter()
    for _ in range(max_steps):
        state, reward, done, info = env.step()
        infos.append(info)
        step = env.step_index
        idx = list(info["current_subtask_idx"])
        if idx != last_idx:
            last_progress_step = step
            last_idx = idx

        alive = np.asarray(env.alive_mask)
        if not alive.all():
            i = int(np.where(~alive)[0][0])
            k = min(idx[i], len(paths[i]) - 2)
            collision = {
                "agent": i,
                "step": step,
                "position": np.asarray(state[i, position_dims]).tolist(),
                "waypoint_idx": k,
                "edge": [paths[i][k], paths[i][k + 1]],
            }
            outcome = "collision"
            reason = f"agent {i} collided at step {step} flying edge {paths[i][k]}->{paths[i][k + 1]}"
            break
        if rendezvous is not None and rendezvous.failed:
            outcome, reason = "rendezvous_timeout", rendezvous.failure_reason
            break
        if env.all_tasks_completed() and (rendezvous is None or rendezvous.all_released):
            outcome, reason = "success", "all waypoints reached and all rendezvous made"
            break
        waiting = info.get("rendezvous_waiting", [False] * num_agents)
        if step - last_progress_step > stall_timeout and not all(waiting):
            outcome, reason = "stalled", (
                f"no waypoint progress for {stall_timeout} steps "
                f"(subtask idx {idx}, waiting {waiting})"
            )
            break
    wall_s = time.perf_counter() - t0
    dt = float(dyn_cfg.dt)

    # All metrics are taken at this step. On success, keep simulating for a
    # short hold (agents keep station at their final nodes) so the video and
    # the exported trajectories (Blender) don't end mid-arrival.
    completion_step = env.step_index
    fallbacks_at_completion = [int(p.num_fallbacks) for p in policies]
    hold_steps = int(round(float(bcfg.get("hold_after_completion_s", 0.0)) / dt))
    post_completion_collision = None
    if outcome == "success":
        for _ in range(hold_steps):
            state, reward, done, info = env.step()
            infos.append(info)
            alive = np.asarray(env.alive_mask)
            if not alive.all() and post_completion_collision is None:
                post_completion_collision = {
                    "agents": np.where(~alive)[0].tolist(),
                    "step": env.step_index,
                }

    # Metrics
    per_agent = []
    for i in range(num_agents):
        s, _a = env.agents.get_full_history_for_agent(i)
        pos = np.asarray(s)[: completion_step + 1, position_dims]
        flown = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
        nominal = float(
            sum(np.linalg.norm(nodes[b] - nodes[a]) for a, b in zip(paths[i], paths[i][1:]))
        )
        per_agent.append(
            {
                "waypoints_reached": int(min(tasks[i].current_subtask_idx, tasks[i].num_subtasks)),
                "num_waypoints": int(tasks[i].num_subtasks),
                "path_length_flown_m": flown,
                "path_length_nominal_m": nominal,
            }
        )

    result = {
        "instance": instance,
        "cave": cave_name,
        "trial": trial_idx,
        "seed": seed,
        "dynamics_model": dynamics_name,
        "policy_mppi": OmegaConf.to_container(pol_cfg),
        "wait_for_comms": bool(bcfg.wait_for_comms),
        "margin_radius": pol_cfg.get("margin_radius", None),
        "margin_penalty": float(pol_cfg.get("margin_penalty", 0.0)),
        "margin_exempt_radius": float(pol_cfg.get("margin_exempt_radius", 0.0)),
        "safe_fallback": bool(pol_cfg.get("safe_fallback", False)),
        "num_safe_fallbacks": fallbacks_at_completion,
        "outcome": outcome,
        "success": outcome == "success",
        "reason": reason,
        "steps": completion_step,
        "sim_time_s": completion_step * dt,
        "wall_time_s": wall_s,
        "ms_per_step": 1000.0 * wall_s / max(1, completion_step),
        # Extra steps simulated after success (station keeping at final nodes)
        "hold_after_completion_steps": env.step_index - completion_step,
        "post_completion_collision": post_completion_collision,
        "collision": collision,
        "rendezvous_indices": rendezvous_indices,
        "rendezvous_events": rendezvous.events if rendezvous is not None else [],
        "agents": per_agent,
    }

    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    env.export_infos(infos)
    # npz for analysis, json for the Blender integration (trajectories/agent_*.json)
    env.export_agent_trajectories(save_as_numpy=True, save_as_json=True)
    if render:
        video = env.finalize_video(output_name=f"{instance}_trial_{trial_idx:02d}")
        result["video"] = str(video)
        (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def _summarize(results: List[Dict[str, Any]]) -> str:
    lines = [
        "| instance | trials | success | collision | rendezvous_timeout | stalled | timeout | mean sim time (s, successes) | mean ms/step |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    by_inst: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_inst.setdefault(r["instance"], []).append(r)
    for inst, rs in by_inst.items():
        count = lambda o: sum(r["outcome"] == o for r in rs)
        succ = [r["sim_time_s"] for r in rs if r["success"]]
        lines.append(
            f"| {inst} | {len(rs)} | {count('success')}/{len(rs)} ({100 * count('success') / len(rs):.0f}%) "
            f"| {count('collision')} | {count('rendezvous_timeout')} | {count('stalled')} | {count('timeout')} "
            f"| {np.mean(succ):.1f} | {np.mean([r['ms_per_step'] for r in rs]):.0f} |"
            if succ
            else f"| {inst} | {len(rs)} | 0/{len(rs)} (0%) | {count('collision')} | {count('rendezvous_timeout')} "
            f"| {count('stalled')} | {count('timeout')} | - | {np.mean([r['ms_per_step'] for r in rs]):.0f} |"
        )
    lines.append("")
    lines.append("Failures:")
    for r in results:
        if not r["success"]:
            lines.append(f"- {r['instance']} trial {r['trial']}: {r['reason']}")
    return "\n".join(lines)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(args: DictConfig):
    print(f"[{inspect.stack()[0][3]}] configuration:\n{OmegaConf.to_yaml(args)}")

    from prospector.utils.custom_logging import make_log_dir, get_repo_root_dir
    from prospector.caves.utils.build_cavemap import build_cavemap

    bcfg = args.benchmark
    log_dir = make_log_dir(suffix="benchmark_tours")
    print(f"[benchmark_tours] Logging to: {log_dir}")
    cfg_dict = OmegaConf.to_container(args, resolve=True)
    (log_dir / "config.yaml").write_text(OmegaConf.to_yaml(args))

    # Build / warm the cave-map caches once up front so workers only load them.
    repo_root = Path(get_repo_root_dir())
    for inst in bcfg.run_instances:
        cave_name = str(bcfg.instances[inst].cave)
        build_cavemap(cave_name, args.caves[cave_name], repo_root=repo_root, use_3d=True)

    jobs = []
    video_trials = set(int(t) for t in bcfg.video_trials)
    for inst in bcfg.run_instances:
        for trial in range(int(bcfg.num_trials)):
            out = log_dir / inst / f"trial_{trial:02d}"
            jobs.append((cfg_dict, str(inst), trial, str(out), trial in video_trials))

    results: List[Dict[str, Any]] = []
    num_workers = int(bcfg.num_workers)
    if num_workers <= 1:
        for job in jobs:
            results.append(run_trial(*job))
            print(f"[benchmark_tours] {job[1]} trial {job[2]}: {results[-1]['outcome']}")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(run_trial, *job): job for job in jobs}
            for fut in as_completed(futures):
                job = futures[fut]
                try:
                    r = fut.result()
                except Exception:
                    tb = traceback.format_exc()
                    print(f"[benchmark_tours] {job[1]} trial {job[2]} CRASHED:\n{tb}")
                    r = {
                        "instance": job[1],
                        "trial": job[2],
                        "outcome": "crash",
                        "success": False,
                        "reason": tb.strip().splitlines()[-1],
                        "sim_time_s": 0.0,
                        "ms_per_step": 0.0,
                    }
                results.append(r)
                print(f"[benchmark_tours] {job[1]} trial {job[2]}: {r['outcome']} ({r['reason']})", flush=True)
                # Keep the summary current so an interrupted run still has one
                _write_summary(log_dir, results, final=False)

    summary = _write_summary(log_dir, results, final=True)
    print(f"\n[benchmark_tours] Summary ({log_dir}):\n{summary}")


def _write_summary(log_dir: Path, results: List[Dict[str, Any]], *, final: bool) -> str:
    results = sorted(results, key=lambda r: (r["instance"], r["trial"]))
    (log_dir / "results.json").write_text(json.dumps(results, indent=2))
    summary = _summarize(results)
    if not final:
        summary = "(in progress)\n\n" + summary
    (log_dir / "summary.md").write_text(summary)
    return summary


if __name__ == "__main__":
    main()
