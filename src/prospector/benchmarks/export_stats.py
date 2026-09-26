# src/prospector/benchmarks/export_stats.py
"""
Export benchmark_tours statistics to CSV.

    uv run python -m prospector.benchmarks.export_stats <run_dir> [<out_dir>] [--tag NAME]

Reads <run_dir>/config.yaml and <run_dir>/<instance>/trial_XX/{result.json,
infos.json, trajectories/agent_*.npz} and writes (to out_dir, default run_dir):

  <tag>trials.csv      one row per trial
  <tag>agents.csv      one row per (trial, agent): completion / hold / flying
                       time, path length, speed, acceleration, jerk, clearance
  <tag>waypoints.csv   one row per (trial, agent, waypoint): arrival time
  <tag>rendezvous.csv  one row per (trial, rendezvous): arrival / wait / release
  <tag>summary.csv     per instance (and 'ALL'): n, mean, std, min, median,
                       max of every numeric metric, plus success rate with a
                       Wilson 95% interval
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial import cKDTree


def _wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _write(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _clearance_field(args, cave_name: str, repo_root: Path):
    from prospector.caves.utils.build_cavemap import build_cavemap

    cave, _ = build_cavemap(cave_name, args.caves[cave_name], repo_root=repo_root, use_3d=True)
    # KD-tree over occupied voxel centers: continuous distance from the
    # drone's actual position to the nearest obstacle voxel (not quantized to
    # the drone's own voxel)
    occ = np.argwhere(cave.occupancy == 1)
    tree = cKDTree(cave.grid_min + (occ + 0.5) * cave.voxel_size)
    return cave, tree


def export(run_dir: Path, out_dir: Path, tag: str = "") -> None:
    from prospector.utils.custom_logging import get_repo_root_dir

    repo_root = Path(get_repo_root_dir())
    args = OmegaConf.load(run_dir / "config.yaml")
    bcfg = args.benchmark
    dyn_name = str(args.simulation.dynamics_model)
    dt = float(args.dynamics[dyn_name].dt)
    pos_dims = [int(i) for i in args.dynamics[dyn_name].position_dimensions]
    tours = {m["instance"]: m for m in json.loads((repo_root / bcfg.tours_file).read_text())["maps"]}

    trials, agents, waypoints, rendezvous = [], [], [], []
    for inst in bcfg.run_instances:
        inst_cfg = bcfg.instances[inst]
        cave_name = str(inst_cfg.cave)
        nodes = np.asarray(json.loads((repo_root / inst_cfg.graph_file).read_text())["nodes"])
        tour = tours[inst]
        contact_steps = [s["step"] for s in tour["steps"] if s["contact"]]
        paths = [tour[k] for k in sorted(k for k in tour if k.startswith("agent") and k.endswith("_path"))]
        cave, obstacle_tree = _clearance_field(args, cave_name, repo_root)

        for tdir in sorted((run_dir / inst).glob("trial_*")):
            if not (tdir / "result.json").exists():
                continue
            r = json.loads((tdir / "result.json").read_text())
            infos = json.loads((tdir / "infos.json").read_text())
            # Only up to the completion step: runs may continue for a short
            # post-completion hold (benchmark.hold_after_completion_s)
            infos = infos[: int(r["steps"])]
            idx = np.array([i["current_subtask_idx"] for i in infos])  # (T, N)
            waiting = np.array(
                [i.get("rendezvous_waiting", [False] * idx.shape[1]) for i in infos], dtype=bool
            )
            T, N = idx.shape
            fb = r.get("num_safe_fallbacks", [None] * N)

            trial_row = {
                "instance": inst,
                "cave": cave_name,
                "trial": r["trial"],
                "seed": r.get("seed"),
                "dynamics_model": dyn_name,
                "outcome": r["outcome"],
                "success": int(bool(r["success"])),
                "mission_time_s": r["sim_time_s"],
                "steps": r["steps"],
                "wall_time_s": r.get("wall_time_s"),
                "ms_per_step": r.get("ms_per_step"),
                "num_rendezvous_planned": len(contact_steps),
                "num_rendezvous_made": len(r.get("rendezvous_events", [])),
                "total_hold_time_s": float(waiting.sum() * dt),
                "reason": r["reason"],
            }

            for a in range(N):
                npz = np.load(tdir / "trajectories" / f"agent_{a}.npz")
                P = np.asarray(npz["states"])[: T + 1, pos_dims].astype(np.float64)
                nwp = len(paths[a]) - 1
                done = np.where(idx[:, a] >= nwp)[0]
                completion = float((done[0] + 1) * dt) if len(done) else float("nan")
                hold = float(waiting[:, a].sum() * dt)

                v = np.diff(P, axis=0) / dt
                speed = np.linalg.norm(v, axis=1)
                acc = np.linalg.norm(np.diff(v, axis=0) / dt, axis=1)
                jerk = np.linalg.norm(np.diff(np.diff(v, axis=0), axis=0) / dt**2, axis=1)
                moving = ~waiting[: len(speed), a] if len(speed) else np.zeros(0, bool)

                clr, _ = obstacle_tree.query(P)

                flown = float(speed.sum() * dt)
                nominal = float(
                    sum(np.linalg.norm(nodes[q] - nodes[p]) for p, q in zip(paths[a], paths[a][1:]))
                )
                flying = completion - hold if np.isfinite(completion) else float("nan")
                agents.append(
                    {
                        "instance": inst,
                        "trial": r["trial"],
                        "agent": a,
                        "success": trial_row["success"],
                        "waypoints_reached": int(min(idx[-1, a], nwp)),
                        "num_waypoints": nwp,
                        "completion_time_s": completion,
                        "hold_time_s": hold,
                        "flying_time_s": flying,
                        "hold_fraction": hold / completion if np.isfinite(completion) and completion > 0 else float("nan"),
                        "path_flown_m": flown,
                        "path_nominal_m": nominal,
                        "path_ratio": flown / nominal if nominal > 0 else float("nan"),
                        "mean_speed_moving_mps": float(speed[moving].mean()) if moving.any() else float("nan"),
                        "effective_speed_mps": nominal / flying if np.isfinite(flying) and flying > 0 else float("nan"),
                        "max_speed_mps": float(speed.max()) if len(speed) else float("nan"),
                        "mean_accel_mps2": float(acc.mean()) if len(acc) else float("nan"),
                        "p95_accel_mps2": float(np.percentile(acc, 95)) if len(acc) else float("nan"),
                        "max_accel_mps2": float(acc.max()) if len(acc) else float("nan"),
                        "mean_jerk_mps3": float(jerk.mean()) if len(jerk) else float("nan"),
                        "p95_jerk_mps3": float(np.percentile(jerk, 95)) if len(jerk) else float("nan"),
                        "min_clearance_m": float(clr.min()),
                        "p05_clearance_m": float(np.percentile(clr, 5)),
                        "mean_clearance_m": float(clr.mean()),
                        "safe_fallbacks": fb[a] if a < len(fb) else None,
                    }
                )

                # Waypoint arrival times (first step the subtask index passes k)
                for k in range(nwp):
                    hit = np.where(idx[:, a] > k)[0]
                    arrival = float((hit[0] + 1) * dt) if len(hit) else float("nan")
                    waypoints.append(
                        {
                            "instance": inst,
                            "trial": r["trial"],
                            "agent": a,
                            "waypoint_idx": k,
                            "tour_step": k + 1,
                            "node": paths[a][k + 1],
                            "is_rendezvous": int((k + 1) in contact_steps),
                            "passed_time_s": arrival,
                        }
                    )

            for e in r.get("rendezvous_events", []):
                row = {
                    "instance": inst,
                    "trial": r["trial"],
                    "tour_step": e["rendezvous_index"] + 1,
                    "released_at_s": e["released_at_step"] * dt,
                    "max_wait_s": max(e["wait_steps"]) * dt,
                }
                for a, (arr, w) in enumerate(zip(e["arrival_steps"], e["wait_steps"])):
                    row[f"agent{a}_node"] = paths[a][e["rendezvous_index"] + 1]
                    row[f"agent{a}_arrival_s"] = arr * dt
                    row[f"agent{a}_wait_s"] = w * dt
                rendezvous.append(row)

            trials.append(trial_row)

    # Summary
    summary = []

    def describe(group: str, rows: List[Dict[str, Any]], table: str) -> None:
        keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float)) and k not in ("trial", "agent", "seed", "waypoint_idx")]
        for k in keys:
            x = np.array([r[k] for r in rows if r[k] is not None], dtype=float)
            x = x[np.isfinite(x)]
            if len(x) == 0:
                continue
            summary.append(
                {
                    "instance": group,
                    "table": table,
                    "metric": k,
                    "n": len(x),
                    "mean": x.mean(),
                    "std": x.std(ddof=1) if len(x) > 1 else 0.0,
                    "min": x.min(),
                    "median": float(np.median(x)),
                    "max": x.max(),
                }
            )

    for group in list(bcfg.run_instances) + ["ALL"]:
        tr = [t for t in trials if group in ("ALL", t["instance"])]
        if not tr:
            continue
        k, n = sum(t["success"] for t in tr), len(tr)
        lo, hi = _wilson(k, n)
        summary.append(
            {"instance": group, "table": "trials", "metric": "success_rate", "n": n,
             "mean": k / n, "std": float("nan"), "min": lo, "median": float("nan"), "max": hi}
        )
        # Timing / motion stats over successful trials only
        succ = [t for t in tr if t["success"]]
        if succ:
            describe(group, succ, "trials (successes)")
        ag = [a for a in agents if group in ("ALL", a["instance"]) and a["success"]]
        if ag:
            describe(group, ag, "agents (successes)")
        rv = [x for x in rendezvous if group in ("ALL", x["instance"])]
        if rv:
            describe(group, [{"max_wait_s": x["max_wait_s"]} for x in rv], "rendezvous")

    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / f"{tag}trials.csv", trials)
    _write(out_dir / f"{tag}agents.csv", agents)
    _write(out_dir / f"{tag}waypoints.csv", waypoints)
    _write(out_dir / f"{tag}rendezvous.csv", rendezvous)
    _write(out_dir / f"{tag}summary.csv", summary)
    (out_dir / f"{tag}README.txt").write_text(
        f"Source run: {run_dir}\nDynamics: {dyn_name}, dt={dt}\n"
        "success_rate row: mean = rate, min/max = Wilson 95% interval.\n"
        "Timing/motion stats in summary.csv are over successful trials only.\n"
        "Clearance = distance from the drone position to the nearest occupied voxel center "
        f"(continuous KD-tree distance); crash check radius = {args.agents.agent_radius} m.\n"
        "Accel/jerk are finite differences of simulated positions at dt "
        "(for velocity-command dynamics they reflect command changes between steps).\n"
    )
    print(f"[export_stats] wrote CSVs to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    export(run_dir, Path(a.out_dir) if a.out_dir else run_dir, tag=(a.tag + "_") if a.tag else "")


if __name__ == "__main__":
    main()
