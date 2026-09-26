# src/prospector/benchmarks/make_latex_table.py
"""
LaTeX results table (paper style: booktabs, grey environment rows, siunitx
columns) from export_stats CSVs.

    uv run python -m prospector.benchmarks.make_latex_table <csv_dir> <tag> <out.tex>

e.g. make_latex_table <run_dir> full_dynamics <run_dir>/table.tex

Rows per environment: each agent (mean +- std over trials) and the team
(both agents combined per trial, then mean +- std over trials). Each column
gets enough decimals that every nonzero standard deviation is visible; an
exactly-zero spread is printed explicitly as +- 0.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ENVIRONMENTS = [
    # instance, display name
    ("tunnel_3d_025", "DARPA 3D Tunnels-25 (N = 2)"),
    ("chamber_3d_025", "DARPA 3D Chamber-25 (N = 2)"),
    ("grapevine_3d_025", "Grapevine-25 (N = 2)"),
]

# (header, unit, agents.csv column)
METRICS = [
    ("Time", "(s)", "completion_time_s"),
    ("Waiting time", "(s)", "hold_time_s"),
    ("Path flown", "(m)", "path_flown_m"),
    ("Path ratio", "", "path_ratio"),
    ("Average speed", "(m/s)", "mean_speed_moving_mps"),
    ("Clearance", "(m)", "min_clearance_m"),
]
NUM_COLS = 2 + len(METRICS)


def _mean_std(x):
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


def _decimals(stds, lo=2, hi=4):
    """Fewest decimals (>= lo) at which every nonzero std rounds to nonzero."""
    nz = [s for s in stds if s > 0]
    for d in range(lo, hi + 1):
        if all(round(s, d) > 0 for s in nz):
            return d
    return hi


def build(csv_dir: Path, tag: str) -> str:
    rd = lambda name: list(csv.DictReader(open(csv_dir / f"{tag}_{name}.csv")))
    trials, agents = rd("trials"), rd("agents")
    by_trial = defaultdict(dict)
    for a in agents:
        by_trial[(a["instance"], a["trial"])][int(a["agent"])] = a

    # blocks: list of (header, [(label, successes, [(mean, std) per metric])])
    blocks = []
    for inst, name in ENVIRONMENTS:
        tr = [t for t in trials if t["instance"] == inst]
        if not tr:
            continue
        ok = [t for t in tr if t["success"] == "1"]
        n_rv = max(int(t["num_rendezvous_planned"]) for t in tr)
        # tour steps = waypoints per agent (paths are step-aligned)
        steps = max(int(a["num_waypoints"]) for a in agents if a["instance"] == inst)
        rows = []
        mine_all = [a for a in agents if a["instance"] == inst]
        for i in sorted({int(a["agent"]) for a in mine_all}):
            mine = [a for a in mine_all if int(a["agent"]) == i]
            # An agent succeeds if it completed its whole path (collision-free)
            done = sum(a["waypoints_reached"] == a["num_waypoints"] for a in mine)
            r = [a for a in mine if a["success"] == "1"]
            rows.append(
                (f"Agent {i + 1}", f"{done}/{len(mine)}",
                 [_mean_std([float(a[k]) for a in r]) for _, _, k in METRICS])
            )
        # Team: combine both agents per successful trial
        team = defaultdict(list)
        for t in ok:
            pa = list(by_trial[(inst, t["trial"])].values())
            flown = sum(float(a["path_flown_m"]) for a in pa)
            team["completion_time_s"].append(float(t["mission_time_s"]))
            team["hold_time_s"].append(sum(float(a["hold_time_s"]) for a in pa))
            team["path_flown_m"].append(flown)
            team["path_ratio"].append(flown / sum(float(a["path_nominal_m"]) for a in pa))
            team["mean_speed_moving_mps"].append(np.mean([float(a["mean_speed_moving_mps"]) for a in pa]))
            team["min_clearance_m"].append(min(float(a["min_clearance_m"]) for a in pa))
        rows.append(("Team", f"{len(ok)}/{len(tr)}", [_mean_std(team[k]) for _, _, k in METRICS]))
        header = (
            rf"\multicolumn{{{NUM_COLS}}}{{l}}{{\cellcolor{{gray!15}}\textbf{{\textsc{{{name}}}}}"
            rf"\quad {steps} tour steps, {n_rv} communication rendezvous}} \\"
        )
        blocks.append((header, rows))

    # Per-column decimals and siunitx formats
    col_cells = [[cells[m] for _, rows in blocks for _, _, cells in rows] for m in range(len(METRICS))]
    # Cells can be NaN when an environment has no successful trials
    col_cells = [[(m, s) for m, s in cells if np.isfinite(m) and np.isfinite(s)] or [(0.0, 0.0)]
                 for cells in col_cells]
    decimals = [_decimals([s for _, s in cells]) for cells in col_cells]
    formats = []
    for d, cells in zip(decimals, col_cells):
        int_digits = max(len(str(int(abs(m)))) for m, _ in cells)
        unc_digits = max(len(str(int(round(s * 10**d)))) for _, s in cells)
        formats.append(f"S[table-format={int_digits}.{d}({unc_digits})]")

    n_trials = max(sum(t["instance"] == inst for t in trials) for inst, _ in ENVIRONMENTS)

    def cell(m, s, d):
        if not (np.isfinite(m) and np.isfinite(s)):
            return "{--}"
        if round(s, d) == 0:
            # exact zero spread: print it explicitly (siunitx may drop a zero uncertainty)
            return f"{{${m:.{d}f} \\pm {0:.{d}f}$}}"
        return f"{m:.{d}f} +- {s:.{d}f}"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Low-level execution of multi-agent tours in Prospector \citep{ward2025prospector}. "
        rf"Each tour is flown by two quadrotors, {n_trials} trials per environment (mean $\pm$ standard "
        r"deviation). \emph{Time}: launch to arrival at the final node. \emph{Waiting time}: time "
        r"spent waiting for the other agent to get into communications range. \emph{Path ratio}: "
        r"distance actually flown over the total straight line length of the planned node path. "
        r"\emph{Clearance}: closest approach to an occupied cell of the voxelized cave map (the "
        r"collision radius is 0.1\,m).}",
        r"\label{tab:simulation_results}",
        "",
        r"\begin{adjustbox}{max width=\textwidth}",
        r"\begin{tabular}{@{}l|r|" + "|".join(formats) + r"@{}}",
        r"\toprule",
        "& {Successes} & " + " & ".join(f"{{{h}}}" for h, _, _ in METRICS) + r" \\",
        "& & " + " & ".join(f"{{{u}}}" if u else "" for _, u, _ in METRICS) + r" \\",
    ]
    for header, rows in blocks:
        lines += [r"\midrule", header]
        for label, succ, cells in rows:
            lines.append(
                f"{label} & {succ} & "
                + " & ".join(cell(m, s, d) for (m, s), d in zip(cells, decimals))
                + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    csv_dir, tag, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    out.write_text(build(csv_dir, tag))
    print(f"[make_latex_table] wrote {out}")


if __name__ == "__main__":
    main()
