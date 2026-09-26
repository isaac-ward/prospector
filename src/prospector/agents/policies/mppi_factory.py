# src/prospector/agents/policies/mppi_factory.py
"""
Build MPPI policies from the config, shared by test_simulation and the tour
benchmark so both fly with exactly the same controller.

Config (policies.policy_mppi):
  - base settings (num_samples, horizon, lambda_, noise_std, sampling_mode, ...)
  - per_dynamics.<dynamics_model>: overrides for that dynamics model
  - safe_fallback, margin_radius, margin_penalty, margin_exempt_radius
Older run configs kept these under benchmark.policy_mppi / benchmark.safety;
those are still read so old runs can be reproduced.
"""

from __future__ import annotations

import functools
from typing import Any, List, Sequence

import numpy as np
from omegaconf import OmegaConf

from .policy_mppi import PolicyMPPI

_SAFETY_KEYS = ("safe_fallback", "margin_radius", "margin_penalty", "margin_exempt_radius")


def mppi_settings(args: Any, dynamics_name: str):
    """Merged MPPI settings for this dynamics model (OmegaConf node)."""
    base = OmegaConf.to_container(args.policies.policy_mppi, resolve=True)
    per = base.pop("per_dynamics", {}) or {}
    cfg = OmegaConf.create(base)
    legacy = args.get("benchmark", {}) or {}
    if dynamics_name in (legacy.get("policy_mppi", {}) or {}):
        cfg = OmegaConf.merge(cfg, legacy.policy_mppi[dynamics_name])
    if dynamics_name in per:
        cfg = OmegaConf.merge(cfg, per[dynamics_name])
    # Legacy location of the safety options
    for k, v in (legacy.get("safety", {}) or {}).items():
        if k not in base:
            cfg[k] = v
    return cfg


def make_mppi_policies(
    args: Any,
    dynamics_name: str,
    dynamics: Any,
    tasks: Sequence[Any],
    action_limits: np.ndarray,
    seed: int,
) -> List[PolicyMPPI]:
    """One MPPI policy per task (agent), seeded seed + i."""
    cfg = mppi_settings(args, dynamics_name)
    collision_penalty = float(args.rewards.waypoint_following.collision_penalty)
    fallback_drop = 0.1 * collision_penalty if bool(cfg.get("safe_fallback", False)) else None
    margin_radius = cfg.get("margin_radius", None)

    noise_std = cfg.noise_std
    noise_std = OmegaConf.to_container(noise_std) if OmegaConf.is_config(noise_std) else float(noise_std)
    extra = {
        k: cfg[k]
        for k in ("warm_start", "noise_smoothing", "include_nominal_sample", "reward_normalization")
        if k in cfg
    }
    if hasattr(dynamics, "hover_action"):
        extra["nominal_action"] = np.asarray(dynamics.hover_action())

    policies = []
    for i, task in enumerate(tasks):
        pol = PolicyMPPI(
            action_dim=int(action_limits.shape[0]),
            action_limits=action_limits,
            num_samples=int(cfg.num_samples),
            horizon=int(cfg.horizon),
            lambda_=float(cfg.lambda_),
            sampling_mode=str(cfg.sampling_mode),
            noise_std=noise_std,
            seed=seed + i,
            safe_fallback_reward_drop=fallback_drop,
            **extra,
        )
        pol.set_dynamics(dynamics)
        pol.set_reward_function(task.reward_for_active_subtask)
        if margin_radius is not None:
            pol.set_batched_reward_function(
                functools.partial(
                    task.batch_reward_for_active_subtask,
                    margin_radius=float(margin_radius),
                    margin_penalty=float(cfg.get("margin_penalty", 0.0)),
                    margin_exempt_radius=float(cfg.get("margin_exempt_radius", 0.0)),
                ),
                future_only_collisions=True,
            )
        else:
            pol.set_batched_reward_function(task.batch_reward_for_active_subtask)
        policies.append(pol)
    return policies
