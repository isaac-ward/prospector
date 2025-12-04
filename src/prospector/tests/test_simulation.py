# src/prospector/tests/test_simulation.py

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import jax.numpy as jnp
from omegaconf import OmegaConf, DictConfig
import hydra
from tqdm import tqdm
import matplotlib as mpl

from prospector.utils.custom_logging import make_log_dir, get_repo_root_dir
from prospector.caves.utils.build_cavemap import build_cavemap

from prospector.caves.cave_map_2d import CaveMap2D

from prospector.dynamics.dynamics_2d_linear import Dynamics2DLinear
from prospector.dynamics.dynamics_3d_linear import Dynamics3DLinear
from prospector.dynamics.dynamics_3d import Dynamics3D

from prospector.agents.initial_states import sample_initial_states
from prospector.agents.policies.policy_random import PolicyRandom
from prospector.agents.policies.policy_mppi import PolicyMPPI
from prospector.agents.agent_multi import MultiAgent
from prospector.agents.colors import get_agent_color_list
from prospector.environment.environment import ProspectorEnvironment
from prospector.tasks.task_waypoint_following import TaskWaypointFollowing


_DYNAMICS_REGISTRY: Dict[str, Any] = {
    "dynamics_2d_linear": Dynamics2DLinear,
    "dynamics_3d_linear": Dynamics3DLinear,
    "dynamics_3d": Dynamics3D,
}


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(args: DictConfig):
    # ------------------------------------------------------------------ #
    # Print full config                                                  #
    # ------------------------------------------------------------------ #
    yaml_str = OmegaConf.to_yaml(args)
    print(f"[{inspect.stack()[0][3]}] configuration:\n{yaml_str}")

    # ------------------------------------------------------------------ #
    # Setup logging directory                                            #
    # ------------------------------------------------------------------ #
    log_dir = make_log_dir(prefix="test_simulation")
    print(f"[test_simulation] Logging to: {log_dir}")

    repo_root = Path(get_repo_root_dir())

    # ------------------------------------------------------------------ #
    # Simulation config                                                  #
    # ------------------------------------------------------------------ #
    sim_cfg = args.simulation
    dynamics_name: str = sim_cfg.dynamics_model
    cave_name: str = sim_cfg.cave
    policy_name: str = sim_cfg.agents.policy

    # Seed
    seed = args.seed
    rng = np.random.default_rng(seed)
    print(f"[test_simulation] Using seed: {seed}")

    # ------------------------------------------------------------------ #
    # Agent parameters                                                   #
    # ------------------------------------------------------------------ #
    num_agents = int(args.simulation.agents.num_agents)
    agent_colors = get_agent_color_list(num_agents)
    history_len = int(args.agents.history_length)

    # ------------------------------------------------------------------ #
    # Instantiate dynamics                                               #
    # ------------------------------------------------------------------ #
    if dynamics_name not in _DYNAMICS_REGISTRY:
        raise ValueError(
            f"Unknown dynamics_model '{dynamics_name}'. "
            f"Available: {list(_DYNAMICS_REGISTRY.keys())}"
        )

    dyn_class = _DYNAMICS_REGISTRY[dynamics_name]
    dyn_cfg = args.dynamics[dynamics_name]

    dt = float(dyn_cfg.dt)
    #physical_parameters = dyn_cfg.physical_parameters

    dynamics = dyn_class(dt=dt)

    state_dim = int(dyn_cfg.state_dim)
    action_dim = int(dyn_cfg.action_dim)
    action_limits = np.asarray(dyn_cfg.action_limits, dtype=np.float64)  # (action_dim, 2)

    # Position dimensions (indices into the state vector used as [x, y, (z)])
    position_dims = [int(i) for i in dyn_cfg.position_dimensions]
    if len(position_dims) not in (2, 3):
        raise ValueError(
            f"position_dimensions must have length 2 or 3; "
            f"got {position_dims} for dynamics '{dynamics_name}'"
        )

    is_3d = len(position_dims) == 3

    print(
        f"[test_simulation] Dynamics '{dynamics_name}': "
        f"state_dim={state_dim}, action_dim={action_dim}, dt={dt}, "
        f"position_dimensions={position_dims}, is_3d={is_3d}"
    )

    # ------------------------------------------------------------------ #
    # Load / build cave map                                              #
    # ------------------------------------------------------------------ #
    cave_cfg = args.caves[cave_name]
    cave_map, ply_path = build_cavemap(
        cave_name=cave_name,
        cave_cfg=cave_cfg,
        repo_root=repo_root,
        use_3d=is_3d,
    )
    print(f"[test_simulation] Cave '{cave_name}' PLY path: {ply_path}")

    # ------------------------------------------------------------------ #
    # Sample initial position from free space                            #
    # ------------------------------------------------------------------ #
    initial_state, start_points = sample_initial_states(
        cave_map=cave_map,
        num_agents=num_agents,
        state_dim=state_dim,
        position_dims=position_dims,
        agent_radius=args.agents.agent_radius,
        rng=rng,
    )

    print(f"[test_simulation] Initial world position(s): {start_points}")

    # ------------------------------------------------------------------ #
    # Instantiate per-agent waypoint tasks                               #
    # ------------------------------------------------------------------ #
    num_waypoints = sim_cfg.num_waypoints
    agent_radius = float(args.agents.agent_radius)

    # Reward hyperparameters from config
    rw_cfg = args.rewards.waypoint_following
    distance_weight = float(rw_cfg.distance_weight)
    control_weight = float(rw_cfg.control_effort_weight)
    collision_penalty = float(rw_cfg.collision_penalty)

    tasks: List[TaskWaypointFollowing] = []

    for agent_idx in range(num_agents):
        # Sample waypoints in free space; ensure some separation
        waypoint_points_world = cave_map.sample_free_points(
            num_points=num_waypoints,
            min_distance=agent_radius * 2.0,
            rng=rng,
        )  # shape (num_waypoints, world_dim)

        # Compress world coords to state position dimensions
        waypoint_points_task = waypoint_points_world[:, position_dims]

        task = TaskWaypointFollowing(
            waypoints=waypoint_points_task,
            position_dims=position_dims,
            cave_map=cave_map,
            agent_radius=agent_radius,
            distance_weight=distance_weight,
            control_weight=control_weight,
            collision_penalty=collision_penalty,
            multiplier_within_radius_of_goal=args.rewards.waypoint_following.multiplier_within_radius_of_goal,
        )
        tasks.append(task)

    print("[test_simulation] Waypoints per agent (task coords):")
    for i, task in enumerate(tasks):
        # this attribute name matches the implementation we wrote earlier
        print(f"  agent {i}: {np.asarray(task._waypoints)}")  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ #
    # Instantiate per-agent policies + MultiAgent                        #
    # ------------------------------------------------------------------ #
    policies: List[Any] = []

    if policy_name == "policy_random":
        for _ in range(num_agents):
            policies.append(
                PolicyRandom(
                    action_dim=action_dim,
                    action_limits=action_limits,
                )
            )

    elif policy_name == "policy_mppi":
        pol_cfg = args.policies.policy_mppi
        for i in range(num_agents):
            pol = PolicyMPPI(
                action_dim=action_dim,
                action_limits=action_limits,
                num_samples=int(pol_cfg.num_samples),
                horizon=int(pol_cfg.horizon),
                lambda_=float(pol_cfg.lambda_),
                sampling_mode=str(pol_cfg.sampling_mode),
                noise_std=float(pol_cfg.noise_std),
                seed=seed + i,  # different seed per agent for diversity
            )
            pol.set_dynamics(dynamics)
            # Each policy gets its *own* task's reward function
            pol.set_reward_function(tasks[i].reward_for_active_subtask)
            policies.append(pol)

    else:
        raise ValueError(
            f"Unsupported policy '{policy_name}'. "
            f"Expected one of: 'policy_random', 'policy_mppi'."
        )

    agents = MultiAgent(
        num_agents=num_agents,
        state_dim=state_dim,
        action_dim=action_dim,
        history_len=history_len,
        policies=policies,
    )

    env = ProspectorEnvironment(
        config=args,
        dynamics=dynamics,
        cave_map=cave_map,
        agents=agents,
        initial_state=initial_state,
        render=args.simulation.render.enabled,
        log_dir=log_dir,
        cave_name=cave_name,
        agent_colors=agent_colors,
        tasks=tasks,
    )

    print(
        f"[test_simulation] Environment initialized with num_agents={env.num_agents}, "
        f"state_dim={state_dim}, action_dim={action_dim}"
    )

    # ------------------------------------------------------------------ #
    # Rollout (rendering is handled inside env.step)                     #
    # ------------------------------------------------------------------ #
    num_steps = args.environment.max_episode_length
    print(f"[test_simulation] Running simulation for {num_steps} steps ...")
    pbar = tqdm(range(num_steps), desc="Simulation", unit="step")

    num_steps_to_simulate_after_completion = int(args.simulation.num_steps_to_simulate_after_completion)
    for _ in pbar:
        state, reward, done, info = env.step()  # env handles render_frame internally

        if "current_subtask_idx" in info and isinstance(info["current_subtask_idx"], list):
            if num_agents > 0:
                pbar.set_postfix(
                    {
                        "wp0": int(info["current_subtask_idx"][0]),
                        "r0": float(reward[0]),
                    }
                )

        # Also end episode if all agents are done or have completed their tasks
        if done.all():
            print("[test_simulation] All agents done (dead); ending episode early.")
            break
        if env.all_tasks_completed():
            print(f"[test_simulation] All agents completed their tasks; ending episode in {num_steps_to_simulate_after_completion} steps.")
            num_steps_to_simulate_after_completion -= 1
            if num_steps_to_simulate_after_completion <= 0:
                break

    # Export video of the simulation (env owns the orchestrator).
    video_path = env.finalize_video(output_name=f"{cave_name}_simulation")
    print(f"[test_simulation] Simulation video saved to: {video_path}")

    print("[test_simulation] Test complete.")


if __name__ == "__main__":
    main()
