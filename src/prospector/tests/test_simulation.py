# src/prospector/tests/test_simulation.py

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, Any

import numpy as np
import jax.numpy as jnp
from omegaconf import OmegaConf, DictConfig
import hydra
from tqdm import tqdm
import matplotlib as mpl


from prospector.utils.custom_logging import make_log_dir, get_repo_root_dir
from prospector.utils.plotting_orchestrator import PlottingOrchestrator
from prospector.caves.utils.build_cavemap import build_cavemap

from prospector.caves.cave_map_2d import CaveMap2D

from prospector.dynamics.dynamics_2d_linear import Dynamics2DLinear
from prospector.dynamics.dynamics_3d_linear import Dynamics3DLinear
from prospector.dynamics.dynamics_3d import Dynamics3D

from prospector.agents.policies.policy_random import RandomPolicy
from prospector.agents.agent_multi import MultiAgent
from prospector.agents.colors import get_agent_color_list
from prospector.environment.environment import ProspectorEnvironment


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
    seed = int(getattr(args, "seed", 42))
    rng = np.random.default_rng(seed)
    print(f"[test_simulation] Using seed: {seed}")

    # ------------------------------------------------------------------ #
    # Agent parameters will be used shortly to get initialization        #
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
    physical_parameters = dict(getattr(dyn_cfg, "physical_parameters", {}))

    dynamics = dyn_class(dt=dt, physical_parameters=physical_parameters or None)

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
    agent_radius = float(args.agents.agent_radius)
    print(f"[test_simulation] Sampling initial position (agent_radius={agent_radius}) ...")

    start_points = cave_map.sample_free_points(
        num_points=num_agents,
        min_distance=agent_radius * 4.0,
        rng=rng,
    )
    print(f"[test_simulation] Initial world position(s): {start_points}")

    # ------------------------------------------------------------------ #
    # Build initial state for dynamics                                   #
    # ------------------------------------------------------------------ #

    # The state isn't always just the position; we need to build the full state.
    state_dim = args.dynamics[dynamics_name].state_dim
    print(f"[test_simulation] Building initial state with state_dim={state_dim} ...")
    initial_state = np.zeros((num_agents, state_dim), dtype=np.float32)  # (N, state_dim)
    for i, pt in enumerate(start_points):
        initial_state[i, position_dims] = pt

    # ------------------------------------------------------------------ #
    # Instantiate policy, agents, environment                            #
    # ------------------------------------------------------------------ #
    if policy_name != "policy_random":
        raise ValueError(
            f"Only 'policy_random' is supported currently, got '{policy_name}'."
        )

    policy = RandomPolicy(
        action_dim=action_dim,
        action_limits=action_limits,
    )

    agents = MultiAgent(
        num_agents=num_agents,
        state_dim=state_dim,
        action_dim=action_dim,
        history_len=history_len,
        policies=policy,
    )

    env = ProspectorEnvironment(
        config=args,
        dynamics=dynamics,
        cave_map=cave_map,
        agents=agents,
        initial_state=initial_state,
    )

    print(
        f"[test_simulation] Environment initialized with num_agents={env.num_agents}, "
        f"state_dim={state_dim}, action_dim={action_dim}"
    )

    # ------------------------------------------------------------------ #
    # Plotting orchestrator for simulation                               #
    # ------------------------------------------------------------------ #
    orch = PlottingOrchestrator(
        mode="3d" if is_3d else "2d",
        cave_map=cave_map,
        cave_name=f"{cave_name}_sim",
        log_dir=log_dir,
        fps=24,
        max_obstacle_points_3d=cave_cfg.plotting.obstacle_points.max_num,
        alpha_obstacles_3d=cave_cfg.plotting.obstacle_points.alpha,
    )

    # ------------------------------------------------------------------ #
    # Rollout + plotting                                                 #
    # ------------------------------------------------------------------ #
    num_steps = int(getattr(sim_cfg, "num_steps", 200))
    print(f"[test_simulation] Running simulation for {num_steps} steps ...")
    pbar = tqdm(range(num_steps), desc="Simulation", unit="step")

    for t in pbar:
        state, reward, done, info = env.step()  # state shape: (num_agents, state_dim)

        # Agent info for plotting
        agent_positions = np.array(
            [state[i, position_dims].tolist() for i in range(num_agents)]
        )  # (N, 2) or (N, 3)
        alive_mask = np.array(info["alive_mask"])  # (N,)

        frame_path = orch.render_frame(
            frame_idx=t,
            agent_positions=agent_positions,     # (N, 2) or (N, 3)
            agent_colors=agent_colors,           # (N,)
            agent_alives=alive_mask,             # (N,)
            agent_communications_matrix=info["communications_matrix"],  # (N, N)
        )

    # Export video of the simulation
    video_path = orch.finalize_video(output_name=f"{cave_name}_simulation")
    print(f"[test_simulation] Simulation video saved to: {video_path}")

    print("[test_simulation] Test complete.")


if __name__ == "__main__":
    main()
