# src/prospector/environment/environment.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Optional, Sequence

import jax.numpy as jnp
import numpy as np

from ..dynamics.base_dynamics import BaseDynamics
from ..agents.agent_multi import MultiAgent
from ..caves.cave_map_2d import CaveMap2D
from ..caves.cave_map_3d import CaveMap3D
from ..utils.plotting_orchestrator import PlottingOrchestrator


AliveFn = Callable[[jnp.ndarray], bool]


class ProspectorEnvironment:
    """
    Environment wrapper around a dynamics model and a (multi-)agent controller.

    Responsibilities
    ----------------
    - Hold the current state (for one or many agents).
    - Ask the agent(s) for actions each step.
    - Advance the state using the provided `BaseDynamics` instance.
    - Track a simple step counter and per-agent "alive" / "done" flags.
    - Return reward (currently zeros) and a debug/info dict.
    - Optionally manage rendering via an internal PlottingOrchestrator when
      `render=True`.
    """

    def __init__(
        self,
        dynamics: BaseDynamics,
        cave_map: CaveMap2D | CaveMap3D,
        agents: MultiAgent,
        initial_state: jnp.ndarray,
        config: Any,
        *,
        render: bool = False,
        log_dir: Optional[str | Path] = None,
        cave_name: Optional[str] = None,
        agent_colors: Optional[np.ndarray] = None,
    ) -> None:
        """
        Parameters
        ----------
        dynamics      : BaseDynamics
            Underlying system dynamics.
        cave_map      : CaveMap2D or CaveMap3D
            Cave map for line-of-sight calculations and collision checking.
        agents        : MultiAgent
            Controller(s) that produce actions from histories.
        initial_state : (num_agents, state_dim) or (state_dim,)
            Initial state for each agent's copy of the system.
        config        : Dict-like / DictConfig
            Full configuration for reference (Hydra DictConfig is fine).
        render        : bool, optional
            If True, this environment will construct and own a
            PlottingOrchestrator and call `render_frame` on each step.
        log_dir       : str or Path, optional
            Logging directory for rendered frames (required if render=True).
        cave_name     : str, optional
            Name of the cave / scenario (used for naming and config lookup;
            required if render=True).
        agent_colors  : (num_agents,) array-like, optional
            Per-agent colors for plotting (required if render=True).
        """
        self._dynamics = dynamics
        self._cave_map = cave_map
        self._agents = agents
        self.config = config

        self._max_episode_steps = self.config.environment.max_episode_length

        # What are the state elements that define the cartesian position?
        self.position_dims = tuple(
            self.config.dynamics[self.config.simulation.dynamics_model].position_dimensions
        )

        self._state = self._normalize_state_shape(initial_state)
        self._step_index: int = 0

        # Per-agent alive mask; True until collision or global termination.
        self._alive = jnp.ones((self._agents.num_agents,), dtype=bool)

        # Initialize agent histories with the initial state.
        self._agents.reset(self._state)

        # ------------------------------------------------------------------ #
        # Optional rendering setup                                           #
        # ------------------------------------------------------------------ #
        self._render = bool(render)
        self._orch: Optional[PlottingOrchestrator] = None
        self._agent_colors: Optional[np.ndarray] = None
        self._cave_name: Optional[str] = cave_name

        if self._render:
            if log_dir is None:
                raise ValueError("log_dir must be provided when render=True.")
            if cave_name is None:
                raise ValueError("cave_name must be provided when render=True.")
            if agent_colors is None:
                raise ValueError("agent_colors must be provided when render=True.")

            log_dir = Path(log_dir)
            self._agent_colors = np.asarray(agent_colors)

            # Determine 2D vs 3D from the cave map or position dims.
            is_3d = isinstance(cave_map, CaveMap3D) or (len(self.position_dims) == 3)
            mode = "3d" if is_3d else "2d"

            cave_cfg = self.config.caves[cave_name]

            self._orch = PlottingOrchestrator(
                mode=mode,
                cave_map=cave_map,
                cave_name=f"{cave_name}_sim",
                log_dir=log_dir,
                fps=24,
                max_obstacle_points_3d=cave_cfg.plotting.obstacle_points.max_num,
                alpha_obstacles_3d=cave_cfg.plotting.obstacle_points.alpha,
                figure_size_multiplier=config.simulation.render.figure_size_multiplier,
            )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _normalize_state_shape(self, state: jnp.ndarray) -> jnp.ndarray:
        s = jnp.asarray(state)

        if s.ndim == 1:
            if self._agents.num_agents != 1:
                raise ValueError(
                    "Provided state is 1D but agents.num_agents != 1 "
                    f"(agents.num_agents={self._agents.num_agents})."
                )
            s = s[None, :]  # (1, state_dim)

        if s.shape[0] != self._agents.num_agents:
            raise ValueError(
                "Mismatch between number of agents and state batch dimension: "
                f"state.shape[0]={s.shape[0]}, "
                f"agents.num_agents={self._agents.num_agents}"
            )

        return s

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def reset(self, initial_state: jnp.ndarray) -> jnp.ndarray:
        """
        Reset environment state, alive mask, and agent histories.

        Parameters
        ----------
        initial_state : (num_agents, state_dim) or (state_dim,)

        Returns
        -------
        state : (num_agents, state_dim) jnp.ndarray
            The new current state.
        """
        self._state = self._normalize_state_shape(initial_state)
        self._agents.reset(self._state)
        self._step_index = 0
        self._alive = jnp.ones((self._agents.num_agents,), dtype=bool)
        return self._state

    def step(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """
        Advance the environment by one time step.

        Returns
        -------
        next_state : (num_agents, state_dim) jnp.ndarray
        reward     : (num_agents,) jnp.ndarray
        done       : (num_agents,) jnp.ndarray of bool
        info       : dict
        """
        num_agents = self._agents.num_agents

        # 1) Ask agents for actions.
        actions = self._agents.act()  # (num_agents, action_dim)

        # 2) Apply dynamics per agent (no batch assumption).
        next_states_list: List[jnp.ndarray] = []
        debug_list: List[Dict[str, jnp.ndarray]] = []
        new_alive_list: List[bool] = []
        done_list: List[bool] = []

        for i in range(num_agents):
            state_i = self._state[i]
            action_i = actions[i]

            if not bool(self._alive[i]):
                # Dead agents: freeze state, ignore action.
                next_states_list.append(jnp.asarray(state_i))
                debug_list.append({"status": np.asarray("dead")})
                new_alive_list.append(False)
                done_list.append(True)
                continue

            # Alive agent: step dynamics.
            ns_i, dbg_i = self._dynamics.step(state_i, action_i)
            ns_i = jnp.asarray(ns_i)
            next_states_list.append(ns_i)
            debug_list.append(dbg_i)

            # Check for collisions with cave map.
            still_alive = True
            if self._cave_map.is_collision_within_radius(
                [ns_i[dim] for dim in self.position_dims],
                self.config.agents.agent_radius,
            ):
                still_alive = False

            new_alive_list.append(still_alive)
            done_list.append(not still_alive)

        next_state = jnp.stack(next_states_list, axis=0)

        # Update internal alive mask.
        self._alive = jnp.asarray(new_alive_list, dtype=bool)

        # 3) Update histories on the agent side.
        self._agents.update_history(next_state, actions)

        # 4) Update environment state and step index.
        self._state = next_state
        self._step_index += 1

        # 5) Global episode termination by step limit (if configured).
        max_episode_terminated = False
        if self._max_episode_steps is not None:
            if self._step_index >= self._max_episode_steps:
                max_episode_terminated = True
                done_list = [True] * num_agents

        done = jnp.asarray(done_list, dtype=bool)

        # 6) Reward (currently zero for all agents).
        reward = jnp.zeros((num_agents,), dtype=jnp.float32)

        # 7) Communications matrix (line of sight).
        comms_matrix = np.zeros((num_agents, num_agents), dtype=np.int8)
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if not (self._alive[i] and self._alive[j]):
                    continue
                pos_i = self._state[i, self.position_dims]
                pos_j = self._state[j, self.position_dims]
                if self._cave_map.has_line_of_sight(pos_i, pos_j):
                    comms_matrix[i, j] = 1
                    comms_matrix[j, i] = 1
        # Mirror it (note: this will double the 1s to 2s as written).
        comms_matrix += comms_matrix.T

        info: Dict[str, Any] = {
            "actions": actions,
            "dynamics_debug": debug_list,
            "step_index": self._step_index,
            "alive_mask": self._alive,
            "max_episode_terminated": max_episode_terminated,
            "communications_matrix": comms_matrix,
        }

        # 8) Optional rendering
        if self._render and self._orch is not None and self._agent_colors is not None:
            agent_positions = np.asarray(
                [self._state[i, self.position_dims].tolist() for i in range(num_agents)]
            )
            alive_mask = np.asarray(self._alive)

            # frame_idx should match the external "t" used previously (0-based).
            frame_idx = self._step_index - 1

            self._orch.render_frame(
                frame_idx=frame_idx,
                agent_positions=agent_positions,
                agent_colors=self._agent_colors,
                agent_alives=alive_mask,
                agent_communications_matrix=comms_matrix,
            )

        return self._state, reward, done, info

    def finalize_video(self, output_name: Optional[str] = None) -> Optional[Path]:
        """
        If rendering is enabled, finalize and export the simulation video.

        Parameters
        ----------
        output_name : str, optional
            Name for the output video (without extension). If None, a default
            based on the cave name is used.

        Returns
        -------
        video_path : Path or None
        """
        if not self._render or self._orch is None:
            return None

        if output_name is None:
            # Fall back to config-based cave name if we didn't get one in ctor.
            cave_name = (
                self._cave_name if self._cave_name is not None else self.config.simulation.cave
            )
            output_name = f"{cave_name}_simulation"

        video_path = self._orch.finalize_video(output_name=output_name)
        return Path(video_path)

    # Convenience properties
    # ------------------------------------------------------------------ #
    @property
    def num_agents(self) -> int:
        return self._agents.num_agents

    @property
    def state(self) -> jnp.ndarray:
        """Current state, always shape (num_agents, state_dim) internally."""
        return self._state

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def dynamics(self) -> BaseDynamics:
        return self._dynamics

    @property
    def agents(self) -> MultiAgent:
        return self._agents

    @property
    def alive_mask(self) -> jnp.ndarray:
        """Per-agent alive mask, updated each step."""
        return self._alive
