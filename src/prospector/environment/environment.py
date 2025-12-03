# src/prospector/environment/environment.py

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import jax.numpy as jnp
import numpy as np

from ..dynamics.base_dynamics import BaseDynamics
from ..agents.agent_multi import MultiAgent
from ..caves.cave_map_2d import CaveMap2D
from ..caves.cave_map_3d import CaveMap3D


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

    Design choices
    --------------
    - We *do not* assume the dynamics are batchable; instead, we treat
      `num_agents` as independent copies of the same dynamics and step
      them in a Python loop.
    - For num_agents == 1, we still use the same interface, with the
      state stored as shape (1, state_dim) internally.
    - Collision / termination is injected via an optional `alive_fn`
      that maps a state vector to a boolean "still alive" flag.
    """

    def __init__(
        self,
        dynamics: BaseDynamics,
        cave_map: CaveMap2D | CaveMap3D,
        agents: MultiAgent,
        initial_state: jnp.ndarray,
        config: Dict[str, Any],
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
        config       : Dict[str, Any]
            Full configuration dictionary for reference.
        """
        self._dynamics = dynamics
        self._cave_map = cave_map
        self._agents = agents
        self.config = config 

        self._max_episode_steps = self.config.environment.max_episode_length

        # What are the state elements that define the cartesian position? 
        self.position_dims = tuple(self.config.dynamics[self.config.simulation.dynamics_model].position_dimensions)

        self._state = self._normalize_state_shape(initial_state)
        self._step_index: int = 0

        # Per-agent alive mask; True until collision or global termination.
        self._alive = jnp.ones((self._agents.num_agents,), dtype=bool)

        # Initialize agent histories with the initial state.
        self._agents.reset(self._state)

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

        Procedure
        ---------
        - Query the agent(s) for actions using their internal histories.
        - For each agent:
            * If already dead, keep its state fixed, ignore its action,
              and mark done=True.
            * Else, apply dynamics.step(current_state_i, action_i).
              If `alive_fn` is provided and returns False on the new state,
              mark that agent as dead and done=True.
        - Update the agents' histories with (actions, next_state).
        - Optionally terminate all agents if `max_episode_steps` is reached.

        Returns
        -------
        next_state : (num_agents, state_dim) jnp.ndarray
        reward     : (num_agents,) jnp.ndarray
            Currently all zeros; ready to be extended.
        done       : (num_agents,) jnp.ndarray of bool
            True where the agent is dead or episode terminated.
        info       : dict
            Contains at least:
                - "actions": actions taken this step
                - "dynamics_debug": list of per-agent debug dicts
                - "step_index": current step index (after increment)
                - "alive_mask": current alive mask after this step
                - "max_episode_terminated": bool
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
                debug_list.append({"status": np.asarray("dead")})  # lightweight tag
                new_alive_list.append(False)
                done_list.append(True)
                continue

            # Alive agent: step dynamics.
            ns_i, dbg_i = self._dynamics.step(state_i, action_i)
            ns_i = jnp.asarray(ns_i)
            next_states_list.append(ns_i)
            debug_list.append(dbg_i)

            # Evaluate alive_fn on the *next* state if provided.
            still_alive = True
            # Check for collisions with cave map
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

        # 3) Update histories on the agent side (even for dead agents, their
        #    states/actions get tracked so that logs are consistent).
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
                # Keep alive mask as-is (they may still be "alive" physically,
                # but the episode is over).

        done = jnp.asarray(done_list, dtype=bool)

        # 6) Compute reward (currently zero for all agents).
        reward = jnp.zeros((num_agents,), dtype=jnp.float32)

        # 7) Compute a communications matrix. This is a line of sight matrix
        # for all pairs of agents (num_agents, num_agents), with 1 meaning
        # they can see each other, and 0 meaning they cannot (get this information)
        # from the cavemap
        comms_matrix = np.zeros((num_agents, num_agents), dtype=np.int8)
        # It will be symmetric, so only compute half the matrix
        # then mirror it
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                if not (self._alive[i] and self._alive[j]):
                    continue
                pos_i = self._state[i, self.position_dims]
                pos_j = self._state[j, self.position_dims]
                if self._cave_map.has_line_of_sight(pos_i, pos_j):
                    comms_matrix[i, j] = 1
                    comms_matrix[j, i] = 1
        # Mirror it
        comms_matrix += comms_matrix.T

        info: Dict[str, Any] = {
            "actions": actions,
            "dynamics_debug": debug_list,
            "step_index": self._step_index,
            "alive_mask": self._alive,
            "max_episode_terminated": max_episode_terminated,
            "communications_matrix": comms_matrix,
        }

        return self._state, reward, done, info

    # Convenience properties
    # ------------------------------------------------------------------ #
    @property
    def num_agents(self) -> int:
        return self._agents.num_agents

    @property
    def state(self) -> jnp.ndarray:
        """
        Current state, always shape (num_agents, state_dim) internally.
        """
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
        """
        Per-agent alive mask, updated each step.
        """
        return self._alive
