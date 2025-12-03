# src/prospector/environment/environment.py

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import jax.numpy as jnp

from ..dynamics.base_dynamics import BaseDynamics
from ..agents.agent_multi import MultiAgent


class ProspectorEnvironment:
    """
    Environment wrapper around a dynamics model and a (multi-)agent controller.

    Responsibilities
    ----------------
    - Hold the current state (for one or many agents).
    - Ask the agent(s) for actions each step.
    - Advance the state using the provided `BaseDynamics` instance.
    - Track a simple step counter.
    - Return reward (currently zeros) and a debug/info dict.

    Design choices
    --------------
    - We *do not* assume the dynamics are batchable; instead, we treat
      `num_agents` as independent copies of the same dynamics and step
      them in a Python loop.
    - For num_agents == 1, we still use the same interface, with the
      state stored as shape (1, state_dim) internally.
    """

    def __init__(
        self,
        dynamics: BaseDynamics,
        agents: MultiAgent,
        initial_state: jnp.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        dynamics      : BaseDynamics
            Underlying system dynamics.
        agents        : MultiAgent
            Controller(s) that produce actions from histories.
        initial_state : (num_agents, state_dim) or (state_dim,)
            Initial state for each agent's copy of the system.
        """
        self._dynamics = dynamics
        self._agents = agents
        self._state = self._normalize_state_shape(initial_state)
        self._step_index: int = 0

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
        Reset environment state and agent histories.

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
        return self._state

    def step(self) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """
        Advance the environment by one time step.

        Procedure
        ---------
        - Query the agent(s) for actions using their internal histories.
        - Apply dynamics.step(current_state_i, action_i) for each agent i.
        - Update the agents' histories with (actions, next_state).
        - Return (next_state, reward, info).

        Returns
        -------
        next_state : (num_agents, state_dim) jnp.ndarray
        reward     : (num_agents,) jnp.ndarray
            Currently all zeros; ready to be extended.
        info       : dict
            Contains at least:
                - "actions": actions taken this step
                - "dynamics_debug": list of per-agent debug dicts
                - "step_index": current step index (after increment)
        """
        num_agents = self._agents.num_agents

        # 1) Ask agents for actions.
        actions = self._agents.act()  # (num_agents, action_dim)

        # 2) Apply dynamics per agent (no batch assumption).
        next_states_list: List[jnp.ndarray] = []
        debug_list: List[Dict[str, jnp.ndarray]] = []

        for i in range(num_agents):
            state_i = self._state[i]
            action_i = actions[i]
            ns_i, dbg_i = self._dynamics.step(state_i, action_i)
            next_states_list.append(jnp.asarray(ns_i))
            debug_list.append(dbg_i)

        next_state = jnp.stack(next_states_list, axis=0)
        
        # TODO are the agents done?

        # 3) Update histories on the agent side.
        self._agents.update_history(next_state, actions)

        # 4) Update environment state and step index.
        self._state = next_state
        self._step_index += 1

        # TODO has the environment hit the max episode length?

        # 5) Compute reward (currently zero for all agents).
        reward = jnp.zeros((num_agents,), dtype=jnp.float32)

        info: Dict[str, Any] = {
            "actions": actions,
            "dynamics_debug": debug_list,
            "step_index": self._step_index,
        }

        return self._state, reward, info

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
