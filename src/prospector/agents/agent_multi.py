# src/prospector/agents/agent_multi.py

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import jax.numpy as jnp

from .policies.base_policy import BasePolicy


class MultiAgent:
    """
    Multi-agent wrapper that:

    - Tracks per-agent histories of states and actions.
    - Uses per-agent (or shared) policies to compute actions.
    - Enforces a fixed lookback window length (`history_len`) for what
      is passed into each policy.

    Timeline convention
    -------------------
    Let t be the current time index.

    After reset(initial_states):
        state_history[i]   = [s_0]         for agent i
        action_history[i]  = []            (no actions yet)

    At each step:
        1) Agents see:
            history_states  = [s_0, ..., s_t]
            history_actions = [a_0, ..., a_{t-1}]
        2) Policies produce a_t.
        3) Environment applies dynamics to get s_{t+1}.
        4) We call update_history(next_states, actions) so that:
            state_history[i]   = [s_0, ..., s_t, s_{t+1}]
            action_history[i]  = [a_0, ..., a_t]
    """

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        history_len: int,
        policies: BasePolicy | Sequence[BasePolicy],
    ) -> None:
        if num_agents <= 0:
            raise ValueError(f"num_agents must be positive, got {num_agents}")
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if history_len <= 0:
            raise ValueError(f"history_len must be positive, got {history_len}")

        self.num_agents = int(num_agents)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.history_len = int(history_len)

        # Either one shared policy or a list of per-agent policies.
        self._policies = policies

        # Per-agent histories: Python lists of jnp.ndarray.
        self._state_history: List[List[jnp.ndarray]] = [
            [] for _ in range(self.num_agents)
        ]
        self._action_history: List[List[jnp.ndarray]] = [
            [] for _ in range(self.num_agents)
        ]
        self._initialized: bool = False

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _get_policy_for_agent(self, agent_idx: int) -> BasePolicy:
        """
        Return the policy object to use for the given agent index.

        Supports:
        - A single policy object for all agents.
        - A sequence of policies of length == num_agents.
        - A sequence of length 1 (broadcasted to all agents).
        """
        policies = self._policies

        # Treat non-string Sequences specially.
        if isinstance(policies, Sequence) and not isinstance(policies, (str, bytes)):
            if len(policies) == 0:
                raise ValueError("policies sequence must not be empty.")

            if len(policies) == 1:
                return policies[0]

            if len(policies) != self.num_agents:
                raise ValueError(
                    "policies sequence must have length 1 or num_agents; "
                    f"got len(policies)={len(policies)}, num_agents={self.num_agents}"
                )

            return policies[agent_idx]

        # Single shared policy
        policy = policies  # type: ignore[assignment]
        if not isinstance(policy, BasePolicy):
            raise TypeError(
                f"Expected BasePolicy or sequence of BasePolicy, got {type(policies)}"
            )
        return policy

    def _windowed_history_for_agent(
        self,
        agent_idx: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get the truncated history window for a given agent.

        Returns
        -------
        history_states  : (T_s, state_dim)
        history_actions : (T_a, action_dim)
        """
        if not (0 <= agent_idx < self.num_agents):
            raise IndexError(
                f"agent_idx {agent_idx} out of range [0, {self.num_agents})"
            )

        s_hist = self._state_history[agent_idx]
        a_hist = self._action_history[agent_idx]

        if not s_hist:
            raise RuntimeError(
                "Agent histories are not initialized. "
                "Call MultiAgent.reset(initial_states) first."
            )

        # Truncate to last `history_len` entries.
        s_window = s_hist[-self.history_len :]
        a_window = a_hist[-self.history_len :]

        history_states = jnp.stack(s_window, axis=0)  # (T_s, state_dim)

        if a_window:
            history_actions = jnp.stack(a_window, axis=0)  # (T_a, action_dim)
        else:
            history_actions = jnp.zeros(
                (0, self.action_dim), dtype=history_states.dtype
            )

        return history_states, history_actions

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def reset(self, initial_states: jnp.ndarray) -> None:
        """
        Initialize per-agent histories from initial states.

        Parameters
        ----------
        initial_states : (num_agents, state_dim) or (state_dim,)
        """
        states = jnp.asarray(initial_states)

        if states.ndim == 1:
            if self.num_agents != 1:
                raise ValueError(
                    f"Single state provided but num_agents={self.num_agents}"
                )
            states = states[None, :]  # (1, state_dim)

        expected_shape = (self.num_agents, self.state_dim)
        if states.shape != expected_shape:
            raise ValueError(
                "initial_states shape mismatch: "
                f"expected {expected_shape}, got {states.shape}"
            )

        # Reset histories
        self._state_history = [[] for _ in range(self.num_agents)]
        self._action_history = [[] for _ in range(self.num_agents)]

        for i in range(self.num_agents):
            self._state_history[i].append(states[i])
            self._action_history[i] = []

        # Propagate reset to policies if they care.
        for i in range(self.num_agents):
            policy = self._get_policy_for_agent(i)
            policy.reset()

        self._initialized = True

    def act(self) -> jnp.ndarray:
        """
        Compute an action for each agent using its policy and history.

        Returns
        -------
        actions : (num_agents, action_dim) jnp.ndarray
        """
        if not self._initialized:
            raise RuntimeError(
                "MultiAgent not initialized. Call reset(initial_states) first."
            )

        actions_list: List[jnp.ndarray] = []

        for i in range(self.num_agents):
            policy = self._get_policy_for_agent(i)
            history_states, history_actions = self._windowed_history_for_agent(i)

            action = policy.act(history_states, history_actions)
            action = jnp.asarray(action)

            if action.ndim != 1 or action.shape[0] != self.action_dim:
                raise ValueError(
                    f"Policy for agent {i} returned action of shape {action.shape}, "
                    f"expected ({self.action_dim},)"
                )

            actions_list.append(action)

        actions = jnp.stack(actions_list, axis=0)  # (num_agents, action_dim)
        return actions

    def update_history(
        self,
        next_states: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> None:
        """
        Append the latest actions and resulting next states to histories.

        Parameters
        ----------
        next_states : (num_agents, state_dim) or (state_dim,)
        actions     : (num_agents, action_dim) or (action_dim,)
        """
        ns = jnp.asarray(next_states)
        act = jnp.asarray(actions)

        if ns.ndim == 1:
            if self.num_agents != 1:
                raise ValueError(
                    f"Single next_state provided but num_agents={self.num_agents}"
                )
            ns = ns[None, :]
        if act.ndim == 1:
            if self.num_agents != 1:
                raise ValueError(
                    f"Single action provided but num_agents={self.num_agents}"
                )
            act = act[None, :]

        expected_state_shape = (self.num_agents, self.state_dim)
        expected_action_shape = (self.num_agents, self.action_dim)

        if ns.shape != expected_state_shape:
            raise ValueError(
                "next_states shape mismatch: "
                f"expected {expected_state_shape}, got {ns.shape}"
            )
        if act.shape != expected_action_shape:
            raise ValueError(
                "actions shape mismatch: "
                f"expected {expected_action_shape}, got {act.shape}"
            )

        for i in range(self.num_agents):
            self._action_history[i].append(act[i])
            self._state_history[i].append(ns[i])

    # Convenience accessors
    # ------------------------------------------------------------------ #
    def get_full_history_for_agent(
        self, agent_idx: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Return the full stored history (all time steps) for a given agent.

        Returns
        -------
        states  : (T_s, state_dim)
        actions : (T_a, action_dim)
        """
        if not (0 <= agent_idx < self.num_agents):
            raise IndexError(
                f"agent_idx {agent_idx} out of range [0, {self.num_agents})"
            )

        s_hist = self._state_history[agent_idx]
        a_hist = self._action_history[agent_idx]

        if not s_hist:
            raise RuntimeError("No state history stored for this agent yet.")

        states = jnp.stack(s_hist, axis=0)
        if a_hist:
            actions = jnp.stack(a_hist, axis=0)
        else:
            actions = jnp.zeros((0, self.action_dim), dtype=states.dtype)

        return states, actions
