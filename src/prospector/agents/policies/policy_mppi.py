# src/prospector/agents/policies/policy_mppi.py

from __future__ import annotations

from typing import Callable, Optional, Sequence, List

import numpy as np
import jax.numpy as jnp

from .base_policy import BasePolicy
from ...dynamics.base_dynamics import BaseDynamics


RewardFn = Callable[[jnp.ndarray, jnp.ndarray], float | jnp.ndarray]


class PolicyMPPI(BasePolicy):
    """
    Simplified MPPI-like controller with *action sequences*.

    At each call to `act`:

        - Let s_t be the last state in `history_states`.
        - Sample `num_samples` candidate *action sequences* of length H:
            - Each sequence is shape (H, action_dim).
            - Sampling can be:
                - Gaussian around the previous action (per step)
                - Uniform over the full action range (per step)
        - For each candidate sequence a^(k) = [a_0^k, ..., a_{H-1}^k]:
            - Simulate forward for H steps using the provided dynamics
              object (BaseDynamics), starting from s_t.
            - Build extended histories (states + actions).
            - Evaluate scalar reward via `reward_fn(history_states_ext,
              history_actions_ext)`.
        - Compute MPPI-style weights:

                w_k ∝ exp((reward_k - max_reward) / lambda_)

          and form the final *first* action as:

                a_0 = Σ_k w_k * a_0^k

        - Clip `a_0` to the action limits and store as `last_action`.

    This is a simplified MPPI / random-shooting controller over sequences,
    but it respects your requirements:

      - Sampling around the previous action (Gaussian)
      - Or uniform sampling over all allowed values
      - Reward depends on full (past + simulated future) histories
      - Uses a dynamics object implementing BaseDynamics
    """

    def __init__(
        self,
        action_dim: int,
        action_limits: Sequence[Sequence[float]],
        *,
        num_samples: int = 128,
        horizon: int = 10,
        lambda_: float = 1.0,
        sampling_mode: str = "gaussian",  # "gaussian" or "uniform"
        noise_std: float | Sequence[float] = 0.05,
        seed: int | None = None,
    ) -> None:
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if lambda_ <= 0.0:
            raise ValueError(f"lambda_ must be positive, got {lambda_}")

        sampling_mode = sampling_mode.lower()
        if sampling_mode not in ("gaussian", "uniform"):
            raise ValueError(
                f"sampling_mode must be 'gaussian' or 'uniform', got {sampling_mode}"
            )

        self.action_dim = int(action_dim)
        self.num_samples = int(num_samples)
        self.horizon = int(horizon)
        self.lambda_ = float(lambda_)
        self.sampling_mode = sampling_mode

        # Parse action limits: shape (action_dim, 2)
        limits = np.asarray(action_limits, dtype=float)
        if limits.shape != (self.action_dim, 2):
            raise ValueError(
                f"action_limits must have shape ({self.action_dim}, 2), "
                f"got {limits.shape}"
            )

        low = limits[:, 0]
        high = limits[:, 1]

        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            raise ValueError(
                "MPPIPolicy requires finite action limits for all dimensions."
            )
        if np.any(high <= low):
            raise ValueError(
                "Each action limit must satisfy high > low. "
                f"Got lows={low}, highs={high}."
            )

        self._low = low
        self._high = high

        # Noise std per dimension
        if isinstance(noise_std, (float, int)):
            noise_std_arr = np.full((self.action_dim,), float(noise_std), dtype=float)
        else:
            noise_std_arr = np.asarray(noise_std, dtype=float)
            if noise_std_arr.shape != (self.action_dim,):
                raise ValueError(
                    f"noise_std must be scalar or shape ({self.action_dim},), "
                    f"got {noise_std_arr.shape}"
                )

        if np.any(noise_std_arr < 0.0):
            raise ValueError("noise_std entries must be non-negative.")

        self._noise_std = noise_std_arr

        # User-specified reward and dynamics
        self._reward_fn: Optional[RewardFn] = None
        self._dynamics: Optional[BaseDynamics] = None

        # RNG and last_action
        self._rng = np.random.default_rng(seed)

        # Initialize last_action to mid-range
        self._last_action = jnp.asarray(
            0.5 * (self._low + self._high), dtype=jnp.float32
        )

    # ------------------------------------------------------------------ #
    # Configuration setters                                              #
    # ------------------------------------------------------------------ #
    def set_reward_function(self, reward_function: RewardFn) -> None:
        """
        Set the scalar reward function used to evaluate candidate rollouts.

        The callable must have signature:

            reward_function(history_states, history_actions) -> float

        where:
            history_states  : (T_s, state_dim) jnp.ndarray
            history_actions : (T_a, action_dim) jnp.ndarray

        The policy will call this on extended histories that include
        both the real past and the simulated future under a sampled
        candidate action *sequence*.
        """
        self._reward_fn = reward_function

    def set_dynamics(self, dynamics: BaseDynamics) -> None:
        """
        Set the dynamics object used for rollouts inside MPPI.

        The object must implement BaseDynamics (in particular, a `step`
        method with signature:

            next_state, debug = dynamics.step(state, action)

        where:
            state      : (state_dim,) jnp.ndarray
            action     : (action_dim,) jnp.ndarray
            next_state : (state_dim,) jnp.ndarray
        """
        self._dynamics = dynamics

    # ------------------------------------------------------------------ #
    # BasePolicy interface                                               #
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """
        Reset internal memory (only last_action for now).

        The caller may invoke this at episode boundaries.
        """
        self._last_action = jnp.asarray(
            0.5 * (self._low + self._high), dtype=jnp.float32
        )

    def act(
        self,
        history_states: jnp.ndarray,
        history_actions: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute an action using MPPI over *action sequences*.

        Requires that both reward_fn and dynamics have been set via
        `set_reward_function(...)` and `set_dynamics(...)`.
        """
        if self._reward_fn is None:
            raise RuntimeError(
                "MPPIPolicy requires a reward function. "
                "Call set_reward_function(...) before using act()."
            )
        if self._dynamics is None:
            raise RuntimeError(
                "MPPIPolicy requires a dynamics object. "
                "Call set_dynamics(...) before using act()."
            )

        reward_fn = self._reward_fn
        dynamics = self._dynamics

        # Current state = last state in history_states
        if history_states.ndim != 2:
            raise ValueError(
                f"history_states must have shape (T_s, state_dim), "
                f"got {history_states.shape}"
            )
        current_state = history_states[-1]  # (state_dim,)

        # Previous action (for Gaussian center)
        if history_actions.ndim == 2 and history_actions.shape[0] > 0:
            prev_action = history_actions[-1]
        else:
            prev_action = self._last_action

        # Sample candidate action sequences: (num_samples, horizon, action_dim)
        action_sequences = self._sample_candidate_action_sequences(prev_action)

        rewards: List[float] = []

        # Evaluate each candidate sequence by simulating `horizon` steps.
        for k in range(self.num_samples):
            seq_k = action_sequences[k]  # (horizon, action_dim)

            # Start from *copies* of the real histories
            states_list: List[jnp.ndarray] = [
                history_states[i] for i in range(history_states.shape[0])
            ]
            actions_list: List[jnp.ndarray] = [
                history_actions[i] for i in range(history_actions.shape[0])
            ]

            s = current_state
            for t in range(self.horizon):
                a_t = seq_k[t]
                actions_list.append(a_t)
                s, _ = dynamics.step(s, a_t)
                states_list.append(s)

            # Stack extended histories
            states_ext = jnp.stack(states_list, axis=0)
            if actions_list:
                actions_ext = jnp.stack(actions_list, axis=0)
            else:
                actions_ext = jnp.zeros(
                    (0, self.action_dim), dtype=states_ext.dtype
                )

            r_k = reward_fn(states_ext, actions_ext)
            r_k = float(jnp.asarray(r_k))  # ensure scalar float
            rewards.append(r_k)

        # Convert to jnp arrays for weighting
        rewards_jnp = jnp.asarray(rewards, dtype=jnp.float32)  # (num_samples,)
        # First action of each sequence: (num_samples, action_dim)
        first_actions = action_sequences[:, 0, :]

        # MPPI-style weights: w_k ∝ exp((r_k - max_r) / lambda)
        max_r = jnp.max(rewards_jnp)
        scaled = (rewards_jnp - max_r) / self.lambda_
        weights = jnp.exp(scaled)

        weight_sum = jnp.sum(weights)
        weights = jnp.where(
            weight_sum > 0.0,
            weights / weight_sum,
            jnp.full_like(weights, 1.0 / self.num_samples),
        )

        # Weighted average of *first* actions in each sequence
        action = jnp.sum(weights[:, None] * first_actions, axis=0)

        # Clip to action limits and update last_action
        action = jnp.clip(action, jnp.asarray(self._low), jnp.asarray(self._high))
        self._last_action = action

        return action

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _sample_candidate_action_sequences(
        self,
        prev_action: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Sample `num_samples` candidate action sequences.

        Returns
        -------
        sequences : (num_samples, horizon, action_dim) jnp.ndarray
        """
        if self.sampling_mode == "uniform":
            # Uniform over full action range at each time step.
            seq_np = self._rng.uniform(
                low=self._low,
                high=self._high,
                size=(self.num_samples, self.horizon, self.action_dim),
            )
            return jnp.asarray(seq_np, dtype=jnp.float32)

        # Gaussian around prev_action at each step, then clipped.
        prev_np = np.asarray(prev_action, dtype=float)

        # Broadcast prev_action over (num_samples, horizon, action_dim)
        base = np.broadcast_to(
            prev_np, (self.num_samples, self.horizon, self.action_dim)
        )

        # Noise: N(0, noise_std^2) per dimension, independent over time/samples.
        # noise_std shape (action_dim,) ⇒ broadcasts along the first two dims.
        noise = self._rng.normal(
            loc=0.0,
            scale=self._noise_std,
            size=(self.num_samples, self.horizon, self.action_dim),
        )

        seq_np = base + noise
        seq_np = np.clip(seq_np, self._low, self._high)

        return jnp.asarray(seq_np, dtype=jnp.float32)
