# src/prospector/agents/policies/policy_mppi.py

from __future__ import annotations

from typing import Callable, Optional, Sequence, List

import numpy as np
import jax.numpy as jnp

from .base_policy import BasePolicy
from ...dynamics.base_dynamics import BaseDynamics


RewardFn = Callable[[jnp.ndarray, jnp.ndarray], float | jnp.ndarray]
# Batched: (N, T_s, state_dim), (N, T_a, action_dim) -> (N,)
BatchRewardFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


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
        # ---- Optional extensions (defaults reproduce the original behaviour)
        warm_start: bool = False,
        nominal_action: Sequence[float] | None = None,
        noise_smoothing: float = 0.0,
        include_nominal_sample: bool = True,
        use_batched_rollouts: bool = False,
        reward_normalization: str = "none",
        safe_fallback_reward_drop: float | None = None,
    ) -> None:
        """
        Optional extensions
        -------------------
        warm_start : bool
            If True, use standard MPPI (Williams et al. 2017): keep a nominal
            action *sequence* U (H, action_dim), sample U + noise, update the
            whole sequence as the reward-weighted average of the samples,
            apply U[0] and shift U by one step for the next call. In this
            mode `sampling_mode="gaussian"` perturbs U (not prev_action);
            "uniform" still samples uniformly but the update is still the
            weighted average of full sequences.
        nominal_action : sequence, optional
            Action used to initialise / reset `last_action` and U (e.g. the
            hover action). Default: mid-range of `action_limits` (original
            behaviour).
        noise_smoothing : float in [0, 1)
            AR(1) temporal correlation of the Gaussian noise along the
            horizon: eps_t = beta * eps_{t-1} + sqrt(1 - beta^2) * n_t. The
            marginal std is unchanged. 0 = independent (original).
        include_nominal_sample : bool
            With warm_start, replace sample 0 with the unperturbed nominal
            sequence (so the update can never be worse than "keep the plan").
        use_batched_rollouts : bool
            If True, roll out all samples at once with
            `dynamics.rollout_batch(state, sequences)` when available
            (e.g. Dynamics3D), else with a jit+vmap of `dynamics.step`.
            Rewards use the batch reward function if one was set with
            `set_batch_reward_function`, otherwise the per-sample reward
            function is called on each (precomputed) rollout.
        reward_normalization : "none" | "range" | "std" | "median"
            If not "none", rewards are divided by their spread across the
            samples before the exp(. / lambda_) weighting, making lambda_
            scale-free ("range": max - min, "std": standard deviation,
            "median": max - median). "range" and "std" are dominated by
            huge penalties (e.g. a 1e6 collision penalty on a few samples
            flattens the weights of all the others); "median" is robust to
            that as long as fewer than half of the samples are penalised.
        safe_fallback_reward_drop : float, optional
            The executed plan is a weighted average of samples, and an average
            of collision-free samples need not be collision-free. If set, the
            averaged plan (warm start: the updated nominal sequence; else the
            best sample's sequence with the averaged first action) is rolled
            out and scored; if it scores more than this below the best sample
            (i.e. it collides, for a collision penalty >> this value), the
            best sample is executed instead. None = off (original behaviour).
        """
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

        # Optional extensions
        if not (0.0 <= float(noise_smoothing) < 1.0):
            raise ValueError(
                f"noise_smoothing must be in [0, 1), got {noise_smoothing}"
            )
        reward_normalization = str(reward_normalization).lower()
        if reward_normalization not in ("none", "range", "std", "median"):
            raise ValueError(
                "reward_normalization must be 'none', 'range', 'std' or 'median', "
                f"got {reward_normalization}"
            )
        self.warm_start = bool(warm_start)
        self.noise_smoothing = float(noise_smoothing)
        self.include_nominal_sample = bool(include_nominal_sample)
        self.use_batched_rollouts = bool(use_batched_rollouts)
        self.reward_normalization = reward_normalization
        if nominal_action is None:
            self._nominal_action = 0.5 * (self._low + self._high)
        else:
            self._nominal_action = np.asarray(nominal_action, dtype=float)
            if self._nominal_action.shape != (self.action_dim,):
                raise ValueError(
                    f"nominal_action must have shape ({self.action_dim},), "
                    f"got {self._nominal_action.shape}"
                )
        self._batch_reward_fn: Optional[BatchRewardFn] = None
        self._batch_reward_future_only_collisions = False
        self.safe_fallback_reward_drop = safe_fallback_reward_drop
        self.num_fallbacks = 0
        self._generic_rollout_fn = None
        self._nominal_sequence = np.tile(self._nominal_action, (self.horizon, 1))

        # Initialize last_action to mid-range (or nominal_action if given)
        self._last_action = jnp.asarray(self._nominal_action, dtype=jnp.float32)

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
        self._generic_rollout_fn = None

    def set_batch_reward_function(self, batch_reward_function: BatchRewardFn) -> None:
        """
        Optional vectorised reward, used when `use_batched_rollouts=True`.

            batch_reward_function(states_ext, actions_ext) -> (N,) rewards

        where states_ext is (N, T_s + H, state_dim) (real history followed by
        each sample's simulated future) and actions_ext is
        (N, T_a + H, action_dim). It must be equivalent to calling the scalar
        reward function on each sample.
        """
        self._batch_reward_fn = batch_reward_function

    def set_batched_reward_function(
        self,
        batched_reward_function: BatchRewardFn,
        *,
        future_only_collisions: bool = False,
    ) -> None:
        """
        Set the batch reward AND enable batched rollouts.

        If future_only_collisions, the batch reward is called with the extra
        keyword `num_history_states=T_s` so it can skip collision checks on
        the (unchangeable) past states, e.g.
        TaskWaypointFollowing.batch_reward_for_active_subtask.
        """
        self._batch_reward_fn = batched_reward_function
        self._batch_reward_future_only_collisions = bool(future_only_collisions)
        self.use_batched_rollouts = True

    @property
    def nominal_sequence(self) -> np.ndarray:
        """Current nominal (warm-start) action sequence, shape (H, action_dim)."""
        return np.array(self._nominal_sequence)

    # ------------------------------------------------------------------ #
    # BasePolicy interface                                               #
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """
        Reset internal memory (last_action, and the nominal sequence when
        warm-starting).

        The caller may invoke this at episode boundaries.
        """
        self._last_action = jnp.asarray(self._nominal_action, dtype=jnp.float32)
        self._nominal_sequence = np.tile(self._nominal_action, (self.horizon, 1))

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

        if self._uses_extensions():
            return self._act_extended(
                history_states, history_actions, current_state, prev_action
            )

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
        # If lambda is small, this approaches picking the max-reward action.
        # If lambda is large, this approaches uniform weighting.
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

    # ------------------------------------------------------------------ #
    # Optional extensions (only used when enabled in __init__)           #
    # ------------------------------------------------------------------ #
    def _uses_extensions(self) -> bool:
        return (
            self.warm_start
            or self.use_batched_rollouts
            or self.noise_smoothing > 0.0
            or self.reward_normalization != "none"
            or self.safe_fallback_reward_drop is not None
        )

    def _sample_noise(self) -> np.ndarray:
        shape = (self.num_samples, self.horizon, self.action_dim)
        n = self._rng.normal(size=shape)
        beta = self.noise_smoothing
        if beta > 0.0:
            c = np.sqrt(1.0 - beta * beta)
            for t in range(1, self.horizon):
                n[:, t] = beta * n[:, t - 1] + c * n[:, t]
        return n * self._noise_std

    def _sample_sequences_extended(self, prev_action: jnp.ndarray) -> np.ndarray:
        shape = (self.num_samples, self.horizon, self.action_dim)
        if self.sampling_mode == "uniform":
            seq = self._rng.uniform(low=self._low, high=self._high, size=shape)
        else:
            if self.warm_start:
                base = self._nominal_sequence[None]
            else:
                base = np.asarray(prev_action, dtype=float)[None, None, :]
            seq = base + self._sample_noise()
        seq = np.clip(seq, self._low, self._high)
        if self.warm_start and self.include_nominal_sample:
            seq[0] = np.clip(self._nominal_sequence, self._low, self._high)
        return seq

    def _rollout_all(self, current_state: jnp.ndarray, seqs: np.ndarray) -> np.ndarray:
        """(N, H, action_dim) -> (N, H, state_dim) predicted states."""
        dynamics = self._dynamics
        seqs_j = jnp.asarray(seqs, dtype=jnp.float32)
        if self.use_batched_rollouts:
            if hasattr(dynamics, "rollout_batch"):
                return np.asarray(dynamics.rollout_batch(current_state, seqs_j))
            if self._generic_rollout_fn is None:
                import jax

                def _single(s0, actions):
                    def body(s, a):
                        s_next, _ = dynamics.step(s, a)
                        return s_next, s_next

                    _, xs = jax.lax.scan(body, s0, actions)
                    return xs

                self._generic_rollout_fn = jax.jit(
                    jax.vmap(_single, in_axes=(None, 0))
                )
            s0 = jnp.asarray(current_state, dtype=jnp.float32)
            return np.asarray(self._generic_rollout_fn(s0, seqs_j))

        # Sequential (original) rollouts
        out = np.zeros(
            (self.num_samples, self.horizon, current_state.shape[-1]), dtype=np.float32
        )
        for k in range(self.num_samples):
            s = current_state
            for t in range(self.horizon):
                s, _ = dynamics.step(s, seqs_j[k, t])
                out[k, t] = np.asarray(s)
        return out

    def _act_extended(
        self,
        history_states: jnp.ndarray,
        history_actions: jnp.ndarray,
        current_state: jnp.ndarray,
        prev_action: jnp.ndarray,
    ) -> jnp.ndarray:
        seqs = self._sample_sequences_extended(prev_action)   # (N, H, A)
        hs = np.asarray(history_states, dtype=np.float32)
        ha = np.asarray(history_actions, dtype=np.float32).reshape(-1, self.action_dim)

        def score(seqs_: np.ndarray) -> np.ndarray:
            """Roll out (n, H, A) sequences from current_state and score them."""
            n = seqs_.shape[0]
            future_states = self._rollout_all(current_state, seqs_)  # (n, H, S)
            states_ext = np.concatenate(
                [np.broadcast_to(hs[None], (n,) + hs.shape), future_states], axis=1
            )
            actions_ext = np.concatenate(
                [np.broadcast_to(ha[None], (n,) + ha.shape), seqs_.astype(np.float32)], axis=1
            )
            if self._batch_reward_fn is not None:
                kwargs = (
                    {"num_history_states": hs.shape[0]}
                    if self._batch_reward_future_only_collisions
                    else {}
                )
                return np.asarray(
                    self._batch_reward_fn(states_ext, actions_ext, **kwargs), dtype=np.float64
                ).reshape(n)
            return np.array(
                [
                    float(
                        jnp.asarray(
                            self._reward_fn(
                                jnp.asarray(states_ext[k]), jnp.asarray(actions_ext[k])
                            )
                        )
                    )
                    for k in range(n)
                ],
                dtype=np.float64,
            )

        rewards = score(seqs)

        # Treat non-finite rewards (e.g. diverged rollouts) as worst.
        finite = np.isfinite(rewards)
        if not np.any(finite):
            rewards = np.zeros_like(rewards)
        else:
            rewards = np.where(finite, rewards, np.min(rewards[finite]) - 1e6)

        max_r = np.max(rewards)
        scaled = rewards - max_r
        if self.reward_normalization == "range":
            scaled = scaled / max(max_r - np.min(rewards), 1e-9)
        elif self.reward_normalization == "std":
            scaled = scaled / max(np.std(rewards), 1e-9)
        elif self.reward_normalization == "median":
            scaled = scaled / max(max_r - np.median(rewards), 1e-9)
        weights = np.exp(scaled / self.lambda_)
        weights = weights / np.sum(weights)
        self.last_weights = weights
        self.last_rewards = rewards

        if self.warm_start:
            new_seq = np.einsum("k,kha->ha", weights, seqs)
            new_seq = np.clip(new_seq, self._low, self._high)
        else:
            new_seq = np.array(seqs[int(np.argmax(rewards))])
            new_seq[0] = np.clip(
                np.einsum("k,ka->a", weights, seqs[:, 0, :]), self._low, self._high
            )

        if self.safe_fallback_reward_drop is not None:
            best = int(np.argmax(rewards))
            if float(score(new_seq[None])[0]) < max_r - float(self.safe_fallback_reward_drop):
                new_seq = np.array(seqs[best])
                self.num_fallbacks += 1

        action_np = new_seq[0]
        if self.warm_start:
            self._nominal_sequence = np.concatenate([new_seq[1:], new_seq[-1:]], axis=0)

        action = jnp.asarray(action_np, dtype=jnp.float32)
        self._last_action = action
        return action
