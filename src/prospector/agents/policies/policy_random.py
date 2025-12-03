# src/prospector/agents/policies/policy_random.py

from __future__ import annotations

from typing import Sequence

import numpy as np
import jax.numpy as jnp

from .base_policy import BasePolicy


class RandomPolicy(BasePolicy):
    """
    Stateless uniform-random policy within given action limits.

    Action limits
    -------------
    action_limits : (action_dim, 2) array-like
        For each dimension i:
            [low_i, high_i]

        All bounds must be finite. This is designed to be fed directly
        by your config, e.g.:

            cfg.dynamics.dynamics_2d_linear.action_limits
    """

    def __init__(
        self,
        action_dim: int,
        action_limits: Sequence[Sequence[float]],
        *,
        seed: int | None = None,
    ) -> None:
        self.action_dim = int(action_dim)

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
                "RandomPolicy requires finite action limits for all dimensions."
            )

        if np.any(high <= low):
            raise ValueError(
                "Each action limit must satisfy high > low. "
                f"Got lows={low}, highs={high}."
            )

        self._low = low
        self._high = high

        # Simple NumPy RNG; this policy is not JIT-targeted.
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # BasePolicy interface                                               #
    # ------------------------------------------------------------------ #
    def act(
        self,
        history_states: jnp.ndarray,
        history_actions: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Ignore history; just sample uniformly within the action limits.

        history_states/history_actions are accepted to satisfy BasePolicy
        but are not used.
        """
        # Sample in NumPy and convert to jax.numpy
        a_np = self._rng.uniform(self._low, self._high)
        return jnp.asarray(a_np, dtype=jnp.float32)

    def reset(self) -> None:
        """
        No internal state to clear; provided for completeness.
        """
        # Nothing to do.
        pass
