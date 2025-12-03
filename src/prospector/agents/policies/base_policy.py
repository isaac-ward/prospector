# src/prospector/agents/policies/base_policy.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp


class BasePolicy(ABC):
    """
    Abstract base class for all control policies.

    The environment/agents layer will pass a fixed-length history of
    states and actions. The policy returns a single action vector for
    the current step.

    Shapes
    ------
    history_states : (T_s, state_dim)
        Sequence of past and current states. Typically T_s == history_len,
        except at the very beginning of an episode when it's shorter.
    history_actions : (T_a, action_dim)
        Sequence of past actions. Typically T_a == history_len or
        history_len - 1, but it may be 0 at the first step.
    """

    @abstractmethod
    def act(
        self,
        history_states: jnp.ndarray,
        history_actions: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute an action from the given history.

        Returns
        -------
        action : (action_dim,) jnp.ndarray
        """
        raise NotImplementedError
