# seamstress/src/prospector/dynamics/base_dynamics.py

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Sequence

import jax.numpy as jnp


class BaseDynamics(ABC):
    """
    Abstract base class for all dynamics models.

    - Stores a fixed time step `dt`.
    - Stores a dictionary of `physical_parameters` (mass, inertia, aero coeffs, etc.).
    - Exposes metadata for both state variables and control variables.
    """

    def __init__(
        self,
        *,
        dt: float = 0.1,
        physical_parameters: Dict[str, Any] | None = None,
    ):
        self.dt = float(dt)
        self.physical_parameters: Dict[str, Any] = physical_parameters or {}

    @abstractmethod
    def step(
        self, state: jnp.ndarray, action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Advance the system by one time step.

        Parameters
        ----------
        state : (N,) jnp.ndarray
            Current state.
        action : (M,) jnp.ndarray
            Control input at this time step.

        Returns
        -------
        next_state : (N,) jnp.ndarray
        debug_dict : Dict[str, jnp.ndarray]
            Arbitrary debugging info (forces, torques, etc.).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # State metadata
    # ------------------------------------------------------------------ #
    @abstractmethod
    def state_variable_descriptions(self) -> Sequence[str]:
        """
        Human-readable descriptions for each component of the state vector,
        in the same order as the state array.
        """
        raise NotImplementedError

    @abstractmethod
    def state_variable_symbols(self) -> Sequence[str]:
        """
        Single-character (unicode) symbols for each state variable,
        in the same order as the state array.
        """
        raise NotImplementedError

    @abstractmethod
    def state_indices_manifold(self) -> Sequence[int]:
        """
        Indices of the state to visualize as a manifold.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Control metadata
    # ------------------------------------------------------------------ #
    @abstractmethod
    def control_variable_descriptions(self) -> Sequence[str]:
        """
        Human-readable descriptions of each control input, in the same order
        as the action array passed to `step`.
        """
        raise NotImplementedError

    @abstractmethod
    def control_variable_symbols(self) -> Sequence[str]:
        """
        Single-character (or short unicode) symbols for each control input.
        """
        raise NotImplementedError
