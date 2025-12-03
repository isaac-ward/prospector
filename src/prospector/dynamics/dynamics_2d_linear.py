# src/prospector/dynamics/dynamics_2d_linear.py

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import jax.numpy as jnp

from .base_dynamics import BaseDynamics


class Dynamics2DLinear(BaseDynamics):
    """
    Super simple 2D point-mass dynamics for debugging.

    Coordinate system
    -----------------
    - x : forward
    - y : lateral (right)

    State (2D)
    ----------
    s = [x, y]

    Control (2D)
    ------------
    a = [u_x, u_y]

    Interpretation
    --------------
    - If action == 0, the point stays exactly where it is.
    - If action != 0, it moves in that direction via:

          state_{t+1} = state_t + dt * action

      So the action is interpreted as a commanded velocity (m/s).
    - Purely kinematic. No angles, no gravity, no coupling.
    """

    def __init__(
        self,
        *,
        dt: float = 0.02,
        physical_parameters: Dict[str, Any] | None = None,
    ):
        # physical_parameters unused, kept for consistency
        super().__init__(dt=dt, physical_parameters=physical_parameters or {})

    # ------------------------------------------------------------------ #
    # Core dynamics
    # ------------------------------------------------------------------ #
    def step(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Kinematic update in 2D:

            next_state = state + dt * action

        Parameters
        ----------
        state : (2,) jnp.ndarray
            [x, y]

        action : (2,) jnp.ndarray
            [u_x, u_y]

        Returns
        -------
        next_state : (2,) jnp.ndarray
        debug      : Dict[str, jnp.ndarray]
        """
        delta = self.dt * action
        next_state = state + delta

        debug = {
            "delta": delta,
            "action": action,
        }

        return next_state, debug

    # ------------------------------------------------------------------ #
    # State metadata
    # ------------------------------------------------------------------ #
    def state_variable_descriptions(self) -> Sequence[str]:
        return [
            "x position (forward, m)",   # 0
            "y position (lateral, m)",   # 1
        ]

    def state_variable_symbols(self) -> Sequence[str]:
        return ["x", "y"]

    def state_indices_manifold(self) -> Sequence[int]:
        """
        For visualization, the full (x, y) space is the manifold.
        """
        return [0, 1]

    # ------------------------------------------------------------------ #
    # Control metadata
    # ------------------------------------------------------------------ #
    def control_variable_descriptions(self) -> Sequence[str]:
        return [
            "u_x velocity command (m/s)",   # 0
            "u_y velocity command (m/s)",   # 1
        ]

    def control_variable_symbols(self) -> Sequence[str]:
        return ["uₓ", "u_y"]
