# src/prospector/dynamics/dynamics_3d_linear.py

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import jax.numpy as jnp

from .base_dynamics import BaseDynamics


class Dynamics3DLinear(BaseDynamics):
    """
    Super simple 3D point-mass dynamics for debugging.

    Coordinate system
    -----------------
    - x : forward
    - y : lateral (right)
    - z : up

    State (3D)
    ----------
    s = [x, y, z]

    Control (3D)
    ------------
    a = [u_x, u_y, u_z]

    Interpretation
    --------------
    - If a == 0, the point stays exactly where it is.
    - If a != 0, it moves in that direction with:

          x_{t+1} = x_t + dt * a

      So `a` can be interpreted as a velocity command in [m/s].
    - No gravity, no dynamics coupling — purely kinematic.
    """

    # `step` broadcasts over a leading batch dimension, so MPPI can roll out
    # all samples at once.
    supports_batched_step = True

    def __init__(
        self,
        *,
        dt: float = 0.02,
        physical_parameters: Dict[str, Any] | None = None,
    ):
        # physical_parameters is unused here but kept for interface consistency
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
        Kinematic update:

            next_state = state + dt * action

        Parameters
        ----------
        state : (3,) jnp.ndarray
            [x, y, z]
        action: (3,) jnp.ndarray
            [u_x, u_y, u_z]

        Returns
        -------
        next_state : (3,) jnp.ndarray
        debug      : Dict[str, jnp.ndarray]
        """
        # shape checks are up to you / calling code; we keep it minimal here
        delta = self.dt * action
        next_state = state + delta

        debug: Dict[str, jnp.ndarray] = {
            "delta": jnp.asarray(delta),
            "action": jnp.asarray(action),
        }
        return next_state, debug

    # ------------------------------------------------------------------ #
    # State metadata
    # ------------------------------------------------------------------ #
    def state_variable_descriptions(self) -> Sequence[str]:
        return [
            "x position (forward, m)",   # 0
            "y position (lateral, m)",   # 1
            "z position (up, m)",        # 2
        ]

    def state_variable_symbols(self) -> Sequence[str]:
        return ["x", "y", "z"]

    def state_indices_manifold(self) -> Sequence[int]:
        """
        We just use (x, y, z) directly as the manifold coordinates.
        """
        return [0, 1, 2]

    # ------------------------------------------------------------------ #
    # Control metadata
    # ------------------------------------------------------------------ #
    def control_variable_descriptions(self) -> Sequence[str]:
        return [
            "u_x position-rate command (m/s)",  # 0
            "u_y position-rate command (m/s)",  # 1
            "u_z position-rate command (m/s)",  # 2
        ]

    def control_variable_symbols(self) -> Sequence[str]:
        return ["uₓ", "u_y", "u_z"]
