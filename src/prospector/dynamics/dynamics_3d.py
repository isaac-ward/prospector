# src/prospector/dynamics/dynamics_3d.py

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import jax.numpy as jnp

from .base_dynamics import BaseDynamics


class DynamicsQuadcopter3D(BaseDynamics):
    """
    12D rigid-body quadrotor in 3D using Euler angles (Z-Y-X, ψ θ φ).

    Coordinate system
    -----------------
    - x : forward
    - y : lateral (right)
    - z : up   (gravity points in -z)

    State (12D)
    -----------
    s = [x,  y,  z,
         rz, ry, rx,
         vx, vy, vz,
         p,  q,  r]

    where:
        rz = ψ (yaw about world z, rad)
        ry = θ (pitch about world y, rad)
        rx = φ (roll about world x, rad)
        vx, vy, vz : translational velocities in world frame
        p, q, r    : body angular rates about x, y, z

    Control (4D)
    ------------
    a = [w1, w2, w3, w4]

    Rotor layout and sign conventions (matching your original code):
        w1 : left,    CW
        w4 : forward, CCW
        w3 : right,   CW
        w2 : rear,    CCW

    Physical parameters (via physical_parameters dict)
    --------------------------------------------------
    - diameter        : rotor-to-rotor distance [m]
    - mass            : vehicle mass [kg]
    - Ix, Iy, Iz      : moments of inertia [kg m^2]
    - g               : gravity magnitude [m/s^2], POSITIVE (e.g. 9.81)
                        (acceleration is then [0, 0, -g] in world, since z up)
    - thrust_coef     : thrust coefficient k
    - drag_yaw_coef   : yaw drag coefficient b
    - drag_force_coef : translational drag coefficient kd (currently unused)
    """

    def __init__(
        self,
        *,
        dt: float = 0.02,
        physical_parameters: Dict[str, Any] | None = None,
    ):
        super().__init__(dt=dt, physical_parameters=physical_parameters or {})

        params = physical_parameters or {}

        # Core physical properties
        self.diameter: float = float(params.get("diameter", 0.25))
        self.mass: float = float(params.get("mass", 1.0))
        self.Ix: float = float(params.get("Ix", 0.01))
        self.Iy: float = float(params.get("Iy", 0.01))
        self.Iz: float = float(params.get("Iz", 0.02))

        # z up ⇒ gravity is [0, 0, -g], with g > 0 (e.g. 9.81)
        self.g: float = float(params.get("g", 9.81))

        # Thrust & drag coefficients
        self.thrust_coef: float = float(params.get("thrust_coef", 1.0))        # k
        self.drag_yaw_coef: float = float(params.get("drag_yaw_coef", 1e-3))   # b
        self.drag_force_coef: float = float(params.get("drag_force_coef", 0.0))  # kd (unused)

        # Convenience: arm radius
        self._arm_radius: float = self.diameter / 2.0

    # ------------------------------------------------------------------ #
    # Core dynamics
    # ------------------------------------------------------------------ #
    def step(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        One-step explicit Euler integrator:

            x_{t+1} = x_t + dt * f(x_t, u_t)

        Parameters
        ----------
        state : (12,) jnp.ndarray
        action: (4,)  jnp.ndarray

        Returns
        -------
        next_state : (12,) jnp.ndarray
        debug      : Dict[str, jnp.ndarray]
            Includes thrust and body torques, etc.
        """
        state_dot, debug = self._state_derivative_with_debug(state, action)
        next_state = state + self.dt * state_dot
        return next_state, debug

    def _state_derivative_with_debug(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Continuous-time nonlinear dynamics:

            s_dot = f(s, a)

        This mirrors your original DynamicsQuadcopter3D.state_delta logic,
        but with z UP and gravity acting in -z.
        """
        k = self.thrust_coef
        b = self.drag_yaw_coef
        kd = self.drag_force_coef  # currently unused, kept for completeness

        # ------------------------------------------------------------------ #
        # Unpack state & action
        # ------------------------------------------------------------------ #
        # Positions (world frame, z up)
        x = state[0]
        y = state[1]
        z = state[2]

        # Euler angles in Z-Y-X order
        rz = state[3]  # ψ (yaw)
        ry = state[4]  # θ (pitch)
        rx = state[5]  # φ (roll)

        # Translational velocities (world frame)
        vx = state[6]
        vy = state[7]
        vz = state[8]

        # Body rates
        p = state[9]
        q = state[10]
        r_body = state[11]

        # Rotor speeds / commands
        w1 = action[0]
        w2 = action[1]
        w3 = action[2]
        w4 = action[3]

        # Map to ψ, θ, φ for clarity
        ψ = rz
        θ = ry
        φ = rx

        # ------------------------------------------------------------------ #
        # Trig precomputations
        # ------------------------------------------------------------------ #
        s_ψ, c_ψ = jnp.sin(ψ), jnp.cos(ψ)
        s_θ, c_θ = jnp.sin(θ), jnp.cos(θ)
        t_θ = jnp.tan(θ)
        s_φ, c_φ = jnp.sin(φ), jnp.cos(φ)

        # ------------------------------------------------------------------ #
        # Control → total thrust + body torques
        # ------------------------------------------------------------------ #
        w1_sq = w1 ** 2
        w2_sq = w2 ** 2
        w3_sq = w3 ** 2
        w4_sq = w4 ** 2

        arm_r = self._arm_radius

        ft = k * (w1_sq + w2_sq + w3_sq + w4_sq)          # total thrust (body +z)
        tx = k * arm_r * (w3_sq - w1_sq)                  # roll torque
        ty = k * arm_r * (w4_sq - w2_sq)                  # pitch torque
        tz = b * ((w2_sq + w4_sq) - (w1_sq + w3_sq))      # yaw torque

        # ------------------------------------------------------------------ #
        # State derivative
        # ------------------------------------------------------------------ #
        state_dot = jnp.zeros_like(state)

        # Positions: derivative = velocities
        state_dot = state_dot.at[0].set(vx)
        state_dot = state_dot.at[1].set(vy)
        state_dot = state_dot.at[2].set(vz)

        # Euler angle rates from body rates (standard Z-Y-X mapping)
        # Same as your original (it doesn't depend on NED vs z-up)
        state_dot = state_dot.at[3].set(q * s_φ / c_θ + r_body * c_φ / c_θ)       # dψ/dt
        state_dot = state_dot.at[4].set(q * c_φ - r_body * s_φ)                   # dθ/dt
        state_dot = state_dot.at[5].set(p + q * s_φ * t_θ + r_body * c_φ * t_θ)   # dφ/dt

        # Translational accelerations in world frame (z up)
        #
        # a = (1/m) R_b2w * [0, 0, ft]^T + [0, 0, -g]^T
        # with R_b2w = Rz(ψ) Ry(θ) Rx(φ).
        #
        # Third column of R_b2w is:
        #   [cψ sθ cφ + sψ sφ,
        #    sψ sθ cφ - cψ sφ,
        #    cθ cφ]
        #
        # so:
        #   ax = (ft/m) * (cψ sθ cφ + sψ sφ)
        #   ay = (ft/m) * (sψ sθ cφ - cψ sφ)
        #   az = (ft/m) * (cθ cφ) - g
        m = self.mass
        g = self.g

        ax = (ft / m) * (c_ψ * s_θ * c_φ + s_ψ * s_φ)
        ay = (ft / m) * (s_ψ * s_θ * c_φ - c_ψ * s_φ)
        az = (ft / m) * (c_θ * c_φ) - g

        # If you want simple velocity drag, uncomment:
        # ax = ax - kd * vx
        # ay = ay - kd * vy
        # az = az - kd * vz

        state_dot = state_dot.at[6].set(ax)
        state_dot = state_dot.at[7].set(ay)
        state_dot = state_dot.at[8].set(az)

        # Rotational dynamics (body frame)
        Ix = self.Ix
        Iy = self.Iy
        Iz = self.Iz

        pdot = ((Iy - Iz) * q * r_body + tx) / Ix
        qdot = ((Iz - Ix) * p * r_body + ty) / Iy
        rdot = ((Ix - Iy) * p * q + tz) / Iz

        state_dot = state_dot.at[9].set(pdot)
        state_dot = state_dot.at[10].set(qdot)
        state_dot = state_dot.at[11].set(rdot)

        debug: Dict[str, jnp.ndarray] = {
            "ft": jnp.asarray(ft),
            "tx": jnp.asarray(tx),
            "ty": jnp.asarray(ty),
            "tz": jnp.asarray(tz),
            "p": jnp.asarray(p),
            "q": jnp.asarray(q),
            "r": jnp.asarray(r_body),
        }

        return state_dot, debug

    # ------------------------------------------------------------------ #
    # State metadata
    # ------------------------------------------------------------------ #
    def state_variable_descriptions(self) -> Sequence[str]:
        return [
            "x position (forward, m)",          # 0
            "y position (lateral, m)",          # 1
            "z position (up, m)",               # 2
            "yaw ψ about z (rad)",              # 3 (rz)
            "pitch θ about y (rad)",            # 4 (ry)
            "roll φ about x (rad)",             # 5 (rx)
            "x velocity vx in world (m/s)",     # 6
            "y velocity vy in world (m/s)",     # 7
            "z velocity vz in world (m/s)",     # 8
            "roll rate p (rad/s)",              # 9
            "pitch rate q (rad/s)",             # 10
            "yaw rate r (rad/s)",               # 11
        ]

    def state_variable_symbols(self) -> Sequence[str]:
        return [
            "x",
            "y",
            "z",
            "ψ",
            "θ",
            "φ",
            "vₓ",
            "v_y",
            "v_z",
            "p",
            "q",
            "r",
        ]

    def state_indices_manifold(self) -> Sequence[int]:
        """
        Indices to use when visualising the state as a manifold.

        Here we choose the position coordinates (x, y, z).
        """
        return [0, 1, 2]

    # ------------------------------------------------------------------ #
    # Control metadata
    # ------------------------------------------------------------------ #
    def control_variable_descriptions(self) -> Sequence[str]:
        return [
            "w₁ rotor speed (left, CW)",      # 0
            "w₂ rotor speed (rear, CCW)",     # 1
            "w₃ rotor speed (right, CW)",     # 2
            "w₄ rotor speed (front, CCW)",    # 3
        ]

    def control_variable_symbols(self) -> Sequence[str]:
        return ["ω₁", "ω₂", "ω₃", "ω₄"]
