# src/prospector/dynamics/dynamics_3d.py

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .base_dynamics import BaseDynamics


# ---------------------------------------------------------------------- #
# Default physical parameters                                            #
# ---------------------------------------------------------------------- #
# A compact ~1 kg "+"-configuration indoor / cave quadrotor (250 mm
# motor-to-motor, 5" props, prop guards). Reasoning / sources:
#
# - Model structure (k*w^2 thrust, b*w^2 yaw drag, linear translational
#   drag) follows T. Luukkonen, "Modelling and control of quadcopter",
#   Aalto University, 2011: m = 0.468 kg, l = 0.225 m,
#   Ixx = Iyy = 4.856e-3, Izz = 8.801e-3 kg m^2, k = 2.98e-6,
#   b = 1.14e-7, A = 0.25 kg/s.
# - Inertia: scaling Luukkonen's values by m * l^2 to m = 1 kg,
#   l = 0.125 m gives Ixx = Iyy ~ 3.2e-3, Izz ~ 5.8e-3. A point-mass
#   build-up (4 x 50 g motor+prop at 0.125 m plus a 0.8 kg central box)
#   gives Ixx ~ 2.3e-3 (+ arms / guards), Izz ~ 4.5e-3, so we use
#   Ixx = Iyy = 3.2e-3, Izz = 5.5e-3.
# - Thrust coefficient: chosen so that the maximum rotor speed
#   (1500 rad/s ~ 14.3k RPM, reasonable for 5" props on a 4S pack under
#   load) gives thrust-to-weight = 2:
#       k = 2 * m g / (4 w_max^2) = 2.18e-6 N / (rad/s)^2
#   (same order as Luukkonen's 2.98e-6). Hover rotor speed = 1061 rad/s.
# - Yaw drag coefficient: torque/thrust ratio b/k ~ 0.02 m (Luukkonen:
#   0.038 m for larger props) -> b = 4.4e-8 N m / (rad/s)^2.
# - Translational drag: linear, F = -kd v. Rotor-drag identification on
#   ~1 kg quads (Faessler, Franchi, Scaramuzza, RA-L 2018) gives
#   ~0.3 (m/s^2)/(m/s); Luukkonen uses 0.25 kg/s -> kd = 0.3 N/(m/s).
# - Rotor idle speed 100 rad/s (ESCs never fully stop the motor in
#   flight).
DEFAULT_PHYSICAL_PARAMETERS: Dict[str, float] = {
    "mass": 1.0,               # kg
    "diameter": 0.25,          # motor-to-motor distance [m] (arm = 0.125 m)
    "Ix": 3.2e-3,              # kg m^2
    "Iy": 3.2e-3,              # kg m^2
    "Iz": 5.5e-3,              # kg m^2
    "g": 9.81,                 # m/s^2 (positive; gravity acts along -z)
    "thrust_coef": 2.18e-6,    # k   [N / (rad/s)^2]
    "drag_yaw_coef": 4.4e-8,   # b   [N m / (rad/s)^2]
    "drag_force_coef": 0.3,    # kd  [N / (m/s)]
    "rotor_speed_min": 100.0,  # rad/s
    "rotor_speed_max": 1500.0, # rad/s
}

# Default gains / limits for the internal cascaded controller used by the
# "attitude" and "velocity" control modes. The inner loops run at the
# integration sub-step rate (dt / substeps, 200 Hz by default), i.e. the
# same structure as a PX4 / Betaflight autopilot (velocity -> attitude ->
# body rate -> mixer).
DEFAULT_CONTROLLER_PARAMETERS: Dict[str, float] = {
    # Body-rate loop:  tau = I * K_rate * (w_des - w) + w x I w
    "k_rate_rp": 40.0,       # 1/s   (roll / pitch rate bandwidth)
    "k_rate_yaw": 10.0,      # 1/s
    # Attitude loop:   euler_rate_des = K_att * (euler_des - euler)
    #   -> closed loop w_n = sqrt(40 * 10) = 20 rad/s, zeta = 1.0
    "k_att": 10.0,           # 1/s
    # Velocity loop (velocity mode):  a_des = K_vel * (v_des - v) + drag ff
    "k_vel_xy": 2.5,         # 1/s
    "k_vel_z": 3.0,          # 1/s
    # Limits
    "max_tilt_deg": 35.0,    # max commanded roll / pitch (both modes)
    "max_accel_up": 5.0,     # m/s^2 (vertical, velocity mode)
    "max_accel_down": 5.0,   # m/s^2 (vertical, velocity mode)
}

_CONTROL_MODES = ("rotor_speeds", "attitude", "velocity")
_INTEGRATORS = ("rk4", "euler")


class Dynamics3D(BaseDynamics):
    """
    12D rigid-body quadrotor in 3D using Z-Y-X Euler angles (yaw psi,
    pitch theta, roll phi).

    Coordinate system (right-handed, "FLU")
    ---------------------------------------
    - x : forward
    - y : left
    - z : up   (gravity is [0, 0, -g])

    Positive roll (phi) lifts the left (+y) side and tilts thrust towards
    -y; positive pitch (theta) lowers the nose and tilts thrust towards +x;
    positive yaw (psi) turns the nose towards +y.

    State (12D)  -- NOTE the order
    -------------------------------
    s = [x,  y,  z,          0..2   position, world frame [m]
         psi, theta, phi,    3..5   yaw, pitch, roll [rad]
         vx, vy, vz,         6..8   velocity, world frame [m/s]
         p,  q,  r]          9..11  body angular rates [rad/s]

    Physical model
    --------------
    - thrust  T   = k * sum(w_i^2) along body +z
    - torques tau = arm * k * (...), yaw from b * w^2 (see `_mixer`)
    - m a = R_b2w [0, 0, T] - [0, 0, m g] - kd * v
    - I w_dot = tau - w x (I w)
    - Euler kinematics for Z-Y-X angles (singular at theta = +/-90 deg).

    Rotor layout ("+" configuration, speeds w_i in rad/s)
    ------------------------------------------------------
        w1 : left   (+y), spins CW  (viewed from above) -> +z reaction torque
        w2 : rear   (-x), spins CCW (viewed from above) -> -z reaction torque
        w3 : right  (-y), spins CW
        w4 : front  (+x), spins CCW

    Control modes (`control_mode`)
    ------------------------------
    The physical 12D model above is always the ground truth. What changes
    is the *action interface* seen by the policy:

    - "rotor_speeds": a = [w1, w2, w3, w4] in rad/s (raw, open loop).
    - "attitude":     a = [a_z, phi_des, theta_des, yaw_rate_des]
          a_z    : desired vertical acceleration offset from hover [m/s^2]
                   (collective thrust T = m (g + a_z) / (cos phi cos theta)),
          phi/theta_des : desired roll / pitch [rad],
          yaw_rate_des  : desired yaw rate [rad/s].
    - "velocity":     a = [vx_des, vy_des, vz_des, yaw_rate_des]
          world-frame velocity setpoint [m/s] + yaw rate [rad/s].

    In "attitude" and "velocity" modes, a *stateless* cascaded controller
    (velocity -> attitude -> body rate -> mixer) is evaluated at every
    integration sub-step, so `step(state, action)` stays a pure function of
    the 12D state and the action (no hidden controller state), which MPPI
    rollouts require. In all modes an action of zeros (or hover rotor
    speeds for "rotor_speeds") holds a hover.

    Integration
    -----------
    Each call to `step` advances `dt` using `substeps` sub-steps of an
    `integrator` ("rk4" by default, or semi-implicit "euler"); the rotor
    command is held constant over each sub-step (zero-order hold).

    Parameters
    ----------
    dt : float
        Control / environment time step [s].
    physical_parameters : dict, optional
        Overrides for DEFAULT_PHYSICAL_PARAMETERS.
    control_mode : str
        "rotor_speeds" | "attitude" | "velocity".
    controller_parameters : dict, optional
        Overrides for DEFAULT_CONTROLLER_PARAMETERS.
    integrator : str
        "rk4" | "euler".
    substeps : int, optional
        Number of sub-steps per `dt`. Default: round(dt / 0.005), i.e. a
        200 Hz inner loop.
    """

    def __init__(
        self,
        *,
        dt: float = 0.02,
        physical_parameters: Mapping[str, Any] | None = None,
        control_mode: str = "rotor_speeds",
        controller_parameters: Mapping[str, Any] | None = None,
        integrator: str = "rk4",
        substeps: int | None = None,
    ):
        params: Dict[str, Any] = dict(DEFAULT_PHYSICAL_PARAMETERS)
        params.update(dict(physical_parameters or {}))
        super().__init__(dt=dt, physical_parameters=params)

        control_mode = str(control_mode).lower()
        if control_mode not in _CONTROL_MODES:
            raise ValueError(
                f"control_mode must be one of {_CONTROL_MODES}, got {control_mode!r}"
            )
        integrator = str(integrator).lower()
        if integrator not in _INTEGRATORS:
            raise ValueError(
                f"integrator must be one of {_INTEGRATORS}, got {integrator!r}"
            )
        self.control_mode = control_mode
        self.integrator = integrator

        # Core physical properties
        self.diameter = float(params["diameter"])
        self.mass = float(params["mass"])
        self.Ix = float(params["Ix"])
        self.Iy = float(params["Iy"])
        self.Iz = float(params["Iz"])
        self.g = float(params["g"])
        self.thrust_coef = float(params["thrust_coef"])      # k
        self.drag_yaw_coef = float(params["drag_yaw_coef"])  # b
        self.drag_force_coef = float(params["drag_force_coef"])  # kd
        self.rotor_speed_min = float(params["rotor_speed_min"])
        self.rotor_speed_max = float(params["rotor_speed_max"])
        self._arm_radius = self.diameter / 2.0

        if self.rotor_speed_max <= self.rotor_speed_min:
            raise ValueError("rotor_speed_max must exceed rotor_speed_min")

        # Controller parameters
        cparams: Dict[str, Any] = dict(DEFAULT_CONTROLLER_PARAMETERS)
        cparams.update(dict(controller_parameters or {}))
        self.controller_parameters = {k: float(v) for k, v in cparams.items()}

        # Sub-stepping
        if substeps is None:
            substeps = max(1, int(round(self.dt / 0.005)))
        self.substeps = int(substeps)
        if self.substeps < 1:
            raise ValueError("substeps must be >= 1")
        self.dt_sub = self.dt / self.substeps

        # Mixer: [T, tx, ty, tz] = M @ [w1^2, w2^2, w3^2, w4^2]
        k, b, L = self.thrust_coef, self.drag_yaw_coef, self._arm_radius
        self._mixer_np = np.array(
            [
                [k, k, k, k],
                [k * L, 0.0, -k * L, 0.0],   # roll : left (+y) up is +phi
                [0.0, k * L, 0.0, -k * L],   # pitch: rear up = nose down = +theta
                [b, -b, b, -b],              # yaw  : CW rotors react +z
            ],
            dtype=np.float64,
        )
        self._mixer_inv = jnp.asarray(np.linalg.inv(self._mixer_np), dtype=jnp.float32)

        # Jitted entry points
        self._step_jit = jax.jit(self._step_pure)
        self._rollout_batch_jit = jax.jit(
            jax.vmap(self._rollout_pure, in_axes=(None, 0))
        )
        self._rollout_batch_multi_jit = jax.jit(
            jax.vmap(self._rollout_pure, in_axes=(0, 0))
        )

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, dyn_cfg: Any) -> "Dynamics3D":
        """
        Build from a hydra / dict config block such as `dynamics.dynamics_3d`.

        Reads keys: dt, physical_parameters, control_mode,
        controller_parameters, integrator, substeps (all optional except dt).
        """
        def _get(key, default=None):
            if hasattr(dyn_cfg, "get"):
                val = dyn_cfg.get(key, default)
            else:
                val = getattr(dyn_cfg, key, default)
            return default if val is None else val

        def _to_dict(val):
            if val is None:
                return None
            try:
                from omegaconf import OmegaConf  # optional dependency

                if OmegaConf.is_config(val):
                    return OmegaConf.to_container(val, resolve=True)
            except ImportError:  # pragma: no cover
                pass
            return dict(val)

        substeps = _get("substeps", None)
        return cls(
            dt=float(_get("dt", 0.1)),
            physical_parameters=_to_dict(_get("physical_parameters", None)),
            control_mode=str(_get("control_mode", "rotor_speeds")),
            controller_parameters=_to_dict(_get("controller_parameters", None)),
            integrator=str(_get("integrator", "rk4")),
            substeps=None if substeps is None else int(substeps),
        )

    # ------------------------------------------------------------------ #
    # Convenience quantities
    # ------------------------------------------------------------------ #
    @property
    def hover_rotor_speed(self) -> float:
        return float(np.sqrt(self.mass * self.g / (4.0 * self.thrust_coef)))

    @property
    def thrust_to_weight(self) -> float:
        return float(4.0 * self.thrust_coef * self.rotor_speed_max**2 / (self.mass * self.g))

    def hover_action(self) -> np.ndarray:
        """Action that holds a level hover in the current control mode."""
        if self.control_mode == "rotor_speeds":
            return np.full((4,), self.hover_rotor_speed, dtype=np.float32)
        return np.zeros((4,), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Core dynamics
    # ------------------------------------------------------------------ #
    def step(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """
        Advance one control step `dt` (jit-compiled).

        Parameters
        ----------
        state : (12,) array
        action: (4,)  array, interpreted according to `control_mode`.

        Returns
        -------
        next_state : (12,) jnp.ndarray
        debug      : dict with thrust ft, torques tx/ty/tz, rotor speeds w
                     and body rates p/q/r (from the last sub-step).
        """
        state = jnp.asarray(state, dtype=jnp.float32)
        action = jnp.asarray(action, dtype=jnp.float32)
        return self._step_jit(state, action)

    def rollout(self, state: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """
        Roll out a single action sequence.

        state   : (12,)
        actions : (H, 4)
        returns : (H, 12) states s_1..s_H
        """
        state = jnp.asarray(state, dtype=jnp.float32)
        actions = jnp.asarray(actions, dtype=jnp.float32)
        return self._rollout_batch_jit(state, actions[None])[0]

    def rollout_batch(self, state: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """
        Vectorised (vmap + jit) rollout of many action sequences.

        state   : (12,) shared initial state, or (N, 12) one per sequence
        actions : (N, H, 4)
        returns : (N, H, 12) states s_1..s_H for each sequence
        """
        state = jnp.asarray(state, dtype=jnp.float32)
        actions = jnp.asarray(actions, dtype=jnp.float32)
        if state.ndim == 1:
            return self._rollout_batch_jit(state, actions)
        return self._rollout_batch_multi_jit(state, actions)

    # ------------------------------------------------------------------ #
    # Pure (traceable) implementation
    # ------------------------------------------------------------------ #
    def _step_pure(self, state, action):
        h = self.dt_sub

        def body(s, _):
            w_sq = self._rotor_speeds_sq(s, action)
            s_next = self._integrate(s, w_sq, h)
            return s_next, w_sq

        s_final, w_sq_hist = jax.lax.scan(body, state, None, length=self.substeps)
        w_sq_last = w_sq_hist[-1]
        ft, tx, ty, tz = self._wrench(w_sq_last)
        debug = {
            "ft": ft,
            "tx": tx,
            "ty": ty,
            "tz": tz,
            "w": jnp.sqrt(w_sq_last),
            "p": s_final[9],
            "q": s_final[10],
            "r": s_final[11],
        }
        return s_final, debug

    def _rollout_pure(self, state, actions):
        def body(s, a):
            s_next, _ = self._step_pure(s, a)
            return s_next, s_next

        _, states = jax.lax.scan(body, state, actions)
        return states

    def _integrate(self, s, w_sq, h):
        if self.integrator == "rk4":
            k1 = self._state_derivative(s, w_sq)
            k2 = self._state_derivative(s + 0.5 * h * k1, w_sq)
            k3 = self._state_derivative(s + 0.5 * h * k2, w_sq)
            k4 = self._state_derivative(s + h * k3, w_sq)
            return s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        # Semi-implicit Euler: update velocities / rates first, then use the
        # new values for the positions / angles (symplectic, cheap).
        ds = self._state_derivative(s, w_sq)
        vel_new = s[6:12] + h * ds[6:12]
        s_mid = s.at[6:12].set(vel_new)
        ds2 = self._state_derivative(s_mid, w_sq)
        pos_new = s[0:6] + h * ds2[0:6]
        return jnp.concatenate([pos_new, vel_new])

    def _wrench(self, w_sq):
        """Rotor speeds^2 -> (thrust, tau_x, tau_y, tau_z); same as `_mixer`,
        written out so that symmetric inputs cancel exactly in float32."""
        k, b, L = self.thrust_coef, self.drag_yaw_coef, self._arm_radius
        w1, w2, w3, w4 = w_sq[0], w_sq[1], w_sq[2], w_sq[3]
        ft = k * ((w1 + w3) + (w2 + w4))
        tx = (k * L) * (w1 - w3)            # left (+y) up -> +roll
        ty = (k * L) * (w2 - w4)            # rear up (nose down) -> +pitch
        tz = b * ((w1 + w3) - (w2 + w4))    # CW rotors react +z
        return ft, tx, ty, tz

    def _state_derivative(self, s, w_sq):
        """Continuous-time dynamics s_dot = f(s, w^2)."""
        psi, theta, phi = s[3], s[4], s[5]
        vx, vy, vz = s[6], s[7], s[8]
        p, q, r = s[9], s[10], s[11]

        ft, tx, ty, tz = self._wrench(w_sq)

        s_psi, c_psi = jnp.sin(psi), jnp.cos(psi)
        s_th, c_th = jnp.sin(theta), jnp.cos(theta)
        s_ph, c_ph = jnp.sin(phi), jnp.cos(phi)
        t_th = s_th / c_th

        # Z-Y-X Euler angle rates from body rates
        psi_dot = (q * s_ph + r * c_ph) / c_th
        theta_dot = q * c_ph - r * s_ph
        phi_dot = p + (q * s_ph + r * c_ph) * t_th

        # Translational dynamics (world frame, z up), third column of
        # R_b2w = Rz(psi) Ry(theta) Rx(phi)
        m, g, kd = self.mass, self.g, self.drag_force_coef
        ax = (ft / m) * (c_psi * s_th * c_ph + s_psi * s_ph) - (kd / m) * vx
        ay = (ft / m) * (s_psi * s_th * c_ph - c_psi * s_ph) - (kd / m) * vy
        az = (ft / m) * (c_th * c_ph) - g - (kd / m) * vz

        # Rotational dynamics (Euler's equations, body frame)
        Ix, Iy, Iz = self.Ix, self.Iy, self.Iz
        pdot = ((Iy - Iz) * q * r + tx) / Ix
        qdot = ((Iz - Ix) * p * r + ty) / Iy
        rdot = ((Ix - Iy) * p * q + tz) / Iz

        return jnp.stack(
            [vx, vy, vz, psi_dot, theta_dot, phi_dot, ax, ay, az, pdot, qdot, rdot]
        )

    # ------------------------------------------------------------------ #
    # Action interface -> rotor speeds^2
    # ------------------------------------------------------------------ #
    def _rotor_speeds_sq(self, s, action):
        if self.control_mode == "rotor_speeds":
            w = jnp.clip(action, self.rotor_speed_min, self.rotor_speed_max)
            # NOTE: under jit, XLA may contract w*w with the mixer
            # differences into FMAs, so perfectly symmetric rotor speeds give
            # float32 torque residuals of ~1e-8 N m (open-loop hover drifts
            # ~1 cm in 10 s). Irrelevant in the closed-loop modes.
            return w * w
        if self.control_mode == "attitude":
            return self._attitude_controller(
                s,
                accel_z_offset=action[0],
                phi_des=action[1],
                theta_des=action[2],
                yaw_rate_des=action[3],
            )
        return self._velocity_controller(s, action)

    def _velocity_controller(self, s, action):
        c = self.controller_parameters
        m, g, kd = self.mass, self.g, self.drag_force_coef
        psi = s[3]
        v = s[6:9]
        v_des = action[0:3]

        # Desired world acceleration (+ drag feed-forward)
        a_xy = c["k_vel_xy"] * (v_des[0:2] - v[0:2]) + (kd / m) * v[0:2]
        a_z = c["k_vel_z"] * (v_des[2] - v[2]) + (kd / m) * v[2]
        a_z = jnp.clip(a_z, -c["max_accel_down"], c["max_accel_up"])

        # Tilt limit: horizontal accel <= (g + a_z) tan(max_tilt)
        max_tilt = np.deg2rad(c["max_tilt_deg"])
        a_xy_max = (g + a_z) * np.tan(max_tilt)
        a_xy_norm = jnp.sqrt(jnp.sum(a_xy * a_xy) + 1e-12)
        a_xy = a_xy * jnp.minimum(1.0, a_xy_max / a_xy_norm)

        # Desired specific force f = a + g z, expressed in the yaw frame
        fx, fy, fz = a_xy[0], a_xy[1], a_z + g
        c_psi, s_psi = jnp.cos(psi), jnp.sin(psi)
        fx_y = c_psi * fx + s_psi * fy
        fy_y = -s_psi * fx + c_psi * fy
        f_norm = jnp.sqrt(fx * fx + fy * fy + fz * fz)

        # R e3 in the yaw frame is [s_th c_ph, -s_ph, c_th c_ph]
        phi_des = jnp.arcsin(jnp.clip(-fy_y / f_norm, -1.0, 1.0))
        theta_des = jnp.arctan2(fx_y, fz)

        return self._attitude_controller(
            s,
            accel_z_offset=a_z,
            phi_des=phi_des,
            theta_des=theta_des,
            yaw_rate_des=action[3],
        )

    def _attitude_controller(self, s, *, accel_z_offset, phi_des, theta_des, yaw_rate_des):
        c = self.controller_parameters
        m, g = self.mass, self.g
        theta, phi = s[4], s[5]
        omega = s[9:12]

        max_tilt = np.deg2rad(c["max_tilt_deg"])
        phi_des = jnp.clip(phi_des, -max_tilt, max_tilt)
        theta_des = jnp.clip(theta_des, -max_tilt, max_tilt)

        # Collective thrust with tilt compensation (vertical accel = a_z)
        tilt_cos = jnp.maximum(jnp.cos(phi) * jnp.cos(theta), 0.5)
        thrust = m * (g + accel_z_offset) / tilt_cos
        t_min = 4.0 * self.thrust_coef * self.rotor_speed_min**2
        t_max = 4.0 * self.thrust_coef * self.rotor_speed_max**2
        thrust = jnp.clip(thrust, t_min, t_max)

        # Attitude loop -> desired Euler rates -> desired body rates
        phi_dot_d = c["k_att"] * (phi_des - phi)
        theta_dot_d = c["k_att"] * (theta_des - theta)
        psi_dot_d = yaw_rate_des
        s_th, c_th = jnp.sin(theta), jnp.cos(theta)
        s_ph, c_ph = jnp.sin(phi), jnp.cos(phi)
        p_d = phi_dot_d - s_th * psi_dot_d
        q_d = c_ph * theta_dot_d + s_ph * c_th * psi_dot_d
        r_d = -s_ph * theta_dot_d + c_ph * c_th * psi_dot_d
        omega_d = jnp.stack([p_d, q_d, r_d])

        # Rate loop with gyroscopic compensation
        inertia = jnp.asarray([self.Ix, self.Iy, self.Iz], dtype=jnp.float32)
        k_rate = jnp.asarray([c["k_rate_rp"], c["k_rate_rp"], c["k_rate_yaw"]], dtype=jnp.float32)
        tau = inertia * k_rate * (omega_d - omega) + jnp.cross(omega, inertia * omega)

        return self._mix(thrust, tau)

    def _mix(self, thrust, tau):
        """Wrench -> rotor speeds^2 with attitude-priority desaturation."""
        w_sq = self._mixer_inv @ jnp.concatenate([thrust[None], tau])
        lo = self.rotor_speed_min**2
        hi = self.rotor_speed_max**2
        # Shift collective (keeps torques) to fit into [lo, hi] if possible
        over = jnp.maximum(jnp.max(w_sq) - hi, 0.0)
        under = jnp.maximum(lo - jnp.min(w_sq), 0.0)
        w_sq = w_sq - over + under
        return jnp.clip(w_sq, lo, hi)

    # ------------------------------------------------------------------ #
    # State metadata
    # ------------------------------------------------------------------ #
    def state_variable_descriptions(self) -> Sequence[str]:
        return [
            "x position (forward, m)",          # 0
            "y position (left, m)",             # 1
            "z position (up, m)",               # 2
            "yaw ψ about z (rad)",              # 3
            "pitch θ about y (rad)",            # 4
            "roll φ about x (rad)",             # 5
            "x velocity vx in world (m/s)",     # 6
            "y velocity vy in world (m/s)",     # 7
            "z velocity vz in world (m/s)",     # 8
            "roll rate p (rad/s)",              # 9
            "pitch rate q (rad/s)",             # 10
            "yaw rate r (rad/s)",               # 11
        ]

    def state_variable_symbols(self) -> Sequence[str]:
        return ["x", "y", "z", "ψ", "θ", "φ", "vₓ", "v_y", "v_z", "p", "q", "r"]

    def state_indices_manifold(self) -> Sequence[int]:
        """Position coordinates (x, y, z)."""
        return [0, 1, 2]

    # ------------------------------------------------------------------ #
    # Control metadata
    # ------------------------------------------------------------------ #
    def control_variable_descriptions(self) -> Sequence[str]:
        if self.control_mode == "attitude":
            return [
                "vertical acceleration offset from hover a_z (m/s^2)",
                "roll setpoint φ_des (rad)",
                "pitch setpoint θ_des (rad)",
                "yaw-rate setpoint (rad/s)",
            ]
        if self.control_mode == "velocity":
            return [
                "vx setpoint, world (m/s)",
                "vy setpoint, world (m/s)",
                "vz setpoint, world (m/s)",
                "yaw-rate setpoint (rad/s)",
            ]
        return [
            "w₁ rotor speed (left, CW) rad/s",    # 0
            "w₂ rotor speed (rear, CCW) rad/s",   # 1
            "w₃ rotor speed (right, CW) rad/s",   # 2
            "w₄ rotor speed (front, CCW) rad/s",  # 3
        ]

    def control_variable_symbols(self) -> Sequence[str]:
        if self.control_mode == "attitude":
            return ["a_z", "φ_d", "θ_d", "ψ̇_d"]
        if self.control_mode == "velocity":
            return ["vₓ_d", "v_y_d", "v_z_d", "ψ̇_d"]
        return ["ω₁", "ω₂", "ω₃", "ω₄"]
