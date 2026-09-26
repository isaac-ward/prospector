# src/tasks/task_waypoint_following.py

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np
import jax.numpy as jnp

from prospector.caves.cave_map_2d import CaveMap2D
from prospector.caves.cave_map_3d import CaveMap3D
from prospector.agents.policies.policy_mppi import RewardFn


SubtaskSatisfiedFn = Callable[[jnp.ndarray], bool]


class TaskWaypointFollowing:
    """
    Sequential waypoint-following task with MPPI-style rewards.

    For each waypoint, we define:

      - A reward over full histories:

            reward(history_states, history_actions)

        with components:
          * Distance term: sum_t ||p_t - p_goal||^2
          * Control term : sum_t ||a_t|| (L2)
          * Collision term: huge penalty if ANY state collides

        reward = - (distance_cost + control_cost + collision_cost)

      - A subtask satisfaction function:

            is_satisfied(current_state) -> bool

        which is True when within `agent_radius * multiplier_within_radius_of_goal`
        of the waypoint.

    Internally, we create a *synthetic* extra subtask that duplicates the
    final waypoint. This lets `_current_subtask_idx` advance *past* the last
    real waypoint when it has been reached, while:

      - `num_subtasks` still reports the number of *real* waypoints K.
      - `current_waypoint` always returns one of the real waypoints (the last
        one once the task is completed).

    This keeps `is_task_completed()` well-defined without breaking external
    expectations about `num_subtasks`.
    """

    def __init__(
        self,
        waypoints: Sequence[Sequence[float]],
        position_dims: Sequence[int],
        cave_map: CaveMap2D | CaveMap3D,
        agent_radius: float,
        *,
        distance_weight: float = 1.0,
        control_weight: float = 0.1,
        collision_penalty: float = 1e6,
        multiplier_within_radius_of_goal: float = 3.0,
        hold_at_indices: Sequence[int] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        waypoints : sequence of (2,) or (3,) sequences
            Positional waypoints to be reached in order.

        position_dims : sequence of int
            Indices of the state vector that correspond to (x, y[, z])
            for the agent's position.

        cave_map : CaveMap2D or CaveMap3D
            Used for collision checks in the reward.

        agent_radius : float
            Base radius used for collision checks; waypoint satisfaction uses
            agent_radius * multiplier_within_radius_of_goal.

        distance_weight : float
            Weight on summed squared distance-to-waypoint term.

        control_weight : float
            Weight on summed action magnitude term.

        collision_penalty : float
            Big positive scalar added to cost when ANY state collides;
            reward = -cost, so this becomes a huge negative reward.

        multiplier_within_radius_of_goal : float
            Multiplier on agent_radius to define the "goal reached" radius for
            each waypoint.

        hold_at_indices : sequence of int, optional
            Waypoint indices at which the agent must hold (keep targeting that
            waypoint) once reached, until `release_hold(idx)` is called. Used
            for communication rendezvous: an agent waits at a comms node until
            it has heard from the other agent(s).
        """
        wp_np = np.asarray(waypoints, dtype=float)
        if wp_np.ndim != 2:
            raise ValueError(
                f"waypoints must be a 2D array of shape (K, D), got {wp_np.shape}"
            )

        num_waypoints, dim = wp_np.shape
        if dim not in (2, 3):
            raise ValueError(
                f"waypoints must be 2D or 3D points, got D={dim} (shape={wp_np.shape})"
            )
        if num_waypoints == 0:
            raise ValueError("At least one waypoint is required.")

        # Real (external) waypoints: shape (K, D)
        self._waypoints = jnp.asarray(wp_np, dtype=jnp.float32)
        self._num_subtasks: int = int(num_waypoints)            # K
        self._num_internal_subtasks: int = self._num_subtasks + 1  # K + 1 (synthetic)

        self._agent_radius = float(agent_radius)
        self._distance_weight = float(distance_weight)
        self._control_weight = float(control_weight)
        self._collision_penalty = float(collision_penalty)
        self._multiplier_within_radius_of_goal = float(
            multiplier_within_radius_of_goal
        )

        self._cave_map = cave_map
        self._position_dims = position_dims
        self._current_subtask_idx: int = 0

        # Hold points (e.g. comms rendezvous). `_holding_idx` is set once the
        # agent has reached a hold waypoint that hasn't been released yet.
        self._hold_indices = set(int(i) for i in (hold_at_indices or []))
        for i in self._hold_indices:
            if not (0 <= i < self._num_subtasks):
                raise ValueError(
                    f"hold index {i} out of range for {self._num_subtasks} waypoints"
                )
        self._released_holds: set[int] = set()
        self._holding_idx: int | None = None

        # ------------------------------------------------------------------ #
        # Build per-waypoint reward + satisfaction functions                 #
        # ------------------------------------------------------------------ #
        # Internal lists have length K + 1: the final entry duplicates the
        # last real waypoint.
        self._reward_fns: List[RewardFn] = []
        self._is_satisfied_fns: List[SubtaskSatisfiedFn] = []

        # Real waypoints
        for k in range(self._num_subtasks):
            target = self._waypoints[k]  # (D,)
            self._reward_fns.append(self._make_reward_fn_for_waypoint(target))
            self._is_satisfied_fns.append(
                self._make_is_satisfied_fn_for_waypoint(target)
            )

        # Synthetic extra subtask: duplicates the last waypoint
        last_target = self._waypoints[-1]
        self._reward_fns.append(self._make_reward_fn_for_waypoint(last_target))
        self._is_satisfied_fns.append(
            self._make_is_satisfied_fn_for_waypoint(last_target)
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _make_reward_fn_for_waypoint(self, target_pos: jnp.ndarray) -> RewardFn:
        """
        Reward for a single waypoint, using FULL histories.

        Given:
            history_states  : (T_s, state_dim)
            history_actions : (T_a, action_dim)

        Cost components:
          - distance_cost = distance_weight * sum_t ||p_t - p_goal||^2
          - control_cost  = control_weight * sum_t ||a_t||
          - collision_cost = collision_penalty if ANY state collides

        reward = - (distance_cost + control_cost + collision_cost)
        """
        target_pos = jnp.asarray(target_pos, dtype=jnp.float32)
        pos_idx = jnp.asarray(self._position_dims, dtype=jnp.int32)
        agent_radius = float(self._agent_radius)
        distance_weight = self._distance_weight
        control_weight = self._control_weight
        collision_penalty = self._collision_penalty
        cave_map = self._cave_map

        def reward_fn(
            history_states: jnp.ndarray,
            history_actions: jnp.ndarray,
        ) -> jnp.ndarray:
            if history_states.ndim != 2:
                raise ValueError(
                    f"history_states must have shape (T_s, state_dim), "
                    f"got {history_states.shape}"
                )

            # 1) Distance term over the full path
            positions = history_states[:, pos_idx]  # (T_s, D)
            diff = positions - target_pos  # (T_s, D)
            dist_sq = jnp.sum(diff * diff, axis=-1)  # (T_s,)
            distance_cost = distance_weight * jnp.sum(dist_sq)

            # 2) Control term over the full action sequence
            if history_actions.ndim == 2 and history_actions.shape[0] > 0:
                norms = jnp.linalg.norm(history_actions, axis=-1)  # (T_a,)
                control_cost = control_weight * jnp.sum(norms)
            else:
                control_cost = jnp.asarray(0.0, dtype=jnp.float32)

            # 3) Collision penalty: any state along the path colliding
            collision_flag = False
            for t in range(history_states.shape[0]):
                state_t = history_states[t]
                pos_t = np.array(
                    [float(state_t[int(d)]) for d in self._position_dims],
                    dtype=float,
                )
                if cave_map.is_collision_within_radius(pos_t, agent_radius):
                    collision_flag = True
                    break

            collision_cost = collision_penalty if collision_flag else 0.0

            total_cost = distance_cost + control_cost + collision_cost
            return -total_cost  # higher is better

        return reward_fn

    def _make_is_satisfied_fn_for_waypoint(
        self,
        target_pos: jnp.ndarray,
    ) -> SubtaskSatisfiedFn:
        """
        Subtask is satisfied when current position is within
        (agent_radius * multiplier_within_radius_of_goal) of the waypoint:

            ||p - p_goal|| <= agent_radius * multiplier_within_radius_of_goal
        """
        target_pos = jnp.asarray(target_pos, dtype=jnp.float32)
        pos_idx = jnp.asarray(self._position_dims, dtype=jnp.int32)
        radius_sq = float(
            self._agent_radius * self._multiplier_within_radius_of_goal
        ) ** 2

        def is_satisfied(current_state: jnp.ndarray) -> bool:
            if current_state.ndim != 1:
                raise ValueError(
                    f"current_state must be a 1D vector (state_dim,), "
                    f"got {current_state.shape}"
                )

            pos = current_state[pos_idx]  # (D,)
            diff = pos - target_pos
            dist_sq = jnp.sum(diff * diff)

            return bool(dist_sq <= radius_sq)

        return is_satisfied

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    @property
    def num_subtasks(self) -> int:
        """
        Number of *real* subtasks (real waypoints), K.
        """
        return self._num_subtasks

    @property
    def current_subtask_idx(self) -> int:
        """
        Current internal subtask index.

        This can range from 0 up to K (inclusive), where K corresponds to the
        synthetic "completed" subtask that duplicates the last waypoint.
        """
        return self._current_subtask_idx

    @property
    def current_waypoint(self) -> jnp.ndarray:
        """
        Return the waypoint for the currently active subtask.

        Shape: (D,), where D is 2 or 3 depending on how the task was
        constructed. Even when the internal index has advanced into the
        synthetic subtask, this returns the *last real waypoint*.
        """
        idx = self._current_subtask_idx
        # Clamp to last real waypoint if we've advanced into the synthetic one.
        clamped_idx = min(idx, self._num_subtasks - 1)
        return self._waypoints[clamped_idx]

    @property
    def waypoints(self) -> jnp.ndarray:
        """
        All real waypoints as a (K, D) array (mainly for inspection/debug).
        """
        return self._waypoints

    def update_subtask_from_state(self, current_state: jnp.ndarray) -> None:
        """
        Advance the active subtask index if the current one is satisfied.

        - Internal index ranges from 0 up to K (inclusive), where K is
          the synthetic "completed" subtask.
        - Does NOT advance past the synthetic subtask; the last reward
          remains active there as well.
        """
        idx = self._current_subtask_idx
        max_internal_idx = self._num_internal_subtasks - 1  # K

        # While current subtask is satisfied and we're not at the internal last
        while idx < max_internal_idx and self._is_satisfied_fns[idx](current_state):
            # Reached an unreleased hold point: stay on it (keep targeting
            # this waypoint) until release_hold(idx) is called.
            if idx in self._hold_indices and idx not in self._released_holds:
                self._holding_idx = idx
                break
            idx += 1

        self._current_subtask_idx = idx

    @property
    def holding_idx(self) -> int | None:
        """Index of the hold waypoint the agent is waiting at, else None."""
        return self._holding_idx

    @property
    def hold_indices(self) -> List[int]:
        return sorted(self._hold_indices)

    def release_hold(self, idx: int) -> None:
        """Allow the agent to advance past hold waypoint `idx`."""
        self._released_holds.add(int(idx))
        if self._holding_idx == idx:
            self._holding_idx = None

    def reward_for_active_subtask(
        self,
        history_states: jnp.ndarray,
        history_actions: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Reward function for the *currently active* subtask.

        This uses the internal reward list (length K+1), so even when the task
        is "completed" (index == K), the same reward as the last real waypoint
        is used.
        """
        fn = self._reward_fns[self._current_subtask_idx]
        return fn(history_states, history_actions)

    def batch_reward_for_active_subtask(
        self,
        history_states: np.ndarray,
        history_actions: np.ndarray,
        *,
        num_history_states: int = 0,
        margin_radius: float | None = None,
        margin_penalty: float = 0.0,
        margin_exempt_radius: float = 0.0,
    ) -> np.ndarray:
        """
        Vectorized `reward_for_active_subtask` over K candidate histories.

        history_states  : (K, T_s, state_dim)
        history_actions : (K, T_a, action_dim)
        returns         : (K,) rewards. With the default keyword arguments this
                          is identical in definition to the per-history reward.

        num_history_states : the first this-many states are the (unchangeable)
                             past; they are excluded from the collision check
                             so a planning margin can't make every candidate
                             "collide" because of where the agent already was.
        margin_radius      : optional soft safety margin (> agent_radius).
                             Candidates coming within this radius of an
                             obstacle (but not within agent_radius) pay
                             `margin_penalty`. It is soft on purpose: in a
                             corridor narrower than 2*margin_radius a hard
                             margin makes every candidate equally
                             "infeasible", and the planner loses the signal
                             that keeps it off the walls.
        margin_exempt_radius : planned states within this distance of the
                             active waypoint skip the soft margin (the hard
                             crash check still applies). Graph nodes can sit
                             closer to rock than the margin; without this the
                             margin penalty outweighs the pull of the last
                             metre and the agent parks short of the node.
        """
        idx = min(self._current_subtask_idx, self._num_internal_subtasks - 1)
        target = np.asarray(self._waypoints[min(idx, self._num_subtasks - 1)], dtype=np.float64)
        S = np.asarray(history_states, dtype=np.float64)
        A = np.asarray(history_actions, dtype=np.float64)
        pos = S[:, :, list(self._position_dims)]  # (K, T, D)

        distance_cost = self._distance_weight * np.sum((pos - target) ** 2, axis=(1, 2))
        if A.ndim == 3 and A.shape[1] > 0:
            control_cost = self._control_weight * np.sum(np.linalg.norm(A, axis=-1), axis=1)
        else:
            control_cost = np.zeros(S.shape[0])

        check = pos[:, int(num_history_states):]

        def collisions(radius: float) -> np.ndarray:
            """(K, T) per-state collision flags for the checked states."""
            K, T, D = check.shape
            if T == 0:
                return np.zeros((K, 0), dtype=bool)
            if hasattr(self._cave_map, "batch_is_collision_within_radius"):
                return self._cave_map.batch_is_collision_within_radius(
                    check.reshape(-1, D), radius
                ).reshape(K, T)
            return np.array(
                [
                    [self._cave_map.is_collision_within_radius(p, radius) for p in traj]
                    for traj in check
                ]
            )

        collision_cost = np.where(collisions(self._agent_radius).any(axis=1), self._collision_penalty, 0.0)
        if margin_radius is not None and margin_penalty > 0.0:
            in_margin = collisions(float(margin_radius))
            if margin_exempt_radius > 0.0:
                near_goal = np.linalg.norm(check - target, axis=-1) <= float(margin_exempt_radius)
                in_margin = in_margin & ~near_goal
            collision_cost = collision_cost + np.where(
                in_margin.any(axis=1), float(margin_penalty), 0.0
            )

        return -(distance_cost + control_cost + collision_cost)

    @property
    def reward_functions(self) -> List[RewardFn]:
        """
        Reward functions for the *real* subtasks only (length K).

        The synthetic extra subtask is not exposed here.
        """
        return list(self._reward_fns[: self._num_subtasks])

    @property
    def is_subtask_satisfied_fns(self) -> List[SubtaskSatisfiedFn]:
        """
        Satisfaction functions for the *real* subtasks only (length K).

        The synthetic extra subtask is not exposed here.
        """
        return list(self._is_satisfied_fns[: self._num_subtasks])

    def is_task_completed(self) -> bool:
        """
        Return True if all waypoints have been reached.

        This is defined as: the internal index has advanced into the
        synthetic "completed" subtask, i.e. current_subtask_idx >= K.
        """
        return self._current_subtask_idx >= self._num_subtasks
