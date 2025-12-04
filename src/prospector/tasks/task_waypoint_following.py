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
            idx += 1

        self._current_subtask_idx = idx

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
