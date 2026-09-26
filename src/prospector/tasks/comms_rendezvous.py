# src/prospector/tasks/comms_rendezvous.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from prospector.tasks.task_waypoint_following import TaskWaypointFollowing


class CommsRendezvous:
    """
    Communication rendezvous between agents following step-aligned tours.

    Agents fly their own waypoint sequences at their own pace. At a shared
    rendezvous index k (a tour step after which the agents are meant to be in
    contact), each agent holds at its k-th waypoint until:

      1) every agent has reached (and is holding at) its k-th waypoint, and
      2) the agents actually have communications (line of sight) according to
         the environment's communications matrix.

    Only then are all agents released to continue. Drones can only talk when
    they are at comms nodes with line of sight, so nothing downstream of a
    rendezvous may start before that contact has happened.

    Barriers are processed in order; `timeout_steps` bounds how long any agent
    may wait at a barrier before the rendezvous is declared failed.
    """

    def __init__(
        self,
        tasks: Sequence[TaskWaypointFollowing],
        rendezvous_indices: Sequence[int],
        *,
        timeout_steps: Optional[int] = None,
    ) -> None:
        self._tasks = list(tasks)
        self._barriers: List[int] = sorted(set(int(k) for k in rendezvous_indices))
        for i, task in enumerate(self._tasks):
            missing = set(self._barriers) - set(task.hold_indices)
            if missing:
                raise ValueError(
                    f"Task for agent {i} does not hold at rendezvous indices {sorted(missing)}"
                )
        self._next = 0  # position in self._barriers
        self._timeout_steps = timeout_steps
        self._arrival_step: List[Optional[int]] = [None] * len(self._tasks)
        self.events: List[Dict[str, Any]] = []
        self.failed: bool = False
        self.failure_reason: Optional[str] = None

    @property
    def current_barrier(self) -> Optional[int]:
        if self._next < len(self._barriers):
            return self._barriers[self._next]
        return None

    @property
    def all_released(self) -> bool:
        return self._next >= len(self._barriers)

    @staticmethod
    def _connected(comms_matrix: np.ndarray) -> bool:
        """True if the comms graph over all agents is connected."""
        n = comms_matrix.shape[0]
        seen = {0}
        frontier = [0]
        while frontier:
            i = frontier.pop()
            for j in range(n):
                if j not in seen and comms_matrix[i, j]:
                    seen.add(j)
                    frontier.append(j)
        return len(seen) == n

    def update(self, comms_matrix: np.ndarray, step_index: int) -> Dict[str, Any]:
        """
        Call once per environment step, after tasks have been updated.

        Returns a small dict for the step's info (who is waiting, releases).
        """
        info: Dict[str, Any] = {"rendezvous_released": None}
        k = self.current_barrier
        if k is None:
            info["rendezvous_waiting"] = [False] * len(self._tasks)
            return info

        holding = [t.holding_idx == k for t in self._tasks]
        for i, h in enumerate(holding):
            if h and self._arrival_step[i] is None:
                self._arrival_step[i] = step_index

        in_contact = self._connected(np.asarray(comms_matrix))
        if all(holding) and in_contact:
            for t in self._tasks:
                t.release_hold(k)
            event = {
                "rendezvous_index": k,
                "released_at_step": step_index,
                "arrival_steps": list(self._arrival_step),
                "wait_steps": [step_index - a for a in self._arrival_step],
            }
            self.events.append(event)
            info["rendezvous_released"] = event
            self._arrival_step = [None] * len(self._tasks)
            self._next += 1
            holding = [False] * len(self._tasks)
        elif self._timeout_steps is not None:
            waited = [
                step_index - a for a in self._arrival_step if a is not None
            ]
            if waited and max(waited) > self._timeout_steps:
                self.failed = True
                self.failure_reason = (
                    f"rendezvous {k} timed out after {max(waited)} steps "
                    f"(holding={holding}, in_contact={in_contact})"
                )

        info["rendezvous_waiting"] = holding
        info["rendezvous_barrier"] = k
        return info
