# src/prospector/tests/test_comms_rendezvous.py
"""
Checks the comms-rendezvous hold/release logic without a cave:

    uv run python -m prospector.tests.test_comms_rendezvous
"""

from __future__ import annotations

import numpy as np

from prospector.tasks.task_waypoint_following import TaskWaypointFollowing
from prospector.tasks.comms_rendezvous import CommsRendezvous


class _FreeSpace:
    def is_collision_within_radius(self, p, r):
        return False


def main() -> None:
    free = _FreeSpace()
    # Agent 0 has a short leg to its rendezvous waypoint (index 1), agent 1 a long one.
    wps0 = [[1, 0, 0], [2, 0, 0], [3, 0, 0]]
    wps1 = [[0, 5, 0], [0, 10, 0], [0, 11, 0]]
    tasks = [
        TaskWaypointFollowing(w, [0, 1, 2], free, 0.1, hold_at_indices=[1])
        for w in (wps0, wps1)
    ]
    rv = CommsRendezvous(tasks, [1], timeout_steps=100)
    no_los = np.zeros((2, 2), dtype=np.int8)
    los = np.array([[0, 1], [1, 0]], dtype=np.int8)

    def at(i, k):
        tasks[i].update_subtask_from_state(np.asarray(([wps0, wps1][i])[k], dtype=np.float32))

    # Agent 0 reaches wp0 then wp1 -> must hold at 1
    at(0, 0); at(0, 1)
    assert tasks[0].current_subtask_idx == 1 and tasks[0].holding_idx == 1
    rv.update(los, step_index=1)
    assert tasks[0].holding_idx == 1, "released before other agent arrived"

    # Being on wp1 repeatedly must not advance past it
    at(0, 1); at(0, 2)
    assert tasks[0].current_subtask_idx == 1

    # Agent 1 arrives, but no line of sight -> still held
    at(1, 0); at(1, 1)
    rv.update(no_los, step_index=5)
    assert tasks[0].holding_idx == 1 and tasks[1].holding_idx == 1

    # Line of sight -> both released, and both can now advance
    info = rv.update(los, step_index=7)
    assert info["rendezvous_released"]["wait_steps"] == [6, 2], info
    assert rv.all_released
    at(0, 1); at(1, 1)
    assert tasks[0].current_subtask_idx == 2 and tasks[1].current_subtask_idx == 2

    # Timeout
    tasks2 = [TaskWaypointFollowing(w, [0, 1, 2], free, 0.1, hold_at_indices=[0]) for w in (wps0, wps1)]
    rv2 = CommsRendezvous(tasks2, [0], timeout_steps=3)
    tasks2[0].update_subtask_from_state(np.asarray(wps0[0], dtype=np.float32))
    for s in range(1, 6):
        rv2.update(los, step_index=s)
    assert rv2.failed, "expected timeout"

    print("[test_comms_rendezvous] all checks passed")


if __name__ == "__main__":
    main()
