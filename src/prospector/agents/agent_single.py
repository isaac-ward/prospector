# src/prospector/agents/agent_single.py

from __future__ import annotations

from typing import Any

from .agent_multi import MultiAgent
from .policies.base_policy import BasePolicy


class SingleAgent(MultiAgent):
    """
    Thin wrapper around MultiAgent for the single-agent case.

    This fixes num_agents = 1 and reuses all the history / policy logic
    from MultiAgent, while giving a nicer constructor for the common
    single-agent case.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        history_len: int,
        policy: BasePolicy,
    ) -> None:
        super().__init__(
            num_agents=1,
            state_dim=state_dim,
            action_dim=action_dim,
            history_len=history_len,
            policies=policy,
        )

    @property
    def policy(self) -> BasePolicy:
        """
        Convenience accessor for the single underlying policy.
        """
        return self._get_policy_for_agent(0)
