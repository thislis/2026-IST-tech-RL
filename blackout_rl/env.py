"""Project-level BlackOut environment adapter with a reproducible seed contract."""

from __future__ import annotations

from typing import Any

from blackout_env import BlackOutEnv


class ContractBlackOutEnv(BlackOutEnv):
    """BlackOutEnv with seed delivery completed before Unity's reset command.

    ML-Agents carries side-channel messages and the environment-reset command in
    one exchange. In the fixed upstream revisions, Unity handles the reset before
    `SeedChannel.OnMessageReceived`, so `BlackOutEnv.reset(seed=N)` seeds the next
    episode rather than the one returned by that call. Flushing the seed through
    one hidden Unity step first makes the public reset follow the Gymnasium and
    PettingZoo expectation that `seed` applies to the returned episode.
    """

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict[str, Any]]:
        if seed is not None:
            # UnityEnvironment.step() aliases to reset() for its first message,
            # which would put the seed back in the same ordering race. Complete
            # that handshake without a queued seed before the explicit flush.
            if self._unity_env._is_first_message:
                self._unity_env.reset()
            self._seed_channel.send_seed(seed)
            # This exchange delivers the side-channel payload. ML-Agents fills
            # blank actions for all current decision agents; its state is then
            # discarded by the immediately following environment reset.
            self._unity_env.step()
        # The upstream wrapper keeps these caches across reset. Clear them so
        # the returned observation cannot silently reuse the previous episode's
        # semantic map or ML-agent routing table.
        self._team_graphics.clear()
        self._agent_name_cache.clear()
        return super().reset(seed=None, options=options)
