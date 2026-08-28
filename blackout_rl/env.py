"""Project-level BlackOut environment adapter with a reproducible seed contract."""

from __future__ import annotations

import importlib
import threading
from typing import Any

from blackout_env import BlackOutEnv
from mlagents_envs.environment import UnityEnvironment


_UPSTREAM_ENV_MODULE = importlib.import_module(BlackOutEnv.__module__)
_UNITY_ENV_PATCH_LOCK = threading.Lock()


class _BackgroundUnityEnvironment(UnityEnvironment):
    """Launch a rendered Unity player without its normal interactive UI."""

    def _executable_args(self) -> list[str]:
        args = super()._executable_args()
        if "-batchmode" not in (arg.lower() for arg in args):
            args.append("-batchmode")
        return args


class ContractBlackOutEnv(BlackOutEnv):
    """BlackOutEnv with seed delivery completed before Unity's reset command.

    ML-Agents carries side-channel messages and the environment-reset command in
    one exchange. In the fixed upstream revisions, Unity handles the reset before
    `SeedChannel.OnMessageReceived`, so `BlackOutEnv.reset(seed=N)` seeds the next
    episode rather than the one returned by that call. Flushing the seed through
    one hidden Unity step first makes the public reset follow the Gymnasium and
    PettingZoo expectation that `seed` applies to the returned episode.
    """

    def __init__(self, *args: Any, background: bool = False, **kwargs: Any) -> None:
        """Create the environment, optionally keeping the Unity player in background.

        ``background=True`` adds Unity's ``-batchmode`` argument but deliberately
        does not use ``-nographics``. BlackOut's semantic map is a rendered visual
        observation, so disabling the graphics device would replace that map with
        zeros and silently invalidate training.
        """

        if not background:
            super().__init__(*args, **kwargs)
            return

        # The upstream wrapper constructs UnityEnvironment internally and does not
        # expose ``additional_args``. Substitute the launch class only for the
        # duration of its constructor. Environment construction in this process is
        # serialized so two simultaneous callers cannot observe a half-restored
        # upstream module global.
        with _UNITY_ENV_PATCH_LOCK:
            original = _UPSTREAM_ENV_MODULE.UnityEnvironment
            _UPSTREAM_ENV_MODULE.UnityEnvironment = _BackgroundUnityEnvironment
            try:
                super().__init__(*args, **kwargs)
            finally:
                _UPSTREAM_ENV_MODULE.UnityEnvironment = original

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
