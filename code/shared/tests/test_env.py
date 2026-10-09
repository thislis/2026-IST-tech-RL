"""Tests for Unity process launch behavior in the project environment adapter."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from blackout_env import BlackOutEnv

from blackout_rl.env import (
    ContractBlackOutEnv,
    _BackgroundUnityEnvironment,
    _UPSTREAM_ENV_MODULE,
)


class BackgroundUnityEnvironmentTests(unittest.TestCase):
    @staticmethod
    def _environment(*, no_graphics: bool) -> _BackgroundUnityEnvironment:
        env = _BackgroundUnityEnvironment.__new__(_BackgroundUnityEnvironment)
        env._no_graphics = no_graphics
        env._port = 50123
        env._log_folder = None
        env._worker_id = 0
        env._additional_args = []
        return env

    def test_background_mode_keeps_graphics_and_adds_batchmode(self) -> None:
        args = self._environment(no_graphics=False)._executable_args()

        self.assertIn("-batchmode", args)
        self.assertNotIn("-nographics", args)

    def test_no_graphics_does_not_duplicate_batchmode(self) -> None:
        args = self._environment(no_graphics=True)._executable_args()

        self.assertEqual(args.count("-batchmode"), 1)
        self.assertIn("-nographics", args)

    def test_adapter_restores_upstream_launch_class(self) -> None:
        observed: list[type] = []

        def capture_constructor(_self: BlackOutEnv, *args: object, **kwargs: object) -> None:
            observed.append(_UPSTREAM_ENV_MODULE.UnityEnvironment)

        original = _UPSTREAM_ENV_MODULE.UnityEnvironment
        with patch.object(BlackOutEnv, "__init__", capture_constructor):
            ContractBlackOutEnv(env_path="unused", background=True)

        self.assertEqual(observed, [_BackgroundUnityEnvironment])
        self.assertIs(_UPSTREAM_ENV_MODULE.UnityEnvironment, original)


if __name__ == "__main__":
    unittest.main()
