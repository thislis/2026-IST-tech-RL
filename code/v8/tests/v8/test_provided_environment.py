"""Restoration checks use mocks and files only; no actual Unity launches."""

from project_paths import project_root, project_path

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil
import unittest
from unittest.mock import Mock, patch

from blackout_rl.v8 import provided_environment as original
from blackout_rl.v8.config import load_config


class ProvidedEnvironmentTests(unittest.TestCase):
    def test_background_launcher_keeps_graphics_and_forwards_arguments(self):
        from mlagents_envs.env_utils import validate_environment_path
        self.assertEqual(validate_environment_path(str(original.RENDERED_LAUNCHER)),str(original.LAUNCHER_EXECUTABLE))
        with tempfile.TemporaryDirectory(prefix='v8 launcher ') as tmp:
            root=Path(tmp);launcher=root/original.LAUNCHER_EXECUTABLE.relative_to(original.ROOT)
            launcher.parent.mkdir(parents=True);shutil.copy2(original.LAUNCHER_EXECUTABLE,launcher)
            marker=root/'code/shared/project_paths.py'
            marker.parent.mkdir(parents=True,exist_ok=True);marker.touch()
            player=root/'artifacts/builds/BlackOut.app/Contents/MacOS/RLGame2026'
            player.parent.mkdir(parents=True)
            player.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n');player.chmod(0o755)
            result=subprocess.run([str(launcher),'--mlagents-port','28400','-logFile','a b.log'],capture_output=True,text=True,check=True)
            self.assertEqual(result.stdout.splitlines(),['-batchmode','--mlagents-port','28400','-logFile','a b.log'])
            for flag in ('-nographics','-no-graphics','-force-gfx-null'):
                result=subprocess.run([str(launcher),flag],capture_output=True,text=True)
                self.assertEqual(result.returncode,64);self.assertEqual(result.stdout,'')

    def test_original_sources_and_installed_api_are_unchanged(self):
        report=original.verify_original()
        self.assertTrue(report['provided_api_unchanged'])
        self.assertFalse(report['research_environment_enabled'])
        self.assertFalse(report['training_ready'])
        self.assertEqual(report['executable_sha256'],'49172f88b1ba2429677e7ec4a7876c4589876090be4888aeaef39269e29f4a69')

    def test_factory_returns_provided_instance_with_exact_reset_step_results(self):
        import blackout_env
        sentinel=Mock();observation={'vector':object(),'graphic':object()}
        reset_result=(observation,object());step_result=(observation,object(),object(),object(),object())
        sentinel.reset.return_value=reset_result;sentinel.step.return_value=step_result
        with patch.object(original,'verify_original'),patch.object(blackout_env,'BlackOutEnv',return_value=sentinel) as cls:
            env=original.make_original_env(time_scale=50,worker_id=2,base_port=26000)
            self.assertIs(env,sentinel)
            cls.assert_called_once_with(env_path=str(original.RENDERED_LAUNCHER),no_graphics=False,
                                        time_scale=50,worker_id=2,base_port=26000)
            env.reset.assert_not_called();env.step.assert_not_called()
            self.assertIs(env.reset(seed=123),reset_result)
            self.assertIs(env.step({'action':0}),step_result)
            env.reset.assert_called_once_with(seed=123);env.step.assert_called_once_with({'action':0})

    def test_custom_channel_and_fixture_paths_cannot_launch(self):
        from blackout_rl.v8.environment import UnityResearchEnv,make_channel
        from blackout_rl.v8.live_validation import validate
        with patch('subprocess.Popen',side_effect=AssertionError('must not start processes')):
            for fn,args in [(UnityResearchEnv,()),(make_channel,()),(validate,({},))]:
                with self.assertRaisesRegex(RuntimeError,'철회'):fn(*args)

    def test_active_configs_use_original_game_without_native_overlay(self):
        for path in (project_path('code/v8/configs', root=original.ROOT)).rglob('*.yaml'):
            if path.name=='protocol.yaml':continue
            cfg=load_config(path,require_build=False)
            self.assertEqual(cfg['environment']['build'],'builds/BlackOut.app')
            self.assertEqual(cfg['environment']['timeout_seconds'],0)
            self.assertEqual(cfg['runtime']['backend'],'original')
            self.assertEqual(cfg['evaluation']['primary'],'provided_api_unverified')

    def test_retired_entrypoints_refuse_execution_from_external_directory(self):
        env=dict(os.environ);env.pop('PYTHONPATH',None)
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('prepare_v8_unity.py','register_v8_build.py','validate_v8_timer.py'):
                result=subprocess.run([sys.executable,str(project_path(Path('scripts') / name, root=original.ROOT))],cwd=tmp,env=env,
                                      capture_output=True,text=True,timeout=60)
                self.assertEqual(result.returncode,2,result.stderr)
                self.assertIn('철회',result.stderr)
        self.assertFalse((project_path('unity/v8', root=original.ROOT)).exists())
        self.assertFalse((project_path('artifacts/build/v8_unity', root=original.ROOT)).exists())
        self.assertEqual(list((project_path('artifacts/builds', root=original.ROOT)).glob('BlackOut-v8*.app')),[])

    def test_model_code_was_not_rewritten_to_accommodate_rollback(self):
        record=json.loads(original.MANIFEST.read_text())
        for name,h in record['model_files_before'].items():
            from project_paths import verify_relocated_source
            verify_relocated_source(name,h)


if __name__=='__main__':unittest.main()
