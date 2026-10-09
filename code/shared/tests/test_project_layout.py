"""Regression checks for physical layout, imports and archived artifact paths."""
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from project_paths import project_path, project_root

ROOT = project_root()


class LayoutTests(unittest.TestCase):
    def test_versioned_packages_import_from_physical_directories(self):
        for module, version in [('blackout_rl.mappo_curriculum_v5', 'v5'),
                                ('blackout_rl.mappo_v6', 'v6'),
                                ('blackout_rl.v7_registry', 'v7'),
                                ('blackout_rl.v8.config', 'v8'),
                                ('blackout_v9.policy', 'v9')]:
            source = Path(importlib.import_module(module).__file__)
            self.assertTrue(source.is_relative_to(ROOT / 'code' / version))
            self.assertFalse(source.is_symlink())

    def test_historical_paths_resolve_without_creating_aliases(self):
        examples = {
            'checkpoints/win_70_vs_scripted.pt': 'artifacts/checkpoints/pre_v1/win_70_vs_scripted.pt',
            'scripts/v9_experiments.py': 'code/v9/scripts/v9_experiments.py',
            'reports/v7/main_study_preregistration.md': 'docs/v7/reports/main_study_preregistration.md',
        }
        for old, new in examples.items():
            self.assertEqual(project_path(old, root=ROOT), ROOT / new)
            self.assertEqual(project_path(ROOT / old), ROOT / new)
            self.assertFalse((ROOT / old).exists())
        with tempfile.TemporaryDirectory() as folder:
            external = Path(folder) / 'checkpoints/model.pt'
            self.assertEqual(project_path(external), external)

    def test_cli_bootstraps_outside_project_without_pythonpath(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTHONPATH', None)
        with tempfile.TemporaryDirectory() as folder:
            run = subprocess.run([sys.executable, str(ROOT/'code/v9/scripts/v9_experiments.py'), '--help'],
                                 cwd=folder, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn('--status', run.stdout)
