"""Launcher checks and map exclusion must not start experiments."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from blackout_v12 import finetune_runner as runner
from blackout_v12.io import atomic_json, file_hash, read_json
from blackout_v12.map_splits import MapSplitCollision, register_map


class FineTuneLauncherTests(unittest.TestCase):
    def test_new_splits_do_not_reuse_source_study_ranges(self):
        cfg = runner.configuration(runner.CONFIG)
        old = read_json(runner.ROOT / 'code/v12/configs/splits.json')
        for name in ('train', 'dev', 'confirmation', 'test'):
            interval = cfg['splits'][name]
            for previous in ('train', 'data', 'dev', 'confirmation', 'test'):
                self.assertGreater(interval[0], old[previous][1])
        self.assertEqual(len(cfg['opponents']), 9)
        self.assertEqual(cfg['ppo']['minibatch_events'], 128)

    def test_dry_run_never_starts_or_registers_an_experiment(self):
        cfg = runner.configuration(runner.CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / 'must_not_exist'
            with patch.object(runner, 'preflight', return_value={'ready': True}), \
                 patch.object(runner, 'configuration', return_value=cfg), \
                 patch('blackout_v12.runner.directory_for', return_value=target), \
                 patch.object(runner, 'worker') as worker, \
                 patch.object(runner, 'supervise') as supervisor, \
                 patch.object(runner.subprocess, 'Popen') as launch, \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.main(['--dry-run'])
            worker.assert_not_called()
            supervisor.assert_not_called()
            launch.assert_not_called()
            self.assertFalse(target.exists())

    def test_prior_physical_maps_are_excluded_without_editing_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            prior = directory / 'prior.json'
            atomic_json(prior, dict(maps={'old-layout': dict(split='dev')}))
            original = prior.read_bytes()
            cfg = dict(prior_map_registry=str(prior), map_split_registry=str(directory / 'map_splits.json'))
            manifest = dict(prior_map_registry_sha256=file_hash(prior))
            runner.prepare_map_registry(directory, cfg, manifest)
            with self.assertRaises(MapSplitCollision):
                register_map(cfg, 'dev', 'old-layout', 'observation', 5100001, 0)
            self.assertTrue(register_map(cfg, 'dev', 'new-layout', 'observation', 5100002, 0)['valid'])
            runner.prepare_map_registry(directory, cfg, manifest)
            self.assertIn('new-layout', read_json(directory / 'map_splits.json')['maps'])
            self.assertEqual(prior.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
