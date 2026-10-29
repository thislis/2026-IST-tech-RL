"""Adaptive real-Unity probe orchestration checks; Unity is mocked here."""
from contextlib import ExitStack, nullcontext
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from blackout_v12 import throughput_probe as probe
from blackout_v12.io import atomic_json, read_json


class ProbeScalingTests(unittest.TestCase):
    def sweep(self, rates, allowed=lambda workers: True, **options):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            study = root/'logs/v12/test_study'
            study.mkdir(parents=True)
            source = root/'code/v12/blackout_v12/throughput_probe.py'
            source.parent.mkdir(parents=True)
            source.write_text('# test source\n', encoding='utf-8')
            attempted = []

            def candidate(output, templates, workers, seconds, warmup, collector_workload=False):
                attempted.append(workers)
                self.assertEqual(seconds, 12.)
                self.assertTrue(collector_workload)
                return dict(workers=workers, aggregate_env_steps_per_second=rates[workers],
                            resources={}, initial_map_hash='same-map',
                            initial_observation_sha256='same-observation')

            stack.enter_context(patch.object(probe, 'ROOT', root))
            stack.enter_context(patch.object(probe, '__file__', str(source)))
            stack.enter_context(patch.object(probe, '_templates', return_value=(
                [{'arm':'Q2'}], [{'path':'immutable-plan'}], [23, 11, 0, 'scripted-collector', 'map'])))
            stack.enter_context(patch.object(probe, '_candidate', side_effect=candidate))
            stack.enter_context(patch.object(probe, '_memory_check', side_effect=lambda n, reports:
                                           dict(allowed=allowed(n), requested_workers=n)))
            stack.enter_context(patch.object(probe, 'FileLease', side_effect=lambda *a, **k:nullcontext()))
            stack.enter_context(patch.object(probe, 'process_tree_guard', side_effect=nullcontext))
            stack.enter_context(patch('blackout_v12.runner.is_active', return_value=False))
            stack.enter_context(patch('builtins.print'))
            path = probe.run('test_study', scale_up=True, probe_arm='Q2', **options)
            return attempted, read_json(path)

    def test_sweep_passes_twelve_cores_and_stops_after_two_plateaus(self):
        counts, summary = self.sweep({8:100, 10:125, 12:130, 14:130.5, 16:129})
        self.assertEqual(counts, [8, 10, 12, 14, 16])
        self.assertEqual(summary['measured_best_workers'], 14)
        self.assertEqual(summary['selected_workers'], 14)
        self.assertEqual(summary['stop_reason'], 'two_consecutive_candidates_improved_less_than_2_percent')
        self.assertIsNone(summary['maximum_workers'])
        self.assertEqual(len(summary['candidates']), 5)

    def test_memory_bound_prevents_next_launch_and_retains_measurements(self):
        counts, summary = self.sweep({8:100, 10:130}, allowed=lambda n:n < 12)
        self.assertEqual(counts, [8, 10])
        self.assertEqual(summary['selected_workers'], 10)
        self.assertIn('safe_resource_bound', summary['stop_reason'])
        self.assertEqual(summary['memory_checks'][-1]['requested_workers'], 12)

    def test_severe_regression_stops_immediately(self):
        counts, summary = self.sweep({8:100, 10:79})
        self.assertEqual(counts, [8, 10])
        self.assertEqual(summary['selected_workers'], 8)
        self.assertEqual(summary['stop_reason'], 'severe_throughput_regression_over_20_percent')

    def test_explicit_maximum_is_honored(self):
        counts, summary = self.sweep({8:100, 10:120, 12:140}, max_workers=12)
        self.assertEqual(counts, [8, 10, 12])
        self.assertEqual(summary['stop_reason'], 'maximum_requested_workers_reached')

    def test_one_percent_tie_prefers_more_workers_but_five_percent_does_not(self):
        rows = [dict(workers=n, aggregate_env_steps_per_second=rate)
                for n,rate in [(8,100), (10,99.5), (12,96)]]
        selected = probe.select_workers(rows)
        self.assertEqual(selected['measured_best_workers'], 8)
        self.assertEqual(selected['selected_workers'], 10)

    def test_memory_uses_measured_python_unity_peak_with_headroom(self):
        fake_psutil = SimpleNamespace(virtual_memory=lambda:SimpleNamespace(available=25*probe.GIB),
                                      cpu_count=lambda logical:2)
        reports = [dict(workers=8, resources={'sum_process_lifetime_peak_rss_bytes':16*probe.GIB})]
        with patch.dict(sys.modules, {'psutil':fake_psutil}):
            check = probe._memory_check(10, reports)
        self.assertEqual(check['reserve_bytes'], 4*probe.GIB)
        self.assertEqual(check['estimated_worker_bytes'], int(2*probe.GIB*1.2))
        self.assertEqual(check['safe_max_workers'], 8)
        self.assertFalse(check['allowed'])
        # The logical CPU count is recorded, not used as a concurrency cap.
        self.assertEqual(check['logical_cores'], 2)

    def test_q2_uses_latest_immutable_data_scripted_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study = root/'logs/v12/test_study'
            plan = root/'artifacts/data/v12/test_study/runs/data/wave_00007/plan.json'
            original = dict(id='q2-data', arm='Q2', requested_seed=123, action_seed=456,
                            side=1, config={'stop_file':'study-stop'},
                            opponent={'id':'scripted-collector', 'family':'scripted'})
            atomic_json(plan, {'tasks':[original]})
            before = plan.read_bytes()
            with patch.object(probe, 'ROOT', root):
                templates, sources, sample = probe._templates(study, 'Q2')
            self.assertEqual(templates[0], original)
            self.assertEqual(sample[:3], (123, 456, 1))
            self.assertEqual(sources[0]['path'], str(plan))
            self.assertEqual(plan.read_bytes(), before)
            templates[0]['config'].pop('stop_file')
            self.assertEqual(plan.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
