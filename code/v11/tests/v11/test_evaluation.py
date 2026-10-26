import tempfile
from pathlib import Path
import unittest
from blackout_v11.io import atomic_json, append, read_json
from blackout_v11.statistics import summarize, economic_comparison
from blackout_v11.study import _gate, _map_audit, StudyGateError


def row(i,side=0):
    return dict(winner=0,visible_score_winner=1,side=side,replicate=i,
                initial_map_hash=f'map-{i}',opponent=dict(id='v4-test',family='planner'),
                delivery_proxy=1,inference_p95_ms=2.)


class EvaluationContracts(unittest.TestCase):
    def test_below_target_diagnostics_do_not_abort_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('G1','G2'):
                report=_gate(tmp,name,False,blocking=False,planner_macro=.775,reference_macro=.8893)
                self.assertFalse(report['passed'])
                self.assertFalse(read_json(Path(tmp)/'gates'/(name+'.json'))['blocking'])
            self.assertFalse((Path(tmp)/'summary.json').exists())

    def test_execution_and_final_selection_failures_still_abort(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('G0','technical_pilot_F','technical_pilot_evaluation','G3','confirmation'):
                with self.assertRaises(StudyGateError):
                    _gate(tmp,name,False)
                self.assertEqual(read_json(Path(tmp)/'summary.json')['stopped_at_gate'],name)

    def test_raw_and_visible_score_remain_separate(self):
        report=summarize([row(i,s) for i in range(10) for s in (0,1)])
        self.assertEqual(report['score_rate'],1.)
        self.assertEqual(report['visible_score_rate'],0.)
        self.assertEqual(report['unique_maps'],10)
        self.assertLess(report['map_interval'][0],1.)

    def test_economic_comparison_requires_actual_matched_maps(self):
        rows=[row(i,s) for i in range(12) for s in (0,1)]
        result=economic_comparison(rows,rows)
        self.assertTrue(result['passed'])
        baseline=[dict(r,initial_map_hash='different-'+r['initial_map_hash']) for r in rows]
        self.assertFalse(economic_comparison(rows,baseline)['passed'])

    def test_train_map_audit_reads_worker_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cells=root/'dev/cells';cells.mkdir(parents=True)
            atomic_json(cells/'one.json',row(0))
            append(root/'runs/F_s11/episodes.jsonl',dict(result=dict(initial_map_hash='map-0')))
            with self.assertRaises(StudyGateError):
                _map_audit(root,dict(dev=[dict(rows_path=str(cells))]),strict=True)


if __name__=='__main__': unittest.main()
