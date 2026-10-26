"""Evaluation reduction contracts; all match execution is mocked."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from blackout_v11.io import read_json, atomic_json, digest
from blackout_v11.evaluation import evaluate_bundle
from blackout_v11.evaluation_schedule import build_evaluation_schedule, record_evaluation_schedule
from blackout_v11.opponents import dev_catalog, frozen_catalog, reference_spec
from blackout_v11.study import run_study


def config():
    return read_json(Path(__file__).parents[2]/'configs/default.json')


def fake_matches(tasks, *args, **kwargs):
    return [dict(id=t['id'],valid=True,result=dict(
        valid=True,natural_terminal=True,winner=0,visible_score_winner=0,
        side=t['side'],replicate=t['replicate'],opponent=t['opponent'],
        initial_map_hash=f"{t['split']}-{t['opponent']['id']}-{t['replicate']}",
        requested_seed=t['requested_seed'],delivery_proxy=1,inference_p95_ms=1.)) for t in tasks]


class FakePolicy:
    arm='F'
    decoder='greedy'
    max_interventions=1
    goal_period=25
    residual_period=4
    use_memory=True
    def eval(self): return self
    def load_state_dict(self, state): return None
    def parameters(self): return iter([torch.zeros(1)])


class EvaluationScheduleTests(unittest.TestCase):
    def test_counts_and_training_budget_remain_explicit_and_separate(self):
        cfg=config(); original=copy.deepcopy(cfg)
        schedule=build_evaluation_schedule(cfg)
        self.assertEqual(cfg,original)
        self.assertEqual(schedule['main_dev']['total_games'],1800)
        self.assertEqual(schedule['confirmation']['total_games'],1000)
        self.assertEqual(schedule['frozen_test']['submission_games'],500)
        self.assertEqual(schedule['frozen_test']['reference_games'],440)
        self.assertEqual(schedule['total_remaining_evaluation_games'],3740)
        self.assertEqual(schedule['confirmation']['training_seeds'],[11,22,33,44,55])
        self.assertFalse(schedule['confirmation']['full_league_G4_G5_G6_applied'])
        self.assertFalse(schedule['confirmation']['performance_gate_blocking'])
        budget=cfg['teacher_steps']+4*cfg['pilot_steps']+16*cfg['training_steps']
        self.assertEqual(budget,34340864)

    def test_recording_schedule_never_rewrites_the_learning_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            original=b'{"schema":"v11.study.2","total_learning_steps":34340864}\n'
            (root/'task_plan.json').write_bytes(original)
            schedule,checksum=record_evaluation_schedule(root,config())
            self.assertEqual((root/'task_plan.json').read_bytes(),original)
            self.assertEqual(checksum,digest(schedule))
            self.assertEqual(record_evaluation_schedule(root,config()),(schedule,checksum))
            changed=config(); changed['seeds']=[11,22]
            with self.assertRaisesRegex(ValueError,'evaluation schedule changed'):
                record_evaluation_schedule(root,changed)

    def test_original_default_plan_shape_and_completed_cells_are_reused(self):
        cfg=config(); pool=dev_catalog()[:2]
        with tempfile.TemporaryDirectory() as tmp:
            with patch('blackout_v11.runtime.run_tasks',side_effect=fake_matches) as execute:
                first=evaluate_bundle(None,tmp,cfg,replicates=2,split='baseline',opponents=pool)
                self.assertEqual(execute.call_count,1)
            plan_path=Path(tmp)/'plan.json'; original=plan_path.read_bytes()
            plan=read_json(plan_path)
            self.assertEqual(set(plan),{'tasks','split'})
            self.assertEqual(len(plan['tasks']),8)
            self.assertEqual(set(plan['tasks'][0]),{
                'id','kind','config','bundle','bundle_hashes','candidate_spec','opponent','side',
                'replicate','requested_seed','action_seed','split','audit_windows'})
            with patch('blackout_v11.runtime.run_tasks') as execute:
                second=evaluate_bundle(None,tmp,cfg,replicates=2,split='baseline',opponents=pool)
                execute.assert_not_called()
            self.assertEqual(plan_path.read_bytes(),original)
            self.assertEqual(first,second)
            self.assertNotIn('requested_map_pairs_by_opponent',second)

    def test_final_candidate_and_reference_share_exact_440_cells(self):
        cfg=config(); schedule=build_evaluation_schedule(cfg)
        counts=schedule['frozen_test']['map_pairs_by_opponent']
        with tempfile.TemporaryDirectory() as tmp, patch('blackout_v11.runtime.run_tasks',side_effect=fake_matches):
            root=Path(tmp)
            candidate=evaluate_bundle(None,root/'candidate',cfg,replicates=10,split='test',
                replicates_by_opponent=counts,evaluation_scope='locked_submission')
            baseline=evaluate_bundle(None,root/'reference',cfg,replicates=10,split='test',
                candidate_spec=reference_spec(),replicates_by_opponent=counts,evaluation_scope='reference')
            self.assertEqual((candidate['n'],baseline['n']),(500,440))
            self.assertIsNone(candidate['requested_map_pairs'])
            self.assertEqual(candidate['requested_map_pairs_by_opponent'],counts)
            candidate_tasks=read_json(root/'candidate/plan.json')['tasks']
            baseline_tasks=read_json(root/'reference/plan.json')['tasks']
            candidate_ids={t['id']:t for t in candidate_tasks}
            for task in baseline_tasks:
                other=candidate_ids[task['id']]
                self.assertEqual((task['opponent'],task['replicate'],task['side'],task['requested_seed'],task['action_seed']),
                                 (other['opponent'],other['replicate'],other['side'],other['requested_seed'],other['action_seed']))
            self.assertEqual(len(baseline_tasks),440)
            self.assertFalse(any(t['opponent']['id']==reference_spec()['id'] for t in baseline_tasks))

    def test_poor_confirmation_is_diagnostic_then_only_locked_candidate_gets_final_gates(self):
        cfg=config(); evaluations=[]; trainings=[]
        def train(path,configuration,arm,seed,registration,device,**kwargs):
            trainings.append((arm,seed,kwargs.get('pilot',False),copy.deepcopy(configuration)))
            return FakePolicy(),dict(complete=True,technical_passed=True)
        def evaluate(bundle,path,configuration,replicates=None,split='dev',*,opponents=None,
                     candidate_spec=None,replicates_by_opponent=None,evaluation_scope=None,**kwargs):
            pool=frozen_catalog() if opponents is None else opponents
            if candidate_spec: pool=[s for s in pool if s['id']!=candidate_spec['id']]
            reps=configuration[split+'_replicates'] if replicates is None else replicates
            counts={s['id']:(replicates_by_opponent or {}).get(s['id'],reps) for s in pool}
            evaluations.append(dict(path=str(path),split=split,pool=[s['id'] for s in pool],counts=counts,
                                    scope=evaluation_scope,bundle=str(bundle)))
            rate=0. if split=='confirmation' else .95
            return dict(complete=True,n=2*sum(counts.values()),invalid=0,
                        opponents={s['id']:dict(score_rate=rate) for s in pool},
                        opponent_macro=rate,family_macro=rate,worst_side=rate,inference_p95_ms=1.,
                        rows_path=str(Path(path)/'cells'))
        def rows(report):
            return [dict(side=s,delivery_proxy=1,initial_map_hash=f'map-{s}') for s in (0,1)]
        with tempfile.TemporaryDirectory() as tmp, \
             patch('blackout_v11.training.train_run',side_effect=train), \
             patch('blackout_v11.demonstrations.prepare_demonstrations',return_value=('teacher',{'complete':True})), \
             patch('blackout_v11.study._policy',return_value=FakePolicy()), \
             patch('blackout_v11.study._bundle',side_effect=lambda policy,path,*args:Path(path)), \
             patch('blackout_v11.study.ensure_export',side_effect=lambda policy,path:Path(path)), \
             patch('blackout_v11.study.verify_export',return_value={'passed':True}), \
             patch('blackout_v11.study.export_onnx',return_value={'passed':True}), \
             patch('blackout_v11.study.file_hash',return_value='frozen'), \
             patch('blackout_v11.study.evaluate_bundle',side_effect=evaluate), \
             patch('blackout_v11.study.load_rows',side_effect=rows), \
             patch('blackout_v11.study._map_audit',return_value={}), \
             patch('blackout_v11.study.final_gates',return_value={g:{'passed':True} for g in ('G4','G5','G6')}) as gates, \
             patch('blackout_v11.policy.MyPolicy',FakePolicy), \
             patch('blackout_v11.study.torch.load',return_value={'policy_state':{}}):
            report=run_study(tmp,cfg,'original-registration')
            self.assertTrue(report['complete'])
            self.assertEqual(len(trainings),20)
            self.assertEqual(sum(c['pilot_steps'] if pilot else c['training_steps'] for _,_,pilot,c in trainings)
                             +cfg['teacher_steps'],34340864)
            self.assertTrue(all(c==cfg for _,_,_,c in trainings))
            main=[e for e in evaluations if e['scope']=='main_dev_15_opponents_5_map_pairs_selection']
            self.assertEqual(len(main),12)
            self.assertEqual(sum(2*sum(e['counts'].values()) for e in main),1800)
            confirmation=[e for e in evaluations if e['split']=='confirmation']
            self.assertEqual(len(confirmation),10)
            self.assertEqual(sum(2*sum(e['counts'].values()) for e in confirmation),1000)
            final=[e for e in evaluations if e['split']=='test']
            self.assertEqual(sorted(2*sum(e['counts'].values()) for e in final),[440,500])
            self.assertEqual(gates.call_count,1)
            self.assertEqual(len(report['frozen_test']),1)
            self.assertFalse(report['all_seed_full_league_performance_assessed'])
            self.assertEqual(report['selection']['registration_sha256'],'original-registration')
            self.assertTrue(all(e['scope'] is None for e in evaluations if 'pilot/' in e['path'] or e['split']=='baseline'))
            self.assertTrue(all(set(e['counts'].values())=={cfg['pilot_replicates']} for e in evaluations if 'pilot/' in e['path']))
            diagnostic=read_json(Path(tmp)/'gates/confirmation.json')
            self.assertFalse(diagnostic['blocking'])
            self.assertIsNone(diagnostic['performance_passed'])
            self.assertTrue(read_json(Path(tmp)/'gates/G3.json')['blocking'])


if __name__=='__main__': unittest.main()
