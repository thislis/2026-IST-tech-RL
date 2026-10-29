"""Mocked cancellation, cached-artifact and fixed-wave checks; no Unity."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import torch
from blackout_v12 import collector, evaluation, training
from blackout_v12.io import atomic_json, atomic_torch, digest, read_json


def result(task):
    return dict(valid=True,natural_terminal=True,episode_id=task['id'],opponent=task['opponent'],
                side=task['side'],replicate=task['replicate'],split=task['split'],
                requested_seed=task['requested_seed'],initial_map_hash='map',
                initial_observation_sha256='observation',raw_points=1.,score_valid=True,score_points=1.)


class ResumeWorkerTests(unittest.TestCase):
    @staticmethod
    def config(root):
        return dict(splits={'dev':[1,100],'train':[101,1000]},stop_file=str(root/'STOP'),
                    workers_per_run=2,actor_threads=1,map_split_registry=str(root/'maps.json'))

    def test_interrupted_cells_retry_and_real_failure_is_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); cfg=self.config(root); output=root/'evaluation'
            calls=[]
            def worker(task):
                calls.append(task['id'])
                if len(calls)==2: raise ValueError('genuine invalid transition')
                if len(calls)==3: raise InterruptedError('evaluation interrupted')
                return result(task)
            with patch.object(evaluation,'_evaluate_worker',side_effect=worker):
                with self.assertRaises(InterruptedError):
                    evaluation.evaluate(output,cfg,None,'Q2',[{'id':'noop'}],2,'dev')
            self.assertEqual(len(calls),3)
            cells=output/'cells'
            self.assertEqual(len(list(cells.glob('*.json'))),2)
            failure=cells/(calls[1]+'.json'); failure_bytes=failure.read_bytes()
            plan_bytes=(output/'plan.json').read_bytes()
            summary=read_json(output/'summary.json')
            self.assertFalse(summary['completeness'])
            self.assertEqual(summary['pending_games'],2)
            self.assertEqual(len((output/'interrupted_attempts.jsonl').read_text().splitlines()),1)
            newer=dict(cfg,workers_per_run=4,rollout_episodes_per_wave=2,actor_threads=2)
            with patch.object(evaluation,'_evaluate_worker',side_effect=result) as resumed, \
                    patch.object(evaluation,'register_map'):
                report=evaluation.evaluate(output,newer,None,'Q2',[{'id':'noop'}],2,'dev')
            self.assertEqual(resumed.call_count,2)
            self.assertEqual({call.args[0]['id'] for call in resumed.call_args_list},
                             {'noop_r001_s0','noop_r001_s1'})
            self.assertEqual(report['failed_games'],1)
            self.assertEqual(failure.read_bytes(),failure_bytes)
            self.assertEqual((output/'plan.json').read_bytes(),plan_bytes)

    def test_stop_before_submission_leaves_all_cells_pending(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cfg=self.config(root);Path(cfg['stop_file']).touch()
            with patch.object(evaluation,'_evaluate_worker') as worker:
                with self.assertRaises(InterruptedError):
                    evaluation.evaluate(root/'evaluation',cfg,None,'Q2',[{'id':'noop'}],2,'dev')
            worker.assert_not_called()
            report=read_json(root/'evaluation'/'summary.json')
            self.assertEqual(report['games'],0)
            self.assertEqual(report['pending_games'],4)
            self.assertFalse(report['completeness'])

    def test_parallel_stop_keeps_completed_cell_and_submits_no_more(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cfg=self.config(root);calls=[];release=threading.Event()
            def worker(task):
                calls.append(task['id'])
                if task['side']==0:
                    if not release.wait(5): raise RuntimeError('test synchronization failed')
                    return result(task)
                Path(cfg['stop_file']).touch();release.set()
                raise InterruptedError('evaluation interrupted')
            def pool(**kwargs): return ThreadPoolExecutor(max_workers=kwargs['max_workers'])
            with patch.object(evaluation,'ProcessPoolExecutor',side_effect=pool), \
                    patch.object(evaluation,'_evaluate_worker',side_effect=worker):
                with self.assertRaises(InterruptedError):
                    evaluation.evaluate(root/'evaluation',cfg,None,'Q2',[{'id':'noop'}],3,'dev',workers=2)
            self.assertEqual(len(calls),2)
            saved=read_json(root/'evaluation'/'rows.json')
            self.assertEqual(len(saved),1)
            self.assertTrue(saved[0]['valid'])
            self.assertEqual(read_json(root/'evaluation'/'summary.json')['pending_games'],5)

    def test_old_interruption_classification_is_exact(self):
        self.assertTrue(evaluation.interrupted_row({'valid':False,'error':'evaluation interrupted'}))
        self.assertTrue(evaluation.interrupted_row({'valid':False,'type':'CancelledError'}))
        self.assertFalse(evaluation.interrupted_row({'valid':False,'type':'ValueError',
                                                     'error':'evaluation interrupted','interrupted':True}))
        self.assertFalse(evaluation.interrupted_row({'valid':False,'error':'evaluation watchdog is not a draw'}))

    def test_scheduler_can_grow_allocation_within_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'allocation.json'
            cfg=dict(worker_allocation_file=str(path),evaluation_worker_capacity=8)
            self.assertEqual(evaluation._worker_limit(cfg,4),4)
            atomic_json(path,{'workers':8})
            self.assertEqual(evaluation._worker_limit(cfg,4),8)
            atomic_json(path,{'workers':100})
            self.assertEqual(evaluation._worker_limit(cfg,4),8)

    def test_legacy_cached_episode_survives_resource_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cfg=self.config(root)
            task=dict(id='one',directory=str(root/'episode'),config=cfg,action_seed=3,
                      policy_sha256='policy')
            directory=Path(task['directory']);directory.mkdir()
            atomic_json(directory/'task.json',task)
            episode=dict(natural_terminal=True,task_sha256=digest(task),policy_sha256='policy',
                         episode_id='one',events=[],trip_ledger={'rows':[]},env_steps=7)
            atomic_torch(directory/'episode.pt',episode)
            task_bytes=(directory/'task.json').read_bytes();episode_bytes=(directory/'episode.pt').read_bytes()
            newer=dict(task,config=dict(cfg,workers_per_run=4,rollout_episodes_per_wave=2,actor_threads=2))
            with patch('blackout_v12.env.make_env') as make_env:
                summary=collector.run_episode(newer)
            make_env.assert_not_called()
            self.assertEqual(summary['env_steps'],7)
            self.assertEqual((directory/'task.json').read_bytes(),task_bytes)
            self.assertEqual((directory/'episode.pt').read_bytes(),episode_bytes)
            with self.assertRaisesRegex(ValueError,'immutable episode task'):
                collector.run_episode(dict(newer,config=dict(newer['config'],minibatch_size=123)))

    def test_worker_count_does_not_change_wave_tasks_or_seed_serial(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            cfg=dict(self.config(root),rollout_episodes_per_wave=2,
                     opponents=[{'id':'scripted-noop'}],train_family_weights={'scripted':1.})
            observed=[]
            def wave(tasks,workers):
                observed.append((workers,[{key:task[key] for key in
                    ('id','side','family','requested_seed','action_seed')} for task in tasks]))
                summaries=[]
                for task in tasks:
                    path=Path(task['directory'])/'episode.pt'
                    atomic_torch(path,dict(policy_sha256=task['policy_sha256'],episode_id=task['id'],events=[]))
                    summaries.append(dict(trajectory=str(path),env_steps=1,event_count=0,
                                          family=task['family'],side=task['side']))
                return summaries
            with patch.object(training,'ROOT',root),patch.object(training,'run_wave',side_effect=wave):
                training.train(root/'run1',cfg,3,11,'Q2',workers=1)
                training.train(root/'run4',cfg,3,11,'Q2',workers=4)
            self.assertEqual([count for count,_ in observed],[1,1,4,4])
            self.assertEqual([tasks for _,tasks in observed[:2]],[tasks for _,tasks in observed[2:]])
            self.assertTrue(all(len(tasks)==2 for _,tasks in observed))
            for wave_index,(_,tasks) in enumerate(observed[:2]):
                for index,task in enumerate(tasks):
                    serial=wave_index*2+index
                    h=int(digest(['v12',11,serial])[:12],16)
                    self.assertEqual(task['action_seed'],h%(2**31-1))

    def test_legacy_checkpoint_signature_allows_resources_but_rejects_learning_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);output=root/'run'
            cfg=dict(self.config(root),opponents=[{'id':'scripted-noop'}],train_family_weights={'scripted':1.})
            def wave(tasks,workers):
                summaries=[]
                for task in tasks:
                    path=Path(task['directory'])/'episode.pt'
                    atomic_torch(path,dict(policy_sha256=task['policy_sha256'],episode_id=task['id'],events=[]))
                    summaries.append(dict(trajectory=str(path),env_steps=1,event_count=0,
                                          family=task['family'],side=task['side']))
                return summaries
            with patch.object(training,'ROOT',root),patch.object(training,'run_wave',side_effect=wave):
                training.train(output,cfg,1,11,'Q2',workers=2)
            latest=output/'latest.pt';payload=torch.load(latest,weights_only=True)
            payload['signature']=digest(dict(config=cfg,seed=11,arm='Q2',budget=1,initial=None))
            atomic_torch(latest,payload);original=latest.read_bytes()
            newer=dict(cfg,workers_per_run=4,rollout_episodes_per_wave=2,actor_threads=2)
            with patch.object(training,'ROOT',root),patch.object(training,'run_wave') as worker:
                resumed=training.train(output,newer,1,11,'Q2',workers=4)
                worker.assert_not_called()
                self.assertEqual(resumed['episodes'],2)
                self.assertEqual(latest.read_bytes(),original)
                with self.assertRaisesRegex(ValueError,'training configuration'):
                    training.train(output,dict(newer,ppo_epochs=3),1,11,'Q2',workers=4)
            self.assertEqual(latest.read_bytes(),original)


if __name__=='__main__': unittest.main()
