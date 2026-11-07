import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from blackout_v13.io import ROOT,read_json,atomic_json,atomic_torch,digest,file_hash
from blackout_v13.evaluation import schedule,cached_matches,seed_cache,evaluate,MapRegistry
from blackout_v13.migrate_evaluation import (continuation_configuration,retarget_checkpoint,
                                            validate_source_changes,best_common_screen,prepare)
from blackout_v13.study import Study,run_configuration


def record(task,score=1.):
    return dict(id=task['id'],seed=task['seed'],side=task['side'],opponent=task['opponent']['id'],
                baseline=task['baseline'],natural_terminal=True,map_sha256=str(task['seed']),
                score_rate=score,outcome=2*score-1)


class FakePool:
    def __init__(self):self.executed=[];self.score=1.;self.last_weights=None
    def evaluate(self,tasks,weights,on_result,stop):
        self.last_weights=weights
        self.executed.extend(tasks)
        for task in tasks:on_result(record(task,self.score))
    def reset(self):pass


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.config=read_json(ROOT/'code/v13/configs/default.json')
        self.config.update(evaluation_interval=1048576,evaluation_opponents=[dict(id=f'o{i}') for i in range(22)])
        self.config.pop('intermediate_dev_pairs',None)

    def test_only_evaluation_configuration_changes(self):
        cfg=continuation_configuration(self.config,'new',{'run':'old'},{'identity':'old'})
        self.assertEqual(cfg['evaluation_interval'],2097152);self.assertEqual(cfg['intermediate_dev_pairs'],2)
        for k in ('teacher_steps','worker_steps','manager_steps','alternating_steps','seeds','dev_pairs','confirmation_pairs'):
            self.assertEqual(cfg[k],self.config[k])
        self.assertEqual(len(schedule(cfg['evaluation_opponents'],2,cfg['dev_seed_base'])),88)
        self.assertEqual(len(schedule(cfg['evaluation_opponents'],5,cfg['dev_seed_base'])),220)

    def test_training_state_is_preserved(self):
        payload=dict(config=self.config,identity='old',version=123,
                     policy={'w':torch.tensor([1.,2.])},critics={'w':torch.tensor([3.])},
                     optimizers={'step':torch.tensor(7)},torch_rng=torch.get_rng_state(),
                     progress=dict(stage=24,stage_steps=131072,env_steps=5242880,best_score=.8))
        new=retarget_checkpoint(payload,dict(self.config,name='new'),'new','old/latest.pt','hash')
        for k in ('policy','critics','optimizers','torch_rng'):self.assertIs(new[k],payload[k])
        self.assertEqual(new['progress'],payload['progress']);self.assertEqual(new['version'],payload['version'])
        new['progress']['best_score']=.5;self.assertEqual(payload['progress']['best_score'],.8)

    def test_non_evaluation_source_change_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_source_changes({'code/v13/blackout_v13/policy.py':'a'},{'code/v13/blackout_v13/policy.py':'b'})
        self.assertEqual(validate_source_changes({'code/v13/blackout_v13/study.py':'a'},
                                                 {'code/v13/blackout_v13/study.py':'b'}),['code/v13/blackout_v13/study.py'])

    def test_screen_expands_to_220_without_replaying_88(self):
        cfg=self.config;screen=schedule(cfg['evaluation_opponents'],2,cfg['dev_seed_base'])
        full=schedule(cfg['evaluation_opponents'],5,cfg['dev_seed_base'])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);registry=MapRegistry(root/'maps.sqlite3');pool=FakePool()
            for task in screen:atomic_json(root/'screen'/(task['id']+'.json'),record(task))
            seed_cache(full,[root/'screen'],root/'full')
            report,rows=evaluate(pool,full,b'',root/'full',registry,'dev')
            self.assertEqual(len(pool.executed),132);self.assertEqual(report['completed'],220)
            # Restart is entirely cached, including both paired sides.
            evaluate(pool,full,b'',root/'full',registry,'dev');self.assertEqual(len(pool.executed),132)
            self.assertEqual({(r['opponent'],r['seed'],r['side']) for r in rows},
                             {(t['opponent']['id'],t['seed'],t['side']) for t in full})
            registry.close()

    def test_common_screen_not_mixed_with_full_rate(self):
        cfg=dict(self.config,intermediate_dev_pairs=2)
        full=schedule(cfg['evaluation_opponents'],5,cfg['dev_seed_base'])
        screen_ids={t['id'] for t in schedule(cfg['evaluation_opponents'],2,cfg['dev_seed_base'])}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for version in (10,20):
                for t in full:
                    # v10 has the better 220-game rate but v20 has the better common 88.
                    score=(.5 if t['id'] in screen_ids else 1.) if version==10 else (.75 if t['id'] in screen_ids else 0.)
                    atomic_json(root/'dev'/str(version)/(t['id']+'.json'),record(t,score))
            selected,_=best_common_screen(cfg,root,{10:root/'ten.pt',20:root/'twenty.pt'})
            self.assertEqual(selected['version'],20);self.assertEqual(selected['score'],.75)

    def test_wrong_cached_side_is_rejected(self):
        task=schedule([dict(id='a')],1,100)[0]
        with tempfile.TemporaryDirectory() as d:
            row=record(task);row['side']=1
            atomic_json(Path(d)/(task['id']+'.json'),row)
            with self.assertRaises(ValueError):cached_matches([task],[Path(d)])

    def test_final_evaluation_uses_screen_winner_and_full_score(self):
        from types import SimpleNamespace
        from blackout_v13.io import cpu_state
        from blackout_v13.collector import deserialize
        with tempfile.TemporaryDirectory() as d,patch('blackout_v13.study.ROOT',Path(d)):
            root=Path(d);study=Study(dict(name='case'),root/'logs');pool=FakePool()
            cfg=dict(self.config,intermediate_dev_pairs=2)
            directory=root/'runs/main_B1_s11';directory.mkdir(parents=True)
            latest=root/'latest.pt';best=root/'best.pt';progress={'best_score':-1.}
            learner=SimpleNamespace(version=10,policy=torch.nn.Linear(1,1),critics={'worker':torch.nn.Linear(1,1)})
            atomic_torch(latest,dict(version=10,policy=cpu_state(learner.policy),critics={k:cpu_state(v) for k,v in learner.critics.items()}))
            study.dev_evaluation(pool,learner,cfg,directory,progress,latest,best)
            learner.version=20;pool.score=0.
            atomic_torch(latest,dict(version=20,policy=cpu_state(learner.policy),critics={k:cpu_state(v) for k,v in learner.critics.items()}))
            study.dev_evaluation(pool,learner,cfg,directory,progress,latest,best)
            self.assertEqual(torch.load(best,weights_only=True)['version'],10)
            pool.score=.5
            report=study.final_dev_evaluation(pool,cfg,directory,best)
            self.assertEqual(deserialize(pool.last_weights)['version'],10)
            self.assertEqual(report['completed'],220)
            self.assertAlmostEqual(report['macro_score_rate'],(88+132*.5)/220)
            self.assertEqual(progress['best_score'],1.)
            self.assertEqual(learner.version,20)
            self.assertEqual(len(pool.executed),88+88+132);study.registry.close()

    def test_preparation_does_not_launch_or_rewrite_source(self):
        from blackout_v13 import migrate_evaluation as migration
        from agent_league.new_game_runtime import runtime_spec
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'logs/v13/old';old_artifacts=root/'artifacts/checkpoints/v13/old'
            cfg=copy.deepcopy(self.config);cfg.update(name='old',arms=['B1'],seeds=[11],scope='full')
            sources={'code/v13/blackout_v13/policy.py':'fixed'};runtime=runtime_spec()
            manifest=dict(config=cfg,sources=sources,runtime=runtime);old_id=digest(manifest)
            atomic_json(source/'request.json',cfg);atomic_json(source/'manifest.json',dict(identity=old_id,**manifest))
            atomic_json(source/'status.json',dict(phase='stopped'))
            resources=dict(sequence_minibatch=8,selected_workers=8,selected_device='cuda')
            atomic_json(source/'resources/selected.json',resources)
            registry=MapRegistry(source/'map_splits.sqlite3');registry.add('trainmap','train');registry.close()
            data=root/'artifacts/data/v13/old/teacher'
            atomic_json(data/'dataset.json',dict(identity=old_id,steps=cfg['teacher_steps'],files=[]))
            for kind in ('pilot','main'):
                tag=f'{kind}_B1_s11';rc=run_configuration(cfg,'B1',11,resources)
                identity=digest(dict(study=old_id,config=rc,tag=tag))
                p=dict(config=rc,identity=identity,version=30,policy={'weight':torch.tensor([4.])},
                       critics={},optimizers={'step':torch.tensor(23)},torch_rng=torch.get_rng_state(),
                       progress=dict(completed=kind=='pilot',stage=24,stage_steps=131072,bc_phase=1,
                                     env_steps=5242880,best_score=.9,next_eval=5242880))
                for filename in ('latest.pt','best.pt'):atomic_torch(old_artifacts/tag/filename,p)
                if kind=='pilot':
                    atomic_json(source/'runs'/tag/'result.json',dict(run=tag,arm='B1',seed=11,pilot=True,
                        best_score=.8,checkpoint=str(old_artifacts/tag/'best.pt')))
                else:
                    for task in schedule(cfg['evaluation_opponents'],5,cfg['dev_seed_base']):
                        atomic_json(source/'runs'/tag/'dev/30'/(task['id']+'.json'),record(task))
            protected=[source/'request.json',source/'manifest.json',data/'dataset.json',old_artifacts/'main_B1_s11/latest.pt']
            before={p:file_hash(p) for p in protected}
            with patch.object(migration,'ROOT',root),patch.object(migration,'source_identity',return_value=sources),\
                 patch.object(migration,'preflight',return_value={'game_started':False}) as check,\
                 patch('blackout_v13.collector.Pool.resize',side_effect=AssertionError('must not launch')):
                receipt=prepare('old','new')
            self.assertFalse(receipt['experiment_started']);self.assertFalse(receipt['training_changed'])
            self.assertEqual(before,{p:file_hash(p) for p in protected})
            updated=torch.load(root/'artifacts/checkpoints/v13/new/main_B1_s11/latest.pt',weights_only=True)
            self.assertEqual(updated['progress']['env_steps'],5242880)
            self.assertEqual(updated['progress']['next_eval'],6291456)
            self.assertEqual(updated['progress']['best_score'],1.)
            torch.testing.assert_close(updated['policy']['weight'],torch.tensor([4.]))
            self.assertEqual(updated['optimizers']['step'].item(),23)
            self.assertEqual(read_json(root/'logs/v13/new/status.json')['phase'],'prepared')
            check.assert_called_once()


if __name__=='__main__':unittest.main()
