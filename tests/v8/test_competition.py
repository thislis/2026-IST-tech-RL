"""Competition-contract tests using synthetic games only, never Unity."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import shutil
from unittest.mock import patch
import numpy as np
import torch
from tests.test_mappo_v6 import all_observations
from blackout_rl.v8.competition.policy import MyPolicy
from blackout_rl.v8.competition.training import create_policy,ActorCritic,Collector,train,pack,joint_log_prob
from blackout_rl.v8.competition.exporting import export_policy,verify_export
from blackout_rl.v8.competition.evaluation import evaluate,ObservedPolicy
from blackout_rl.v8.competition import study,runner
from blackout_rl.v8.checkpoints import CheckpointStore,fingerprints


class Noop:
    def reset(self):pass
    def act(self,obs,names=None):return {n:np.zeros(2,np.float32) for n in (obs if names is None else names)}


class Game:
    def __init__(self,length=3,fail_after=None):
        self.possible_agents=[f'unit_{i}' for i in range(10)];self.length=length;self.fail=fail_after;self.closed=False;self.resets=0
    def reset(self,seed=None):
        self.resets+=1;self.requested_seed=seed;self.agents=list(self.possible_agents);self.steps=0
        self.obs=all_observations();return self.obs,{n:{} for n in self.agents}
    def step(self,actions):
        self.steps+=1
        if self.steps==self.fail:raise RuntimeError('synthetic transport crash')
        done=self.steps>=self.length
        rewards={n:float(int(n.split('_')[1])<5) for n in self.possible_agents}
        terms={n:done for n in self.possible_agents};truncs={n:False for n in self.possible_agents}
        infos={n:{'winner':1} for n in self.possible_agents}  # Deliberately differs from provided run_match's metric.
        if done:self.agents=[]
        return self.obs,rewards,terms,truncs,infos
    def close(self):self.closed=True


class CompetitionTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def config(self):
        cfg=copy.deepcopy(study.read(study.CONFIG)['training'])
        cfg.update(steps=8,rollout=4,save_every=4,seed=11,arm='c1',opponent={},side_cycle=[0,1])
        cfg['ppo'].update(epochs=1,minibatch_size=2)
        return cfg
    def model(self,arm='c1'):
        torch.manual_seed(11)
        return ActorCritic(create_policy(arm,11,study.ROOT/'checkpoints/win_70_vs_scripted.pt'))
    def collector(self,model,cfg,path,factory):
        return Collector(model,cfg,11,factory,path,opponent_factory=lambda *args:(Noop(),'fixture-opponent'))

    def test_tensor_contract_all_batches_no_state_or_batch_identity(self):
        p=self.model().policy.eval();obs=all_observations();v,g=pack(obs,list(obs)[:5]);before=v.clone(),g.clone()
        a=p(v,g);order=torch.tensor([3,1,4,0,2])
        torch.testing.assert_close(p(v[order],g[order]),a[order],rtol=0,atol=0)
        torch.testing.assert_close(torch.cat([p(v[i:i+1],g[i:i+1]) for i in range(5)]),a,rtol=0,atol=0)
        p(torch.zeros_like(v),torch.zeros_like(g));torch.testing.assert_close(p(v,g),a,rtol=0,atol=0)
        for n in (0,1,3,5):
            out=p(v[:n],g[:n]);self.assertEqual(out.shape,(n,2));self.assertEqual(out.dtype,torch.float32)
            self.assertTrue(torch.isfinite(out).all() and (out.abs()<=1).all())
        torch.testing.assert_close(v,before[0]);torch.testing.assert_close(g,before[1])
        with self.assertRaises(TypeError):p(v.double(),g)

    def test_c1_flat_capacity_and_probability_empty_mask(self):
        a=self.model('c1').policy;b=self.model('flat').policy
        self.assertEqual(sum(p.numel() for p in a.parameters() if p.requires_grad),sum(p.numel() for p in b.parameters() if p.requires_grad))
        features=torch.zeros(5,177);valid=torch.zeros(5,8,dtype=torch.bool)
        for p in (a,b):
            logs=p.log_distribution(features,valid)
            self.assertTrue(torch.equal(logs[:,0],torch.zeros(5)))
            torch.testing.assert_close(logs.exp().sum(-1),torch.ones(5))
            self.assertTrue(torch.equal(p.decode(logs),torch.zeros(5,dtype=torch.long)))

    def test_raw_api_collection_reward_ratio_rollout_and_fresh_episode(self):
        cfg=self.config();model=self.model();games=[]
        def factory():games.append(Game(3));return games[-1]
        with tempfile.TemporaryDirectory() as tmp:
            collector=self.collector(model,cfg,tmp,factory)
            batch,_=collector.collect(2);self.assertEqual(len(games),1)
            torch.testing.assert_close(batch['reward'],torch.ones(2))
            logs=model.policy.log_distribution(batch['features'],batch['valid'])
            torch.testing.assert_close(joint_log_prob(logs,batch['action']),batch['old_log_prob'],rtol=0,atol=0)
            batch,_=collector.collect(2)
            self.assertEqual(len(games),2);self.assertEqual([g.resets for g in games],[1,1]);self.assertTrue(games[0].closed)
            self.assertEqual(float(batch['next_value'][0]),0.);self.assertEqual(float(batch['reward'][1]),0.)
            collector.close()

    def test_ppo_frozen_encoder_checkpoint_resume_and_export_loader(self):
        cfg=self.config();model=self.model();encoder=copy.deepcopy(model.policy.encoder.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            c=self.collector(model,cfg,tmp,lambda:Game(100,6))
            with self.assertRaisesRegex(RuntimeError,'transport crash'):train(model,c,cfg,tmp,'fixture-registration')
            saved,_=CheckpointStore(Path(tmp)/'checkpoints').load();self.assertEqual(saved['global_step'],4)
            model=self.model();c=self.collector(model,cfg,tmp,lambda:Game(100))
            h=train(model,c,cfg,tmp,'fixture-registration');saved,_=CheckpointStore(Path(tmp)/'checkpoints').load(h)
            rows=[json.loads(x) for x in (Path(tmp)/'exposure.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in rows],list(range(1,9)))
            for k,v in encoder.items():torch.testing.assert_close(model.policy.encoder.state_dict()[k],v,rtol=0,atol=0)
            bundle=export_policy(saved,Path(tmp)/'submission');report=verify_export(bundle,saved['policy_state'])
            self.assertTrue(report['provided_loader']);self.assertFalse(report['live_match_executed'])
            self.assertEqual({p.name for p in bundle.iterdir() if p.is_file()},{'policy.py','checkpoint.pt'})

    def test_evaluation_uses_provided_runner_result_including_side_swap(self):
        cfg=self.config();cfg['max_episode_steps']=10;games=[]
        def factory():games.append(Game(2));return games[-1]
        with tempfile.TemporaryDirectory() as tmp,patch('blackout_rl.v8.competition.evaluation.HistoricalOpponent',side_effect=lambda *args:Noop()):
            result=evaluate(None,cfg,[0,1],factory,tmp,planner=True)
        self.assertEqual(result['n'],4);self.assertEqual(result['wins'],2)
        self.assertEqual([r['winner'] for r in result['episodes']],[0,1,0,1])
        self.assertTrue(all(g.closed and g.resets==1 for g in games))
        self.assertTrue(result['replicates_are_not_paired_maps'])

    def test_watchdog_is_invalid_and_does_not_create_result(self):
        cfg=self.config();cfg['max_episode_steps']=2
        with tempfile.TemporaryDirectory() as tmp,patch('blackout_rl.v8.competition.evaluation.HistoricalOpponent',side_effect=lambda *args:Noop()):
            with self.assertRaisesRegex(RuntimeError,'watchdog'):evaluate(None,cfg,[0],lambda:Game(100),tmp,planner=True)
            self.assertFalse((Path(tmp)/'result.json').exists())
            self.assertIn('invalid',(Path(tmp)/'attempts.jsonl').read_text())

    def test_untrained_export_rejected_and_original_study_jobs_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'untrained'):export_policy(dict(sources=fingerprints(),global_step=0),Path(tmp)/'out')
        r={'config':study.read(study.CONFIG)};jobs=study.tasks(r)
        self.assertEqual(len(jobs),84);self.assertEqual(sum(len(j.get('replicates',[]))*2 for j in jobs),780)
        self.assertNotIn('accelerated_pilot_v1',str(study.DIRECTORY))

    def test_flat_submission_loads_mode_from_checkpoint_and_keeps_only_two_files(self):
        policy=self.model('flat').policy
        with torch.no_grad():
            policy.gate[-1].bias.fill_(-8)
            policy.correction.bias.copy_(torch.arange(8,dtype=torch.float32))
        payload=dict(global_step=1,sources=fingerprints(),policy_state=policy.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            bundle=export_policy(payload,Path(tmp)/'submission')
            self.assertTrue(verify_export(bundle,payload['policy_state'])['passed'])
            self.assertEqual({p.name for p in bundle.iterdir()},{'policy.py','checkpoint.pt'})

    def test_saved_observation_batches_replay_through_isolated_provided_loader(self):
        cfg=self.config();model=self.model()
        with tempfile.TemporaryDirectory() as tmp:
            collector=self.collector(model,cfg,tmp,lambda:Game(100))
            try:collector.collect(8)
            finally:collector.close()
            fixture=Path(tmp)/'policy_inputs.pt'
            self.assertEqual(len(torch.load(fixture,weights_only=True)),8)
            payload=dict(global_step=8,sources=fingerprints(),policy_state=model.policy.state_dict())
            bundle=export_policy(payload,Path(tmp)/'submission')
            report=verify_export(bundle,payload['policy_state'],fixture)
            self.assertEqual(report['recorded_observation_batches'],8)
            self.assertEqual(report['recorded_inputs_sha256'],study.file_hash(fixture))

    def test_aggregate_refuses_missing_jobs_before_producing_submission(self):
        r={'config':study.read(study.CONFIG)}
        with patch.object(study,'completed',return_value=False),patch.object(runner,'export_policy') as export:
            with self.assertRaisesRegex(ValueError,'missing complete job'):runner.aggregate(r)
            export.assert_not_called()

    def test_complete_aggregation_selects_final_arm_and_exports_once(self):
        config=copy.deepcopy(study.read(study.CONFIG))
        config.update(endpoints=[8],evaluation_replicates=2,replicates_per_shard=2,bootstrap_replicates=20)
        config['training']['steps']=8;r={'config':config,'sources':{}}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'study';registration=root/'registration.json'
            registration.write_text('{}')
            for job in study.tasks(r):
                artifact=root/'artifacts'/job['id']/'result.json';artifact.parent.mkdir(parents=True)
                if job['kind']=='eval':
                    lock=dict(job=job,sources=r['sources'],registration_sha256=study.file_hash(registration))
                    lock_sha=runner.digest(lock)
                    study.atomic_json(artifact.with_name('model_lock.json'),dict(lock=lock,sha256=lock_sha))
                    rows=[dict(replicate=i,side=s,status='valid',winner=0 if job['run'].startswith('c1') else 1,
                               initial_observation_sha256=f'{i}-{s}',lock_sha256=lock_sha) for i in job['replicates'] for s in (0,1)]
                    artifact.write_text(json.dumps(dict(episodes=rows,lock_sha256=lock_sha)))
                else:artifact.write_text('{}')
                study.atomic_json(directory/'jobs'/job['id']/'done.json',dict(artifact=str(artifact.relative_to(root)),artifact_sha256=study.file_hash(artifact)))
            study.atomic_json(directory/'runs/c1_s11/checkpoint_index/8.json',dict(sha256='fixture-checkpoint'))
            source=Path(runner.__file__).with_name('policy.py')
            def export(payload,output):
                output.mkdir(parents=True);shutil.copyfile(source,output/'policy.py');(output/'checkpoint.pt').write_bytes(b'fixture')
            with patch.object(study,'ROOT',root),patch.object(study,'DIRECTORY',directory),patch.object(study,'REGISTRATION',registration),\
                 patch.object(study,'completed',return_value=True),patch.object(runner,'CheckpointStore') as store,\
                 patch.object(runner,'export_policy',side_effect=export) as exporter,patch.object(runner,'verify_export',return_value={'passed':True}):
                store.return_value.load.return_value=({'policy_state':{}},'fixture-checkpoint')
                runner.aggregate(r);exporter.assert_called_once()
            summary=study.read(directory/'summary.json')
            self.assertEqual(summary['dev_games'],28);self.assertEqual(summary['training_steps'],48)
            self.assertEqual(summary['selected']['run'],'c1_s11')
            self.assertEqual(summary['analyses']['8']['c1_vs_flat']['delta'],1)
            self.assertFalse(summary['external_submission_sent'])


if __name__=='__main__':unittest.main()
