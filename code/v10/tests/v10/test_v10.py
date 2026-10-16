"""Critical v10 regressions. Synthetic games only; never starts Unity."""
import copy
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
from blackout_v10.benchmark import fixture
from blackout_v10.collector import collect_episode
from blackout_v10.contracts import config
from blackout_v10.conversion import convert_state
from blackout_v10.export import export_policy, verify_export
from blackout_v10.io import atomic_json, atomic_torch, cpu_state, file_hash, load_checkpoint, state_hash
from blackout_v10.opponents import StrategyOpponent
from blackout_v10.policy import MyPolicy
from blackout_v10.ppo import optimizers, update
from blackout_v10.targets import commit_targets
from blackout_v10.training import train_run
from blackout_v10.trajectory import Episode, EpisodeWriter, WaveBatch, pack
from blackout_v10.value import Critic
from blackout_v10.schedule import task_spec
from tests.v9.test_v9 import Game, observations


def small_config():
    c=copy.deepcopy(config())
    c.update(workers=1,worker_cap=1,max_episode_steps=5,training_steps=6,pilot_steps=6,
        episode_wall_seconds=10,stall_seconds=5,stop_grace_seconds=.5,disk_stop_gib=0,
        diagnostic_games=4,diagnostic_fit_games=3,diagnostic_epochs=1,diagnostic_samples_per_episode=3,value_batch=2)
    c['ppo'].update(epochs=1,minibatch=2)
    return c


def collect(tmp,cfg=None,game=None,actor=None,critic=None):
    cfg=cfg or small_config(); actor=actor or MyPolicy().eval(); critic=critic or Critic('C1')
    snapshot=Path(tmp)/'behavior.pt';atomic_torch(snapshot,{'policy_state':cpu_state(actor),'critic_state':cpu_state(critic),'critic_arm':critic.arm})
    task=dict(id='test',config=cfg,side=0,action_seed=11,requested_seed=99,policy_sha256=file_hash(snapshot),
        trajectory=str(Path(tmp)/'episode'),invalid_path=str(Path(tmp)/'invalid.json'),opponent={'kind':'noop'},audit_windows=False)
    record=collect_episode(actor,None,StrategyOpponent('noop'),task,lambda:game or Game())
    return task,snapshot,record


def mock_tasks(tasks,directory,cfg,stop=lambda:False,progress=lambda **_:None):
    from blackout_v10.evaluation import evaluate
    result=[]
    for task in tasks:
        path=Path(directory)/task['id'];path.mkdir(parents=True)
        task=dict(task,trajectory=str(path/'trajectory'),invalid_path=str(path/'invalid.json'))
        atomic_json(path/'task.json',task)
        if task['kind']=='collect':
            payload=torch.load(task['policy_path'],weights_only=True)
            actor=MyPolicy().eval();actor.load_state_dict(payload['policy_state'])
            record=collect_episode(actor,None,StrategyOpponent('noop'),task,Game)
        else:
            record=evaluate(task,Game)
        row=dict(id=task['id'],task=task,directory=str(path),valid=True,result=record,error=None,
                 attempt_steps=3,step_count_complete=True,interrupted=False)
        atomic_json(path/'attempt.json',row);result.append(row)
    return result


class V10Tests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1);torch.manual_seed(11)

    def test_parameters_conversion_and_rejection(self):
        from blackout_v9.policy import MyPolicy as Old
        old=Old().eval(); new=convert_state(old.state_dict());v,g=fixture()
        self.assertEqual(sum(p.numel() for p in new.parameters()),120905)
        for arm,n in [('C0',29057),('C1',128769),('C1-blind',128769)]:
            self.assertEqual(sum(p.numel() for p in Critic(arm).parameters()),n)
        torch.testing.assert_close(new.log_probs(v,g),old.log_probs(v,g),atol=2e-6,rtol=1e-5)
        torch.manual_seed(22);a=old(v,g);torch.manual_seed(22);b=new(v,g)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        old.configure('C')
        with self.assertRaises(ValueError):convert_state(old.state_dict())

    def test_actor_batch_all_channels_and_input_immutability(self):
        p=MyPolicy();v,g=fixture();before=(v.clone(),g.clone())
        for n in [0,1,3,5,10]:
            a=p(v.repeat(2,1)[:n],g.repeat(2,1,1,1)[:n])
            self.assertEqual(a.shape,(n,2));self.assertTrue(torch.isfinite(a).all() and (a.abs()<=1).all())
        order=torch.tensor([3,0,4,2,1]);logs=p.log_probs(v,g)
        torch.testing.assert_close(p.log_probs(v[order],g[order]),logs[order],atol=2e-6,rtol=1e-5)
        torch.testing.assert_close(torch.cat([p.log_probs(v[i:i+1],g[i:i+1]) for i in range(5)]),logs,atol=2e-6,rtol=1e-5)
        dense=torch.rand_like(g,requires_grad=True);(-p.log_probs(v,dense)[:,1].mean()).backward()
        self.assertTrue((dense.grad.abs().sum((0,2,3))>0).all())
        torch.testing.assert_close(v,before[0]);torch.testing.assert_close(g,before[1])
        torch.testing.assert_close(p.directions[0],torch.zeros(2))

    def test_critic_independence_popart_and_map_effect(self):
        p=MyPolicy();state=cpu_state(p);v,g=fixture();tv=v[None].repeat(2,1,1);tg=g[:1].repeat(2,1,1,1)
        for arm in ['C0','C1','C1-blind']:
            c=Critic(arm);opt=torch.optim.Adam(c.parameters(),lr=.001)
            for _ in range(3):
                opt.zero_grad();loss=(c(tv,tg)-.2).square().mean();loss.backward();opt.step()
            output=c(tv,tg).detach();c.update_scale(torch.tensor([3.,-2.]))
            torch.testing.assert_close(c(tv,tg),output,atol=1e-6,rtol=1e-5)
            changed=tg.clone();changed[:,0]=0;changed[:,1]=1
            if arm=='C1':self.assertGreater(float((c(tv,changed)-c(tv,tg)).abs().max().detach()),1e-7)
            else:torch.testing.assert_close(c(tv,changed),c(tv,tg),atol=0,rtol=0)
        for k in state:torch.testing.assert_close(p.state_dict()[k],state[k],atol=0,rtol=0)

    def test_validated_target_frozen_critic_parity_and_hash_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=Critic('C1');torch.nn.init.normal_(c.output.weight,std=.1)
            task,snapshot,record=collect(tmp,critic=c)
            self.assertTrue(record['natural_terminal']);self.assertFalse((Path(task['trajectory'])/'committed.json').exists())
            with self.assertRaises(FileNotFoundError):WaveBatch([task['trajectory']])
            ep=Episode(task['trajectory'],validated=True)
            with torch.inference_mode():
                expected=np.array([float(c(torch.from_numpy(np.array(ep.arrays['vectors'][i:i+1])),torch.from_numpy(ep.graphic([i])))[0]) for i in range(ep.length)])
            commit_targets([task['trajectory']],snapshot,task['policy_sha256'],small_config(),'cpu')
            batch=WaveBatch([task['trajectory']]);np.testing.assert_allclose(batch.episodes[0].arrays['values'],expected,atol=2e-6)
            with self.assertRaisesRegex(ValueError,'checksum'):commit_targets([task['trajectory']],snapshot,'bad',small_config(),'cpu')

    def test_ppo_encoder_ratios_and_hard_gate(self):
        for hard in [.04,-1.]:
            with tempfile.TemporaryDirectory() as tmp:
                p=MyPolicy();c=Critic();cfg=small_config();task,snapshot,_=collect(tmp,actor=p,critic=c)
                commit_targets([task['trajectory']],snapshot,task['policy_sha256'],cfg,'cpu')
                batch=WaveBatch([task['trajectory']]);row=batch.minibatch(np.arange(len(batch)),'cpu')
                chosen=p.team_log_probs(row['vectors'],row['graphic']).gather(-1,row['actions'][...,None]).squeeze(-1)
                torch.testing.assert_close(chosen,row['log_probs'],atol=2e-6,rtol=1e-5)
                opts=optimizers(p,c,cfg['ppo']);before=cpu_state(p);cfg['ppo']['hard_kl']=hard
                if hard<0:
                    with self.assertRaisesRegex(RuntimeError,'hard post-step'):update(p,c,*opts,batch,cfg['ppo'],np.random.default_rng(11),'C1')
                    for k in before:torch.testing.assert_close(p.state_dict()[k],before[k],atol=0,rtol=0)
                    self.assertFalse(opts[0].state)
                else:
                    report=update(p,c,*opts,batch,cfg['ppo'],np.random.default_rng(11),'C1')
                    self.assertGreater(report['parameter_delta']['patch.weight'],0)
                    self.assertGreater(report['critic_parameter_delta']['encoder.patch.weight'],0)

    def test_invalid_episodes_never_validated(self):
        for game in [Game(length=100),Game(fail=2),Game(partial=True)]:
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises((RuntimeError,ValueError)):collect(tmp,game=game)
                self.assertFalse((Path(tmp)/'episode/validated.json').exists())
                self.assertTrue(game.closed)

    def test_nonbinary_graphic_lossless(self):
        vectors,graphic=pack(observations(),[f'unit_{i}' for i in range(5)])
        graphic[0,3,3]=.75;graphic[1,3,3]=.25
        with tempfile.TemporaryDirectory() as tmp:
            w=EpisodeWriter(Path(tmp)/'ep',2)
            w.add(vectors=vectors,graphic=graphic,actions=np.zeros(5,np.int64),log_probs=np.zeros(5,np.float32),
                  old_distribution=np.zeros((5,9),np.float32),values=0,potentials=0,aux_targets=np.zeros(3,np.float32))
            w.commit(1,1,1,dict(natural_terminal=True,policy_sha256='one'))
            ep=Episode(w.path,validated=True);np.testing.assert_array_equal(ep.graphic([0])[0],graphic)

    def test_training_resume_and_paired_schedule(self):
        with tempfile.TemporaryDirectory() as tmp,patch('blackout_v10.training.run_tasks',mock_tasks),patch('blackout_v10.windows.visible_windows',return_value={'verified':True,'onscreen_owned_windows':[]}):
            cfg=small_config();hashes=[];tasks=[]
            for arm in ['C0','C1']:
                run=Path(tmp)/arm;p,r=train_run(run,cfg,arm,11,'registration','cpu')
                self.assertTrue(r['complete']);self.assertEqual(r['step'],6)
                payload,_=load_checkpoint(run/'checkpoints');hashes.append(payload['initial_actor_sha256'])
                tasks.append([json.loads(f.read_text()) for f in sorted(run.glob('waves/*/*/task.json'))])
                again,r2=train_run(run,cfg,arm,11,'registration','cpu')
                self.assertEqual(state_hash(cpu_state(p)),state_hash(cpu_state(again)))
                with self.assertRaisesRegex(ValueError,'provenance'):train_run(run,cfg,arm,11,'wrong','cpu')
            self.assertEqual(hashes[0],hashes[1])
            for a,b in zip(*tasks):
                for k in ['opponent','action_seed','side','requested_seed']:self.assertEqual(a[k],b[k])

    def test_export_actual_isolated_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=MyPolicy();bundle=export_policy(p,Path(tmp)/'bundle')
            self.assertTrue(verify_export(bundle,p,fixture())['passed'])
            self.assertEqual({x.name for x in bundle.iterdir()},{'policy.py','checkpoint.pt'})

    def test_entire_study_all_arms_both_confirmation_and_frozen_test(self):
        from blackout_v10 import study
        cfg=small_config();cfg.update(seeds=[11],confirmation_seeds=[44],dev_replicates=1,test_replicates=1,
            baseline_replicates=1,pilot_replicates=1,lifecycle_games=2,evaluation_families=['noop'])
        with tempfile.TemporaryDirectory() as tmp,patch.object(study,'ROOT',Path(tmp)),\
            patch.object(study,'verify_original',return_value={}),\
            patch.object(study,'benchmark_inference',return_value={'measurements':[dict(device='cpu',threads=1,p95=.01)]}),\
            patch.object(study,'choose_learner',return_value={'selected_device':'cpu'}),\
            patch.object(study,'benchmark_values',return_value={'selected_batch':2}),\
            patch.object(study,'run_tasks',mock_tasks),\
            patch('blackout_v10.training.run_tasks',mock_tasks),\
            patch('blackout_v10.value_diagnostic.run_tasks',mock_tasks),\
            patch('blackout_v10.windows.visible_windows',return_value={'verified':True,'onscreen_owned_windows':[]}):
            directory=Path(tmp)/'study';directory.mkdir()
            report=study.run_study(directory,cfg,'registration',lambda:False,lambda **_:None,lambda:0.)
            self.assertTrue(report['complete']);self.assertEqual(set(report['confirmation']),{'C0','C1'})
            self.assertEqual(report['confirmation_steps'],12);self.assertEqual(report['pilot_steps'],12)
            self.assertEqual(set(report['frozen_tests']),{'C0','C1','initial'})
            self.assertFalse(report['value_diagnostic']['weights_reused_for_training'])
            self.assertEqual(report['selected']['arm'],'C1');self.assertTrue((directory/'selection_lock.json').exists())

    def test_viewer_real_previous_models_with_synthetic_game(self):
        from blackout_v10.viewer import play_task
        from blackout_v10.watch_adapters import available_models
        with tempfile.TemporaryDirectory() as tmp:
            bundle=export_policy(MyPolicy(),Path(tmp)/'bundle')
            for version,spec in available_models().items():
                self.assertTrue(spec['available'],spec.get('error'))
                for side in [0,1]:
                    task=dict(bundle=str(bundle),bundle_hashes={n:file_hash(bundle/n) for n in ('policy.py','checkpoint.pt')},
                        config=small_config(),opponent_spec=spec,action_seed=11,speed=1,side=side,model_labels={'version':version})
                    from tests.v8.test_competition import Game as ViewerGame
                    game=ViewerGame(2)
                    report=play_task(task,env_factory=lambda speed:game)
                    self.assertTrue(report['valid']);self.assertTrue(game.closed);self.assertTrue(report['exhibition_only'])

    def test_match_matches_original_and_rejects_invalid_terminal(self):
        from blackout_v10.match import validated_match
        from blackout_env import run_match
        from dataclasses import asdict
        for side in [False,True]:
            a=StrategyOpponent('noop');b=StrategyOpponent('noop')
            self.assertEqual(asdict(validated_match(Game(),a,b,side)),asdict(run_match(Game(),a,b,side)))
        with self.assertRaises(ValueError):validated_match(Game(partial=True),StrategyOpponent('noop'),StrategyOpponent('noop'))
        class Truncated(Game):
            def step(self,actions):
                o,r,t,tr,i=super().step(actions)
                tr={n:True for n in tr}
                return o,r,t,tr,i
        with self.assertRaises(ValueError):validated_match(Truncated(),StrategyOpponent('noop'),StrategyOpponent('noop'))

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable in this process')
    def test_mps_optimizer_and_cpu_export(self):
        with tempfile.TemporaryDirectory() as tmp,patch('blackout_v10.training.run_tasks',mock_tasks),patch('blackout_v10.windows.visible_windows',return_value={'verified':True,'onscreen_owned_windows':[]}):
            p,r=train_run(Path(tmp)/'run',small_config(),'C1',11,'mps-registration','mps')
            self.assertTrue(r['complete'])
            bundle=export_policy(p,Path(tmp)/'bundle')
            self.assertTrue(verify_export(bundle,p,fixture())['passed'])

    def test_runtime_detach_and_watchdog(self):
        from blackout_v10.runtime import run_tasks
        with tempfile.TemporaryDirectory() as tmp:
            cfg=small_config()
            rows=run_tasks([{'id':'one'}],Path(tmp)/'a',cfg,worker_module='tests.v10.runtime_fixture')
            self.assertTrue(rows[0]['valid']);self.assertEqual(rows[0]['result']['pid'],rows[0]['result']['session'])
            cfg.update(stall_seconds=1,episode_wall_seconds=1)
            rows=run_tasks([{'id':'hung','hang':True}],Path(tmp)/'b',cfg,worker_module='tests.v10.runtime_fixture')
            self.assertFalse(rows[0]['valid']);self.assertIn('watchdog',rows[0]['error'])

    def test_background_supervisor_returns_and_stops(self):
        from blackout_v10 import runner
        original=subprocess.Popen;children=[]
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);directory=tmp/'fixture'
            def launch(command,**kwargs):
                if any(str(x).endswith('v10_experiments.py') for x in command):
                    command=[sys.executable,'-u','-m','tests.v10.supervisor_fixture',*command[3:]]
                    kwargs['env']=dict(kwargs['env'],V10_FIXTURE_DIRECTORY=str(directory))
                    proc=original(command,**kwargs);children.append(proc);return proc
                return original(command,**kwargs)
            try:
                with patch.object(runner,'ROOT',tmp),patch.object(runner,'directory_for',return_value=directory),\
                    patch.object(runner.subprocess,'Popen',launch),patch('blackout_v10.runtime.check_legacy_processes'),\
                    patch.object(sys,'argv',['v10','--name','fixture']):runner.main()
                self.assertEqual(len(children),1);self.assertIsNone(children[0].poll())
                self.assertTrue(runner.is_active(directory));self.assertEqual(os.getsid(children[0].pid),children[0].pid)
                (directory/'stop.request').touch();self.assertEqual(children[0].wait(timeout=15),0)
                self.assertFalse(runner.is_active(directory))
            finally:
                for child in children:
                    if child.poll() is None:child.terminate();child.wait(timeout=10)


if __name__=='__main__':unittest.main()
