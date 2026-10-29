import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from agent_league.common import DEFAULTS,atomic,digest,schedule
from agent_league.check import fixture
from agent_league.match import play
from agent_league.trace import NAMES,Trace,load_chunk
from contextlib import nullcontext
from agent_league.report import generate
from agent_league.adapters import Adapter


class FakeEnv:
    def __init__(self,fail=None):self.fail=fail;self.agents=list(NAMES);self.possible_agents=list(NAMES);self.closed=False
    def reset(self,seed=None):self.i=0;self.obs=fixture();self.agents=list(NAMES);return copy.deepcopy(self.obs),{}
    def step(self,actions):
        self.i+=1
        for n,o in self.obs.items():
            o['vector'][95]=1-self.i*.01;o['vector'][0]+=.001
            if n in NAMES[:5]:o['vector'][93]=self.i*.01
            else:o['vector'][94]=self.i*.01
        done=self.i==3;terms={n:done for n in NAMES};truncs={n:False for n in NAMES}
        if self.fail=='truncation':truncs[NAMES[0]]=True
        if self.fail=='partial':terms[NAMES[0]]=True
        if done:self.agents=[]
        rewards={n:1. if n in NAMES[:5] else .5 for n in NAMES}
        obs=copy.deepcopy(self.obs)
        if self.fail=='empty_terminal' and done:obs={}
        return obs,rewards,terms,truncs,{}
    def close(self):self.closed=True


class ConstantAdapter:
    def __init__(self,spec,team,seed,episode,device='cpu'):
        self.team=team;self.names=NAMES[team*5:team*5+5];self.metadata={}
    def act(self,obs,step):return {n:np.array([1.,0.],np.float32) for n in self.names}


def specs():
    return [dict(id=m,kind='strategy',config={'kind':'random'},label=m,step=0,seed=0,contract='legacy',selection_reason='fixture',assets={},path=None) for m in ('a','b','c')]

def task(side=0):
    m=specs()[:2];return dict(schedule(m,dict(DEFAULTS,rounds=1))[side],config=dict(DEFAULTS,chunk_steps=2),models=m,audit_windows=False)


class LeagueTests(unittest.TestCase):
    def test_round_robin_paired_rng(self):
        tasks=schedule(specs(),DEFAULTS);self.assertEqual(len(tasks),30);self.assertEqual(len({t['id'] for t in tasks}),30)
        for a,b in zip(tasks[::2],tasks[1::2]):
            self.assertEqual(a['action_seeds'],b['action_seeds']);self.assertEqual(a['environment_seed'],b['environment_seed']);self.assertEqual((a['side'],b['side']),(0,1))

    def test_trace_alignment_and_terminal(self):
        with tempfile.TemporaryDirectory() as d:
            env=FakeEnv();r=play(task(),Path(d),lambda:env,ConstantAdapter)
            self.assertTrue(env.closed);self.assertEqual(r['side_rewards'],[15.,7.5]);self.assertEqual(r['winner'],'a')
            idx=json.loads((Path(d)/'trace/index.json').read_text());self.assertEqual([c['count'] for c in idx['chunks']],[2,1]);self.assertTrue(idx['complete'])
            with nullcontext(load_chunk(Path(d)/'trace/steps-000000.npz')) as x:
                np.testing.assert_array_equal(x['after'][0],x['before'][1]);self.assertEqual(x['actions'].shape,(2,10,2));self.assertTrue(x['transition_valid'].all())
            with nullcontext(load_chunk(Path(d)/'trace/steps-000002.npz')) as x:self.assertTrue(x['terminated'].all())

    def test_side_swap_reward_mapping(self):
        with tempfile.TemporaryDirectory() as d:r=play(task(1),Path(d),FakeEnv,ConstantAdapter)
        self.assertEqual(r['winner'],'b')

    def test_official_result_is_recorded_without_changing_raw_reward_ranking(self):
        class OfficialEnv(FakeEnv):
            def step(self,actions):
                obs,rewards,terms,truncs,_=super().step(actions)
                return obs,rewards,terms,truncs,{n:{'winner':1} for n in NAMES}
        with tempfile.TemporaryDirectory() as d:
            result=play(task(),Path(d),OfficialEnv,ConstantAdapter)
        self.assertEqual(result['winner'],'a')
        self.assertEqual(result['official_winner'],'b')
        self.assertFalse(result['winner_criteria_agree'])
        self.assertEqual(len(result['initial_map_sha256']),64)

    def test_provided_match_scoring_parity(self):
        from blackout_v10.match import validated_match
        class Policy:
            def act(self,obs):return {n:np.array([1.,0.],np.float32) for n in obs}
        reference=validated_match(FakeEnv(),Policy(),Policy())
        with tempfile.TemporaryDirectory() as d:r=play(task(),Path(d),FakeEnv,ConstantAdapter)
        self.assertEqual(r['side_rewards'],[reference.team_a_total_reward,reference.team_b_total_reward]);self.assertEqual(r['episode_steps'],reference.episode_steps)

    def test_invalid_last_transition_preserved(self):
        for failure in ('truncation','partial'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as d:
                env=FakeEnv(failure)
                with self.assertRaises(ValueError):play(task(),Path(d),lambda:env,ConstantAdapter)
                idx=json.loads((Path(d)/'trace/index.json').read_text());self.assertFalse(idx['complete']);self.assertEqual(idx['steps'],1);self.assertTrue(env.closed)
                with nullcontext(load_chunk(Path(d)/'trace/steps-000000.npz')) as x:self.assertFalse(x['transition_valid'][0])

    def test_terminal_without_observation(self):
        with tempfile.TemporaryDirectory() as d:
            r=play(task(),Path(d),lambda:FakeEnv('empty_terminal'),ConstantAdapter)
            self.assertTrue(r['valid'])
            with nullcontext(load_chunk(Path(d)/'trace/steps-000002.npz')) as x:self.assertFalse(x['after_present'][0]);self.assertTrue(np.isnan(x['after']).all())

    def test_stop_never_valid(self):
        with tempfile.TemporaryDirectory() as d:
            env=FakeEnv()
            with self.assertRaises(InterruptedError):play(task(),Path(d),lambda:env,ConstantAdapter,stop=lambda:True)
            self.assertTrue(env.closed)

    def test_adapter_isolation(self):
        s=specs()[0];p=Adapter(s,0,3);q=Adapter(s,0,3);intruder=Adapter(s,1,5);obs=fixture()
        for i in range(4):
            a=p.act(obs,i);intruder.act(obs,i);b=q.act(obs,i)
            for n in p.names:np.testing.assert_array_equal(a[n],b[n])

    def test_trace_does_not_change_actions(self):
        s=specs()[0];p=Adapter(s,0,3);q=Adapter(s,0,3);obs=fixture()
        with tempfile.TemporaryDirectory() as d:
            trace=Trace(d,DEFAULTS)
            for step in range(3):
                a=p.act(obs,step);trace.snapshot(obs,step);trace.event(dict(step=step,kind='fixture'));b=q.act(obs,step)
                for n in p.names:np.testing.assert_array_equal(a[n],b[n])
            trace.close()

    def test_report_round_results_and_partial(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=specs()[:2];manifest=dict(roster={'models':m,'excluded':[]},config=dict(DEFAULTS,rounds=1),tasks=schedule(m,dict(DEFAULTS,rounds=1)))
            atomic(root/'manifest.json',manifest)
            for side in (0,1):
                folder=root/'attempts'/str(side);folder.mkdir(parents=True);t=task(side)
                r=play(t,folder,FakeEnv,ConstantAdapter)
                atomic(root/'cells'/(t['id']+'.json'),dict(task=t,result=r,directory=str(folder)))
            summary=generate(root);self.assertTrue(summary['complete']);self.assertEqual([r['rank'] for r in summary['ranking']],[1,1])
            self.assertTrue((root/'figures/score_matrix.svg').exists());self.assertTrue((root/'pairs/a__b.md').exists())
            next((root/'cells').glob('*.json')).unlink();summary=generate(root);self.assertFalse(summary['complete']);self.assertIn('잠정',(root/'report.md').read_text(encoding='utf-8'))

    def test_completed_cells_not_reexecuted_and_manifest_guard(self):
        from agent_league.runtime import run_league
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=specs()[:2];manifest=dict(roster={'models':m,'excluded':[]},config=dict(DEFAULTS,rounds=1),tasks=schedule(m,dict(DEFAULTS,rounds=1)))
            atomic(root/'manifest.json',manifest)
            for side in (0,1):
                folder=root/'attempts'/str(side);folder.mkdir(parents=True);t=task(side);r=play(t,folder,FakeEnv,ConstantAdapter)
                atomic(root/'cells'/(t['id']+'.json'),dict(task=t,result=r,directory=str(folder),manifest_sha256=digest(manifest)))
            with patch('agent_league.runtime.execute') as executor:
                run_league(root,manifest,lambda:False);executor.assert_not_called()
            changed=copy.deepcopy(manifest);changed['config']['rounds']=2
            with self.assertRaises(ValueError):run_league(root,changed,lambda:False)

    def test_retry_preserves_failed_attempt_and_counts_once(self):
        from agent_league.runtime import run_league
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=specs()[:2];cfg=dict(DEFAULTS,rounds=1,workers=2)
            manifest=dict(roster={'models':m,'excluded':[]},config=cfg,tasks=schedule(m,cfg))
            atomic(root/'manifest.json',manifest);calls=[]
            def execute(tasks,directory,cfg,stop,progress):
                calls.append(len(tasks));out=[]
                for t in tasks:
                    folder=directory/t['id'];folder.mkdir(parents=True)
                    if len(calls)==1:
                        row=dict(id=t['id'],task=t,valid=False,result=None,error='fixture transient',directory=str(folder))
                    else:
                        r=play(dict(t,audit_windows=False),folder,FakeEnv,ConstantAdapter)
                        row=dict(id=t['id'],task=t,valid=True,result=r,error=None,directory=str(folder))
                    atomic(folder/'attempt.json',row);out.append(row)
                return out
            with patch('agent_league.runtime.execute',execute),patch('agent_league.runtime.resource_sample',return_value=dict(rss_gib=1,free_gib=100,pressure={'critical':False})):
                run_league(root,manifest,lambda:False)
            self.assertEqual(calls,[2,2]);self.assertEqual(len(list((root/'cells').glob('*.json'))),2)
            self.assertEqual(len(list((root/'attempts').glob('*/*/attempt.json'))),4)
            self.assertEqual(read_json(root/'status.json')['state'],'complete')

    def test_resource_abort_salvages_finished_cells(self):
        from agent_league.runtime import run_league
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=specs()[:2];cfg=dict(DEFAULTS,rounds=1,workers=2)
            manifest=dict(roster={'models':m,'excluded':[]},config=cfg,tasks=schedule(m,cfg));atomic(root/'manifest.json',manifest)
            def execute(tasks,directory,cfg,stop,progress):
                t=tasks[0];folder=directory/t['id'];folder.mkdir(parents=True)
                r=play(dict(t,audit_windows=False),folder,FakeEnv,ConstantAdapter)
                atomic(folder/'attempt.json',dict(id=t['id'],task=t,valid=True,result=r,error=None,directory=str(folder)))
                raise RuntimeError('resource threshold')
            with patch('agent_league.runtime.execute',execute),self.assertRaises(RuntimeError):run_league(root,manifest,lambda:False)
            self.assertEqual(len(list((root/'cells').glob('*.json'))),1)



    def test_crash_recovery_promotes_verified_natural_result(self):
        from agent_league.runtime import recover_completed
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m=specs()[:2];cfg=dict(DEFAULTS,rounds=1)
            manifest=dict(roster={'models':m,'excluded':[]},config=cfg,tasks=schedule(m,cfg))
            base=manifest['tasks'][0];t=dict(base,models=m,config=cfg,audit_windows=False)
            folder=root/'attempts/one'/t['id'];folder.mkdir(parents=True)
            result=play(t,folder,FakeEnv,ConstantAdapter)
            atomic(folder/'task.json',t);atomic(folder/'result.json',dict(task_sha256=digest(t),result=result))
            recover_completed(root,manifest)
            self.assertEqual(len(list((root/'cells').glob('*.json'))),1)
            stored=read_json(next((root/'cells').glob('*.json')))
            self.assertTrue(stored['recovered_after_supervisor_exit'])
            # Repeating recovery cannot duplicate a match.
            recover_completed(root,manifest);self.assertEqual(len(list((root/'cells').glob('*.json'))),1)

    def test_compact_trace_bit_exact_for_discontinuous_observations(self):
        with tempfile.TemporaryDirectory() as d:
            tr=Trace(d,dict(DEFAULTS,chunk_steps=2));rng=np.random.default_rng(6)
            expected=[]
            for step in range(2):
                before=rng.normal(size=(10,96)).astype(np.float32);after=rng.normal(size=(10,96)).astype(np.float32)
                before[0,0]=-0.;expected.append((before,after));tr.append(step=step,before=before,after=after)
            tr.close();loaded=load_chunk(Path(d)/'steps-000000.npz')
            for step,(before,after) in enumerate(expected):
                np.testing.assert_array_equal(loaded['before'][step].view(np.uint32),before.view(np.uint32))
                np.testing.assert_array_equal(loaded['after'][step].view(np.uint32),after.view(np.uint32))

    def test_changed_checkpoint_asset_is_rejected(self):
        from agent_league.common import sha,verify_assets
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'checkpoint.pt';p.write_bytes(b'old');spec={'assets':{str(p):sha(p)}}
            verify_assets(spec);p.write_bytes(b'new')
            with self.assertRaises(ValueError):verify_assets(spec)


def read_json(path):return json.loads(path.read_text())

if __name__=='__main__':unittest.main()
