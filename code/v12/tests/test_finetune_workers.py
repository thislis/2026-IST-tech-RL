"""Persistent actor lifecycle and frozen snapshots, without starting Unity."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import copy
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import torch

from blackout_v12 import collector, env, finetune_workers as workers
from blackout_v12.io import atomic_torch, file_hash, read_json


def _actor_state():
    """Spawn-safe probe of Python worker configuration; never constructs a game."""
    return dict(pid=os.getpid(), threads=torch.get_num_threads(),
                cuda_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                slot=workers._SLOT, visible=workers._VISIBLE,
                guarded=workers._GUARD is not None)


class PersistentWorkerTests(unittest.TestCase):
    @staticmethod
    def task(root, name='a', snapshot=None, seed=7):
        task = dict(id=name, directory=str(root/name), config={'actor_threads':4},
                    action_seed=seed, requested_seed=seed, opponent={'id':'noop'},
                    side=0, policy_sha256='snapshot')
        if snapshot is not None:
            task.update(checkpoint=str(snapshot), policy_sha256=file_hash(snapshot))
        return task

    def test_only_slot_zero_visible_and_executors_survive_waves(self):
        created, calls, closed, gate = [], [], [], threading.Barrier(3)

        class Executor:
            def __init__(self, **kwargs):
                self.slot = kwargs['initargs'][0]
                self.pool = ThreadPoolExecutor(max_workers=1)
                self.arguments = kwargs
                created.append(self)

            def submit(self, function, *args):
                def execute():
                    if function is workers._close_worker:
                        closed.append(self.slot)
                        return
                    task = args[0]
                    calls.append((self.slot, task['id'], task['policy_sha256']))
                    if task['id'].startswith('first'):
                        gate.wait(timeout=10)
                    return dict(episode_id=task['id'], policy_sha256=task['policy_sha256'],
                                natural_terminal=True)
                return self.pool.submit(execute)

            def shutdown(self, **kwargs):
                self.pool.shutdown(**kwargs)

        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(workers, 'ProcessPoolExecutor', Executor):
            root = Path(temporary)
            tasks = [self.task(root, f'first-{name}') for name in ('c','a','b')]
            with workers.PersistentEpisodePool(3) as pool:
                first = pool.run(tasks)
                second = [dict(self.task(root, f'next-{i}'), policy_sha256='new') for i in range(3)]
                pool.run(second)
                self.assertEqual(len(created),3)
                self.assertFalse(closed)
            self.assertEqual([row['episode_id'] for row in first],['first-a','first-b','first-c'])
            self.assertEqual([row.arguments['initargs'][:2] for row in created],
                             [(0,True),(1,False),(2,False)])
            self.assertTrue(all(row.arguments['max_workers']==1 for row in created))
            self.assertEqual(sorted(closed),[0,1,2])
            self.assertEqual(sorted((slot,name) for slot,name,_ in calls[:3]),
                             [(0,'first-c'),(1,'first-a'),(2,'first-b')])
            self.assertTrue(pool._closed)
            pool.close()
            with self.assertRaisesRegex(RuntimeError,'closed'):
                pool.run(tasks)

    def test_spawned_actor_is_cpu_only_and_owns_a_cleanup_guard(self):
        with workers.PersistentEpisodePool(1) as pool:
            pool._ensure_pools()
            pool._started.add(0)
            actor = pool._pools[0].submit(_actor_state).result(timeout=30)
            self.assertNotEqual(actor['pid'],os.getpid())
            self.assertEqual(actor['threads'],1)
            self.assertEqual(actor['cuda_devices'],'')
            self.assertEqual(actor['slot'],0)
            self.assertTrue(actor['visible'])
            self.assertTrue(actor['guarded'])

    def test_partial_resume_dispatches_remaining_training_to_visible_slot(self):
        calls = []
        class Executor:
            def __init__(self, **kwargs):
                self.slot = kwargs['initargs'][0]
                self.pool = ThreadPoolExecutor(max_workers=1)
            def submit(self, function, *args):
                def execute():
                    if function is workers._close_worker: return
                    task = args[0]
                    calls.append((self.slot,task['id']))
                    return dict(episode_id=task['id'],policy_sha256=task['policy_sha256'],
                                natural_terminal=True)
                return self.pool.submit(execute)
            def shutdown(self, **kwargs):
                self.pool.shutdown(**kwargs)
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(workers,'ProcessPoolExecutor',Executor):
            root = Path(temporary)
            cached, fresh = self.task(root,'cached'), self.task(root,'fresh')
            Path(cached['directory']).mkdir()
            (Path(cached['directory'])/'episode.pt').touch()
            with workers.PersistentEpisodePool(2) as pool:
                rows = pool.run([cached,fresh])
            self.assertIn((0,'fresh'),calls)
            self.assertEqual([row['episode_id'] for row in rows],['cached','fresh'])

    def test_worker_reuses_unity_but_loads_each_immutable_snapshot(self):
        class Policy:
            instances = []
            def __init__(self, config, **kwargs):
                self.scorer = Mock()
                self.random = torch.rand(1).item()
                self.instances.append(self)
            def eval(self):
                return self

        game = Mock()
        def collect(policy, opponent, task, factory):
            lease = factory()
            lease.reset(seed=task['requested_seed'])
            lease.close()
            return dict(episode_id=task['id'], policy_sha256=task['policy_sha256'],
                        events=[], trip_ledger={'rows':[]}, natural_terminal=True,
                        on_policy=True, env_steps=1)

        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            before, after = root/'before.pt', root/'after.pt'
            atomic_torch(before,{'scorer_state':{'weight':torch.tensor([1.])}})
            atomic_torch(after,{'scorer_state':{'weight':torch.tensor([2.])}})
            stack.enter_context(patch.object(workers,'_ENV',None))
            stack.enter_context(patch.object(workers,'_ENV_KEY',None))
            stack.enter_context(patch.object(workers,'_STOP',None))
            stack.enter_context(patch.object(workers,'_SLOT',0))
            stack.enter_context(patch.object(workers,'_VISIBLE',True))
            factory = stack.enter_context(patch.object(workers,'make_env',return_value=game))
            stack.enter_context(patch('blackout_v12.policy.MyPolicy',Policy))
            stack.enter_context(patch('blackout_v12.opponents.make_opponent',return_value=Mock()))
            stack.enter_context(patch.object(collector,'collect_episode',side_effect=collect))
            first = self.task(root,'first',before)
            second = self.task(root,'second',after)
            unchanged = copy.deepcopy(second)
            with torch.random.fork_rng(devices=[]):
                workers._run_task(first)
                workers._run_task(second)
                workers._run_task(second)  # Cached episode must not reset Unity.
            self.assertEqual(factory.call_count,1)
            factory.assert_called_once_with(first,background=False,worker_id=0)
            self.assertEqual(game.reset.call_count,2)
            self.assertEqual(game.close.call_count,0)
            self.assertEqual(len(Policy.instances),2)
            self.assertEqual(Policy.instances[0].random,Policy.instances[1].random)
            self.assertEqual(Policy.instances[0].scorer.load_state_dict.call_args.args[0]['weight'].item(),1.)
            self.assertEqual(Policy.instances[1].scorer.load_state_dict.call_args.args[0]['weight'].item(),2.)
            self.assertEqual(second,unchanged)
            self.assertEqual(read_json(root/'second'/'task.json'),second)
            workers._destroy_environment()
            game.close.assert_called_once()

    def test_failed_episode_destroys_game_and_next_task_can_recreate_it(self):
        game, replacement = Mock(), Mock()
        with ExitStack() as stack:
            stack.enter_context(patch.object(workers,'_ENV',game))
            stack.enter_context(patch.object(workers,'_ENV_KEY',None))
            stack.enter_context(patch.object(workers,'_STOP',None))
            stack.enter_context(patch.object(workers,'_SLOT',1))
            stack.enter_context(patch.object(workers,'_VISIBLE',False))
            run = stack.enter_context(patch.object(workers,'run_episode',side_effect=ValueError('bad transition')))
            with self.assertRaisesRegex(ValueError,'bad transition'):
                workers._run_task({'action_seed':3})
            game.close.assert_called_once()
            self.assertIsNone(workers._ENV)
            factory = stack.enter_context(patch.object(workers,'make_env',return_value=replacement))
            run.side_effect = lambda task,env_factory,actor_threads:env_factory()
            task = {'config':{},'action_seed':4}
            workers._run_task(task)
            factory.assert_called_once_with(task,background=True,worker_id=1)
            workers._destroy_environment()
            replacement.close.assert_called_once()

    def test_unity_launcher_proxy_is_scoped_to_environment_construction(self):
        from mlagents_envs import env_utils
        original = env_utils.subprocess
        def fail(*args, **kwargs):
            self.assertIs(env_utils.subprocess.Popen,workers.managed_popen)
            self.assertIs(env_utils.subprocess.DEVNULL,original.DEVNULL)
            raise OSError('mock constructor failed')
        with patch.object(workers,'_ENV',None), patch.object(workers,'_STOP',None), \
                patch.object(workers,'make_env',side_effect=fail):
            with self.assertRaisesRegex(OSError,'mock constructor failed'):
                workers._environment({'config':{}})
        self.assertIs(env_utils.subprocess,original)

    def test_stop_prevents_launch_and_rejects_ongoing_lease_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stop = root/'STOP'; stop.touch()
            task = self.task(root)
            task['config']['stop_file'] = str(stop)
            with workers.PersistentEpisodePool(2) as pool, \
                    patch.object(workers,'ProcessPoolExecutor') as constructor:
                with self.assertRaises(InterruptedError):
                    pool.run([task])
                constructor.assert_not_called()
        stopped = threading.Event(); stopped.set(); game = Mock()
        with patch.object(workers,'_STOP',stopped):
            lease = workers._EpisodeLease(game)
            with self.assertRaises(InterruptedError): lease.reset(seed=2)
            with self.assertRaises(InterruptedError): lease.step({})
            game.reset.assert_not_called()
            game.step.assert_not_called()

    def test_invalid_wave_and_capacity_do_not_launch_workers(self):
        for count in (0,17,1.5,True):
            with self.assertRaises(ValueError): workers.PersistentEpisodePool(count)
        with tempfile.TemporaryDirectory() as temporary, workers.PersistentEpisodePool() as pool, \
                patch.object(workers,'ProcessPoolExecutor') as constructor:
            root = Path(temporary)
            first = self.task(root,'first')
            with self.assertRaisesRegex(ValueError,'duplicate episode ids'):
                pool.run([first,first])
            second = dict(self.task(root,'second'),policy_sha256='different')
            with self.assertRaisesRegex(ValueError,'mixed behavior'):
                pool.run([first,second])
            constructor.assert_not_called()

    def test_make_env_defaults_headless_and_visible_keeps_renderer(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'Blackout.exe'; path.touch()
            task = {'config':{'env_path':str(path),'time_scale':50}}
            with patch('blackout_rl.env.ContractBlackOutEnv') as constructor:
                env.make_env(task)
                env.make_env(task,background=False,worker_id=0)
            self.assertEqual([call.kwargs['background'] for call in constructor.call_args_list],[True,False])
            self.assertTrue(all(call.kwargs['no_graphics'] is False for call in constructor.call_args_list))
            self.assertEqual([call.kwargs['time_scale'] for call in constructor.call_args_list],[50.,50.])


if __name__ == '__main__':
    unittest.main()
