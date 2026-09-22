"""Acceleration keeps checkpoints, independent runs, wire schemas and brain math intact."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from scripts.connectome_fast_runtime import queue_pipe
from scripts import accelerated_connectome as fast


class RuntimeTests(unittest.TestCase):
    def test_queue_duplex_poll_and_message_order(self):
        a,b=queue_pipe()
        self.assertFalse(a.poll(.001))
        def server():
            for i in range(100):
                request=b.recv()
                b.send((request['sequence'],request['data']))
        thread=threading.Thread(target=server)
        thread.start()
        for i in range(100):
            a.send(dict(sequence=i,data=bytes(10000)))
            self.assertTrue(a.poll(2))
            self.assertTrue(a.poll(0))
            self.assertEqual(a.recv(),(i,bytes(10000)))
        thread.join(2)
        self.assertFalse(thread.is_alive())
        a.close();b.close()
        with self.assertRaises(OSError):a.send(0)

    def test_scheduling_enforces_shared_mps_limit_and_refills_cpu(self):
        pending=[dict(index=i,suite='v7-2' if i<3 else 'v7-1',phase='train') for i in range(6)]
        direct=fast.choose(pending,{},4,1)
        self.assertEqual(direct['suite'],'v7-2')
        active={123:direct}
        next_job=fast.choose([j for j in pending if j!=direct],active,4,1)
        self.assertEqual(next_job['suite'],'v7-1')
        self.assertIsNone(fast.choose(pending,{i:next_job for i in range(4)},4,1))

    def test_native_generated_schemas_match_original_wire_descriptors(self):
        code='''
import hashlib,importlib,json
from pathlib import Path
import mlagents_envs.communicator_objects as p
result={}
for name in sorted(Path(p.__file__).parent.glob('*_pb2.py')):
 m=importlib.import_module('mlagents_envs.communicator_objects.'+name.stem)
 result[name.name]=hashlib.sha256(m.DESCRIPTOR.serialized_pb).hexdigest()
print(json.dumps(result,sort_keys=True))
'''
        baseline=subprocess.check_output([sys.executable,'-c',code],cwd=fast.ROOT,text=True)
        native=subprocess.check_output([sys.executable,'-c',
            'from scripts.connectome_fast_runtime import install;install();\n'+code],cwd=fast.ROOT,text=True)
        self.assertEqual(json.loads(baseline),json.loads(native))

    def test_completed_and_stopped_main_checkpoints_remain_unmodified(self):
        records=fast.read(fast.ROOT/'reports/v7/acceleration_resume_points.json')
        for record in records:
            self.assertEqual(fast.sha256(Path(record['run'])/'checkpoints/latest.pt'),record['checkpoint_sha256'])

    def test_state_refresh_accepts_prior_scheduler_metadata(self):
        # Main-study completed seed 11 remains ready for evaluation, never retraining.
        job=fast.jobs()[0]
        job.update(phase='train',step=0,complete=False,checkpoint_sha256='stale',log='old')
        state=fast.inspect_job(job)
        self.assertTrue(state['complete'])
        self.assertEqual(state['step'],2_000_000)
        self.assertIn(state['phase'],('evaluate','done'))

    def test_supervisor_failure_stops_other_workers(self):
        class Process:
            def __init__(self,pid,code=None):self.pid,self.code,self.signals=pid,code,[]
            def poll(self):return self.code
            def send_signal(self,sig):self.signals.append(sig);self.code=130
            def wait(self):return self.code
            def terminate(self):self.code=0
        auxiliary,broken,other=Process(1),Process(2,1),Process(3)
        pending=[dict(index=i,suite='v7-1',phase='train',run='unused') for i in (0,1)]
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            with (directory/'lock').open('a') as lock, patch.object(fast,'DIRECTORY',directory), \
                    patch.object(fast,'validate_runtime',return_value=dict(workers=2,mps_workers=1)), \
                    patch.object(fast,'preflight',return_value=pending), \
                    patch.object(fast,'sha256',return_value='fixture'), patch.object(fast.signal,'signal'), \
                    patch.object(fast.time,'sleep'), \
                    patch.object(fast.subprocess,'Popen',side_effect=[auxiliary,broken,other]):
                fast.supervisor([os.dup(lock.fileno())])
            self.assertTrue(other.signals)
            self.assertEqual(fast.read(directory/'status.json')['status'],'failed')
            self.assertFalse(fast.read(directory/'summary.json')['complete'])
            self.assertFalse((directory/'experiment.pid').exists())

    def test_stop_request_does_not_launch_new_jobs(self):
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);(directory/'stop.request').touch()
            with (directory/'lock').open('a') as lock, patch.object(fast,'DIRECTORY',directory), \
                    patch.object(fast,'validate_runtime',return_value=dict(workers=2,mps_workers=1)), \
                    patch.object(fast,'preflight',return_value=[dict(phase='train')]), \
                    patch.object(fast,'sha256',return_value='fixture'), patch.object(fast.signal,'signal'), \
                    patch.object(fast.subprocess,'Popen',return_value=MagicMock()) as launch:
                fast.supervisor([os.dup(lock.fileno())])
                self.assertEqual(launch.call_count,1)  # Only caffeinate, no experiment worker.
            self.assertEqual(fast.read(directory/'status.json')['status'],'stopped')


class MetalTests(unittest.TestCase):
    def test_real_metal_csr_includes_transfers_and_preserves_empty_rows(self):
        import numpy as np
        import torch
        from scipy.sparse import csr_matrix
        if not torch.backends.mps.is_available():self.skipTest('Metal device access required')
        from scripts.connectome_metal import MetalCSR
        matrix=csr_matrix(np.array([[0,1,-.5,0],[0,0,0,0],[.25,0,0,2],[1,0,0,0]],np.float32))
        fast_matrix=MetalCSR(matrix)
        state=np.array([[1,0],[0,1],[1,1],[0,1]],np.float32)
        np.testing.assert_array_equal(fast_matrix.dot(state),matrix.dot(state))


if __name__=='__main__':unittest.main()
