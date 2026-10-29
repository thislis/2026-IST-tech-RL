"""Portable deterministic scorer+MAP event core with a real host call boundary.

Navigation and the explicit FSM remain on the CPU host. This artifact does
not claim to be a complete obs/state -> action/state submission graph.
"""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


def scorer_hash(scorer):
    digest = hashlib.sha256()
    for name, tensor in sorted(scorer.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((array.dtype, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class EventCore(nn.Module):
    def __init__(self, scorer, tasks=24, units=5, learned=True):
        super().__init__()
        self.scorer = copy.deepcopy(scorer).cpu().eval()
        self.tasks, self.units, self.learned = int(tasks), int(units), bool(learned)
        sets = torch.arange(1 << units)
        members = ((sets[:, None] >> torch.arange(units)) & 1).float()
        self.register_buffer('members', members)
        self.register_buffer('counts', members.sum(-1))
        self.register_buffer('contained', (sets[:, None] & sets[None]) == sets[None])
        self.register_buffer('previous', sets[:, None] ^ sets[None])
        self.register_buffer('initial_valid', sets == 0)

    def forward(self, features, prior, legal, capacities):
        scores = self.scorer(features, prior) if self.learned else prior
        weights = scores.transpose(0, 1) @ self.members.transpose(0, 1)
        allowed = ((~legal).float().transpose(0, 1) @ self.members.transpose(0, 1)) == 0
        allowed = allowed & (self.counts[None] <= capacities[:, None])
        value = torch.zeros_like(self.counts)
        valid = self.initial_valid
        history, validity = [value], [valid]
        for task in range(self.tasks):
            ok = self.contained & valid[self.previous] & allowed[task, None]
            candidates = value[self.previous] + weights[task, None]
            value = torch.where(ok, candidates, torch.full_like(candidates, -1e30)).max(-1).values
            valid = ok.any(-1)
            history.append(value)
            validity.append(valid)
        state = torch.full((), (1 << self.units)-1, dtype=torch.long, device=scores.device)
        assignment = torch.full((self.units,), -1, dtype=torch.long, device=scores.device)
        for task in range(self.tasks-1, -1, -1):
            predecessor = self.previous[state]
            ok = self.contained[state] & validity[task][predecessor] & allowed[task]
            candidates = history[task][predecessor] + weights[task]
            subset = torch.where(ok, candidates, torch.full_like(candidates, -1e30)).argmax(-1)
            assignment = torch.where(self.members[subset] > 0, task, assignment)
            state = predecessor[subset]
        return assignment, scores, history[-1][-1], validity[-1][-1]


class OnnxEventBackend:
    """Optional host adapter, invoked only at deterministic joint decision events."""
    def __init__(self, path):
        import onnxruntime as ort
        path = Path(path)
        self.manifest = json.loads(path.with_name('manifest.json').read_text(encoding='utf-8'))
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                           providers=['CPUExecutionProvider'])
        self.inputs = {item.name for item in self.session.get_inputs()}
        self.calls = 0
        self._validated = None

    def validate_policy(self, policy):
        parameters = tuple(policy.scorer.parameters())
        versions = (None if any(torch.is_inference(p) for p in parameters) else
                    (id(policy), tuple(p._version for p in parameters)))
        if versions is not None and self._validated == versions:
            return
        if policy.arm != self.manifest['arm']:
            raise ValueError('ONNX event backend arm does not match policy')
        if not policy.arm.startswith('Q') and scorer_hash(policy.scorer) != self.manifest['scorer_sha256']:
            raise ValueError('ONNX scorer differs from the current policy weights')
        if self.manifest['tasks'] != int(policy.config.get('max_tasks', policy.config.get('task_limit',24))):
            raise ValueError('ONNX event task count does not match policy')
        self._validated = versions

    def __call__(self, features, prior, legal, capacities):
        values = (features, prior, legal, capacities)
        arrays = [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
                  for v in values]
        feeds = {name: value for name, value in zip(
            ('features','prior','legal','capacities'), arrays) if name in self.inputs}
        assignment, scores, _, feasible = self.session.run(None, feeds)
        if not bool(feasible):
            raise ValueError('ONNX event has no legal complete assignment')
        self.calls += 1
        return assignment, scores


def export_event_core(policy, directory, probes=6):
    if policy.config.get('policy_mode') == 'learned':
        raise ValueError('learned variable task support is supported by the Python export, not this fixed-width ONNX core')
    import onnx
    import onnxruntime as ort
    from .assignment import map_assignment
    if policy.arm in ('Q0', 'Q1', 'S') or policy.decoder == 'sample':
        raise ValueError('this development ONNX core supports deterministic joint MAP only')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tasks = int(policy.config.get('max_tasks', policy.config.get('task_limit',24)))
    core = EventCore(policy.scorer, tasks, learned=not policy.arm.startswith('Q')).eval()
    generator = torch.Generator().manual_seed(1206)
    features = torch.randn((5,tasks,64), generator=generator)
    prior = torch.randn((5,tasks), generator=generator)
    legal = torch.ones(5,tasks,dtype=torch.bool)
    capacities = torch.ones(tasks,dtype=torch.long)
    path = directory/'event_core.onnx'
    with torch.no_grad():
        torch.onnx.export(core,(features,prior,legal,capacities),str(path),opset_version=17,
                          dynamo=False,input_names=['features','prior','legal','capacities'],
                          output_names=['assignment','scores','map_value','feasible'])
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads=options.inter_op_num_threads=1
    session = ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    input_names = {item.name for item in session.get_inputs()}
    max_error = 0.
    with torch.no_grad():
        for index in range(int(probes)):
            features = torch.randn((5,tasks,64),generator=generator)
            prior = torch.randn((5,tasks),generator=generator)
            legal = torch.rand((5,tasks),generator=generator) > .3
            capacities = torch.randint(0,3,(tasks,),generator=generator)
            # Dedicated legal holds guarantee complete support; also exercise
            # capacity two and a forced locked row without duplicating tasks.
            legal[:,-5:] = torch.eye(5,dtype=torch.bool)
            capacities[-5:] = 1
            if index % 2:
                legal[0] = False
                legal[0,-5] = True
            args = (features,prior,legal,capacities)
            expected = core(*args)
            score = core.scorer(features,prior) if core.learned else prior
            torch.testing.assert_close(expected[0],map_assignment(score,capacities,legal))
            feeds = {name:arg.numpy() for name,arg in zip(['features','prior','legal','capacities'],args)
                     if name in input_names}
            actual = session.run(None,feeds)
            for a,b in zip(actual,expected):
                np.testing.assert_allclose(a,b.numpy(),atol=1e-5,rtol=1e-4)
            max_error=max(max_error,float(np.max(np.abs(actual[1]-expected[1].numpy()))))
    report = dict(schema='v12.onnx.event_core.1',opset=17,provider='CPUExecutionProvider',
                  torch=torch.__version__,onnx=onnx.__version__,onnxruntime=ort.__version__,
                  tasks=tasks,units=5,arm=policy.arm,scorer_sha256=scorer_hash(policy.scorer),
                  decoder='joint_map',parity_probes=int(probes),
                  max_score_error=max_error,bytes=path.stat().st_size,
                  scope='features/prior/legal/capacities -> assignment/scores/map_value/feasible',
                  event_skip='host must call only on a decision event; no Where-based fake skip',
                  public_submission=False,full_observation_state_graph=False,
                  state_contract='CPU host owns explicit runtime_state()/load_runtime_state(); no implicit ONNX persistence')
    (directory/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
