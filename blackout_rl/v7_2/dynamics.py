"""Explicit Fly64-inspired proxy, not a physiological LIF reproduction.

v'=exp(-dt/tau)*v + gain*W*spike + tonic + Bernoulli(noise_hz*dt)*amplitude + retina;
spike'=v'>=threshold; v'[spike']=0. W[dst,src] uses signed incoming count normalization.
Every retained edge participates. SciPy CSR computes exact sparse multiplication.
"""
import copy
from dataclasses import asdict, dataclass
import numpy as np
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class DynamicsConfig:
    dynamics_id: str = 'fly64_proxy_v1'
    dt: float = .02
    tau: float = .1
    recurrent_gain: float = 1.5
    tonic: float = .180
    noise_hz: float = 1.2
    noise_amplitude: float = .22
    threshold: float = 1.
    sensory_gain: float = .8
    sensory_ema: float = .5
    window_ticks: int = 13


class WholeBrain:
    def __init__(self, graph, *, slots=1, seed=0, config=None):
        self.config = config or DynamicsConfig()
        if self.config.dynamics_id != 'fly64_proxy_v1' or self.config.dt != .02:
            raise ValueError('unsupported dynamics or game time interval')
        self.n, self.slots, self.seed = graph.n, slots, seed
        self.input_indices = np.array(graph.metadata['input_ports'][0]['indices'], dtype=np.int64)
        self.output_ports = graph.metadata['output_ports']
        sign = np.array([-1 if a['transmitter'].lower() in ('gaba','glutamate','histamine') else 1 for a in graph.metadata['node_attributes']], dtype=np.float32)
        counts = graph.synapse_count.astype(np.float32)
        normalizer = np.maximum(1, np.bincount(graph.edge_dst, weights=counts, minlength=graph.n))
        weights = (counts * sign[graph.edge_src] / normalizer[graph.edge_dst]).astype(np.float32)
        self.matrix = coo_matrix((weights,(graph.edge_dst,graph.edge_src)),shape=(graph.n,graph.n)).tocsr()
        self.matrix.data.flags.writeable = False
        self.matrix.indices.flags.writeable = False
        self.matrix.indptr.flags.writeable = False
        self.reset(seed)

    def reset(self, seed=None):
        self.rng = np.random.default_rng(self.seed if seed is None else seed)
        self.voltage = np.zeros((self.slots,self.n),np.float32)
        self.spikes = np.zeros_like(self.voltage)
        self.sensory_filter = np.zeros((self.slots,len(self.input_indices)),np.float32)
        self.previous_sensory = np.zeros_like(self.sensory_filter)
        self.history = np.zeros((self.config.window_ticks,self.slots,len(self.output_ports)),np.float32)
        self.ticks = 0

    def step(self, sensory):
        cfg = self.config
        sensory = np.asarray(sensory,dtype=np.float32)
        if sensory.shape != self.sensory_filter.shape or not np.isfinite(sensory).all():
            raise ValueError('invalid sensory current')
        self.sensory_filter *= cfg.sensory_ema
        self.sensory_filter += (1-cfg.sensory_ema)*(sensory + np.abs(sensory-self.previous_sensory))
        self.previous_sensory[:] = sensory
        self.voltage *= np.float32(np.exp(-cfg.dt/cfg.tau))
        self.voltage += cfg.recurrent_gain*self.matrix.dot(self.spikes.T).T + cfg.tonic
        self.voltage += (self.rng.random(self.voltage.shape)<cfg.noise_hz*cfg.dt)*cfg.noise_amplitude
        self.voltage[:,self.input_indices] += cfg.sensory_gain*self.sensory_filter
        self.spikes[:] = self.voltage >= cfg.threshold
        self.voltage[self.spikes.astype(bool)] = 0
        if not np.isfinite(self.voltage).all():
            raise RuntimeError('non-finite whole-brain state')
        rates = np.stack([self.spikes[:,p['indices']].mean(axis=1) for p in self.output_ports],axis=-1)
        self.history[self.ticks % cfg.window_ticks] = rates
        self.ticks += 1
        return self.history.mean(axis=0).copy()

    def state_dict(self):
        return dict(config=asdict(self.config), voltage=self.voltage.copy(), spikes=self.spikes.copy(),
                    sensory_filter=self.sensory_filter.copy(), previous_sensory=self.previous_sensory.copy(),
                    history=self.history.copy(), ticks=self.ticks, rng=copy.deepcopy(self.rng.bit_generator.state))

    def load_state_dict(self, state):
        if state['config'] != asdict(self.config):
            raise ValueError('dynamics configuration mismatch')
        for key in ('voltage','spikes','sensory_filter','previous_sensory','history'):
            value = np.asarray(state[key],np.float32)
            if value.shape != getattr(self,key).shape or not np.isfinite(value).all():
                raise ValueError('brain state shape/value mismatch')
            setattr(self,key,value.copy())
        self.ticks = int(state['ticks'])
        self.rng.bit_generator.state = copy.deepcopy(state['rng'])
