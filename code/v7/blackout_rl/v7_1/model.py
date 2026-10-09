import math
import torch
from torch import nn
from blackout_rl.mappo_v6 import V6Model, CONTEXT_SIZE
from blackout_rl.connectome.sparse_ops import propagate


class GraphResidualHead(nn.Module):
    def __init__(self, feature_dim, graph, *, channels=4, rounds=4, leak=.5, initial_override_probability=.1):
        super().__init__()
        graph.validate()
        if rounds < 1 or not 0 < leak <= 1 or not all(graph.reachable(rounds)):
            raise ValueError('every output pool needs an input path of at most K-1 edges')
        self.n, self.channels, self.rounds, self.leak = graph.n, channels, rounds, leak
        self.register_buffer('src', torch.tensor(graph.edge_src.astype('int64')))
        self.register_buffer('dst', torch.tensor(graph.edge_dst.astype('int64')))
        self.edge_weight = nn.Parameter(torch.randn(len(self.src)) * .1)
        self.bias = nn.Parameter(torch.zeros(graph.n, channels))
        inputs, outputs = graph.metadata['input_ports'], graph.metadata['output_ports']
        pin = torch.zeros(graph.n, len(inputs))
        qout = torch.zeros(len(outputs), graph.n)
        for j, port in enumerate(inputs):
            pin[port['indices'], j] = 1.
        for j, port in enumerate(outputs):
            qout[j, port['indices']] = 1. / len(port['indices'])
        self.register_buffer('pin', pin)
        self.register_buffer('qout', qout)
        self.input_adapter = nn.Linear(feature_dim, len(inputs) * channels)
        self.output_head = nn.Linear(len(outputs) * channels, 8)
        nn.init.zeros_(self.output_head.weight)
        nn.init.constant_(self.output_head.bias, math.log(initial_override_probability / (40 * (1-initial_override_probability))))

    def forward(self, features):
        ports = self.input_adapter(features).reshape(*features.shape[:-1], self.pin.shape[1], self.channels)
        drive = torch.einsum('np,...pc->...nc', self.pin, ports)
        state = torch.zeros_like(drive)
        normalizer = self.edge_weight.new_zeros(self.n).index_add(0, self.dst, self.edge_weight.abs()).clamp_min(1)
        weights = self.edge_weight / normalizer[self.dst]
        for _ in range(self.rounds):
            state = (1-self.leak)*state + self.leak*torch.tanh(propagate(self.src, self.dst, weights, state, self.n)+drive+self.bias)
        return self.output_head(torch.einsum('pn,...nc->...pc', self.qout, state).flatten(-2))


class V7ResidualModel(V6Model):
    def __init__(self, actor, graph=None, *, variant='graph', channels=4, rounds=4, leak=.5, initial_override_probability=.1):
        super().__init__(actor, initial_override_probability)
        self.variant = variant
        if variant == 'mlp':
            return
        head = GraphResidualHead(int(actor.model_config['hidden_dim']) + CONTEXT_SIZE, graph,
                                channels=channels, rounds=rounds, leak=leak,
                                initial_override_probability=initial_override_probability)
        if variant == 'matched_mlp':
            target = sum(p.numel() for p in head.parameters())
            inputs = int(actor.model_config['hidden_dim']) + CONTEXT_SIZE
            width = max(1, round((target-8)/(inputs+9)))
            self.residual = nn.Sequential(nn.Linear(inputs, width), nn.Tanh(), nn.Linear(width, 8))
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.constant_(self.residual[-1].bias, math.log(initial_override_probability/(40*(1-initial_override_probability))))
            if abs(sum(p.numel() for p in self.residual.parameters())/target-1) > .05:
                raise ValueError('matched MLP exceeds 5% parameter tolerance')
        elif variant in ('graph', 'rewired'):
            self.residual = head
        else:
            raise ValueError(f'unknown variant: {variant}')
