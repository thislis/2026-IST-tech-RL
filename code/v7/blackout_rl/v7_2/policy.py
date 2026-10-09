"""No planner import or fallback for controlled slots. Explicit clock, never observation deduplication."""
import copy
import numpy as np
import torch
from .sensory import SemanticRetina
from .dynamics import WholeBrain
from .decoder import DIRECTIONS, fixed_decode


class DirectPolicy:
    def __init__(self, graph, model, *, controlled_slots=(0,), noise_seed=0, sensor_interval=5, intervention='normal', dynamics=None, decoder_config=None):
        if not controlled_slots or len(set(controlled_slots)) != len(controlled_slots) or any(s not in range(5) for s in controlled_slots):
            raise ValueError('controlled slots must be a unique subset of 0..4')
        if intervention not in ('normal','sensory_block','frozen_frame','output_block'):
            raise ValueError('unknown causal intervention')
        self.slots = tuple(controlled_slots)
        self.brain = WholeBrain(graph,slots=len(self.slots),seed=noise_seed,config=dynamics)
        self.mapping = graph.metadata['retinal_mapping']
        if [m['index'] for m in self.mapping] != self.brain.input_indices.tolist():
            raise ValueError('retinal mapping does not match input neuron order')
        self.model, self.noise_seed, self.sensor_interval, self.intervention = model, noise_seed, sensor_interval, intervention
        self.decoder_config = decoder_config or dict(dt=.02, turn_gain=40., turn_limit=2., movement_threshold=.02)
        self.sensor = SemanticRetina()
        self.episode = None
        self.reset_episode_state('initial',0)

    def reset_episode_state(self, episode_id, team_id, *, noise_seed=None):
        if episode_id is None or team_id not in (0,1):
            raise ValueError('explicit episode and team required')
        self.episode, self.team = str(episode_id), team_id
        self.brain.reset(self.noise_seed if noise_seed is None else noise_seed)
        self.headings = np.full(len(self.slots), 0. if team_id == 0 else np.pi/2)
        self.last_step, self.last_actions, self.last_features = -1, None, None
        self.current = np.zeros_like(self.brain.sensory_filter)
        self.frames, self.neural_ticks = 0, 0

    def features(self, observations, *, episode_id, env_step_id, team_id):
        if str(episode_id) != self.episode or team_id != self.team:
            raise ValueError('call reset_episode_state explicitly for a new episode/team')
        if env_step_id == self.last_step:
            return self.last_features.copy()
        if env_step_id != self.last_step + 1:
            raise ValueError('game steps must be consecutive; no dropped or reversed neural ticks')
        if env_step_id % self.sensor_interval == 0 and (self.intervention != 'frozen_frame' or env_step_id == 0):
            images = [self.sensor.render(observations[f'unit_{team_id*5+s}'],team_id*5+s,self.headings[i]) for i,s in enumerate(self.slots)]
            self.current = np.stack([self.sensor.sample(img,self.mapping) for img in images])
            self.frames += len(images)
        current = np.zeros_like(self.current) if self.intervention == 'sensory_block' else self.current
        rates = self.brain.step(current)
        if self.intervention == 'output_block':
            rates[:] = 0
        self.last_step, self.last_features, self.last_actions = env_step_id, rates.copy(), None
        self.neural_ticks += len(self.slots)
        return rates

    def act(self, observations, *, episode_id, env_step_id, team_id, deterministic=True):
        rates = self.features(observations,episode_id=episode_id,env_step_id=env_step_id,team_id=team_id)
        if self.last_actions is not None:
            return copy.deepcopy(self.last_actions)
        if self.model is None:
            indices,self.headings = fixed_decode(rates,[p['name'] for p in self.brain.output_ports],self.headings,**self.decoder_config)
        else:
            with torch.no_grad():
                logits = self.model.readout(torch.as_tensor(rates))
                indices = (logits.argmax(-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()).numpy()
            self.commit_actions(indices)
        self.last_actions = {f'unit_{team_id*5+s}':DIRECTIONS[int(indices[i])].copy() for i,s in enumerate(self.slots)}
        return copy.deepcopy(self.last_actions)

    def commit_actions(self, indices):
        for i,index in enumerate(indices):
            if int(index):
                self.headings[i] = (int(index)-1)*np.pi/4

    def state_dict(self):
        return dict(brain=self.brain.state_dict(), episode=self.episode, team=self.team, headings=self.headings.copy(),
                    last_step=self.last_step, current=self.current.copy(), last_features=copy.deepcopy(self.last_features),
                    last_actions=copy.deepcopy(self.last_actions), frames=self.frames, neural_ticks=self.neural_ticks)

    def load_state_dict(self,state):
        self.brain.load_state_dict(state['brain'])
        for key in ('episode','team','headings','last_step','current','last_features','last_actions','frames','neural_ticks'):
            setattr(self,key,copy.deepcopy(state[key]))
