from collections import Counter
from .contracts import ContractError
from .telemetry import append

class ExposureLedger:
    def __init__(self,path):
        self.path=path;self.counts=Counter();self.steps=0
    def record(self,context,opponent_hash):
        if not opponent_hash:raise ContractError('opponent identity required even for partial episodes')
        key=(context.map_seed,context.team,opponent_hash)
        self.counts[key]+=1;self.steps+=1
        append(self.path,dict(**context.__dict__,opponent_hash=opponent_hash,global_step=self.steps))
    def state_dict(self):
        return dict(steps=self.steps,cells=[dict(map_seed=k[0],side=k[1],opponent_hash=k[2],steps=v) for k,v in self.counts.items()])
    def load_state_dict(self,state):
        self.steps=state['steps'];self.counts=Counter({(c['map_seed'],c['side'],c['opponent_hash']):c['steps'] for c in state['cells']})
