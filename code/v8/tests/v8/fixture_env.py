"""Explicit synthetic transport for engineering tests; never a game performance benchmark."""
import copy
import numpy as np
from tests.test_mappo_v6 import all_observations
from blackout_rl.v8.contracts import StepContext,OutcomeEvent

class FixtureEnv:
    def __init__(self,length=7,fail_after=None):
        self.length=length;self.fail_after=fail_after;self.total=0;self.counter=0;self.run='synthetic';self.reset_count=0
    def reset(self,seed):
        self.map_seed=seed;self.episode=f'synthetic:{self.counter}';self.counter+=1
        self.decision=0;self.reset_count+=1;self.obs=all_observations()
        return copy.deepcopy(self.obs),dict(score_points=(0,0))
    def context(self,team,version=0):
        return StepContext(self.run,'fixture',self.episode,self.decision,self.decision,self.decision*.02,self.map_seed,team,version)
    def step(self,actions):
        if self.fail_after is not None and self.total>=self.fail_after:raise RuntimeError('injected transport crash')
        before=self.decision;self.total+=1;self.decision+=1
        terminal=self.decision==self.length
        points=(100,42) if terminal else (self.decision,0)
        event=OutcomeEvent(self.run,self.episode,self.episode+':end',before,self.decision,.02*self.decision,
            points,0,'engine','target_score',True,False,'a'*64,'b'*64) if terminal else None
        return copy.deepcopy(self.obs),terminal,False,dict(score_points=points,outcome=event)
    def close(self):pass
