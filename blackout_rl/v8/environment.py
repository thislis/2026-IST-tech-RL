"""Historical outcome validation; live environment is the unchanged provided API."""
from .contracts import ContractError, OutcomeEvent
from .provided_environment import make_original_env, reject_research_environment

class OutcomeBroker:
    def __init__(self, build_sha256, protocol_sha256):
        self.build,self.protocol=build_sha256,protocol_sha256
        self.events={};self.by_episode={}

    def receive(self, event):
        if not isinstance(event,OutcomeEvent):event=OutcomeEvent(**event)
        if (event.build_sha256,event.protocol_sha256)!=(self.build,self.protocol):
            raise ContractError('outcome build/protocol mismatch')
        if event.key in self.events:
            if self.events[event.key]!=event:raise ContractError('conflicting duplicate event')
            return False
        key=(event.run_id,event.episode_id)
        if key in self.by_episode:raise ContractError('two outcomes for one episode')
        self.events[event.key]=event;self.by_episode[key]=event
        return True

    def join(self, run_id, episode_id, decision_id):
        event=self.by_episode.get((run_id,episode_id))
        if event is None:raise ContractError('terminal has no authoritative outcome; no implicit gameplay step')
        if event.decision_id!=decision_id:raise ContractError('terminal decision/outcome mismatch')
        return event



make_channel = reject_research_environment

class UnityResearchEnv:
    """Retired API name: never starts the modified game."""
    def __init__(self, *args, **kwargs):
        reject_research_environment()
