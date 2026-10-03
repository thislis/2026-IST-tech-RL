"""The only construction path for train, evaluation, and research export."""
import copy
import time
import torch
from blackout_rl.model_contract import load_checkpoint
from .contracts import ContractError, digest, file_hash
from .decoders import decode, DECODERS
from .model import ResidualModel
from .planner_adapter import PlannerAdapter, REVISION
from .candidates import build_mask, execute, STABLE_IDS
from .observation import validate_and_hash

DEFAULT_POLICY = dict(mode='c1',width=64,initial_q=.1,decoder='gate_then_conditional_argmax_v1',
                      threshold=.5,controlled_slots=[0,1,2,3,4],action_repeat=1,
                      planner_revision=REVISION,executor='identity_executor_v1',intervention='normal')

def resolve_policy(config):
    if set(config)-set(DEFAULT_POLICY):
        raise ContractError('unused policy config keys: '+str(sorted(set(config)-set(DEFAULT_POLICY))))
    result = dict(DEFAULT_POLICY,**config)
    if result['mode'] not in ('c1','flat','planner') or result['decoder'] not in DECODERS:
        raise ContractError('conditional C2/C3 modes are not enabled')
    for key in ('controlled_slots','action_repeat','planner_revision','executor','intervention'):
        if result[key] != DEFAULT_POLICY[key]:
            raise ContractError('unsupported effective '+key)
    if not 0 <= result['threshold'] <= 1:
        raise ContractError('invalid decoder threshold')
    return result

class V8Policy:
    def __init__(self, model, config, seed=0):
        self.model,self.config = model,resolve_policy(config)
        self.generator=torch.Generator().manual_seed(seed)
        self.key=None
        self.last_decision=-1

    def reset(self, context):
        if context.decision_id != 0:
            raise ContractError('episode starts at decision 0')
        self.key=context.key
        self.last_decision=-1
        self.context=PlannerAdapter(context.team)
        self.cached=None

    @torch.no_grad()
    def decide(self, observations, context, *, behavior=False):
        if self.key != context.key:
            raise ContractError('explicit reset required for new env/episode/team')
        obs_hash=validate_and_hash(observations,context.team)
        call=(obs_hash,behavior,context.behavior_version)
        if context.decision_id == self.last_decision:
            if call != self.last_call:
                raise ContractError('conflicting duplicate decision')
            return copy.deepcopy(self.cached)
        if context.decision_id != self.last_decision+1:
            raise ContractError('skipped/reversed decision ID')
        t0=time.perf_counter()
        features,_,planner,alternatives=self.context.prepare(observations,self.model)
        t1=time.perf_counter()
        valid,reasons=build_mask(observations,context.team,planner,alternatives)
        t2=time.perf_counter()
        distribution=self.model.distribution(features,valid)
        decoder='sampled_v1' if behavior else self.config['decoder']
        chosen=decode(distribution,decoder,self.config['threshold'],self.generator)
        if self.config['mode']=='planner':
            if behavior:
                raise ContractError('fixed planner is not a trainable arm')
            chosen=torch.tensor(0)
        index=int(chosen)
        actuators,executor=execute(index,planner,alternatives,valid)
        t3=time.perf_counter()
        greedy=int(decode(distribution,self.config['decoder'],self.config['threshold'],self.generator)) if self.config['decoder']!='sampled_v1' else index
        result=dict(actions={f'unit_{context.team*5+s}':actuators[s] for s in range(5)},
                    features=features.cpu(),valid=valid,planner=planner,alternatives=alternatives,
                    action=index,old_log_prob=distribution.log_prob(chosen).cpu(),
                    q=float(distribution.q),p_keep=float(distribution.probs[0]),max_correction=float(distribution.probs[1:].max()),
                    margin=float(distribution.probs[1:].max()-distribution.probs[0]),entropy=float(distribution.entropy()),
                    log_probs=distribution.log_probs.cpu(),greedy_action=greedy,mask_reasons=reasons,
                    observation_hash=obs_hash,planner_state_digest=self.context.state_digest(),
                    teacher_valid=self.context.teacher_valid,executor=executor,candidate_ids=STABLE_IDS,
                    effective_config_sha256=digest(self.config),
                    spans=dict(planner_encoder_features_seconds=t1-t0,mask_seconds=t2-t1,distribution_decoder_executor_seconds=t3-t2))
        self.last_decision=context.decision_id; self.last_call=call; self.cached=copy.deepcopy(result)
        return result


def make_policy(config, checkpoint=None, capabilities=None, *, seed=0, actor=None):
    policy_config=resolve_policy(config['policy'])
    caps=capabilities or {}
    if not caps.get('explicit_episode_decision_id') or not caps.get('canonical_agent_ids') or not caps.get('stateful'):
        raise ContractError('v8 requires explicit identity/clock/state capabilities')
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        if actor is None:
            path=config['encoder']['path']
            if file_hash(path)!=config['encoder']['sha256']:
                raise ContractError('frozen encoder hash mismatch')
            actor,_=load_checkpoint(path)
            actor=actor.actor_critic
        model=ResidualModel(actor,'flat' if policy_config['mode']=='flat' else 'c1',policy_config['width'],policy_config['initial_q'])
    if checkpoint is not None:
        if checkpoint['config_sha256']!=digest(config):
            raise ContractError('checkpoint/config mismatch')
        model.load_state_dict(checkpoint['model'])
    return V8Policy(model.eval(),policy_config,seed+40000)
