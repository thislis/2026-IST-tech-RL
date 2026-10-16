"""Public-API match loop with the provided runner's scoring and strict validity checks.

No supplied environment, observation, transport or game code is modified.
The provided runner ignores truncations; this research runner rejects them.
"""
import numpy as np
from .trajectory import pack


def validated_match(env, model_a, model_b, swap_teams=False):
    from blackout_env.competition.match import MatchResult
    names={f'unit_{i}' for i in range(10)}
    team_a={f'unit_{i}' for i in range(5)}; team_b=names-team_a
    own=team_b if swap_teams else team_a
    obs,_=env.reset(); totals={n:0. for n in env.possible_agents}; steps=0; previous=None
    if set(totals)!=names: raise ValueError('match agent contract changed')
    while env.agents:
        if set(env.agents)!=names or set(obs)!=names: raise ValueError('partial match observations/agents')
        # Validate both public observations and time without changing them.
        for side in (team_a,team_b): pack(obs,sorted(side))
        current=float(next(iter(obs.values()))['vector'][95])
        if previous is not None and current>previous+1e-3: raise ValueError('time reset without termination')
        previous=current
        actions=model_a.act({n:obs[n] for n in env.agents if n in own})
        actions.update(model_b.act({n:obs[n] for n in env.agents if n not in own}))
        if set(actions)!=names: raise ValueError('incomplete actions')
        for a in actions.values():
            if a.shape!=(2,) or a.dtype!=np.float32 or not np.isfinite(a).all() or (np.abs(a)>1).any():
                raise ValueError('invalid match action')
        obs,rewards,terms,truncs,_=env.step(actions)
        if set(rewards)!=names or set(terms)!=names or set(truncs)!=names: raise ValueError('partial match transition')
        if any(truncs.values()) or len({bool(x) for x in terms.values()})!=1: raise ValueError('truncated/partial terminal match')
        if not all(np.isfinite(float(x)) for x in rewards.values()): raise ValueError('nonfinite reward')
        for n in rewards: totals[n]+=float(rewards[n])
        steps+=1
        done=all(terms.values())
        if done != (not env.agents): raise ValueError('match termination/active agents mismatch')
    if steps==0: raise ValueError('empty match')
    # Match summation order to the installed run_match (sets of agent names).
    a=sum(totals[n] for n in team_a);b=sum(totals[n] for n in team_b)
    if swap_teams:a,b=b,a
    return MatchResult(winner=0 if a>b else 1 if b>a else None,
        team_a_total_reward=a,team_b_total_reward=b,episode_steps=steps)
