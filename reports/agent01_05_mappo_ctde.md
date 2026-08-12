# AGENT-01~05 MAPPO/CTDE

## Implementation

- `CentralizedStateBuilder`: ten unit blocks, ten self-class vectors, absolute
  team scores, time, and team-perspective semantic map (`123` vector fields).
- `MAPPOActorCritic`: decentralized IPPO actor and training-only centralized
  critic.
- `JointRolloutBuffer`: one team value target with per-slot alive masks; death
  masks actor rows without cutting the team GAE chain, so respawn is supported.
- `initialize_mappo_from_checkpoint`: copies the complete IPPO actor exactly and
  initializes only the centralized critic from scratch.
- `MAPPOParallelRolloutCollector` and `mappo_update`: actual PettingZoo/Unity
  collection and clipped PPO update path.

## Evidence

- Unit/regression evidence: `tests/test_phase3_mappo.py`.
- Live Unity smoke: 8 time steps / 40 actor rows, centralized state `(40,123)`,
  finite KL (`2.898e-05`) and value loss (`5.973e-06`).
- Fair live ablation: `logs/phase3_mappo_ablation_128.json`.
  Both arms used initial `base_r13_bc_warm_start.pt`, train seed `3001`, frozen
  `scripted-battery-v1`, 128 environment steps, and five side-swapped dev seed
  pairs (`3101..3105`). Both were 0/10 with mean score difference `-96.5`;
  therefore the short-budget result is a tie and does not claim an improvement.

The ablation establishes the comparison contract and executable training path.
Long-budget model selection remains governed by the fixed dev/test protocol.
