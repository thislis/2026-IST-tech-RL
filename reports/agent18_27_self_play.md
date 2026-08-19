# AGENT-18~27 Self-play Infrastructure

Implemented and verified now:

- inference-only checkpoint opponent with immutable artifact SHA and no
  optimizer/trainable state;
- bounded snapshot pool with configurable latest/history mixture;
- prefix-balanced physical-side sampler;
- entropy/KL/value-error stability gate;
- directed paired-seed checkpoint matrix builder;
- past-opponent regression detector;
- opponent suite covering random, scripted, IPPO, historical MAPPO, latest;
- PSRO decision gate that stays false before at least eight generations, a
  four-generation plateau, and repeated cyclic regressions.

AGENT-21~24 now have an actual eight-generation run. PPO stability passed, the
major-generation directed matrix is complete on a paired 5,000-step dev
horizon, the defined past/exploiter regression suite passed, and the snapshot
pool saturated. PSRO remains deferred because cyclic regressions were zero.
See `reports/agent21_24_self_play_empirical.md` and
`logs/phase3_agent21_24_self_play.json`.
