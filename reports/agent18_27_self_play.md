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

AGENT-21~24 remain unchecked because the repository does not yet contain an
actual multi-generation self-play run. The code intentionally does not treat a
unit-level gate as empirical stability or snapshot-pool saturation evidence.
