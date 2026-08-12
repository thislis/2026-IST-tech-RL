# Phase 3 Completion Audit

Audited against all 32 `AGENT-*` rows in `README.md`.

## Checked with evidence (21)

`01-05, 07, 09-10, 12-13, 15-20, 25-28, 30`

- Seed isolation, paired bootstrap, MAPPO/CTDE, fair 128-step live ablation,
  representation features, real trajectory role analysis, curriculum gates,
  frozen self-play infrastructure, deterministic submission, and two-file
  clean-room loading have executable tests/evidence.
- Full suite: 139 tests passed.
- Live MAPPO smoke: 8 steps / 40 actor rows, central vector `(40,123)`, finite
  KL and value loss.
- Live IPPO/MAPPO ablation: five side-swapped dev seed pairs, common opponent
  and 128-step budget; both arms 0/10 and mean score difference `-96.5`.

## Intentionally still open (11)

- `06, 08, 11`: attention, global/local crop, and auxiliary modules exist and
  are tested, but no trained held-out ablation establishes their game effect.
- `14`: potential navigation shaping and annealing exist, but score-only versus
  shaping sample-efficiency curves have not been run.
- `21-23`: stability/matrix/regression automation exists, but there is no real
  multi-generation snapshot-mixture training history to evaluate.
- `24`: PSRO must not be considered before the snapshot pool has saturated;
  the decision gate correctly remains false.
- `29`: latency benchmarking and conservative selection exist, but trained
  encoder-removal candidates do not yet exist.
- `31`: CPU passes; this host reports MPS built but unavailable and CUDA false,
  so an actual GPU smoke cannot be claimed.
- `32`: final promotion is blocked by the empirical items above and correctly
  requires CPU+GPU, side bias, score, and past-opponent regression evidence.

No unchecked item is silently represented as complete. Phase 3's parent
checkbox therefore remains open.
