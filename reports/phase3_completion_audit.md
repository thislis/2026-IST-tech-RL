# Phase 3 Completion Audit

Audited against all 32 `AGENT-*` rows in `README.md`.

## Checked with evidence (32)

`01-32`

- Seed isolation, paired bootstrap, MAPPO/CTDE, fair 128-step live ablation,
  representation features, real trajectory role analysis, curriculum gates,
  frozen self-play infrastructure, deterministic submission, and two-file
  clean-room loading have executable tests/evidence.
- Full suite: 164 tests passed after adding the MAPPO-vs-win70 training path.
- Live MAPPO smoke: 8 steps / 40 actor rows, central vector `(40,123)`, finite
  KL and value loss.
- Live IPPO/MAPPO ablation: five side-swapped dev seed pairs, common opponent
  and 128-step budget; both arms 0/10 and mean score difference `-96.5`.

## Completion outcome

- AGENT-29 selected the trained legacy encoder: 31.3% fewer parameters and
  39.8% lower CPU median latency with no held-out score regression.
- AGENT-31 passed standalone deterministic smoke on CPU and actual Apple MPS.
- AGENT-32 completed the promotion review but did not promote a model. The
  latest learned candidate lost 0/10 to the incumbent, while the guarded
  incumbent cannot be reproduced by the current two-file standalone policy.

Every planned Phase 3 experiment/review is executed. Completion does not imply
that a final-submission-ready model exists; that gate remains fail-closed.
