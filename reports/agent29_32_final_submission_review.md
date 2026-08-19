# AGENT-29~32 Final Submission Review

## AGENT-29: lightweight encoder selection

Three already-trained, equal-budget representation arms were benchmarked with
the official `(5,96)` vector and `(5,11,96,96)` map inputs for 50 CPU runs.

| Candidate | Parameters | CPU median | CPU p95 | Dev score difference |
| --- | ---: | ---: | ---: | ---: |
| global/local baseline | 229,594 | 1.240 ms | 1.380 ms | -95.4 |
| legacy, no local encoder | 157,818 | 0.746 ms | 0.827 ms | -90.3 |
| attention legacy | 166,202 | 0.865 ms | 0.968 ms | -93.4 |

All arms had 0/10 wins, so selection also required no mean-score regression.
The legacy encoder was selected: 31.3% fewer parameters, 39.8% lower median
latency, and a 5.1-point better held-out score difference than the baseline.
Optimizer state was removed without changing policy weights, producing
`checkpoints/phase3_lightweight_legacy.pt` (641,930 bytes).

## AGENT-31: CPU and GPU smoke

The standalone two-file policy passed on both CPU and Apple MPS in the same
privileged process used for live training.

| Device | Median / p95 | Deterministic | Contract checks |
| --- | ---: | --- | --- |
| CPU | 0.375 / 0.422 ms | 3 repeats identical | dtype and batch mismatch rejected |
| MPS | 0.782 / 1.465 ms | 3 repeats identical | dtype, batch, and device mismatch rejected |

Both devices returned `float32[5,2]`. The lightweight checkpoint also passed the
clean-room test containing only `policy.py` and `checkpoint.pt`.

## AGENT-32: final promotion decision

The latest learned candidate, `phase3_selfplay_gen_08.pt`, played the existing
`win_70_vs_scripted.pt` benchmark on all five dev seeds with side swapping.
It lost 0/10 with mean score difference `-94.9`; side win-rate gap was 0.0 and
the earlier past-opponent regression suite was clean. It therefore fails the
strength portion of the final promotion gate despite passing determinism,
clean-room, CPU, MPS, and regression checks.

The existing benchmark remains 10/10 against scripted on both its selection and
independent audit series, but it is a hybrid planner-guardrail/global-local
artifact. The current standalone two-file policy cannot load it and cannot
reproduce the planner override. Consequently:

- no new learned candidate is promoted;
- `win_70_vs_scripted.pt` remains the evaluation benchmark only;
- no checkpoint is declared final-submission-ready.

AGENT-32 is complete as a fail-closed promotion review, not as a successful
model promotion. Full measurements and all ten head-to-head episodes are in
[`logs/phase3_agent29_32_submission.json`](../logs/phase3_agent29_32_submission.json).

