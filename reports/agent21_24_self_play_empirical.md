# AGENT-21~24 Empirical Self-play Run

The run starts from the AGENT-14 score/terminal-only selection and performs
eight generations of live Unity PPO, 128 environment steps per generation.
Only one learner is updated. Every opponent is an immutable older checkpoint
sampled from an 8-slot pool with 0.5 latest probability; learner sides alternate
and finish 4:4.

## AGENT-21: PPO stability

All eight updates passed the declared thresholds.

| Metric | Observed range | Threshold |
| --- | ---: | ---: |
| entropy | 1.6994–1.9294 | at least 0.25 |
| approximate KL | 0.001716–0.008117 | at most 0.05 |
| value error (PPO value loss) | 0.000119–0.001320 | at most 10.0 |

The pool sampled generations `0, 0, 2, 0, 1, 0, 6, 7`. Snapshot SHA checks
passed before and after each generation.

## AGENT-22: checkpoint matrix

Major generations 0, 2, 4, and 8 were compared on dev seed 3101 with both
physical sides. Because these weak neural policies usually run to the full time
limit, the matrix uses an explicitly labelled 5,000-step fixed score horizon.
Each unordered pair is played once with side swapping; the reciprocal directed
entry is the exact inverse attribution of those same games. This produces all
12 directed cells from 12 live games without duplicate simulation.

| Model | vs gen 0 | vs gen 2 | vs gen 4 | vs gen 8 |
| --- | ---: | ---: | ---: | ---: |
| gen 0 | — | 0.50 / +3.5 | 0.50 / +3.5 | 0.50 / +0.5 |
| gen 2 | 0.00 / -3.5 | — | 0.50 / +3.5 | 0.50 / +3.5 |
| gen 4 | 0.00 / -3.5 | 0.00 / -3.5 | — | 0.00 / 0.0 |
| gen 8 | 0.50 / -0.5 | 0.00 / -3.5 | 0.00 / 0.0 | — |

Cells are `win rate / mean score difference`. One dev seed is adequate for the
requested automation evidence but not for a high-confidence strength ranking.

## AGENT-23: regression audit

Generation 8 was compared with generation 4 against past generations 0 and 2,
using a 0.05 win-rate tolerance. No past-opponent regression was detected. Both
generation 4 and 8 scored 0/2 in full games against
`checkpoints/win_70_vs_scripted.pt`, so there was also no relative exploiter
regression. This passes the defined narrow regression suite; it does **not** show
that generation 8 is stronger than the existing guarded checkpoint.

## AGENT-24: PSRO decision

The pool reached its full capacity of eight snapshots and the fixed-exploiter
curve had a six-generation plateau. There were zero cyclic regressions, below
the required two. The decision is therefore `expand_to_psro = false`: snapshot
self-play has been audited after saturation, but population/PSRO expansion is
not justified by this evidence.

Full evidence, raw episodes, checkpoint hashes, and decisions are in
[`logs/phase3_agent21_24_self_play.json`](../logs/phase3_agent21_24_self_play.json).

