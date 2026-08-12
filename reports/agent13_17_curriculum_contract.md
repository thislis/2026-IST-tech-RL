# AGENT-13~17 Curriculum Contract

- Fixed order: random -> weak scripted -> full scripted -> frozen RL.
- Every stage has its own opponent ID, win-rate/score threshold, and side-gap
  threshold; results cannot be merged across stages.
- Navigation shaping uses the potential-based form
  `gamma * Phi(next) - Phi(current)`.
- Linear annealing reaches zero so final reward is score plus terminal outcome.
- Special items/raiding use a common-seed/common-budget no-regression gate.
  Existing `BASE-S15` evidence failed that gate, so these features remain
  disabled rather than being promoted.

The fixed-shaping sample-efficiency experiment (AGENT-14) remains open; the
annealer is implemented but must not be claimed to improve learning before that
experiment is run.
