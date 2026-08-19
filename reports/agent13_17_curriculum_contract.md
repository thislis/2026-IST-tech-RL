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

AGENT-14 is now complete. The equal-budget 512-step live ablation produced 0/10
wins for both arms, while navigation shaping reduced held-out mean score
difference from `-89.7` to `-94.3`. The final shaping weight was zero and the
navigation arm was not promoted. See `reports/agent14_navigation_shaping_ablation.md`.
