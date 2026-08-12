# AGENT-25/26 Evaluation Protocol

- Fixed split: `configs/seed_splits_v1.json`
- Test seeds are rejected for training and model selection by
  `SeedSplits.seeds_for`.
- Split overlap and duplicates fail closed.
- `paired_seed_bootstrap_ci` groups the two physical-side games by `pair_id`
  and resamples complete seed pairs only.
- Determinism is controlled by the bootstrap RNG seed and the result records
  confidence, resample count, pair count, estimate, and interval.

Evidence: `tests/test_phase3_evaluation.py`.
