# AGENT-28~31 Submission Validation

- `submission/policy.py` is self-contained apart from PyTorch.
- Deterministic argmax is bit-identical across repeated inference.
- A clean-room subprocess copied exactly `policy.py` and `checkpoint.pt`, loaded
  the checkpoint, and returned `(5,2)` actions.
- CPU latency/parameter benchmarking and performance-constrained lightweight
  candidate selection are implemented.
- Non-five batch sizes fail with a clear contract error.

AGENT-29 and AGENT-31 are now complete. Equal-budget trained encoder candidates
were benchmarked at the official input size; the legacy encoder reduced CPU
median latency from 1.240 ms to 0.746 ms without held-out score regression.
The stripped 641,930-byte artifact passed deterministic inference and mismatch
checks on both CPU and actual Apple MPS, plus the two-file clean-room test.

AGENT-32 was also executed and rejected the latest learned candidate on
strength. The guarded incumbent remains an evaluation benchmark but is not
standalone-submission-compatible. See
`reports/agent29_32_final_submission_review.md`.
