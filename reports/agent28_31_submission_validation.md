# AGENT-28~31 Submission Validation

- `submission/policy.py` is self-contained apart from PyTorch.
- Deterministic argmax is bit-identical across repeated inference.
- A clean-room subprocess copied exactly `policy.py` and `checkpoint.pt`, loaded
  the checkpoint, and returned `(5,2)` actions.
- CPU latency/parameter benchmarking and performance-constrained lightweight
  candidate selection are implemented.
- Non-five batch sizes fail with a clear contract error.

AGENT-29 remains unchecked until trained encoder-removal candidates are
benchmarked. AGENT-31 remains unchecked because MPS is built but unavailable in
this host session and CUDA is not installed. AGENT-32 consequently remains
unchecked; the promotion gate requires both CPU and an actual GPU smoke plus no
past-opponent regressions.
