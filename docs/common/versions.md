# PREP-01 Reproducible Environment

Recorded on 2026-08-05 (Asia/Seoul). All experiment runs must retain the two
source commits, Unity executable hash, Python lock, and container tag below.

## Source revisions

| Component | Path | Revision |
| --- | --- | --- |
| Unity game | `/Users/safeailab_macmini/Desktop/blackout` | `d2220a7d01be88d413f551efd529f4758833be8b` |
| Python API | `/Users/safeailab_macmini/Desktop/blackout-env` | `6ba7d9993cf1bdefe1ed480c8efbcabcb923f539` |

Both repositories were clean when the revisions were recorded. The Python API
is installed into this repository's `.venv` from the second path.

## Unity runtime and executable

| Item | Fixed value |
| --- | --- |
| Unity Editor | `6000.3.8f1` |
| Unity changeset | `1c7db571dde0` |
| Unity ML-Agents package | `com.unity.ml-agents 4.0.2` |
| Player target | macOS Universal (`x86_64`, `arm64`) |
| App bundle | `builds/BlackOut.app` |
| Executable | `builds/BlackOut.app/Contents/MacOS/RLGame2026` |
| Executable SHA-256 | `49172f88b1ba2429677e7ec4a7876c4589876090be4888aeaef39269e29f4a69` |

`builds/` and the locally installed `.unity/` Editor are reproducible local
artifacts and are intentionally ignored by Git. The hash above, not merely the
bundle name, identifies the player used by the experiment log.

## Local Python runtime

| Item | Fixed value |
| --- | --- |
| Host | macOS 26.5 (25F71), arm64 |
| Virtual environment | `.venv` at the repository root |
| Python | `3.10.12` |
| PyTorch | `2.13.0` |
| PyTorch CUDA build | none (`torch.version.cuda is None`) |
| MPS | built, unavailable on this host session |
| `mlagents-envs` | `1.1.0` |
| `blackout-env` | source revision above; package version `0.47.0` |
| PettingZoo | `1.26.1` |
| Gymnasium | `1.3.0` |
| NumPy | `1.23.5` |
| protobuf | `3.20.3` |
| grpcio | `1.51.3` |

The complete transitive runtime set is pinned in `requirements.lock`. Install
`mlagents-envs` and `blackout-env` separately as described at the top of that
file because their declared PettingZoo constraints conflict. `grpcio 1.51.3`
is used because the upstream `1.48.2` pin does not provide a usable Apple
Silicon wheel for this setup.

All experiment entry points use the repository's
`blackout_rl.env.ContractBlackOutEnv` adapter. It completes the first ML-Agents
handshake, delivers the seed before the Unity reset, and clears reset-spanning
map/routing caches. Direct use of the fixed upstream `BlackOutEnv.reset(seed)`
does not satisfy this seed contract.

## Linux / Docker identity

The fixed experiment image identity is stored in `docker-image.env`:

```text
BLACKOUT_DOCKER_IMAGE=blackout-rl:prep01-d2220a7-6ba7d99-cu124
BLACKOUT_DOCKER_BASE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
```

The tag encodes the short game and Python API revisions and the CUDA 12.4
runtime family. Docker is not installed on this macOS host, so PREP-02 was
validated with the local Universal player rather than a locally built image.

## Verified smoke match

`logs/prep04_07_contract.json` records the successful seeded reset-to-terminal
run against the exact executable hash above:

- environment seed: `20260805`
- random-policy seed: `20260805`
- episode steps: `21003`
- final score: Team A `19`, Team B `24`
- winner: `1` (`team_b`)
- terminal/score consistency check: passed

The Unity scene resets score scalars to zero on the terminal frame. The runner
therefore records both those terminal-frame zeros and the final pre-terminal
score used to verify the terminal winner.

The earlier `logs/prep02_random_seed_20260805.json` was produced before the
same-exchange seed ordering and stale-map cache were detected. Its executable
hash remains valid, but it is superseded and must not be used as seed evidence.

## Reproduction commands

From this repository root:

```bash
uv venv .venv --python 3.10.12
uv pip install --python .venv/bin/python -r requirements.lock
uv pip install --python .venv/bin/python 'mlagents-envs==1.1.0' --no-deps
uv pip install --python .venv/bin/python ../blackout-env --no-deps

.venv/bin/python scripts/run_random_match.py \
  --build builds/BlackOut.app \
  --seed 20260805 \
  --policy-seed 20260805 \
  --time-scale 50 \
  --output logs/prep02_random_seed_20260805.json
```
