# BASE-R13/R15 Reproducible Registry and Evaluation Matrix

Phase 2-2의 마지막 Task 묶음으로 checkpoint와 experiment identity를 결합하고, 등록된
checkpoint를 random/scripted baseline과 공통 조건에서 비교하는 평가 matrix를 완성했다.

## BASE-R13 · Checkpoint와 experiment registry

[`ExperimentIdentity`](../blackout_rl/experiment_registry.py)는 다음 다섯 항목을 checkpoint
payload 내부와 append-only JSONL registry 양쪽에 저장한다.

```text
run_id, complete config, training seed, full git SHA, opponent ID
```

Registry entry에는 checkpoint SHA-256, canonical JSON config SHA-256, global step도 추가한다.
읽을 때마다 hash 형식, config hash, identity field와 run ID 중복을 fail-closed 검증한다. 기존
checkpoint v1은 `experiment`가 선택 항목이므로 PREP-12 checkpoint와 호환되지만, registry에
등록하려는 checkpoint에는 experiment identity가 반드시 있어야 한다.

실제 등록 artifact:

| field | value |
| --- | --- |
| run ID | `base-r13-bc-warm-start-seed-1212` |
| checkpoint | [`checkpoints/base_r13_bc_warm_start.pt`](../checkpoints/base_r13_bc_warm_start.pt) |
| checkpoint SHA-256 | `d185fbe497e9d0752ae98552b91b3de902965dbbb59fbeea0c8cb01b40140842` |
| config SHA-256 | `ca15ade35f69f9465e750587a821e9d46d1f921502ed881f7e84b0eda8192e77` |
| seed | `1212` |
| git SHA | `5c1d39e6524dc41de6bc02ad03f27af81511509c` |
| opponent ID | `scripted-battery-v1` |
| PPO global step | `0` |

Registry 원본은 [`experiments/registry.jsonl`](../experiments/registry.jsonl), JSON Schema는
[`experiment_registry_v1.schema.json`](../schemas/experiment_registry_v1.schema.json)이다.
`register_base_r13_checkpoint.py`는 BASE-R12 trajectory를 다시 BC 학습하고 provenance-complete
checkpoint를 생성하는 재현 entry point다.

## BASE-R15 · Common-condition policy matrix

[`policy_matrix.py`](../eval/policy_matrix.py)는 세 개의 paired-series log에서 지정한 공통 seed만
선택한 후 다음 조건을 검증한다.

- random, scripted, IPPO 세 policy가 모두 존재
- 각 seed에 A/B side swap 2경기가 정확히 존재
- 동일 random opponent config SHA와 opponent RNG seed 사용
- 동일 Unity executable SHA 사용
- 승패는 모두 `terminal_info.winner`에서 계산되고 Unity shaping을 사용하지 않음
- model-side attribution audit 통과

이번 smoke matrix의 공통 조건은 environment seed `1401`, opponent policy seed `515403`,
executable SHA `49172f...e29f4a69`이다. 각 후보가 A/B에서 한 경기씩 총 2경기를 수행했다.

| Policy | W-D-L | 승률 | 평균 model score diff | A side | B side |
| --- | ---: | ---: | ---: | --- | --- |
| scripted battery | 2-0-0 | 100% | +95.5 | 승, +97 | 승, +94 |
| random | 1-0-1 | 50% | -10.0 | 승, +2 | 패, -22 |
| IPPO BC warm start | 0-0-2 | 0% | -32.0 | 패, -21 | 패, -43 |

원본은 다음과 같다.

- [`base_r15_random_vs_random_seed1401.json`](../logs/base_r15_random_vs_random_seed1401.json)
- [`base_r15_scripted_vs_random_seed1401.json`](../logs/base_r15_scripted_vs_random_seed1401.json)
- [`base_r15_ippo_vs_random_seed1401.json`](../logs/base_r15_ippo_vs_random_seed1401.json)
- 통합 [`base_r15_policy_matrix_seed1401.json`](../logs/base_r15_policy_matrix_seed1401.json)

IPPO artifact는 BASE-R12 BC warm start 직후이며 registry상 PPO global step이 `0`이다. 따라서 이
결과는 학습된 IPPO 성능 주장이 아니라 evaluator/registry/checkpoint 경로의 end-to-end baseline
이다. 또한 seed 1개의 smoke matrix라 통계적 성능 판정에는 부족하다. 이후 PPO 학습 checkpoint는
같은 matrix에 추가 seed를 사용해 재평가해야 한다.

## 검증

`tests/test_registry_matrix.py`는 checkpoint 저장/복원, embedded identity와 registry 일치,
duplicate run rejection, common-condition matrix validation과 실제 evidence log를 검증한다.

```text
python -m unittest discover -s tests -v
Ran 101 tests
OK
```
