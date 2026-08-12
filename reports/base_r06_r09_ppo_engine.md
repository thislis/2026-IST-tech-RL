# BASE-R06~R09 PPO Training Engine

Phase 2-2의 두 번째 Task 묶음으로 PettingZoo parallel rollout부터 PPO update와 JSONL
diagnostics까지 하나의 time-major rollout schema로 연결했다.

## 구현 산출물

| ID | 구현 | 완료 근거 |
| --- | --- | --- |
| BASE-R06 | [`ParallelRolloutCollector`](../blackout_rl/rollout.py)가 학습 팀의 canonical observation을 batch하고 stochastic action·log-prob·local value를 계산한 뒤 상대 policy action과 합쳐 parallel step 실행 | agent별 vector, graphic, slot, action index, log-prob, value, reward와 mask shape 검증 |
| BASE-R07 | [`EpisodicRolloutBuffer`](../blackout_rl/rollout.py)와 `generalized_advantage_estimate()` | 여러 episode를 한 rollout에 저장하고 termination/truncation 및 episode-start 경계를 보존 |
| BASE-R08 | [`ppo_update()`](../blackout_rl/ppo.py)의 clipped policy objective, clipped value loss, entropy bonus, minibatch epoch, gradient clipping | 실제 backward/optimizer step 후 parameter 변화와 finite loss 검증 |
| BASE-R09 | `PPODiagnostics`와 [`PPODiagnosticLogger`](../blackout_rl/ppo.py) | KL, clip fraction, entropy, explained variance, gradient norm을 strict JSONL로 round-trip 검증 |

## Rollout tensor 계약

Collector 내부 buffer는 시간 우선 `(T,5,...)`, PPO batch는 이를 `(T×5,...)`로 평탄화한다.

| field | dtype | flattened shape | 의미 |
| --- | --- | --- | --- |
| `vector` | float32 | `(T×5,96)` | action 이전 agent observation |
| `graphic` | float32 | `(T×5,11,H,W)` | semantic map CHW |
| `slot_id` | int64 | `(T×5,)` | team-local slot `0~4` |
| `action_index` | int64 | `(T×5,)` | sampled categorical action |
| `old_log_prob`, `old_value` | float32 | `(T×5,)` | rollout policy 통계 |
| `reward`, `next_value` | float32 | `(T×5,)` | transition reward와 bootstrap value |
| `terminated`, `truncated` | bool | `(T×5,)` | 서로 분리된 episode boundary |
| `episode_start` | bool | `(T×5,)` | reset 직후 첫 transition 표시 |
| `advantage`, `return_` | float32 | `(T×5,)` | GAE 결과와 critic target |

Collector는 한 호출이 긴 episode의 일부만 포함하거나 여러 episode를 가로질러도 동작한다.
첫 `collect()`에만 seed를 받을 수 있고, 이후 episode reset에는 임의로 reseed하지 않는다.

## Termination과 truncation

두 mask의 의미를 다음처럼 분리했다.

```text
bootstrap_mask    = not terminated
continuation_mask = not (terminated or truncated)
```

- termination: terminal state이므로 `next_value`를 사용하지 않는다.
- truncation: 실제 terminal이 아니므로 반환된 final observation의 value를 TD delta에 사용한다.
- 두 경우 모두 GAE recursion은 경계에서 끊어 다음 episode의 advantage가 섞이지 않는다.
- 부분적인 team boundary와 한 transition의 termination+truncation 중복은 fail-closed 처리한다.

## PPO와 diagnostics

PPO update는 policy ratio clipping, 선택 가능한 value clipping, entropy bonus, Adam 등 외부
optimizer, configurable epoch/minibatch, global gradient-norm clipping을 지원한다. 매 update에서
다음을 집계한다.

```text
policy_loss, value_loss, entropy
approximate_kl, clip_fraction
explained_variance, gradient_norm
epochs_completed, minibatches, samples, early_stopped
```

`target_kl`을 설정하면 epoch 경계에서 조기 종료할 수 있다. JSONL record에는 schema version,
update index, global environment step, PPO config와 위 metrics가 함께 저장되며 NaN/Infinity는
기록 전에 거부한다. return variance가 0이라 explained variance가 정의되지 않으면 JSON `null`을
사용한다.

## 검증 결과

`tests/test_ppo_training.py`는 두 step마다 termination과 truncation을 번갈아 발생시키는 mock
PettingZoo parallel environment를 사용한다. 실제 observation parser와 IPPO CNN/entity model을
그대로 통과시키며 다음을 검증했다.

- 10개 agent action이 한 parallel step에 모두 전달됨
- 학습 팀 5명의 canonical slot과 agent별 rollout field 수집
- 여러 episode를 가로지르는 buffer와 호출 간 rollout 연속성
- truncation bootstrap과 episode 간 GAE 차단
- clipped PPO update, parameter 변화, gradient clipping 및 전체 diagnostics
- diagnostics strict JSONL round-trip

```text
python -m unittest discover -s tests -v
Ran 89 tests
OK
```

실제 Unity의 reset/step/동시 terminal 계약은 기존 PREP-05 live evidence 회귀 테스트가 함께
통과했다. Collector의 기본 reward는 Unity 반환값을 유지하며, BASE-R10에서 구현한
`TeamTrainingReward`를 전달하면 score-delta/terminal reward 또는 shaping 조합을 선택할 수 있다.
