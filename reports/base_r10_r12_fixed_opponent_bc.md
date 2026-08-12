# BASE-R10~R12 Fixed-Opponent Training and BC Warm Start

Phase 2-2의 세 번째 Task 묶음으로 score-delta/terminal training reward, 완전히 고정된
scripted opponent 경계, recorded scripted trajectory 기반 behavior cloning을 PPO collector 앞에
연결했다.

## BASE-R10 · Training reward

[`TeamTrainingReward`](../blackout_rl/training_reward.py)는 기존
`ScoreDeltaRewardTracker`를 collector용 stateful reward transform으로 감싼다. 모든 학습 팀원에게
동일한 team reward를 주며 다음 세 mode를 명시적으로 선택할 수 있다.

| mode | reward |
| --- | --- |
| `score_delta` | `(Δown_score - Δopponent_score) / 100` + 실제 winner 기반 terminal bonus |
| `unity_shaping` | Unity가 agent별로 반환한 shaping reward |
| `combined` | 가중 score-delta/terminal reward + 가중 Unity shaping reward |

기본값은 `score_delta`이고 Unity shaping은 꺼져 있다. Terminal frame에서 score가 `(0,0)`으로
reset되는 기존 Unity 동작 때문에 terminal에는 score를 다시 읽지 않고 `infos[*]["winner"]`만
사용한다. Collector가 episode를 reset할 때 reward tracker도 함께 reset된다.

Mock parallel environment 통합 테스트에서 첫 score 증가 `+0.01`이 다섯 agent에 동일하게
들어가고, 다음 terminal win이 각각 `+1.0`으로 buffer에 기록되는 것을 검증했다. Unity-only와
combined mode도 별도로 검증했다.

## BASE-R11 · Frozen scripted opponent

[`FrozenScriptedOpponent`](../blackout_rl/frozen_opponent.py)는 battery-only
`ScriptedTeamController`를 training opponent protocol로 노출한다.

- trainable parameter와 optimizer state가 없음
- policy ID, team, seed, special-item 설정을 SHA-256 fingerprint로 고정
- collector는 learning model만 PPO optimizer에 전달
- scripted FSM의 episode-local runtime state는 정상적으로 진행되지만 episode마다 초기화
- reset 후 동일 observation의 action 재현 및 config fingerprint 불변 검증

상대 FSM의 runtime까지 정지시키면 매 step 초기 행동만 반복하므로, 여기서 freeze는 상대의 정책
정의와 학습 가능한 state를 변경하지 않는다는 뜻이다. Episode-local navigation/FSM state는 action
계산에 필요한 비학습 상태로 유지된다.

## BASE-R12 · Scripted trajectory BC

[`ScriptedTrajectoryDataset`](../blackout_rl/behavior_cloning.py)은 versioned JSONL의 header/footer와
record count를 검증하고 다음 agent-row tensor를 복원한다.

```text
vector[96], graphic[11,96,96], slot_id, action_index
```

연속 scripted action은 크기가 거의 0이면 NoOp, 아니면 정규화 후 가장 가까운 8방향으로
양자화한다. BC는 actor logits의 categorical cross-entropy만 최적화하며 critic은 PPO 단계에서
학습한다. Gradient clipping과 deterministic DataLoader seed를 사용한다.

### Recorded-data warm-start 실험

재현 명령:

```text
python scripts/run_base_r12_bc.py \
  --train logs/base_s08_s10_role_fsm_trajectory.jsonl \
  --held-out logs/base_s13_coordination_trajectory.jsonl \
  --epochs 5 --batch-size 32 --learning-rate 0.0001 \
  --environment-steps 0 --seed 1212 \
  --output logs/base_r12_bc_warm_start.json
```

| metric | scratch | BC warm start |
| --- | ---: | ---: |
| held-out categorical NLL | 2.1738 | 2.1145 |
| held-out argmax accuracy | 26.45% | 26.45% |
| downstream environment steps | 0 | 0 |

Train trajectory는 325 agent rows, 별도 held-out trajectory는 155 agent rows이고 BC는 1,625
demonstration samples를 소비했다. 같은 초기 weights와 같은 downstream environment-step budget
`0`에서 비교했으므로 이번 결과는 PPO 이전의 순수 warm-start 효과만 측정한다. NLL은 2.7%
개선됐지만 argmax accuracy는 개선되지 않았으므로 BC를 성능 우위로 승격하지는 않는다. 후속 실제
PPO 실험에서는 log의 `downstream_environment_step_budget`을 scratch/BC에 동일하게 설정해야 한다.

결과 원본은 [`logs/base_r12_bc_warm_start.json`](../logs/base_r12_bc_warm_start.json)에 저장했다.

## 검증

`tests/test_fixed_opponent_bc.py`와 collector integration test가 다음을 검증한다.

- score 감소/증가와 winner terminal reward의 팀 공유
- Unity shaping off/on/combined 선택
- frozen opponent에 trainable parameter가 없고 reset 전후 정책 설정이 불변
- trajectory shape, slot, semantic map, action label 복원
- BC backward/update와 scratch 대비 NLL 감소
- scratch/BC의 동일 environment-step budget 기록

```text
python -m unittest discover -s tests -v
Ran 96 tests
OK
```
