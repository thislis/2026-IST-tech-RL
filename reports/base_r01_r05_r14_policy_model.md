# BASE-R01~R05/R14 Policy Model Vertical Slice

Phase 2-2의 첫 Task 묶음은 학습 시 사용하는 stochastic categorical policy부터 제출 시
사용하는 deterministic `(B,2)` action까지 하나의 모델 계약으로 구현했다.

## 구현 산출물

| ID | 구현 | 완료 근거 |
| --- | --- | --- |
| BASE-R01 | [`action_distribution.py`](../blackout_rl/action_distribution.py)의 9-way `Categorical` sampling, argmax, stored-action 재평가 | index/action/log-prob/entropy shape와 PPO gradient 검증 |
| BASE-R02 | [`EntityVectorEncoder`](../blackout_rl/ippo_model.py)가 10개의 9-field unit block을 shared entity MLP로 각각 처리하고 class·score·time 6-field context를 별도 처리 | `(B,10,entity_dim)` entity와 `(B,context_dim)` context 분리 및 독립성 검증 |
| BASE-R03 | [`SemanticCNNEncoder`](../blackout_rl/ippo_model.py)의 2-layer CNN과 adaptive pooling | 공식 `(B,11,96,96)` 입력을 고정 길이 latent로 변환 |
| BASE-R04 | team-local slot `0~4`의 learned embedding | 동일 observation의 slot별 actor 출력 분화와 다섯 embedding row의 gradient 검증 |
| BASE-R05 | [`IPPOActorCritic`](../blackout_rl/ippo_model.py)의 단일 shared actor head와 agent-local value head | logits `(B,9)`, value `(B,)`; 같은 observation/slot row의 동일 출력 검증 |
| BASE-R14 | [`SubmissionPolicy`](../blackout_rl/model_contract.py)의 canonical-row slot 주입과 deterministic argmax adapter | `forward(vector, graphic) -> (5,2)`, normalized action과 `[-1,1]` 범위 검증 |

## 모델 구조

```text
10 × unit[9] ─ shared entity MLP ─ flatten[10×32] ─┐
class/score/time[6] ─ context MLP[32] ─────────────┤
semantic map[11,96,96] ─ CNN ─ latent[128] ────────┼─ fusion MLP[128]
team-local slot[0..4] ─ embedding[16] ─────────────┘       ├─ shared actor logits[9]
                                                           └─ local value[1]
```

기본 모델은 157,818 parameters이며 serialized checkpoint payload는 약 0.64 MB다. entity
latent를 pooling하지 않고 unit block 순서대로 보존하므로 현재 actor가 각 unit 위치를 구분할 수
있고, 이후 attention encoder로 교체할 때도 `(B,10,D)` 중간 표현을 그대로 사용할 수 있다.

## Action 계약

- index `0`: NoOp `(0,0)`
- index `1~8`: E, NE, N, NW, W, SW, S, SE
- 대각선 action은 L2 norm 1로 정규화
- rollout: categorical sample과 해당 `log_prob`, `entropy` 저장
- PPO update: 저장한 `int64` index를 현재 logits에서 재평가
- evaluation/submission: logits argmax만 사용

학습과 제출 경로 모두 같은 index-to-vector table을 사용하므로 action 의미가 갈라지지 않는다.

## 검증 결과

`tests/test_ippo_model.py`의 신규 테스트 8개와 전체 regression suite를 실행했다.

```text
python -m unittest discover -s tests -v
Ran 84 tests in 0.688s
OK
```

기존 checkpoint 저장/복원 및 upstream `blackout_env.load_checkpoint` round-trip도 함께
통과했다. 제출 wrapper의 row 기반 slot fallback은 PREP-13에서 확인한 현재 evaluator의 canonical
team order를 전제로 하며, 프로젝트 내부 평가는 계속 agent name 기반 `CanonicalTeamModel`을
사용한다.
