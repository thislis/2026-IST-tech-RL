# PREP-12 첫 actor/critic 인터페이스

구현은 [`blackout_rl/model_contract.py`](../blackout_rl/model_contract.py), checkpoint 명세는
[`schemas/checkpoint_v1.schema.json`](../schemas/checkpoint_v1.schema.json)에 고정했다.

## Tensor 계약

| 이름 | dtype | shape | 설명 |
| --- | --- | --- | --- |
| `vector` | `torch.float32` | `(B, 96)` | agent별 processed vector |
| `graphic` | `torch.float32` | `(B, 11, 96, 96)` | 환경 HWC를 contiguous CHW로 변환 |
| `slot_id` | `torch.int64` | `(B,)` | 팀 내부 `unit_0~4 → 0~4`, `unit_5~9 → 0~4` |
| `action_logits` | `torch.float32` | `(B, 9)` | NoOp와 8방향 categorical actor 출력 |
| `value` | `torch.float32` | `(B,)` | agent-local critic value |
| 제출 action | `torch.float32` | `(B, 2)` | deterministic argmax를 `(dx,dy)∈[-1,1]`로 변환 |

한 팀 추론에서 `B=5`다. 학습 minibatch에서는 `B`를 확장할 수 있지만 각 행의 `slot_id`를
명시적으로 함께 전달해야 한다. `team_model_input()`은 입력 dict 순서를 사용하지 않고 agent
이름을 기준으로 canonical order를 만들며, 이 단계에서 shape/dtype/누락 agent도 검증한다.

## PREP-12 당시 첫 reference 구조

PREP-12에서는 다음 shared actor/local critic smoke baseline으로 인터페이스를 먼저 검증했다.
현재 `ReferenceActorCritic` 이름은 하위 호환 alias로 유지되지만 실제 구현은 Phase 2-2에서
완료한 entity-aware `IPPOActorCritic`이다. 현재 구조와 검증 결과는
[`base_r01_r05_r14_policy_model.md`](base_r01_r05_r14_policy_model.md)를 따른다.

```text
vector[96] ─ MLP(128) ───────────┐
graphic[11,96,96] ─ CNN(128) ────┼─ fusion(128) ─ actor logits[9]
slot_id ─ embedding(16) ─────────┘              └─ local value[1]
```

당시 기본 구성은 123,450 parameters였다. CPU에서 팀 5명 batch의 reference deterministic
inference는 로컬 100회 단순 측정 평균 약 0.517 ms였고, metadata 포함 checkpoint는 약 0.50
MB였다. 이는 PREP-12 시점의 개발 기준값이며 현재 entity-aware 모델의 크기와 혼동하지 않는다.

## Action adapter

categorical index는 다음과 같다.

| index | 방향 | `(dx,dy)` |
| ---: | --- | --- |
| 0 | NoOp | `(0,0)` |
| 1~8 | E, NE, N, NW, W, SW, S, SE | 축/대각선 단위 방향 |

대각선은 L2 norm 1로 정규화한다. 학습 시 categorical sampling을 사용할 수 있고, 평가 시에는
반드시 logits의 argmax를 사용한다. PREP-07에서 확인한 Unity의 nonzero normalization과
일치하며 모든 출력은 공식 연속 action 범위 안에 있다.

## 제출 facade와 batch-order 안전성

- `SubmissionPolicy.forward(vector, graphic) -> action`은 공식 `load_checkpoint`의 2-input
  `nn.Module` 계약과 호환된다.
- `CanonicalTeamModel`은 `BaseModel`을 구현하고 agent 이름으로 slot을 붙인 뒤 한 번에 추론한다.
  프로젝트 내부 평가에는 이 adapter를 사용한다.
- 2-input 공식 facade만 사용할 때는 agent 이름을 받을 수 없어 batch row `0~4`를 slot으로
  간주한다. 따라서 공식 evaluator가 팀 batch 순서를 보장하기 전에는 이 fallback을 slot identity의
  최종 제출 근거로 삼지 않는다. 이 항목은 PREP-13 확인 목록에 남겼다.

## Checkpoint v1

PyTorch payload의 필수 key는 다음과 같다.

- `schema_version = blackout.checkpoint.v1`
- `policy_state`: 공식 upstream loader가 그대로 읽을 수 있는 state dict
- `model_config`: vector/channel/slot embedding/hidden/action 크기
- `training`: `global_step`, training seed
- `observation_contract`, `action_contract`
- `source`: game/API commit 등 provenance
- 선택 `experiment`: run ID, 전체 config, seed, git SHA, 상대 ID; registry 등록 시 필수
- 선택 `optimizer_state`: 학습 재개용이며 제출 시 제거 가능

`save_checkpoint()`는 저장 전에 schema를 검증하고, `load_checkpoint()`는 `weights_only=True`,
`map_location`을 사용한 뒤 architecture를 metadata에서 재생성한다. 실제 torch checkpoint 저장 →
로드 → 동일 입력 action 비교가 `test_checkpoint_round_trip`에서 통과한다.
