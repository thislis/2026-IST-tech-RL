# BlackOut RL Game Specification

## 1. 범위와 기준 버전

이 문서는 BlackOut을 강화학습 환경의 **state / action / transition / event / reward / terminal** 관점에서 고정한다. 기준 소스는 다음 두 저장소의 clean commit이다.

- Unity 게임: `d2220a7d01be88d413f551efd529f4758833be8b`
- Python API: `6ba7d9993cf1bdefe1ed480c8efbcabcb923f539`
- Unity Editor: `6000.3.8f1 (1c7db571dde0)`
- Unity ML-Agents: `com.unity.ml-agents 4.0.2`
- Python ML-Agents: `mlagents-envs 1.1.0`

문서 설명과 실행 코드가 다르면 위 commit의 `Prototype.unity`, `GameBalanceConfig.asset`, C# 코드 및 Python wrapper를 실행 기준으로 삼는다.

## 2. 게임 상태와 종료

- 맵은 `24 × 24` grid이며 에피소드마다 seed를 사용해 창고 후보와 아이템 위치를 생성한다.
- 두 팀 A/B가 경쟁하며 각 팀은 5개 유닛, 총 10개 유닛을 동시에 제어한다.
- 모든 유닛은 Collector(노동자)로 시작한다.
- 제한 시간은 420초(7분), 목표 점수는 100점이다.
- 어느 팀의 현재 점수가 100 이상이 되면 즉시 종료한다.
- 제한 시간이 끝나면 점수가 높은 팀이 승리하며 동점이면 무승부다.
- Python terminal `winner`는 `0=Team A`, `1=Team B`, `-1=draw`다.
- 모든 10개 에이전트는 같은 step에 termination되며 truncation은 사용하지 않는다.

상태 전이는 다음 순서로 이해한다.

```text
reset(seed)
  → seed 적용
  → procedural map / storage / item 배치
  → 유닛·점수·timer 초기화
  → 매 step: 10개 action 동시 적용 → 이동/충돌/지역 이벤트 → obs/reward
  → 100점 도달 또는 420초 경과
  → score 기반 winner + terminal reward
```

## 3. 맵과 지역

### 일반/본진 지역

- 각 팀 본진에는 spawn point, 보호 창고, Carrier 성소가 있다.
- 본진 바닥은 적 팀에 대해 `BlockEnemy`이므로 상대가 진입할 수 없다.
- 중앙의 `2 × 2` Hunter 성소는 중립 지역이다.
- Team A/B의 spawn cell은 각각 `(1,22)`, `(22,1)`이다.

### 창고

- 현재 scene에는 팀별 고정 보호 창고 1개가 있다.
- `StorageCountPerTeam=4`이고 5개 후보 중 4개를 매 episode 무작위 선택해 팀 A 창고를 만들고, y=x 대칭 위치에 Team B 창고를 만든다.
- 따라서 **현재 코드 기준 팀당 창고 수는 고정 1 + 절차 생성 4 = 5개**다.
- 저장 타일은 spawn anchor에서 Manhattan distance가 먼 순서로 채운다. 동률은 x, y 오름차순이다.
- 같은 아이템은 한 타일의 최대 stack까지 합쳐진다. Battery 최대 stack은 10, 특수 아이템은 1이다.
- 전체 수량을 원자적으로 수용할 수 없으면 적재가 전부 취소되고 유닛은 아이템을 계속 보유한다.
- 아군은 자기 창고에 놓인 아이템을 다시 집을 수 없고, 적군은 공개 창고의 아이템을 약탈할 수 있다. 보호 본진에는 적이 들어갈 수 없다.

> `gameplay_ko.md`의 “팀당 4개(보호 1 + 공개 3)” 설명과 scene의 실제 `1 + 4` 설정이 다르다. 이 버전의 실험에서는 실행 scene의 5개를 기준으로 기록한다.

## 4. 유닛과 전투

| 클래스 | class id | 충돌 폭 | 기본 속도 | 아이템 수집 | 변신 |
| --- | ---: | ---: | ---: | --- | --- |
| Collector / 노동자 | 0 | 0.55 | 4 | 가능 | 중앙→Hunter, 본진→Carrier |
| Hunter / 경비원 | 1 | 0.65 | 6 | 불가 | 추가 변신 불가 |
| Carrier / 전달자 | 2 | 0.45 | 6 | 가능 | 추가 변신 불가 |

- action 크기가 아니라 방향만 사용한다. Unity는 non-zero `(dx,dy)`를 정규화한 뒤 `speed × fixedDeltaTime`만큼 이동시킨다.
- `action=(0,0)`이면 이동량이 0이다.
- 벽, 맵 밖, 적 본진과 충돌하면 이동 가능 위치까지만 이동한다.
- Hunter로 변신할 때 아이템을 들고 있었다면 해당 아이템은 파괴된다.
- 사망하면 보유 아이템이 파괴되고 즉시 본진 spawn에서 Collector로 부활한다.
- Carrier 성소는 팀별 동시 1기 제한이며 해당 Carrier가 사망해야 lock이 풀린다.

### 접촉 전투 상성

`UnitData.Beats`와 `UnitInteractionSystem.ResolveCombat` 기준 결과다. 양쪽이 서로를 이기면 둘 다 사망한다.

| 행 공격/열 상대 | Collector | Hunter | Carrier |
| --- | --- | --- | --- |
| Collector | 변화 없음 | Collector 사망 | Carrier 사망 |
| Hunter | Collector 사망 | 둘 다 사망 | Carrier 사망 |
| Carrier | Carrier 사망 | Carrier 사망 | 둘 다 사망 |

동일 팀 접촉에는 전투가 없다.

## 5. 아이템, 점수, 20초 이벤트

### Battery

- episode 시작 시 중립의 유효 타일에 총합 약 200점이 될 때까지 생성한다.
- 각 stack 수량은 1~8이고 마지막 spawn 때문에 총합이 200을 조금 넘을 수 있다.
- Battery를 아군 창고에 넣는 순간 수량만큼 현재 점수가 증가한다.
- 흡수 전에 적이 약탈하면 원래 팀 점수에서 같은 수량이 감소한다.
- 약탈한 Battery를 자기 창고에 넣으면 약탈 팀 점수가 증가한다.

### 특수 아이템

| item id / map channel | 효과 | 적용 대상 |
| --- | --- | --- |
| 1 / ch7 BuffSpeed | 이동속도 +50% | 소유 팀 전체 |
| 2 / ch8 DebuffSpeed | 이동속도 -90% | 상대 Collector만 |
| 3 / ch9 BuffSize | 충돌 크기 +50% | 소유 팀 전체 |
| 4 / ch10 DebuffSize | 충돌 크기 -30% | 상대 팀 전체 |

- 특수 효과는 아이템이 창고에 존재하는 동안만 유지된다.
- 약탈·파괴·흡수로 창고에서 빠지면 modifier가 즉시 제거된다.
- 실제 spawner는 동시에 하나의 `currentSpecialItem`만 관리한다. 활성 특수 아이템이 없어진 후 10초가 지나면 4종 중 하나를 무작위로 중립 타일에 생성한다.

> “4종이 각각 1개씩 존재”한다는 게임 설명과 달리 현재 코드에는 전역적으로 특수 아이템 1개만 활성화되는 경로가 구현되어 있다.

### 20초 흡수 이벤트

- 매 20초마다 모든 창고의 아이템에 `OnAbsorbed`를 호출하고 아이템을 제거한다.
- Battery는 적재 시 이미 점수에 반영되며, 흡수 시 그 점수를 차감하지 않음으로써 영구 확정한다.
- 따라서 흡수는 새 점수를 더하는 이벤트가 아니라 **기존 임시 점수를 약탈 불가능한 점수로 lock**하는 이벤트다.
- 특수 아이템 modifier는 흡수 시 제거된다.
- 전략적으로 `time mod 20초`는 적재 안전성, 방어, 약탈의 핵심 phase다. 현재 관측에는 흡수 timer가 직접 제공되지 않고 episode `time_left`에서 추론해야 한다.

## 6. RL state / observation

각 PettingZoo agent는 같은 형태의 dict observation을 받는다.

### `vector: float32[96]`

현재 `n_items=5`, `n_classes=3` 기준이다.

```text
unit_0 block [pos_x, pos_y, team_sign, holding_item_one_hot(6)]  = 9
...
unit_9 block [pos_x, pos_y, team_sign, holding_item_one_hot(6)]  = 9
self_class_one_hot(3)
own_score_norm, opponent_score_norm, time_left_norm
총 10×9 + 3 + 3 = 96
```

- `pos_x,pos_y`: map origin 기준 절대 좌표를 map bounds로 나눈 값.
- `team_sign`: 관찰 agent 기준 아군 `+1`, 적군 `-1`.
- holding item one-hot index 0은 없음, 1~5는 Battery/BuffSpeed/DebuffSpeed/BuffSize/DebuffSize다.
- self class만 직접 주어지며 다른 유닛의 class는 vector에 없다.
- score는 100으로 나눈 값, `time_left`는 `[0,1]`이다.
- unit order는 항상 이름 기준 `unit_0..unit_9` 의미를 갖지만 batch row order 안정성은 별도 PREP-06에서 검증한다.

### `graphic: float32[96,96,11]` (HWC)

각 channel은 0/1 binary mask다.

| ch | 의미 |
| ---: | --- |
| 0 | empty |
| 1 | wall |
| 2 | ally storage |
| 3 | enemy storage |
| 4 | ally unit |
| 5 | enemy unit |
| 6 | Battery |
| 7 | BuffSpeed |
| 8 | DebuffSpeed |
| 9 | BuffSize |
| 10 | DebuffSize |

Team B map은 Python에서 storage와 unit의 ally/enemy channel을 swap해 만든다. 모델 CNN 입력에는 HWC를 CHW로 변환해야 한다.

### `infos`

- 매 step: `score_0`, `score_1`, `time_left`.
- terminal step: 위 필드와 `winner`.
- `score_0/score_1`은 이름과 달리 현재 Python wrapper에서 0~1 normalized score다. 실제 점수는 목표 점수 100을 곱해 복원한다.
- 현재 scene은 `EndEpisode()` 직후 다음 episode 초기화를 시작하므로 terminal frame의 score가 `(0,0)`으로 관측된다. 최종 점수는 마지막 non-terminal `infos`에서 보존하고, 승자는 terminal `winner`에서 읽는다. PREP-02 로그는 두 값을 모두 저장한다.

## 7. RL action

- PettingZoo action: agent별 `float32[2]`, 각 원소 범위 `[-1,1]`.
- 의미: `(dx,dy)` 이동 방향.
- Python wrapper는 범위 밖 값을 clip한다.
- Unity는 non-zero vector를 normalize하므로 `(0.1,0)`과 `(1,0)`의 속도가 같다.
- 대각선도 normalize되므로 축 이동보다 빠르지 않다.
- 명시적 pickup/deposit/attack/transform action은 없다. 모두 위치 overlap 또는 region enter event로 자동 발생한다.

## 8. 이벤트와 학습 reward

실제 승패와 점수는 `infos`를 기준으로 평가하고 shaping reward 합으로 winner를 추론하지 않는다.

| 이벤트 | reward 수신자 | 값 |
| --- | --- | ---: |
| 필드/적 창고에서 아이템 pickup | 집은 agent | item reward (`Battery=0.2`, 그 외 기본 0.1) |
| 자기 아이템을 적이 약탈 | 피해 팀의 각 agent | item reward의 음수 |
| 아이템 deposit | 놓은 agent | item reward |
| 아군 score 증가 | 아군의 각 agent | +0.1 |
| 상대 score 증가 | 상대편 각 agent | -0.1 |
| kill | killer | +0.3 |
| 아군 death | 피해 팀의 각 agent | -0.2 |
| 아군 창고 item 흡수 | 아군의 각 agent | item reward |
| terminal 승/패/무 | 각 agent | +1 / -1 / 0 |

Battery 약탈로 score가 감소하는 경우 `OnScoreChanged` guard가 `score > 0`만 확인하므로 값이 0보다 큰 감소 후 score도 “득점” 이벤트처럼 처리될 가능성이 있다. PREP-09의 Python score-delta reward에서는 반드시 이전 score와 새 score의 차를 직접 계산한다.

## 9. Seed와 재현성 경계

- 실험 코드에서는 반드시 `blackout_rl.env.ContractBlackOutEnv`를 사용한다. 이 adapter는 최초 ML-Agents handshake를 끝낸 뒤 seed SideChannel을 한 exchange 먼저 전달하고, 이전 episode의 map/routing cache를 비운 다음 reset한다.
- 고정된 upstream `BlackOutEnv.reset(seed=N)`을 직접 호출하면 seed 메시지와 reset command의 Unity 처리 순서 때문에 N이 반환 episode보다 늦게 적용될 수 있으므로 사용하지 않는다.
- 같은 게임 commit/build/config와 같은 seed라면 procedural storage 선택 및 item 위치/수량이 같아야 한다.
- random policy action은 별도 policy seed로 고정해야 전체 trajectory를 재현할 수 있다.
- Unity physics/실행 플랫폼 차이까지 bitwise 동일하다고 가정하지 않는다. 초기 map/item 재현과 최종 평가의 paired seed를 별도로 검증한다.

PREP-05에서 seed `40407`의 wall/storage map을 연속 reset했을 때 bitwise 동일했고, `40408`은 다른 storage layout을 생성했다.
