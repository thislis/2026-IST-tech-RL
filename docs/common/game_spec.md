# BlackOut RL Game Specification

## 1. 범위와 기준 버전

이 문서는 BlackOut을 강화학습 환경의 **state / action / transition / event / reward / terminal** 관점에서 정리한다. 2026-11-01에 확인한 최신 로컬 소스 기준은 다음과 같다.

- Unity 게임: `af4c5faaf2ef9eae24f536860737099ad56acba3`
- Python API: `bcd8aadad2f293bf45f3999286b1f174bae5586b`
- Unity Editor: `6000.3.8f1 (1c7db571dde0)`
- Unity ML-Agents: `com.unity.ml-agents 4.0.2`

문서 설명과 실행 코드가 다르면 위 commit의 `Prototype.unity`, `GameBalanceConfig.asset`, C# 코드 및 Python wrapper를 실행 기준으로 삼는다.

이 기준은 소스 검토 결과이며 최신 소스로 빌드하거나 실험을 실행한 결과가 아니다. 로컬에 설치된 Windows `v1.1` player가 위 Unity commit으로 만들어졌다는 대응 관계도 확인하지 않았다. 업데이트 범위와 설치 환경은 [2026-11-01 환경 업데이트](environment_update_2026-11-01.md)를 참고한다.

**과거 실험 기준:** PREP 실험은 Unity `d2220a7d01be88d413f551efd529f4758833be8b`, Python API `6ba7d9993cf1bdefe1ed480c8efbcabcb923f539`, Python ML-Agents `1.1.0`을 사용했다. 해당 macOS player의 SHA-256, 패키지 고정값과 검증 로그는 [versions.md](versions.md)에 보존한다. 이 문서의 최신 소스 설명을 과거 실험의 실행 버전으로 소급 적용하지 않는다.

### 이전 Unity 기준 이후 변경

- Hunter 성소가 무제한 변신에서 **팀당 동시 2기**로 바뀌었다. Carrier의 팀당 1기 제한은 유지되며, 두 제한 모두 `MaxUnitsPerTeam`과 클래스 변경 추적으로 구현된다.
- 기본 맵에서 `(1,16)`, `(16,1)`을 통로로 열고 `(5,18)`, `(6,18)`, `(18,5)`, `(18,6)`을 벽으로 바꿨다. `y=x` 대칭을 유지한다.
- semantic texture에 `Point` 필터와 `Clamp`를 지정해 렌더링 시 이웃 semantic ID가 섞이는 문제를 수정했다. 관측 shape와 channel 정의는 바뀌지 않았다.
- timer callback 중 episode가 재시작되면 기존 `Tick()`을 중단하도록 해 다음 episode의 timer가 제거되는 문제를 수정했다. 제한 시간 420초와 흡수 간격 20초는 그대로다.
- UI의 흡수 표시 문자열은 `[Absorp]`에서 `[Absorb]`로 수정됐다.

근거는 Unity `d2220a7..af4c5fa` 전체 diff의 `Scenes/Prototype.unity`, `Scripts/Region/Sanctuary.cs`, `Scripts/Map/RegionData.cs`, `Scripts/ML/SemanticMapRenderer.cs`, `Scripts/TimerManager.cs`다. 경로는 `Assets/Project/Runtime/` 기준이다. 아래 특수 아이템 초기 생성 설명의 정정은 새 코드 변경이 아니라 기존 명세 오류를 바로잡은 것이다.

## 2. 게임 상태와 종료

- 맵은 `24 × 24` grid이며 에피소드마다 seed를 사용해 창고 후보와 아이템 위치를 생성한다.
- 두 팀 A/B가 경쟁하며 각 팀은 5개 유닛, 총 10개 유닛을 동시에 제어한다.
- 모든 유닛은 Collector(노동자)로 시작한다.
- 제한 시간은 420초(7분), 목표 점수는 100점이다.
- 어느 팀의 현재 점수가 100 이상이 되면 즉시 종료한다.
- 제한 시간이 끝나면 점수가 높은 팀이 승리하며 동점이면 무승부다.
- Python terminal `infos.winner`의 표기는 `0=Team A`, `1=Team B`, `-1=draw`다. 다만 최신 wrapper도 이 필드를 terminal raw reward의 부호에서 추론하므로 평가 승패는 최종 `score_0/score_1` 비교를 기준으로 한다. 자세한 구분은 6절과 8절에 기록한다.
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
- 최신 기본 맵은 `(1,16)`, `(16,1)`이 통로이고 `(5,18)`, `(6,18)`, `(18,5)`, `(18,6)`이 벽이다. 이전 Unity 기준의 반대 설정과 혼용하지 않는다.

### 창고

- 현재 scene에는 팀별 고정 보호 창고 1개가 있다.
- `StorageCountPerTeam=4`이고 5개 후보 중 4개를 매 episode 무작위 선택해 팀 A 창고를 만들고, y=x 대칭 위치에 Team B 창고를 만든다.
- 따라서 **현재 코드 기준 팀당 창고 수는 고정 1 + 절차 생성 4 = 5개**다.
- 저장 타일은 spawn anchor에서 Manhattan distance가 먼 순서로 채운다. 동률은 x, y 오름차순이다.
- 같은 아이템은 한 타일의 최대 stack까지 합쳐진다. Battery 최대 stack은 10, 특수 아이템은 1이다.
- 전체 수량을 원자적으로 수용할 수 없으면 적재가 전부 취소되고 유닛은 아이템을 계속 보유한다.
- 아군은 자기 창고에 놓인 아이템을 다시 집을 수 없고, 적군은 공개 창고의 아이템을 약탈할 수 있다. 보호 본진에는 적이 들어갈 수 없다.

> `gameplay_ko.md`의 “팀당 4개(보호 1 + 공개 3)” 설명과 scene의 실제 `1 + 4` 설정이 다르다. 현재 소스 명세는 실행 scene의 5개를 기준으로 한다.

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
- Hunter는 중앙 성소에서 **팀별 동시 2기**, Carrier는 본진 성소에서 **팀별 동시 1기**까지 변신할 수 있다. Hunter 제한은 두 팀을 합한 2기가 아니다.
- 성소는 자신이 변신시킨 유닛을 팀별로 추적한다. 해당 유닛이 사망 후 Collector로 부활하거나 episode reset으로 클래스가 바뀌면 자리가 반환된다. 제한에 막혔다면 성소에 다시 진입해야 변신을 재시도한다.

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
- 초기 map 생성은 4종을 **각각 1개씩** 중립 유효 타일에 배치한다. 현재 `SymmetricBatterySpawn=0`이므로 초기 특수 아이템도 추가 대칭 생성하지 않는다.
- 별도의 동적 spawner는 초기 4개를 추적하지 않는다. reset 후 `currentSpecialItem=null`에서 시작하므로 약 10초 뒤 4종 중 무작위 1개를 추가 생성할 수 있다.
- 동적 spawner는 자신이 마지막으로 생성한 `currentSpecialItem`이 활성 상태인 동안 추가 생성을 멈춘다. 그 객체가 흡수·파괴 등으로 비활성화되면 다시 10초 cooldown을 센다. 따라서 전역 특수 아이템 수를 1개로 제한하거나 종류별 1개를 보장하지 않는다.

> **기존 명세 정정:** 이전 문서의 “전역적으로 특수 아이템 1개만 활성화” 설명은 잘못됐다. 초기 4개와 동적 추가 생성 경로는 이전 Unity 기준에도 존재했다. 근거: `ProceduralMapGenerator.SpawnItemObjects`, `LevelDirector.PerformReset`, `UpdateSpecialItemSpawner`.

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
- Unity raw vector는 45개 float이며 마지막 `[44]`가 routing용 `unitIndex`다. Python은 item/class를 one-hot으로 바꾸고 이 routing 값을 제거해 96차원 policy vector를 만든다.
- 최신 API는 observation dict를 `unit_0..unit_9` 숫자 순서로 반환하고 `CheckpointModel.act()`도 `unit_index`로 정렬한다. 팀별 정상 5행 batch는 Team A `unit_0..4`, Team B `unit_5..9`에 대응한다. 이 정렬은 소스에서 확인했으며 최신 player와의 실행 검증은 하지 않았다.
- ONNX 대회 경로는 `slot=unit_index(agent) % 5`를 명시적으로 전달하고 slot별 모델·frame stack·recurrent state를 선택한다. 이 identity 정보는 다른 유닛의 클래스를 추가하는 관측이 아니다.

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

최신 Unity source는 semantic ID를 보간하지 않도록 source texture에 `FilterMode.Point`와 `TextureWrapMode.Clamp`를 지정한다. 이는 특히 Xvfb/Mesa 경로에서 Battery `6`이 enemy unit `5`로 섞이는 등의 오류를 막는 수정이다. 최신 설치 player에 이 수정이 포함됐는지는 별도로 확인해야 한다.

### `infos`

- 매 step: `score_0`, `score_1`, `time_left`.
- terminal step: 위 필드와 `winner`.
- `score_0/score_1`은 실제 정수 점수를 목표 점수 100으로 나눈 값이다. 100점을 넘겨 종료하면 값도 1을 넘을 수 있다. 실제 점수는 100을 곱해 복원한다.
- 최신 API는 같은 exchange에 다음 episode의 decision observation과 terminal observation이 함께 오면 **terminal observation의 score scalar를 우선**한다. 따라서 소스상 최종 `infos.score_0/score_1`을 평가에 사용하는 경로가 마련돼 있다. 실제 최신 player와의 terminal 계약은 아직 실행 검증하지 않았다.
- `infos.winner`는 여전히 첫 terminal agent의 raw reward 부호에서 추론한다. 같은 step의 shaping reward가 섞일 수 있으므로 최종 점수 비교와 독립적으로 대조해야 하며, 이를 Unity의 별도 winner 통신으로 해석하지 않는다.
- upstream `competition.run_match()`는 최종 `infos` 점수로 승패를 판정하고 team swap을 model identity에 반영한다. `MatchResult.winner`는 `0=model_a`, `1=model_b`, `None=draw`여서 환경의 `infos.winner` 표기와 구분한다.

> **과거 실행 관측:** 이전 API/기록된 macOS player 조합에서는 다음 episode의 `(0,0)` score가 terminal frame에 노출됐다. PREP 로그는 마지막 non-terminal 점수와 terminal `winner`를 보존했다. 이 결과는 [versions.md](versions.md)의 이전 실행 조합에 한정한다.

## 7. RL action

- PettingZoo action: agent별 `float32[2]`, 각 원소 범위 `[-1,1]`.
- 의미: `(dx,dy)` 이동 방향.
- Python wrapper는 범위 밖 값을 clip한다.
- Unity는 non-zero vector를 normalize하므로 `(0.1,0)`과 `(1,0)`의 속도가 같다.
- 대각선도 normalize되므로 축 이동보다 빠르지 않다.
- 명시적 pickup/deposit/attack/transform action은 없다. 모두 위치 overlap 또는 region enter event로 자동 발생한다.
- 최신 대회 API는 `run_match()`에서 매 episode 시작 전 두 모델의 `BaseModel.reset()`을 호출한다. frame stack이나 recurrent state를 가진 모델은 이 hook에서 episode 상태를 비운다. 기본 구현은 아무 작업도 하지 않는다.

## 8. 이벤트와 학습 reward

실제 승패와 점수는 `infos`를 기준으로 평가하고 shaping reward 합으로 winner를 추론하지 않는다.

프로젝트의 기존 Python 학습 reward는 `blackout_rl.reward.ScoreDeltaRewardTracker`가
`Team A=(Δscore_A-Δscore_B)/100`, `Team B=-Team A`로 계산해 각 팀 5개 agent에
공유한다. 이 adapter는 이전 실행 계약에 맞춰 terminal score 변화를 항상 무시하고
전달받은 `winner`의 `+1/-1/0` bonus만 추가한다. 최신 API가 최종 terminal score를
보존하는 계약으로 바뀌어도 이 로컬 코드는 자동으로 변경되지 않는다. 마지막 step의
점수 변화와 `infos.winner` 추론을 포함한 호환성 검증이 필요하며, 최신 조합에서
검증됐다고 간주하지 않는다.

다음 표는 **Unity raw shaping reward**다. 위 Python score-delta reward와 구분하며,
값과 이벤트 경로는 이전 Unity 기준 이후 변경되지 않았다.

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

Battery 약탈로 score가 감소하는 경우 `OnScoreChanged` guard가 `score > 0`만 확인하므로 값이 0보다 큰 감소 후 score도 “득점” 이벤트처럼 처리된다. Python score-delta reward의 non-terminal 경로는 이전 score와 새 score의 차를 직접 계산한다. 평가 승패에는 raw shaping reward의 합이나 부호를 사용하지 않는다.

## 9. Seed와 재현성 경계

- 기존 실험 코드의 `blackout_rl.env.ContractBlackOutEnv`는 최초 ML-Agents handshake를 끝낸 뒤 seed SideChannel을 한 exchange 먼저 전달하고, 이전 episode의 map/routing cache를 비운 다음 reset한다. 이 adapter의 실행 검증은 이전 소스/기록된 macOS player 조합에 한정된다.
- 최신 upstream `BlackOutEnv.reset(seed=N)`도 seed 메시지를 queue한 다음 같은 호출에서 Unity reset을 요청한다. 이전 조합에서 발견한 seed 적용 순서 문제와 cache 호환성은 최신 조합으로 재검증해야 한다. 최신 API로 업데이트했다는 사실만으로 adapter가 불필요해졌거나 그대로 검증됐다고 판단하지 않는다.
- 같은 게임 commit/build/config와 같은 seed라면 procedural storage 선택 및 item 위치/수량이 같아야 한다.
- random policy action은 별도 policy seed로 고정해야 전체 trajectory를 재현할 수 있다.
- Unity physics/실행 플랫폼 차이까지 bitwise 동일하다고 가정하지 않는다. 초기 map/item 재현과 최종 평가의 paired seed를 별도로 검증한다.

**과거 검증:** PREP-05에서 seed `40407`의 wall/storage map을 연속 reset했을 때 bitwise 동일했고, `40408`은 다른 storage layout을 생성했다. 이는 이전 기준의 결과이며, 벽 배치가 달라진 최신 Unity source 또는 로컬 Windows `v1.1` player에 대한 검증 결과가 아니다.
