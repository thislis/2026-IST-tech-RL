# 1차 이슈 점검 — v0~v7

점검일: **2026-09-27 (KST)**. 사용자 제공 질의응답 18개와 프로젝트의 코드·Unity 원본·실행 로그를 대조했다. 구현 수정이나 재학습을 수행한 문서가 아니라, 현재 문제와 대응 상태를 기록한 감사 문서다.

## 1. 먼저 확인할 사항

1. **v7 평가 점수 기록에 실제 문제가 있다.** 본실험 1,380경기 중 1,211경기의 점수가 `0:0`으로 저장됐고, 그중 1,204경기는 승자가 있다. 종료 프레임 점수를 그대로 읽는 경로를 수정해야 한다. 이 사실만으로 승패까지 잘못됐다고 결론 내리지는 않는다.
2. **시간초과 타이머 재진입 버그가 사용 중인 Unity 소스에 남아 있다.** Python 학습·평가는 종료 후 명시적으로 `reset()`하므로 Unity 자동 재시작 경로와 영향이 다르다. 기존 경기들이 종료됐다는 사실은 원본 버그 수정의 증거가 아니다.
3. **배터리 수량은 vector와 graphic 모두에서 직접 제공되지 않는다.** graphic의 배터리 픽셀 수를 배터리 점수로 해석하면 안 된다.
4. **현재 빌드는 창고 후보를 매번 모두 활성화하지 않는다.** 팀 A 후보 5개 중 4개를 선택하고 B에 대칭 생성한다. “고정 맵”과 “매번 같은 활성 창고”는 구분해야 한다.
5. **학습 완료·제출 준비·성능 개선은 별개다.** v7 본실험은 완료됐지만 실제 FlyWire와 재배선의 경기 기록이 같으며, 최종 test와 제출 export 검증은 완료되지 않았다.
6. **README 상태가 오래됐다.** README는 9월 21일 중단 상태를 설명하지만, 최종 관리자 로그와 history는 9월 22일 본실험·dev 평가 완료를 기록한다.

## 2. 점검 범위와 판정 기준

| 대상 | 이번에 확인한 기준 |
| --- | --- |
| 프로젝트 HEAD | `1968ed24b091dd978085c4a7a508b3d32c057a0a` |
| Unity 원본 | `../blackout`, HEAD `d2220a7d01be88d413f551efd529f4758833be8b`, 작업 트리 clean |
| Python API 원본 | `../blackout-env`, HEAD `6ba7d9993cf1bdefe1ed480c8efbcabcb923f539`, 작업 트리 clean |
| 실제 로컬 실행 파일 | `artifacts/builds/BlackOut.app/Contents/MacOS/RLGame2026` |
| 실행 파일 SHA-256 재계산 | `49172f88b1ba2429677e7ec4a7876c4589876090be4888aeaef39269e29f4a69` — 기존 실험 기준과 일치 |
| 확인 방법 | 정적 코드 검사, 기존 실행 JSON 재집계, 관련 테스트 62개, 소규모 합성 예시 |
| 이번에 하지 않은 검증 | 새 Unity 연속 경기 실행, Linux/Xvfb 재현, Windows 비교, 공식 최신 규정 조회 |

**세대 범위:** 저장소는 MAPPO v1~v6와 v7-1/v7-2를 명시적으로 관리한다. `v0`는 공식 세대명으로 확인되지 않아 이 문서에서는 PREP/BASE/IPPO 기반 작업을 가리키는 편의상 명칭으로만 쓴다.

**판정:** `확인`은 코드 또는 로그에서 확인, `보정됨`은 프로젝트 경로에서 대응, `미검증`은 해당 조건에서 실행 증거 부족, `규정 확인 필요`는 사용자 제공 답변만으로 제출 계약을 확정할 수 없다는 뜻이다. 소스의 결함과 실제 학습 피해는 별도로 표기한다.

**우선순위:** P0은 결과 신뢰성·경기 종료·제출을 막는 항목, P1은 학습/관측 품질에 영향을 주는 항목, P2는 성능·문서·해석 관리 항목이다. 사용자 제공 답변은 당시 외부 답변으로 인용하며, 우리 빌드에 수정이 반영됐다는 증거로 취급하지 않는다.

## 3. 사용자 제공 18개 이슈 대조

| ID | 점검 항목 / 전달받은 답변 | 우리 프로젝트 판정과 근거 | 대응 / 우선순위 |
| --- | --- | --- | --- |
| Q01 | 이미지 1번은 창고, 2·3번도 창고이며 차이가 없고 그래픽을 수정했다고 함 | **번호 대응·수정 반영 미검증.** 번호가 붙은 원본 이미지가 없어 실제 타일 대응은 확인 불가. 코드의 semantic ID `1=wall`, `2=ally_storage`, `3=enemy_storage`는 질문의 이미지 번호와 전혀 다른 체계다. [관측 명세](../pre_v1/reports/prep04_observation_contract.md) | 이미지 번호와 채널 ID 혼동 금지. UI 텍스처 수정은 별도 빌드/이미지 대조 필요. P2 |
| Q02 | 매크로는 처음에는 불허, 이후 모델에 길찾기 로직을 넣는 방식은 허용 | **우리에게 직접 관련.** scripted planner와 planner override/residual을 사용한다. 모델 내부 로직 허용 답변은 참고할 수 있으나 state/reset·의존성·입출력 계약까지 해결하지 않는다. [추론 정책](../blackout_rl/policy.py), [제출 체크리스트](../pre_v1/reports/prep13_evaluation_contract_checklist.md) | 전체 planner 실행을 포함하는 최종 제출물의 호환성 검증. P0 |
| Q03 | 맵은 고정, 아이템 위치는 바뀌며 창고 위치 등 구조는 고정 | **기본 지형·후보 좌표 고정, 활성 창고는 변동.** 현재 생성기는 scene 후보를 seed RNG로 선택한다. “모든 semantic 채널이 고정”이라는 해석은 맞지 않는다. [게임 명세](../game_spec.md), 아래 §4.2 | 고정 지형과 episode별 활성 영역을 분리. P1 |
| Q04 | 길찾기가 경로를 제공하고 모델이 액션을 내는 방식은 모델 내부에 포함하면 허용 | **현재 구조와 관련.** 내부 planner가 관측으로 경로/행동을 만들고 residual이 최종 `(dx,dy)`를 결정한다. v7-2의 제어 슬롯은 직접 행동을 낸다. [v6 정책](../blackout_rl/mappo_v6.py), [v7-2 정책](../blackout_rl/v7_2/policy.py) | 허용 답변과 실제 제출 loader 호환성을 별도로 확인. P0 |
| Q05 | 매크로와 모델을 대결시켜 학습해도 되는지는 별도 명확한 답변 없음 | **실제로 수행 중인 방식.** scripted/weak/target 혼합, teacher BC/DAgger를 사용한다. 기존 답변만으로 학습 상대·teacher 사용까지 공식 확정할 수 없다. [학습 상대 구성](../blackout_rl/v7_training.py), [v3 계획](../v3/reports/mappo_teacher_curriculum_v3_plan.md) | 학습 시 scripted 상대·teacher와 제출 시 로직의 허용 범위를 구분해 질의. P1 |
| Q06 | 모델 추론 시간 제한은 미정, 추후 정한다고 함 | **공식 한도 미확인.** 로컬 속도 자료는 있지만 실제 전체 정책의 공식 하드웨어 p95/p99 지연 보장이 아니다. 특히 전뇌·planner 비용을 포함해야 한다. [제출 체크리스트](../pre_v1/reports/prep13_evaluation_contract_checklist.md), [가속 보고서](../v7/reports/acceleration_and_resume.md) | 팀/step 한도, 경기 wall-clock, 메모리·스레드·패키지 조건 확인 후 측정. P0 |
| Q07 | 배속 이상 의심, 외부 TestSuite에서는 약 140 step/s 보고 | **우리도 처리량 제한을 경험했으나 배속 미동작으로 확정 불가.** 초기 macOS random benchmark는 time scale 50에서 52.278 env step/s, 순차 2환경 합산은 49.736. v7 가속은 통신/전뇌 연산 병목을 개선했다. [초기 benchmark](../pre_v1/reports/prep11_throughput_benchmark.md), [v7 가속](../v7/reports/acceleration_and_resume.md) | 외부 140과 서로 다른 장비·정책·측정 단위를 직접 비교하지 않는다. time scale과 env/agent step/s 분리. P2 |
| Q08 | 타일 24×24에 내부 움직임을 표현하려고 graphic을 96×96으로 제공 | **정상 계약 확인.** 타일당 4×4 픽셀, 최종 HWC `(96,96,11)`; CNN에는 CHW로 변환. 위치의 세부 표현이지 방향 벡터 채널이 따로 생긴 것은 아니다. [semantic decoder](../blackout_rl/semantic_map.py), [관측 로그](../logs/prep04_07_contract.json) | 크기 버그로 취급하지 않는다. 24×24 축소 시 세부 위치 손실 고려. P2 |
| Q09 | 매치마다 다른 seed를 설정할 예정 | **seed별 변화 지원, 초기 wrapper 버그는 보정됨.** upstream reset은 seed 전달 순서와 캐시 문제를 갖고 있어 `ContractBlackOutEnv`가 handshake/seed flush/cache clear를 수행한다. 학습 seed 재표집과 평가 paired seed 재사용은 의도된 실험 설계다. [adapter](../blackout_rl/env.py), [seed 검증](../logs/prep04_07_contract.json) | 모든 실행 경로에서 adapter 유지. “재현용 고정 seed”를 “대회 seed 고정”으로 해석 금지. P1 |
| Q10 | 창고 위치는 고정이고 아이템만 이동 | **후보 위치 고정까지만 일치.** 현재 빌드는 후보 중 일부만 활성화한다. 위치가 새 임의 좌표로 이동하는 것은 아니다. §4.2 | 초기 관측에서 활성 창고를 읽고 episode 간 캐시를 갱신. P1 |
| Q11 | 창고 후보 한 타일 랜덤 비활성화, 이후 변경 예고 | **현 소스에 선택 로직 잔존 확인.** 변경 예고가 현재 고정 빌드에 적용됐다는 근거는 없다. [scene](../../blackout/Assets/Project/Runtime/Scenes/Prototype.unity), [설정](../../blackout/Assets/Project/Runtime/Resources/ScriptableObjects/Config/GameBalanceConfig.asset) | 변경된 빌드를 받으면 재검증하고 환경 버전 분리. P1 |
| Q12 | 정확히 하나만 seed에 의해 비활성화되는 것이 맞다고 함 | **우리 버전은 ‘팀 A 후보 영역 5개 중 1개 미선택 + B 대칭 영역 미선택’.** 후보는 여러 타일로 된 영역이므로 전 맵의 단일 grid cell 하나가 빠진다는 뜻은 아니다. [생성기](../../blackout/Assets/Project/Runtime/Scripts/Map/ProceduralMapGenerator.cs), §4.2 | 문서에 후보 영역·팀·타일의 단위를 명시. P1 |
| Q13 | 비활성 창고가 변경 전 코드 잔존 때문인지는 답변 없음 | **구현 원인은 확인, 기획 경위는 미확인.** `OrderBy(Random.value)` 후 `StorageCountPerTeam`만큼 선택한다. 이를 개발자의 실수나 기획 변경 누락이라고 단정할 수 없다. §4.2 | 공식 확정 사양과 버전별 변경 내역 요청. P2 |
| Q14 | 첫 timeout 뒤 timeout 소실; Tick 콜백의 Clear/Add 후 RemoveAt 재진입 버그 | **동일 결함 코드 확인.** 현재 원본의 `TimerManager.Tick()`/episode 재시작 경로가 일치한다. 자동 재시작은 취약하지만 프로젝트의 명시적 reset이 새 타이머를 다시 만든다. 새 Unity 연속 재현은 미실행. §4.1 | 원본 수정 및 자동/명시 reset 두 경로의 연속 timeout 회귀 검사. P0 |
| Q15 | 제공 observation만 사용, 가공 feature는 자유 | **주요 게임 상태 입력은 vector/graphic 기반.** 다만 agent 이름/slot, 고정 성소 좌표, 관측 이력, 학습 critic의 양 팀 관측을 구분해야 한다. [planner 문맥](../blackout_rl/mappo_v6.py), [team state](../blackout_rl/team_state.py) | agent 메타데이터·사전 지식·CTDE의 허용 범위를 명시. 숨은 런타임 상태를 actor 입력에 추가하면 안 된다. P1 |
| Q16 | 화면의 배터리 수량이 vector에 없는 것에 대해 graphic 사용 안내 | **수량 누락 확인.** raw unit block은 위치·팀·보유 아이템 종류뿐이다. 96-vector의 one-hot 확장에도 수량은 없다. [Unity 관측](../../blackout/Assets/Project/Runtime/Scripts/ML/BlackOutAgent.cs), [파서](../blackout_rl/observation.py) | 수량을 알 수 있다는 전제의 보상/할당 금지. Q17과 함께 사양 확인. P1 |
| Q17 | graphic에도 배터리 수량 대신 존재 여부만 표현; 후속 답변 없음 | **우리도 해당.** renderer는 item 종류 ID를 한 픽셀에 쓰고 수량을 쓰지 않는다. UI 수량 숫자는 semantic observation에 포함되지 않는다. [renderer](../../blackout/Assets/Project/Runtime/Scripts/ML/SemanticMapRenderer.cs) | 수량 채널 요청 또는 관측 가능한 정보만 쓰는 부분관측 정책 유지. P1 |
| Q18 | graphic 가장자리 이상; Xvfb의 Blit 좌표/보간 원인 제시, Windows 미재현 | **우리 Linux/Xvfb 발생 여부 미검증.** 같은 `Graphics.Blit` 경로는 존재한다. macOS의 shape/one-hot 통과만으로 Linux 의미 오염을 배제할 수 없다. §4.3 | raw semantic ID의 플랫폼별 픽셀 대조 필요. P1 |

## 4. 환경·관측 이슈 상세

### 4.1 시간초과: 원본 결함과 프로젝트의 우회 경로

근거는 [TimerManager.cs](../../blackout/Assets/Project/Runtime/Scripts/TimerManager.cs)의 `GameTimer.Tick()`/`TimerManager.Tick()`, [GameScenario.cs](../../blackout/Assets/Project/Runtime/Scripts/GameScenario.cs)의 `EpisodeBegin()`, [BlackOutEpisodeCoordinator.cs](../../blackout/Assets/Project/Runtime/Scripts/ML/BlackOutEpisodeCoordinator.cs)의 `OnGameEnded()`/`NotifyAgentEpisodeBegin()`이다.

```text
timers = [old EpisodeTimer, old AbsorptionTimer]
Tick(i=0) → old EpisodeTimer.OnComplete()
  → TimeExpired → GameEnded → agent.EndEpisode()
  → 모든 agent OnEpisodeBegin → GameScenario.EpisodeBegin()
  → Clear() → timers = [new EpisodeTimer, new AbsorptionTimer]
old Tick 반환 true → RemoveAt(0)
결과: timers = [new AbsorptionTimer]  # 새 EpisodeTimer가 삭제됨
```

목록 변경 순서만 옮긴 Python 합성 예시에서도 새 episode timer가 제거됐다. 이는 **C# 실행 재현이 아니라 소스상의 메커니즘 검증**이다.

반면 [v7 Collector](../blackout_rl/v7_training.py)는 terminal 후 `self.reset()` → `env.reset(seed=...)`를 호출하고, [v6 Collector](../blackout_rl/mappo_v6_training.py)도 명시 reset을 한다. 다음 Python 요청은 이전 Unity Tick이 끝난 뒤 수행되므로 타이머를 다시 생성하는 우회 효과가 예상된다. 기존 v7 본실험 평가 1,380경기는 모두 22,000-step 상한 안에서 완료됐고, 그중 363경기의 길이는 기존 timeout 기준과 같은 21,003 step이다. 따라서 **“v7의 첫 timeout 이후 모든 경기가 무한 진행됐다”는 증거는 없다.** 다만 종료 사유가 따로 저장되지 않아 길이만으로 timeout 사유를 완전히 확정하지 않는다.

완료 조건:

- Unity에서 콜백 도중 목록을 수정해도 새 timer를 삭제하지 않도록 생명주기/삭제 방식을 수정한다.
- 동일 프로세스에서 점수 100 미만인 상태로 3회 이상 연속 timeout을 검사한다.
- Unity 자동 재시작과 Python 명시 reset을 각각 검사하고 `time_left`, timer 등록 상태, 종료 사유를 기록한다.
- 수정 전후 빌드 hash를 분리한다. watchdog은 중단 안전장치이며 타이머 수정이 아니다.

### 4.2 창고: 고정 위치와 랜덤 활성화의 혼동

현재 scene의 `storageCandidatesA`는 `Storage_A2`~`Storage_A6` 5개 영역이다. A2~A4는 2×2, A5는 1×3과 3×1이 겹치는 L자, A6는 5×1이다. `GameBalanceConfig.asset`의 `StorageCountPerTeam=4`가 적용되며 생성기는 A 후보를 섞어 4개를 선택하고 y=x 대칭으로 B 영역을 만든다. 고정 보호 창고를 포함하면 **팀당 창고 영역 1+4=5개**다.

- 고정인 것: 기본 tilemap, 후보 좌표, 본진/성소 위치.
- seed에 따라 달라지는 것: 활성 창고 후보의 조합, 아이템 배치·수량.
- 질문의 “하나 비활성”은 현 구현상 A 후보 영역 하나와 그 B 대칭 영역 하나의 미선택이다.
- 클래스 기본값 `StorageCountPerTeam=3`만 읽으면 실제 scene 설정 4를 놓친다.
- [game_spec.md](../game_spec.md)는 게임 설명의 “보호 1+공개 3”과 실제 “1+4” 차이를 이미 기록한다. 이 차이를 새 빌드에서도 자동 유지되는 규칙으로 가정하면 안 된다.

관측 로그의 seed `40407` 재실행은 static 채널이 같았고 `40408`과는 달랐다. static 채널 묶음에 창고가 포함돼 있으므로 이 결과를 “벽/기본 지형도 랜덤”이라고 해석하면 안 된다. 지형 고정 검사는 wall과 storage를 분리해야 한다.

### 4.3 렌더링: one-hot 정상이어도 잘못된 의미일 수 있음

현재 renderer는 낮은 정수 semantic ID를 grayscale 값으로 쓰고 `Graphics.Blit`한다. upstream [obs_preprocessor.py](../../blackout-env/blackout_env/env/obs_preprocessor.py)는 `round(raw * 255)`로 ID를 복원한 뒤 one-hot으로 만든다.

**이번 합성 확인:** `1/255`, `1.5/255`, `2/255`를 실제 전처리기에 넣으면 ID가 각각 `1,2,2`가 된다. 모두 채널 합은 1이다. 즉 wall ID 1과 storage ID 2 사이의 보간 오염은 **정상 one-hot storage로 바뀌어** [parse_graphic](../blackout_rl/observation.py)의 검사까지 통과할 수 있다. 이것은 Xvfb 재현 자체가 아니라 기존 검사의 한계를 입증한 것이다.

추가로 renderer는 `background → item → unit` 순으로 단일 ID를 덮어쓴다. 반환값이 11채널이어도 원본은 다층 독립 센서가 아니며, 같은 픽셀에 있는 item/storage가 unit에 가려질 수 있다. “배터리 픽셀이 사라짐=누군가 획득함”을 단일 프레임만으로 단정하면 안 된다.

완료 조건은 플랫폼별 raw ID/벽·창고 경계 대조, 가장자리·단일 item·겹침 fixture 검사다. 필터/샘플 좌표 수정은 실제 renderer에서 검증해야 하며 Python에서 오염된 ID를 임의 치환하는 것만으로 해결했다고 보지 않는다. 현재 소스에는 sRGB round-trip 보정은 있지만 이를 Xvfb 좌표/보간 문제의 해결 증거로 쓸 수 없다.

### 4.4 수량·정체성·클래스의 관측 공백

- **배터리 수량:** ground item 수량과 holding stack 수량이 직접 제공되지 않는다. UI를 읽을 수 있다는 답변과 semantic graphic 계약이 일치하지 않는다. 점수 변화는 적재 이후의 정보라서 모든 배터리의 사전 수량을 복원하지 못한다.
- **자기 유닛 ID:** raw 45-vector의 `unitIndex`는 wrapper routing에 쓰고 모델용 96-vector에서 제거한다. 초기 같은 팀 5명의 관측이 동일했다는 실측이 있다. 내부 canonical batching/slot 입력은 보정이나 공식 loader의 순서 계약은 별도다. [PREP-06](../pre_v1/reports/prep06_agent_identity_order.md)
- **상대 클래스:** 각 agent vector에는 self class만 있고 전체 10명의 class 배열은 없다. 아군 5개의 관측을 합치면 아군 class를 얻을 수 있지만, 상대 관측을 actor에 넘겨 상대 self class를 직접 읽는 것은 현재 팀 입력 계약과 다르다. 기존 위험 지도는 적 위치 중심의 근사다. [team state](../blackout_rl/team_state.py), [위험 지도](../pre_v1/reports/base_s12_danger_map.md)
- **관측에 없는 지형 속성:** semantic wall 채널은 `BlockAll`을 표시하고 성소/적 본진 제한을 모두 별도 채널로 표시하지 않는다. planner의 고정 성소·진영 지식은 현재 맵에 의존하므로 새 맵에 대한 일반화로 해석하면 안 된다.

## 5. 추가 발견: 평가·보상·제출·운영

### A01 — v7 평가가 초기화된 terminal 점수를 저장함 (P0, 코드·로그 확인)

[scripts/evaluate_v7.py](../scripts/evaluate_v7.py)의 terminal 분기는 `info['score_0']`, `info['score_1']`를 직접 저장한다. 기존 [eval/evaluator.py](../eval/evaluator.py)는 terminal 값을 버리고 마지막 non-terminal score를 보존한다. **이미 알려진 terminal score reset 대응이 v7 별도 evaluator에 전달되지 않았다.**

[최종 관리자 summary](../logs/v7/main_study/accelerated/summary.json)의 `runs[].result` 23개만 읽어 재집계했다. `*.progress.json`은 중복이므로 제외했다.

| 모델 | 경기 수 | 기록상 승/무/패 | 점수가 0:0인 경기 |
| --- | ---: | --- | ---: |
| A0 MLP | 300 | 46 / 1 / 253 | 267 |
| A1 matched MLP | 300 | 65 / 6 / 229 | 260 |
| A2 rewired | 300 | 150 / 0 / 150 | 270 |
| A3 FlyWire | 300 | 150 / 0 / 150 | 270 |
| F1 전뇌 특징 + PPO 출력층 | 180 | 32 / 0 / 148 | 144 |
| 합계 | 1,380 | 443 / 7 / 930 | **1,211** |

이 중 1,204경기는 `winner != -1`이다. 예를 들어 [FlyWire seed 11 결과](../logs/v7/v7_1_fafb783_flywire_main_2m_v1/11/eval_dev_target_1790021324103477000.json)의 첫 경기는 seed 41000, team 0, 1,410 step, winner 1이지만 score는 0:0이다.

영향과 조치:

- 현재 v7 결과의 score를 최종 득점·득실차 지표로 사용하면 안 된다. 0이 아닌 terminal 값도 최종 점수라는 보장이 없어 전체 경로를 정비해야 한다.
- v7의 승패 집계는 `winner`를 사용하므로 점수 기록 오류만으로 기존 승률을 무효 처리하지 않는다. `winner` 생성 신뢰성은 A03에서 별도로 다룬다.
- 마지막 non-terminal 점수, terminal raw 값, 점수 출처·단위, 종료 사유를 분리한다. score는 normalized 값이므로 실제 점수와 필드명도 구분한다.
- 마지막 non-terminal 값 역시 100점 도달 마지막 행동의 점수 증가를 누락할 수 있다. **정확한 최종 점수**는 Unity 종료 이벤트의 snapshot이 필요하다.
- 이번 작업에서는 원본 결과를 덮어쓰거나 결측 최종 점수를 추정해 채우지 않았다. 재평가 시 수정 evaluator와 새 provenance로 별도 결과를 남겨야 한다.

### A02 — upstream 대회 runner의 승패 기준이 프로젝트와 다름 (P0, 소스 확인)

[competition/match.py](../../blackout-env/blackout_env/competition/match.py)의 `run_match()`는 경기 전체 Unity reward를 합쳐 큰 쪽을 승자로 삼는다. 프로젝트 평가기는 terminal `infos['winner']`를 사용한다. 수집·전투·흡수 shaping이 있으므로 두 기준은 동일하지 않다.

이는 공식 최신 서버가 실제로 잘못 판정한다는 확정이 아니라 **고정된 upstream 코드와 우리 평가 계약의 불일치**다. 공식 순위 판정 기준을 확인하고, same-match 기록에서 누적 보상·최종 점수·winner를 함께 비교해야 한다.

### A03 — `infos['winner']` 자체도 독립적인 게임 결과 전송이 아님 (P1, 코드상 위험 확인 / 실제 오판 미확인)

[blackout_env.py](../../blackout-env/blackout_env/env/blackout_env.py)의 `_collect_obs()`는 첫 terminal agent의 reward 부호로 winner를 만든다. C# coordinator가 종료 보상 `+1/-1/0`을 더하지만, terminal reward에 같은 구간의 shaping이 함께 들어갈 수 있다. 특히 동점 종료에서 작은 양·음의 shaping이 있으면 `0=draw` 가정이 깨질 수 있다.

기존 timeout 검증은 일부 seed에서 점수와 winner가 일치했음을 보여줄 뿐 모든 동점·동시 이벤트를 증명하지 않는다. 전체 경기 누적 보상으로 승자를 정하는 A02와도 다른 문제다. 게임 종료 이벤트의 명시적 winner를 별도 전달하고, 동점+마지막 프레임 pickup/deposit/combat 케이스를 검사하는 것이 완료 조건이다.

### A04 — 점수 감소에도 Unity shaping의 득점 보상 핸들러가 실행됨 (P1, 소스 확인)

[BlackOutAgent.cs](../../blackout/Assets/Project/Runtime/Scripts/ML/BlackOutAgent.cs)의 `onMyTeamScored`는 delta가 아니라 새 점수 `score > 0`이면 양의 reward를 준다. `onOpponentScored`도 새 점수가 양수이면 penalty를 준다. 따라서 약탈로 자기 점수가 감소해도 0보다 크면 해당 score-change 핸들러에서는 양의 보상이 발생한다. 별도 약탈 penalty도 존재하므로 **총보상까지 반드시 양수라고 단정하지 않는다.**

[Python score-delta tracker](../blackout_rl/reward.py)는 감소 부호를 올바르게 처리한다. v6/v7 score-delta 중심 경로는 완화됐지만 Unity shaping을 쓰는 과거 실험과 A02의 reward-sum 판정에는 관련된다. 동일 이벤트에서 각 reward 성분과 합계를 분리해 확인해야 한다.

### A05 — terminal 점수 보존은 마지막 득점 누락까지 해결하지 못함 (P1, 알려진 한계)

[ScoreDeltaRewardTracker](../blackout_rl/reward.py)는 terminal frame에서 점수를 무시하고 winner bonus만 추가한다. reset 때문에 생기는 가짜 큰 음의 delta는 막지만, 종료를 유발한 마지막 적재의 실제 delta는 복원하지 못한다. 현재 학습의 의도된 타협이며 “정확한 최종 score delta를 모두 학습한다”는 설명은 부정확하다. 게임 측 terminal snapshot 제공 전까지 이 한계를 명시한다.

### A06 — agent 순서·state/reset·모델 내부 로직을 제출 계약이 모두 보장하지 않음 (P0, 미해결)

[upstream loader](../../blackout-env/blackout_env/model/loader.py)는 입력 dict 순서대로 batch를 만들고, [기존 체크리스트](../pre_v1/reports/prep13_evaluation_contract_checklist.md)는 stateless 인터페이스와 episode reset hook 부재를 기록한다. 우리 planner와 v7-2는 내부 이력 또는 명시적 episode/step ID를 사용한다. 단순히 모델 안에 planner를 넣는 것만으로 이 차이가 사라지지 않는다.

[export_v7.py](../scripts/export_v7.py)는 아직 export를 명시적으로 차단한다. v6는 격리된 두 파일 export 검사가 있으나 목표 gate를 통과하지 못해 최종 `artifacts/submission/v6` 산출물이 없다. 공식 loader에서 연속 여러 경기, 순서 변경, state reset, 허용 의존성·메모리·지연을 확인해야 제출 준비 완료로 표시할 수 있다.

### A07 — headless 실행이 관측을 무효화할 수 있음 (P1, 프로젝트 보정 있음)

[ContractBlackOutEnv](../blackout_rl/env.py)는 background 실행에 `-batchmode`를 쓰고 `-nographics`를 추가하지 않는다. 렌더링 기반 semantic observation을 유지하기 위해서다. upstream은 map 관측이 없으면 zero graphic을 제공하거나 기존 캐시를 유지할 수 있다. 그러므로 Unity 실행 성공만 확인하는 smoke test는 충분하지 않다. 실제 frame의 유효성·갱신 여부·예상 지형을 함께 검사해야 한다. 이번 62개 검사는 background argument 경로를 포함하지만 실제 GPU/Xvfb 검증은 아니다.

### A08 — 문서의 현재 상태·정책 설명이 서로 다름 (P2, 확인)

- [README 상단](../README.md): 파일럿 완료 및 본실험 9월 21일 중단 상태.
- [history](../history.md), [최종 관리자 상태](../logs/v7/main_study/accelerated/status.json): 9월 22일 본실험 23개와 평가 완료.
- [초기 제출 체크리스트](../pre_v1/reports/prep13_evaluation_contract_checklist.md): stateful 모델을 사용하지 않는다는 당시 결정. 현재 planner/v7-2에는 그대로 적용되지 않는다.
- [history의 v6 crop 설명](../history.md)은 기존 IPPO crop을 변경하지 않았다고 쓰지만, 현재 [GlobalLocalMapEncoder](../blackout_rl/ippo_model.py)는 `1 - 2*y`로 이미 top-down 변환한다. 과거 실행 버전과 현재 코드 상태를 분리해 기술해야 한다.

이번에는 요청된 이슈 문서만 작성한다. README/과거 보고서 수정 시 역사적 실행 사실을 현재 구현으로 덮어쓰지 말고 갱신일·적용 버전을 명시해야 한다.

### A09 — 기본 특수 아이템 정책을 끈 근거의 일반화 범위가 좁음 (P2, 추가 검증 대상)

[BASE-S15 ablation](../pre_v1/reports/base_s15_special_item_ablation.md)은 random 상대 10경기에서 두 정책 모두 10승, 특수 아이템 사용 시 점수 차 -1.7·경기 길이 +74.1 step을 기록해 기본 기능을 껐다. 구현 고장이라는 증거는 없다. 다만 이 실험만으로 강한 target 상대에서도 특수 아이템이 불필요하다고 일반화할 수 없다. 평가 점수 기록을 먼저 정상화한 뒤 target 상대에서 별도 비교할 항목이다.

### A10 — 재개·지표·환경 버전 비교 시 혼동 위험 (P2, 계약상 한계)

- v6/v7 resume은 모델·optimizer·RNG를 복원하지만 Unity 물리 상태는 복원하지 않고 진행 중 경기를 폐기한다. bitwise trajectory 재개라고 부르면 안 된다.
- v5 override율은 agent action 기준 약 0.184%, v6는 team step 기준 46.82% 또는 agent action 기준 9.36%다. 서로 다른 분모를 바로 비교하지 않는다.
- v7은 v6와 seed 집합·예산·curriculum·탐색 설정이 다르다. A2/A3 50%를 v6 50%와 동일 경기에서 비교한 값으로 취급하지 않는다.
- 가속 12프로세스 327.68 step/s는 합산 짧은 수집 성능이며 단일 모델 추론 속도나 전체 학습 평균이 아니다.
- 현재 실행 파일 hash는 일치하지만 최신 공식 빌드와의 일치까지 확인한 것은 아니다. 환경 업데이트 후 기존 결과를 같은 benchmark로 섞지 않는다.

## 6. 세대별 학습 문제와 현재 상태

아래 원인은 이미 실행 기록에서 확인된 것과 추가 분석 가설을 구분했다. 상세 발전 기록은 [history.md](../history.md)에 있다.

| 세대 | 확인된 이상 / 실패 | 현재 대응·남은 문제 | 근거 |
| --- | --- | --- | --- |
| 기반 작업(편의상 v0) | seed가 반환 episode에 적용되지 않음, reset 간 map/routing cache 잔존, self-ID 누락, terminal 점수 reset | seed/cache/canonical batching/score tracker 보정. 배터리 수량, 제출 metadata 계약은 남음 | [versions](../versions.md), [계약 검증](../logs/prep04_07_contract.json) |
| v1 | 512-step rollout마다 경기 초기화; 2,000,384 step에도 완료 학습 경기 0, target 0/10 | v2 persistent collector로 수명 분리. 현재 버그로 재분류하지 않음 | [v1 summary](../logs/mappo_vs_win70/run_summary.json), [v2 변경 분석](../v2/reports/mappo_vs_win70_v2_plan_changes.md) |
| v2 | 2,000,896 step, 학습 1,643전 전패, target 0/10. 강한 win70의 성능이 neural core 대신 planner override에서 나옴 | checkpoint weight만 이전해 원래 정책 성능을 재현할 수 없었음 | [v2 summary](../logs/mappo_vs_win70_v2/run_summary.json), [추론 정책](../blackout_rl/policy.py) |
| v3 | 3,000,320 step, target 0/10. NoOp label 41.4%, guard 93% 이상; BC 정확도와 실전 성적 괴리; 실패 단계 강제 승급 | 단일-step imitation 누적오차와 회귀. v4 planner 보존·fail-closed로 전환 | [v3 summary](../logs/mappo_teacher_curriculum_v3/run_summary.json), [계획/분석](../v3/reports/mappo_teacher_curriculum_v3_plan.md) |
| v4 | 100,352 step에서 중단. scripted 8/10·target 5/10 유지. warm-up은 KEEP BC이고 PPO 꺼짐 | 개선 신호 없는 보존 학습과 baseline보다 높은 초기 gate가 병목. v5에서 수정 | [v4 summary](../logs/mappo_planner_residual_v4/run_summary.json), [v4 분석](../v4/reports/mappo_planner_residual_v4_plan_changes.md) |
| v5 | 317,440 step, target 5/10, rollback 1회. PPO override 약 0.184%/agent action; 표집 후 1명 선택과 학습 확률 정합성 문제 | v6 team 41-way 분포·탐색 하한으로 대응. 기존 성능 회복이 개선을 뜻하지 않음 | [v5 summary](../logs/mappo_planner_residual_v5/run_summary.json), [v6 변경](../v6/reports/mappo_planner_residual_v6_plan_changes.md) |
| v6 | 1,159,168 step, target 5/10. 탐색 증가에도 target-best는 step 0, regression rollback 2회, full_win70 gate 실패 | 탐색 빈도 부족만으로 설명 불가. 수정 행동의 장기 기여·학습/argmax 차이 분석 필요. 최종 test/목표 제출물 없음 | [v6 summary](../logs/mappo_planner_residual_v6/run_summary.json), [최종 dev](../logs/mappo_planner_residual_v6/target_eval_step_1159168.json) |
| v7-1 | A2/A3 각 5시드, 총 10개의 `episodes` 배열이 **모두 동일**. 각각 dev 50%; A0 15.33%, A1 21.67% | 실제 배선 고유 이득 미입증. KEEP 고정/행동 일치 가능성은 가설이며 행동 궤적 확인 전 확정 불가. 점수 동일성 해석에는 A01 결함도 반영 | [본실험 summary](../logs/v7/main_study/accelerated/summary.json), [학습 구현](../blackout_rl/v7_1/model.py) |
| v7-2 파일럿 | 전뇌 담당 슬롯까지 scripted 경로 계산, `(1,20)`에서 `PathNotFound`; 34,123 step에서 실패, 저장은 32,768 | active slots 분리와 도달 불가 처리로 보정. teammate-v2는 별도 실험 ID로 보존 | [복구 보고서](../v7/reports/teammate_v2_recovery.md), [teammate 코드](../blackout_rl/v7_2/teammates.py) |
| v7-2 본실험 | 3시드 × 200만 step 완료, 32승/180경기=17.78%. 1유닛 전뇌+4유닛 scripted | 5유닛 전뇌 성과가 아님. 전뇌 재배선/CNN·GRU 대조군, F2, 최종 test 미실행 | [본실험 범위](../v7/reports/main_study_preregistration.md), [본실험 summary](../logs/v7/main_study/accelerated/summary.json) |

### v7 결과를 해석할 때 특히 남는 질문

- 그래프 residual의 **평가 시 KEEP/override 선택률과 실제 action**이 planner와 같은가? 현재 evaluation JSON에는 이를 확정할 궤적이 없다.
- 학습 중 sampled correction의 보상이 deterministic argmax 행동에도 반영되는가? 학습 override율만으로 답할 수 없다.
- A0/A1의 열세는 encoder/초기화/학습 안정성/행동 변화 중 무엇과 관련되는가? 지금 결과는 원인 식별까지 하지 않는다.
- 진영 편향은 어떤 seed 집합에서 생기는가? v6 dev에서는 B가 낮지만 v7 A2/A3에서는 A 40%, B 60%다. “B 진영이 항상 불리”는 근거가 없다.
- 단일 학습 seed의 paired bootstrap `[0.5,0.5]`는 해당 표본에서 pair별 승률이 같다는 뜻이다. 배선 개선이나 일반화 불확실성 0을 뜻하지 않는다. 독립 학습 seed까지 고려한 비교가 필요하다.

## 7. 조치 우선순위와 완료 조건

| 순서 | 조치 | 완료 조건 |
| --- | --- | --- |
| 1 / P0 | v7 evaluator 점수 기록 및 공통 평가 계약 정비 | 정상·100점 조기 종료·timeout에서 score 출처/단위와 winner 구분, 종료 snapshot 또는 관측 한계 명시, 기존 결과 보존 |
| 2 / P0 | Unity timer 수정·회귀 확인 | 자동 재시작과 명시 reset 모두 3회 이상 연속 timeout 정상; 22,000 watchdog과 종료 사유 구분 |
| 3 / P0 | 공식 판정/제출 계약 확정 | reward-sum vs game winner, model-side planner, training teacher, canonical slot, state/reset, 자원 한도에 대한 명시 답변 |
| 4 / P1 | 관측 정확성 검사 확대 | macOS/Linux-Xvfb/Windows semantic ID 경계 비교, 수량 부재/겹침/자기 ID 한계 문서화 |
| 5 / P1 | v7 행동 수준 감사 | planner/KEEP/override action과 유효 수정 결과를 A0~A3 동일 관측에서 비교; 동일 episodes의 원인 식별 |
| 6 / P1 | reward/winner 근원 검사 | 동점+shaping, 점수 감소, 마지막 득점 이벤트에서 game event와 Python label 비교 |
| 7 / P1 | 최종 성능 검증 | 선택 규칙·모델 lock 이후 confirmation/test; v7-2 구조 우월성 주장에는 별도 대조군 필요 |
| 8 / P2 | 현황 문서 정리 | README/history/실행 상태 및 현재 코드 적용 범위 일치, v0 명칭 확정 |

## 8. 이번 점검의 검증 기록과 재현 방법

### 수행 결과

- Unity/Python API 원본 commit과 clean 상태 확인, 로컬 player hash 재계산 완료.
- 사용자 목록 18개 모두 코드/문서/로그 또는 미검증 사유와 연결.
- 최종 관리자 summary가 지정한 23개 평가 파일을 재집계: 1,380경기, zero score 1,211경기, 승자가 있는 zero score 1,204경기.
- A2/A3의 10개 평가 `episodes` 배열 전체 동일성 재확인. 행동 궤적의 동일성을 검사한 것은 아님.
- 관련 테스트 **62개 통과(6.655초)**. 합성 환경과 기존 로그 검사 포함이며 이번 실행에서 실제 Unity 경기를 새로 돌린 것은 아님.
- 타이머 목록 재진입과 전처리 보간 오인식에 대해 합성 예시 실행. 원본 Unity/Xvfb 재현과 구분.

### 관련 테스트 명령

프로젝트 루트에서:

```bash
.venv/bin/python code/run.py -m unittest \
  tests.test_env tests.test_contract tests.test_mappo_v6 \
  tests.v7.test_contracts tests.v7.test_teammates tests.v7.test_main_study
```

이 테스트가 통과해도 A01이 남아 있다. v7 실제 평가 산출물의 점수 보존까지 현재 테스트가 보장하지 못한다는 뜻이다.

### v7 평가 로그의 중복 없는 재집계

```python
import json
from pathlib import Path

summary = json.loads(Path(
    'logs/v7/main_study/accelerated/summary.json'
).read_text())
episodes = []
graph_episodes = []
for run in summary['runs']:
    result = json.loads(Path(run['result']).read_text())
    episodes.extend(result['episodes'])
    if any(x in run['experiment_id'] for x in ('_flywire_', '_rewired_')):
        graph_episodes.append(result['episodes'])
print('runs / episodes:', len(summary['runs']), len(episodes))
print('zero score:', sum(e['score_0'] == e['score_1'] == 0 for e in episodes))
print('zero score with winner:', sum(
    e['score_0'] == e['score_1'] == 0 and e['winner'] != -1 for e in episodes
))
print('A2/A3 arrays identical:', all(
    x == graph_episodes[0] for x in graph_episodes
))
```

예상 출력: `23 / 1380`, `1211`, `1204`, `True`. summary의 결과 경로는 이 workspace의 절대 경로이므로 다른 호스트에서는 경로를 맞춰야 한다.

**열린 한계:** 공식 답변의 최신화 여부, 새 빌드에 그래픽/창고/timer 수정이 포함됐는지, Linux/Xvfb 현상, 실제 terminal winner 오판 발생 여부는 아직 확인되지 않았다. 이 항목들은 미발생·해결 완료로 처리하지 않는다.
