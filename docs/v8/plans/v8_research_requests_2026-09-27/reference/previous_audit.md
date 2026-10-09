# BlackOut RL 저장소 문제·성능 병목 감사

**대상:** `thislis/2026-IST-tech-RL` · **기준일:** 2026-09-27 KST  
**고정 snapshot:** `main@e6e98a63e9f401f6bf03e463a720c03878d1beb7`

## 1. 결론

현재 가장 큰 병목은 더 큰 신경망이나 학습 시간의 부족으로 확정할 수 없다. **평가의 신뢰성 → 실제 정책 개입 → 과제 정보/행동 표현 → 인과성/통계 검증 → 제출 실행 계약**이 연결되지 않은 것이 더 먼저 해결할 문제다.

v6는 더 많은 탐색을 도입했어도 최종 target 5/10이고 best는 시작 시점에 머물렀다. v7은 23개 run, 총 4,600만 환경 step을 마쳤지만 A2/A3의 요약 경기 기록이 동일하고 전뇌 F1은 17.78%였다. 다만 v6와 v7은 평가 seed·표본·실험 목적이 달라 이 승률을 직접 세대별 향상률로 읽어서는 안 된다.

이 문서는 실제 버그, 고정 환경의 제약, 연구 설계의 한계, 아직 확인되지 않은 병목 가설을 모두 포함한다. 아래 36개를 “현재 발생한 36개 버그”라고 읽으면 안 된다. 가설은 실험 전까지 가설로 유지한다.

### 범위와 확인 수준

GitHub 연결로 전체 tracked tree를 확인하고 핵심 실행 경로, 설정, 보고서, 브랜치와 최신 커밋/PR을 대조했다. 모든 과거 파일을 한 줄도 빠짐없이 읽거나 모든 버그가 없음을 증명한 감사는 아니다. `logs/`, `artifacts/checkpoints/`, `artifacts/builds/`, `artifacts/data/connectomes`가 snapshot에 없어 실제 checkpoint 행동·학습 로그 재집계·Unity 재실행은 하지 못했다. 기존 감사 문서의 62개 테스트 통과는 이번 실행 결과가 아니다.

이번에 별도로 실행한 것은 `audit_sanity_checks.py`의 네 가지 산술/인터페이스 검사뿐이다. 이 검사는 실제 학습 정책의 붕괴나 실제 게임 패배를 재현했다고 주장하지 않는다.

근거 코드: C=현재 코드/설정 확인, R=커밋된 보고 결과, H=실제 기여도 미확정 가설, S=이번 합성 검사.

## 2. 실행 구조와 원인 연결

```text
Unity build
  → BlackOutEnv / ContractBlackOutEnv
  → 관측(vector96 + semantic96×96×11) / winner / score
  ├─ v6·v7-1: planner + frozen encoder + 49차원 문맥
  │              → KEEP + 5×8 joint residual → 최대 한 슬롯 수정
  └─ v7-2: 자기 시점 retina → 고정 MaleCNS proxy dynamics
                 → 6개 output pool rates → 9-way readout → slot0 제어
                 + 나머지 네 슬롯 scripted teammate
  → 팀 score-delta/terminal reward → centralized critic/GAE/PPO
  → 별도 evaluator → dev/confirmation/test/모델 선택
  → export / 실제 제출 loader
```

원인 사슬은 다음처럼 추적해야 한다. 잘못된 종료 정보가 점수와 라벨을 오염시키면 어떤 개선을 평가해야 하는지부터 불명확해진다. 관측/특징에서 필요한 정보를 버리거나 residual이 실제 행동을 바꾸지 않으면 optimizer는 돌아도 새로운 전략이 실행되지 않는다. 그래프 차이를 실제 행동 차이와 분리해 측정하지 않으면 같은 승률의 원인을 알 수 없다. 마지막으로 로컬 정책과 제출 정책의 상태/의존성 계약이 다르면 좋은 로컬 결과도 운영 성능으로 이어지지 않는다.

## 3. 버전별 실패 경로와 현재 상태

아래 수치는 `docs/common/history.md` 및 `docs/issues/1st_issues_v0-v7.md`에 기록된 결과다. 원본 로그를 이번에 재집계한 값이 아니다.

| 세대 | 기록된 결과 | 추적된 원인 | 현재 판정 |
|---|---|---|---|
| PREP/BASE | seed·관측·평가 계약 정비 | seed 전달 순서, reset 캐시, 자기 ID, terminal score | 여러 항목 보정됨. v0는 공식 세대명이 아님 |
| v1 | 2,000,384 step, 완료 경기 0, dev 0/10 | rollout마다 env를 초기화하여 terminal 보상 데이터 없음 | persistent collector로 후속 세대에서 수정 |
| v2 | 2,000,896 step, 학습 1,643전 전패, dev 0/10 | 강한 win70의 planner를 제외하고 약한 neural core만 이전 | 원인 규명, 정책 전체 artifact 계약이 중요 |
| v3 | 3,000,320 step, dev 0/10 | NoOp/guard 편향, 한 step 모방 오차 누적, 실패 단계 강제 승격 | 역사적 실패; 현재 v7 collector와 동일 코드라고 보면 안 됨 |
| v4 | 100,352 step, target 5/10 | KEEP 정답만 BC하고 PPO off인 warm-up에서 개선 gate 요구 | 구조 변경으로 후속 대응 |
| v5 | 317,440 step, target 5/10, rollback 1 | 낮은 개입률과 실행된 joint action/확률 불일치 | v6 joint distribution에서 수정 |
| v6 | 1,159,168 step, target 5/10, rollback 2 | 탐색은 늘었지만 best는 step0; 학습이 개선으로 연결 안 됨 | 미해결 성능 병목 |
| v7-1 | 4구조×5seed×200만 step | A0 15.33%, A1 21.67%, A2/A3 각50%; A2/A3 기록 동일 | wiring 고유 이득 미입증 |
| v7-2 파일럿 | 34,123 step에서 PathNotFound | 전뇌 담당 슬롯도 scripted 경로 계산 | active slots/teammate-v2로 후속 보정 |
| v7-2 | 3seed×200만 step | F1 32/180승=17.78%; 한 슬롯·고정 전뇌·linear readout | 기전/대조군/제출 검증 미완료 |

v5의 agent당 개입률과 v6의 팀 step당 개입률은 분모가 다르다. v6/v7의 50%도 동일 seed 검정 결과가 아니다. B 진영이 항상 불리하다는 일반화 역시 맞지 않는다. 기존 감사의 v7 A2/A3는 A40%/B60%로 보고돼 있다.

## 4. 가장 먼저 확인할 정량적 단서

### 평가 점수

기존 감사는 본실험 1,380경기 중 1,211경기에서 score0=score1=0을 보고했고 1,204경기에는 승자가 있었다. 현재 `evaluate_v7.py`가 terminal info 점수를 그대로 읽는 것을 정적으로 확인했다. 마지막 비종료 점수는 임시 복구 수단일 뿐 최종 득점 snapshot과 동일하지 않다.

### 학습 개입과 평가 개입의 차이

40개 수정 행동에 균등하게 확률이 분산된 경우:

| 수정 총확률 | KEEP 확률 | 수정 하나의 확률 | 평가 argmax |
|---:|---:|---:|---|
| 10% | 90% | 0.25% | KEEP |
| 50% | 50% | 1.25% | KEEP |
| 80% | 20% | 2% | KEEP |
| 97% | 3% | 2.425% | KEEP |
| 98% | 2% | 2.45% | 수정 |

이 균등 예시의 전환점은 40/41≈97.56%다. 모든 실제 학습 정책에 이 수치가 적용되는 것은 아니다. 하나의 수정 방향으로 확률이 집중되면 훨씬 낮은 총 수정 확률에서도 argmax가 수정으로 바뀐다. 중요한 것은 **학습 sampled override 비율만으로 greedy 평가 개입을 판단할 수 없다는 점**이다.

초기 bias가 동일한 상태에서 legal correction이 40개면 총 수정 확률 10%, 20개면 약5.26%, 8개면 약2.17%다. v7의 exploration 하한은 0이라 불리한 상태에서 개입 확률이 보장되지 않는다.

### 전뇌 인과성 gate

`unity_causality_teammate_v2.json`의 저장 결과에서 normal 대비 sensory_block은 22 action step이 다르고, output_block은 74 step이 다르며, frozen_frame은 0 step이 다르다. 이 실험은 F0의 짧은 기술 검증이다. 현재 gate는 frozen_frame 차이를 요구하지 않고 학습된 F1을 시험하지 않는다.

### 실행 성능

가속 보고서의 v7-1 수집은 28.45→57.25 env step/s, v7-2는 13.24→51.52 env step/s다. batch5 encoder는 CPU1.73ms/MPS3.89ms여서 CPU를 유지했고, 전뇌 CSR은 CPU19.10ms/MPS5.15ms여서 MPS를 사용했다. 12개 합산327.68step/s는 짧은 수집 측정이며 PPO update·전체 평가 시간·제출 p99를 나타내지 않는다.

## 5. 문제·병목 추적 목록

P0=결과 신뢰성/경기 종료/제출 게이트, P1=성능·관측·검증의 우선 병목, P2=추가 검증·운영·해석 개선이다. 이는 추천 우선순위이며 특정 담당자가 이미 배정됐다는 뜻이 아니다.

| ID | 우선 | 영역 | 항목 | 근거 |
|---|---|---|---|---|
| F01 | P0 | 평가 | v7 종료 프레임 점수 기록 오류 | C/R |
| F02 | P0 | 판정 | 로컬 승자·upstream 경기 판정 기준 불일치 위험 | R |
| F03 | P1 | 보상 | 종료 순간의 마지막 점수 변화가 보상에서 빠짐 | C |
| F04 | P0 | 환경 | Unity 타이머 재진입 결함의 잔존 위험 | R |
| F05 | P1 | 환경 | 고정 지형과 활성 창고 변동을 혼동할 위험 | R |
| F06 | P1 | 관측 | semantic one-hot가 맞아도 원래 클래스가 맞다는 보장은 없음 | R/C |
| F07 | P1 | 관측 | 배터리 수량·자기 ID·상대 클래스의 정보 한계 | R/C |
| F08 | P1 | 안전/기하 | 하드코딩 금지 영역·근사 action mask와 실제 물리의 차이 | C/H |
| F09 | P1 | 정책 | sampling→argmax와 KEEP 편향: 학습 변화가 평가 행동으로 안 나올 수 있음 | C/H/S |
| F10 | P1 | 정책 | 한 유닛·한 프레임만 바꾸는 residual의 전략 표현 한계 | C/H |
| F11 | P1 | 표현 | 약한 neural core를 frozen encoder로 계속 사용 | C/R/H |
| F12 | P1 | planner | 하나의 PathNotFound가 팀 전체 정지로 확대 | C |
| F13 | P1 | 학습 신호 | 20ms 미세 행동과 지연 팀 보상 사이의 credit assignment | C/H/S |
| F14 | P1 | 실험 목적 | 구조 비교용 고정 schedule을 성능 개선 curriculum으로 혼동 | C/R |
| F15 | P1 | 행동 검증 | A2/A3 동일 경기 기록의 원인을 아직 식별하지 못함 | R/C |
| F16 | P1 | connectome | v7-1은 생물학적 기능 이전이 아니라 topology prior 실험 | C/H |
| F17 | P2 | 실험 통제 | 단일 재배선과 parameter matching만으로 비교가 완결되지 않음 | C/R |
| F18 | P1 | v7-2 인터페이스 | 자기 시점 감각과 절대 방향 readout의 좌표계 불일치 | C/H/S |
| F19 | P1 | v7-2 정보 | 환경이 제공하는 vector 정보 대부분을 감각 경로에서 버림 | C/H |
| F20 | P1 | v7-2 용량 | 전뇌 규모와 실제 학습·제어 용량의 큰 차이 | C/R/H |
| F21 | P2 | v7-2 시간 정렬 | 20ms 행동·100ms 시각 갱신·260ms 출력 창의 지연 | C/H |
| F22 | P1 | 인과성 검증 | 인과성 gate가 동적 화면 활용이나 학습 후 F1을 보장하지 않음 | C/R |
| F23 | P1 | 실험 해석 | 전뇌 재배선·CNN/GRU 대조군 부재 | R/C |
| F24 | P2 | 생물학적 가정 | 전뇌 동역학·망막 위치의 proxy 가정 | C |
| F25 | P1 | 통계/일반화 | dev·학습 seed·세대별 조건을 분리한 최종 검증 미완료 | R/C |
| F26 | P2 | 실행 성능 | 통신 직렬화·planner·CSR 비용: 일부는 이미 개선됨 | R |
| F27 | P1 | 운영 성능 | 짧은 aggregate 처리량은 학습 총시간·제출 지연 보장이 아님 | R/C |
| F28 | P0 | 제출 | v7 export가 의도적으로 비활성화됨 | C |
| F29 | P0 | 제출 계약 | agent 순서·episode/step/reset·패키지 계약 미확정 | C/R |
| F30 | P1 | 재현성 | 핵심 로그·가중치·그래프·Unity build가 저장소에 없음 | C |
| F31 | P2 | 재현성 | resume는 물리 상태의 bitwise 연속 실행이 아님 | C/R |
| F32 | P1 | 검증 체계 | 테스트 통과가 실제 경기 산출물의 정확성을 보장하지 못함 | C/R |
| F33 | P2 | 개발 추적 | README·history·등록 문서의 상태가 어긋남 | C/R |
| F34 | P2 | 잠재 회귀 | v7 평가가 direct intervention 설정을 전달하지 않음 | C |
| F35 | P1 | 보상/판정 | 점수 감소에도 득점 shaping이 발생하는 upstream 경로 | R |
| F36 | P2 | 전략 ablation | 특수 아이템 비활성 선택의 근거가 약한 상대·소표본에 제한됨 | R |

### 항목별 원인·영향·완료 조건

#### F01 · P0 · v7 종료 프레임 점수 기록 오류

**상태:** 현재 코드와 기존 감사 기록에서 확인

**원인:** 구형 code/shared/eval/evaluator.py의 마지막 비종료 점수 보존 계약을 code/shared/scripts/evaluate_v7.py가 재사용하지 않고 종료 info의 score_0/score_1을 바로 기록한다. 종료 직후 초기화된 점수가 평가 파일에 섞인다.

**영향:** 기존 감사 문서의 1,380경기 중 1,211경기가 0:0이며 그중 1,204경기는 승자가 있다. 점수 차·득점 기반 진단을 신뢰할 수 없다. 이것만으로 승패 라벨까지 틀렸다고 단정하지 않는다.

**권고 조치:** 평가 결과 스키마를 통합하고 authoritative terminal snapshot을 우선 사용한다. 불가능하면 마지막 비종료 점수를 preterminal로 명시하며 기존 결과는 덮어쓰지 않는다.

**완료 조건:** 정상 득점·점수 감소·목표 점수 조기 종료·timeout fixture에서 점수 출처/단위/시점이 검증되고 Unity 이벤트와 대조된다.

**근거 파일:** `code/v7/scripts/evaluate_v7.py`, `code/shared/eval/evaluator.py`, `docs/issues/1st_issues_v0-v7.md`

#### F02 · P0 · 로컬 승자·upstream 경기 판정 기준 불일치 위험

**상태:** 고정 upstream에 대한 기존 감사 결과; 현 공식 운영 환경 미확인

**원인:** 기존 감사 문서에 따르면 프로젝트는 info.winner를, upstream competition runner는 누적 Unity reward를 사용한다. wrapper의 winner 자체도 첫 종료 agent의 reward 부호에서 유도된다.

**영향:** 게임 점수상의 승리, shaping reward 우세, 저장된 winner가 다른 의미가 될 수 있다. 실제 운영 서버가 이 코드인지, 실제 오판이 발생했는지는 별도 확인이 필요하다.

**권고 조치:** 게임 엔진이 winner/score/termination_reason을 명시적으로 내보내고 로컬 evaluator와 공식 runner가 같은 필드를 사용하도록 계약을 확정한다.

**완료 조건:** 동점+shaping, 동시 이벤트, 마지막 득점, timeout에서 engine outcome과 양쪽 runner의 판정이 일치한다.

**근거 파일:** `docs/issues/1st_issues_v0-v7.md`, `code/shared/eval/evaluator.py`, `code/shared/blackout_rl/reward.py`

#### F03 · P1 · 종료 순간의 마지막 점수 변화가 보상에서 빠짐

**상태:** 의도된 reset 방어의 잔여 한계

**원인:** ScoreDeltaRewardTracker는 terminated이면 점수 변화분을 0으로 놓고 승패 보너스만 추가한다. reset 점수에 의한 가짜 음의 보상을 피하지만 마지막 deposit의 실제 delta도 관측하지 못한다.

**영향:** 마지막 득점 행동의 dense reward와 기록상 최종 점수가 빠질 수 있다. 승리 보너스가 있으므로 마지막 행동이 완전히 무보상인 것은 아니다.

**권고 조치:** 종료 snapshot에서 마지막 delta와 승패 보너스를 각각 계산하고 보상 이벤트를 분리 기록한다.

**완료 조건:** 99→100점 종료 같은 제어 fixture에서 실제 delta가 한 번만 반영되고 reset 100→0은 보상으로 들어가지 않는다.

**근거 파일:** `code/shared/blackout_rl/reward.py`, `code/v7/blackout_rl/v7_training.py`, `code/shared/eval/evaluator.py`

#### F04 · P0 · Unity 타이머 재진입 결함의 잔존 위험

**상태:** upstream 소스 결함 보고; 현재 학습 경로 피해 미확정

**원인:** 기존 감사 문서에서 TimerManager.Tick의 callback이 timer 목록을 비우고 새 timer를 넣은 뒤 기존 루프의 RemoveAt이 새 timer를 제거하는 경로를 확인했다.

**영향:** Unity 자동 재시작에서는 다음 경기 timeout이 사라질 위험이 있다. Python의 명시적 env.reset은 호출 시점이 달라 같은 영향을 단정할 수 없고 기존 1,380경기는 watchdog 안에서 종료됐다.

**권고 조치:** callback 실행 전 만료 timer를 제거하거나 안정된 ID/지연 큐로 변경하고 새 빌드로 검증한다.

**완료 조건:** 자동 재시작과 명시 reset 각각 최소 3회 연속 timeout, timer ID와 종료 사유·build hash를 함께 보존한다.

**근거 파일:** `docs/issues/1st_issues_v0-v7.md`, `code/shared/blackout_rl/env.py`

#### F05 · P1 · 고정 지형과 활성 창고 변동을 혼동할 위험

**상태:** 고정 로컬 빌드의 조건; 문서와 불일치

**원인:** 기존 감사에서 팀 A 후보 영역 5개 중 4개를 seed로 선택하고 B에 대칭 생성하는 scene 설정을 확인했다. 클래스 기본값, scene asset, 전달받은 설명이 다르다.

**영향:** 에피소드 간 창고 위치·도달성 캐시를 무조건 재사용하거나 다른 빌드의 창고 수를 가정하면 경로와 정책이 틀어진다.

**권고 조치:** 지형·후보·실제 활성 창고를 구분하고 reset마다 현재 관측으로 활성 집합을 갱신한다.

**완료 조건:** 여러 seed와 빌드에서 활성 영역 목록을 기록하고 planner 목표가 해당 에피소드의 합법 영역인지 검사한다.

**근거 파일:** `docs/issues/1st_issues_v0-v7.md`, `docs/common/game_spec.md`, `code/shared/blackout_rl/env.py`

#### F06 · P1 · semantic one-hot가 맞아도 원래 클래스가 맞다는 보장은 없음

**상태:** 보간·가림·플랫폼 차이에 대한 잔여 위험

**원인:** grayscale ID를 보간한 뒤 반올림하면 유효하지만 잘못된 semantic ID가 만들어질 수 있다. 한 픽셀은 한 클래스이므로 유닛이 아이템/배경을 가린다.

**영향:** 벽/창고 경계 오인식, 아이템 소실을 획득으로 오인하는 문제가 가능하다. macOS 결과를 Linux/Xvfb·Windows까지 일반화할 수 없다.

**권고 조치:** one-hot 검사 외에 원본 semantic ID, 렌더링 경계 픽셀, scene 객체 라벨을 비교한다. -batchmode를 사용하되 -nographics로 시각 관측을 없애지 않는다.

**완료 조건:** 플랫폼별 경계 fixture·가림 fixture를 통과하고 그래픽 0/stale 관측을 실행 초기에 차단한다.

**근거 파일:** `docs/issues/1st_issues_v0-v7.md`, `code/shared/blackout_rl/observation.py`, `code/shared/blackout_rl/env.py`, `docs/pre_v1/reports/prep04_observation_contract.md`

#### F07 · P1 · 배터리 수량·자기 ID·상대 클래스의 정보 한계

**상태:** 환경 계약/표현의 한계

**원인:** 배터리 stack 수량은 직접 제공되지 않는다. 96차원 vector는 자기 unitIndex를 포함하지 않아 외부 agent 이름/slot이 필요하며 상대 클래스도 직접 주어지지 않는다.

**영향:** 배터리 픽셀 면적을 가치로 사용하거나 입력 행 순서를 자기 ID로 잘못 해석하면 타깃 배정과 추론이 틀어진다. 더 큰 모델만으로 숨겨진 정보를 복원할 수는 없다.

**권고 조치:** 관측 가능한 사실과 추정을 분리하고 canonical agent/slot 계약을 유지한다. 수량이 필요하면 허용된 관측 확장 또는 불확실성을 가진 기억 추정을 별도 실험한다.

**완료 조건:** 배터리 수량을 픽셀 수로 계산하지 않으며 입력 dict permutation에서도 동일 agent에게 동일 의미의 행동을 반환한다.

**근거 파일:** `code/shared/blackout_rl/observation.py`, `docs/issues/1st_issues_v0-v7.md`, `artifacts/submission/policy.py`

#### F08 · P1 · 하드코딩 금지 영역·근사 action mask와 실제 물리의 차이

**상태:** 정적 제약 확인; 실제 손실 비중 미측정

**원인:** PlannerFeatures는 고정 shrine 영역과 제한된 endpoint/offset 표본으로 대체 방향의 합법성을 판정한다. hidden entry restriction, 유닛 충돌, 속도/크기 효과 전부를 시뮬레이터처럼 계산하지는 않는다.

**영향:** 유효 mask라도 실제로 못 움직이거나, 안전한 우회가 과도하게 제외될 수 있다. map/build 변경에도 민감하다.

**권고 조치:** 마스크 탈락 이유와 요청 행동/실제 displacement를 함께 기록하고 빌드별 기하 fixture로 검증한다.

**완료 조건:** 벽·모서리·금지 영역·버프 상태별 false-positive/false-negative mask 비율과 stuck 비율을 측정한다.

**근거 파일:** `code/v6/blackout_rl/mappo_v6.py`, `docs/pre_v1/reports/prep07_action_semantics.md`, `docs/issues/1st_issues_v0-v7.md`

#### F09 · P1 · sampling→argmax와 KEEP 편향: 학습 변화가 평가 행동으로 안 나올 수 있음

**상태:** 구조/산술 확인; 실제 checkpoint 붕괴는 추가 확인 필요

**원인:** 학습은 41-way 분포에서 sample하지만 V6Policy 평가는 argmax다. KEEP logit은 0, 초기 40개 수정 logit은 log(0.1/(40×0.9)). v7 본실험 exploration 하한은 0이고 legal mask가 수정 확률을 추가로 줄인다.

**영향:** 수정 총확률 80%가 40개에 균등 분산되어도 KEEP 20%가 각 수정 2%보다 커 평가에서는 KEEP만 선택한다. 초기 legal 수정 8개면 총 수정 확률은 약 2.17%다. 이것이 A2/A3 동일 결과의 유력 가설이나 아직 실증은 아니다.

**권고 조치:** 동일 관측에서 p_KEEP, top correction logit, legal count, sampled/greedy action, planner 대비 차이를 기록한다. 계층적 개입 gate+조건부 방향 분포 또는 평가와 일치하는 선택 규칙을 새 실험 ID로 비교한다.

**완료 조건:** 학습/평가의 effective override·argmax KEEP 비율이 보고되고, 학습된 수정이 frozen planner 대비 paired 경기 성능을 개선한다. PPO log_prob는 실제 새 선택 분포와 일치해야 한다.

**근거 파일:** `code/v6/blackout_rl/mappo_v6.py`, `code/v7/blackout_rl/v7_1/model.py`, `code/v7/blackout_rl/v7_training.py`, `code/v7/configs/main_study/v7_1_flywire.yaml`

#### F10 · P1 · 한 유닛·한 프레임만 바꾸는 residual의 전략 표현 한계

**상태:** 의도된 안전 제약; 성능 상한 가설

**원인:** keep_plus_5x8_v6는 매 환경 step 최대 한 슬롯의 방향만 교체한다. action_repeat=1이며 목표·역할·장기 경로는 계속 planner가 정한다.

**영향:** 여러 유닛의 동시 협력이나 수십~수백 step의 전략 전환을 학습하기 어렵고 다음 프레임 planner가 작은 수정을 상쇄할 수 있다. 여러 프레임에 걸친 변경 자체가 불가능한 것은 아니다.

**권고 조치:** 개입 지속시간·목표 선택·역할 배정 residual을 제한된 옵션으로 확장하고 안전성·기존 planner 보존을 별도 평가한다.

**완료 조건:** 미세 방향 수정, 지속 옵션, 목표 residual을 같은 split/예산에서 비교하고 실제 경로·역할 변화와 승률을 함께 보고한다.

**근거 파일:** `code/v6/blackout_rl/mappo_v6.py`, `code/v7/configs/main_study/v7_1_flywire.yaml`

#### F11 · P1 · 약한 neural core를 frozen encoder로 계속 사용

**상태:** 설계 사실; 병목 여부 미검증

**원인:** 강한 win70의 실력은 역사적으로 planner override에 크게 의존했다. v7-1은 그 checkpoint의 actor encoder를 고정한 상태에서 residual만 학습한다.

**영향:** 기존 encoder가 필요한 전략 정보를 충분히 표현하지 못하면 residual 구조를 바꾸어도 학습 가능한 정보가 제한된다. 49차원 planner 문맥이 별도로 있어 완전한 정보 차단으로 단정하지 않는다.

**권고 조치:** frozen encoder, 부분 해제 encoder, planner/context 전용 encoder를 동일 조건으로 비교하고 표현 probe를 수행한다.

**완료 조건:** 표현에서 유효 타깃·막힘·다음 score event를 예측할 수 있는지 확인하며 표현 변경의 실제 성능 효과를 분리한다.

**근거 파일:** `code/v6/blackout_rl/mappo_v6.py`, `code/v7/blackout_rl/v7_1/model.py`, `docs/common/history.md`

#### F12 · P1 · 하나의 PathNotFound가 팀 전체 정지로 확대

**상태:** 현재 예외 처리 경로에서 확인

**원인:** PlannerFeatures.prepare가 팀 planner 호출 전체를 하나의 try/except로 감싸고 PathNotFound에서 planner를 reset한 뒤 5개 기본 행동 모두를 0으로 만든다.

**영향:** 일부 슬롯/목표의 도달 불가능성이 정상 유닛까지 중단시키고 FSM 기억을 지울 수 있다. 이 경로의 실제 빈도는 현재 공개 산출물로 확인하지 못했다.

**권고 조치:** 슬롯/목표 단위 실패 격리, 도달 불가능 타깃 제외, 정상 슬롯 행동 보존으로 바꾼다.

**완료 조건:** 한 슬롯만 도달 불가능한 fixture에서 나머지 네 슬롯의 계획과 행동이 유지되며 실패 횟수·영향 step을 로깅한다.

**근거 파일:** `code/v6/blackout_rl/mappo_v6.py`, `code/v7/blackout_rl/v7_2/teammates.py`

#### F13 · P1 · 20ms 미세 행동과 지연 팀 보상 사이의 credit assignment

**상태:** 수식/입력 구조 확인; 실제 기여도 미측정

**원인:** gamma=.9995, lambda=.99에서 GAE의 기하급수 가중 규모는 약 95.28 step(1.91 게임초)이다. 반면 경기와 전략 보상은 훨씬 길 수 있다. critic은 4×4 pooled map을 사용하고 planner/brain의 모든 내부 상태를 보지 않는다.

**영향:** 한 유닛의 짧은 개입 효과가 나머지 팀 행동과 긴 지연에 묻히고 value 오차가 advantage를 지배할 위험이 있다. GAE가 1.91초 이후를 잘라 버리거나 장기 학습을 불가능하게 한다는 뜻은 아니다.

**권고 조치:** actor/critic gradient를 분리 계측하고 지연별 reward, explained variance, override 전후 counterfactual을 기록한다. 목표/옵션 단위 제어와 더 정보가 충분한 critic을 별도 비교한다.

**완료 조건:** 학습 실패가 무개입·나쁜 개입·value 오차 중 무엇인지 분해되고 신호 개선이 paired 성능으로 이어진다.

**근거 파일:** `code/v7/configs/main_study/v7_1_flywire.yaml`, `code/v7/configs/main_study/v7_2_readout_ppo.yaml`, `code/shared/blackout_rl/rollout.py`, `code/v6/blackout_rl/mappo_v6_training.py`, `docs/pre_v1/reports/prep07_action_semantics.md`

#### F14 · P1 · 구조 비교용 고정 schedule을 성능 개선 curriculum으로 혼동

**상태:** 의도된 비교 설계의 목적 한계

**원인:** v7은 성적과 무관하게 20만/60만 step에서 opponent 비율을 변경하고 200만 latest만 선택한다. 실패 seed 우선순위와 탐색 하한도 끈다.

**영향:** 구조 비교의 통제에는 유리하지만 실패 원인을 고치고 최고 정책을 찾는 적응형 개발 절차는 아니다. 고정 예산 완주는 성공 gate 통과를 뜻하지 않는다.

**권고 조치:** 사전 등록 비교 실험을 보존하고 별도의 성능 개발 실험에서 행동 gate·성능 gate·rollback·진영별 환경-step 노출량을 관리한다.

**완료 조건:** comparison run과 optimization run이 구별되고 단계 승격 이유 및 실제 상대/진영 노출량을 보고한다.

**근거 파일:** `code/v7/blackout_rl/v7_training.py`, `code/v7/configs/main_study/v7_1_flywire.yaml`, `docs/v7/reports/main_study_preregistration.md`

#### F15 · P1 · A2/A3 동일 경기 기록의 원인을 아직 식별하지 못함

**상태:** 기존 로그 재집계 결과; 행동 궤적은 미검증

**원인:** 기존 감사에서 A2 재배선과 A3 실제 배선의 10개 episodes 배열이 모두 같다고 보고했다. 현재 evaluator는 승패·점수·길이만 남기고 행동/분포 추적을 남기지 않는다.

**영향:** 네트워크가 달라도 planner 유지로 동일 행동일 수 있고, 다른 행동이 같은 요약 결과를 낼 수도 있다. 현재 기록으로 두 경우를 분리할 수 없다.

**권고 조치:** 같은 고정 관측 replay와 같은 Unity seed에서 A0~A3, pure planner, 초기/final checkpoint를 행동 수준으로 대조한다.

**완료 조건:** logit/greedy action/요청 action/실제 변위의 최초 분기와 planner 대비 성능 변화가 확인된다.

**근거 파일:** `docs/issues/1st_issues_v0-v7.md`, `code/v7/scripts/evaluate_v7.py`, `code/v7/blackout_rl/v7_training.py`

#### F16 · P1 · v7-1은 생물학적 기능 이전이 아니라 topology prior 실험

**상태:** 명시적 설계 한계

**원인:** GraphResidualHead는 synapse count와 transmitter를 실제 연산 가중치로 쓰지 않고 임의 초기 trainable edge_weight를 사용한다. 호출마다 상태를 0으로 만들고 4회 전파하며 인공 feature port를 주입한다.

**영향:** 실제 신경 회로의 학습된 동작이나 장기 기억이 그대로 이식되는 구조가 아니다. 205-node 부분회로 밖의 많은 입력/출력도 제외되어 있어 생물학적 배선만으로 이득을 기대하기 어렵다.

**권고 조치:** 현재 실험은 topology prior로 정확히 명명한다. sign/count·상태 유지·port 설계를 변경하는 실험은 별도 대조군과 gate를 갖춘 새 프로토콜로 수행한다.

**완료 조건:** 구조 고유 효과를 주장하려면 동일 action 사용량과 통계적 비교 아래 rewired 대비 이득을 보여야 한다.

**근거 파일:** `code/v7/blackout_rl/v7_1/model.py`, `code/v7/blackout_rl/connectome/extract_circuit.py`, `logs/v7/reports/data/fafb783_cx_primary_audit.json`

#### F17 · P2 · 단일 재배선과 parameter matching만으로 비교가 완결되지 않음

**상태:** 설계/해석 범위

**원인:** 차수 보존 재배선은 구현되어 있지만 하나의 고정 topology 대조에 의존한다. parameter 수 ±5%는 optimizer 난이도·FLOPs·지연·유효 신호 깊이를 맞춘다는 뜻이 아니다.

**영향:** 결과 차이를 생물학적 구조 하나의 효과로 해석할 때 한계가 있다. 반대로 같은 결과만으로 모든 connectome 접근을 반증할 수도 없다.

**권고 조치:** 여러 재배선 seed, 동일 frozen planner baseline, compute/latency와 학습 파라미터를 함께 보고한다.

**완료 조건:** 구조 변동성과 학습 seed 변동성을 분리한 비교 결과가 제공된다.

**근거 파일:** `code/v7/blackout_rl/connectome/graph_controls.py`, `docs/v7/reports/main_study_preregistration.md`, `logs/v7/reports/main_study_parameter_counts.json`

#### F18 · P1 · 자기 시점 감각과 절대 방향 readout의 좌표계 불일치

**상태:** 인터페이스 한계 확인; 실제 실패 기여는 실험 필요

**원인:** SemanticRetina.render는 heading에 따라 영상을 회전한다. F1은 output rates만 Linear→9-way에 넣어 절대 동/북 등의 DIRECTIONS를 선택하고 heading은 readout에 제공하지 않는다. F0의 상대 turn decoder와 다르다.

**영향:** 같은 자기 시점 특징이 다른 heading에서 나타나면 적절한 세계 방향은 달라도 readout은 같은 절대 action을 내는 aliasing이 가능하다.

**권고 조치:** 전뇌 출력에서 상대 회전/전진을 선택한 뒤 heading으로 세계 좌표에 변환하는 readout을 새 실험으로 비교한다. 직접 정보 bypass가 생기는지 프로토콜을 명시한다.

**완료 조건:** world/heading을 함께 회전한 fixture에서 action이 적절히 회전하는 equivariance와 장기 경기 성능을 검증한다.

**근거 파일:** `code/v7/blackout_rl/v7_2/sensory.py`, `code/v7/blackout_rl/v7_2/policy.py`, `code/v7/blackout_rl/v7_2/decoder.py`

#### F19 · P1 · 환경이 제공하는 vector 정보 대부분을 감각 경로에서 버림

**상태:** 현재 감각 설계의 정보 손실

**원인:** SemanticRetina는 vector에서 자기 위치만 사용하고 held item, self class, score, remaining time 등을 neural input으로 전달하지 않는다. 전역 semantic map도 시야/거리/가림을 적용한 영상으로 축소한다.

**영향:** 같은 풍경에서 적재 상태·점수·남은 시간에 따라 달라야 하는 전략을 구분하기 어렵다. 기억으로 일부 추정할 수 있으나 직접 관측을 주는 것과 같지 않다.

**권고 조치:** 합법적인 내부 상태를 감각/고유수용 입력으로 인코딩하는 별도 조건을 만들고 순수 영상 조건과 비교한다.

**완료 조건:** 보유 아이템·점수·시간만 다른 fixture를 구분할 수 있고 해당 입력이 행동/승률에 미치는 효과가 측정된다.

**근거 파일:** `code/v7/blackout_rl/v7_2/sensory.py`, `code/shared/blackout_rl/observation.py`, `code/v7/blackout_rl/v7_2/policy.py`

#### F20 · P1 · 전뇌 규모와 실제 학습·제어 용량의 큰 차이

**상태:** 설계 사실; 성능 상한 가설

**원인:** 166,700개 노드/25,582,938개 간선을 실행하지만 정책의 학습 가능 출력부는 6개 output pool rates에서 9개 행동으로 가는 linear readout(가중치 54개+편향 9개=63개)이다. critic은 별도로 학습한다. brain weights는 고정, plasticity off, slot 0 한 유닛만 제어하며 나머지 4개는 scripted다.

**영향:** 계산 비용은 크지만 학습 가능한 정책 인터페이스는 작다. 한 유닛의 기여에 팀 전체 보상이 붙고 모든 전략을 새로 학습하는 실험도 아니다.

**권고 조치:** 동일 정보·슬롯·보상 조건의 가벼운 readout/MLP/GRU baseline과 특징 분리 가능성을 먼저 비교한다.

**완료 조건:** 같은 제어 범위에서 전뇌 특징이 직접 감각 특징보다 유용한지와 추가 계산 비용 대비 효과가 확인된다.

**근거 파일:** `code/v7/blackout_rl/v7_2/decoder.py`, `code/v7/blackout_rl/v7_2/dynamics.py`, `code/v7/configs/main_study/v7_2_readout_ppo.yaml`, `logs/v7/reports/data/malecns_v1_whole_brain_audit.json`

#### F21 · P2 · 20ms 행동·100ms 시각 갱신·260ms 출력 창의 지연

**상태:** 설계상 시간척도 차이; 피해 미측정

**원인:** game action은 매 step, retina 갱신은 5 step, output rate는 13 tick 평균이다. F1의 절대 방향 변경은 heading을 즉시 바꾸지만 다음 시각 refresh까지 이전 감각을 사용한다.

**영향:** 충돌/회피 상황에서 감각·자세·행동 간 지연과 빠른 방향 진동이 생길 수 있다. 긴 평균 창 자체가 잘못이라는 뜻은 아니다.

**권고 조치:** heading 변화·sensor timestamp·출력률·action을 정렬하여 기록하고 상대 방향 decoder와 주기/필터 ablation을 별도 등록한다.

**완료 조건:** 동일 장면에서 감각 지연에 따른 행동 오류와 stuck/oscillation 비율을 정량화한다.

**근거 파일:** `code/v7/blackout_rl/v7_2/policy.py`, `code/v7/blackout_rl/v7_2/dynamics.py`, `code/v7/configs/main_study/v7_2_readout_ppo.yaml`

#### F22 · P1 · 인과성 gate가 동적 화면 활용이나 학습 후 F1을 보장하지 않음

**상태:** 저장된 짧은 F0 시험에서 확인

**원인:** unity_causality_teammate_v2.json의 frozen_frame은 normal과 다른 action step이 0이다. gate는 sensory_block/output_block의 행동 변화와 물리 이동만 검사하며 DirectPolicy(model=None)인 F0를 실행한다.

**영향:** 센서/출력 경로가 존재한다는 증거이지 변하는 화면을 유용하게 이용한다거나 PPO 학습 후 F1이 그 경로를 계속 쓴다는 증거는 아니다.

**권고 조치:** 학습 전후 F1에서 frozen-frame, sensory/output block, spatial shuffle 등을 동일 noise/seed로 비교하고 terminal 경기 성능도 평가한다.

**완료 조건:** 실시간 감각과 전뇌 출력에 대한 행동 민감도뿐 아니라 과제 성능 기여가 확인된다.

**근거 파일:** `code/v7/scripts/validate_v7_2_unity.py`, `logs/v7/reports/unity_causality_teammate_v2.json`, `code/v7/blackout_rl/v7_registry.py`

#### F23 · P1 · 전뇌 재배선·CNN/GRU 대조군 부재

**상태:** 본실험 범위에 명시된 미실행 항목

**원인:** v7-2 주 실행은 F1 세 학습 seed뿐이고 B2/B3 등 구조 대조, F2, 다섯 슬롯 제어, 추가 F0 장기 성능 비교가 포함되지 않았다.

**영향:** 17.78%라는 결과가 전뇌 wiring, 감각 adapter, readout, 제어 범위 중 어디에서 왔는지 분리할 수 없다.

**권고 조치:** 감각·제어 슬롯·상대·보상·budget을 고정하고 관측→MLP/GRU, rewired wholebrain, real wholebrain을 비교한다.

**완료 조건:** 추가 계산 비용과 독립 seed 불확실성을 포함해 구조별 효과를 보고한다.

**근거 파일:** `docs/v7/reports/main_study_preregistration.md`, `docs/common/history.md`, `code/v7/configs/main_study/v7_2_readout_ppo.yaml`

#### F24 · P2 · 전뇌 동역학·망막 위치의 proxy 가정

**상태:** 코드에 명시된 모델링 가정

**원인:** dynamics.py는 physiological LIF 재현이 아닌 fly64-inspired proxy라고 명시한다. transmitter별 sign과 tonic/noise/threshold를 공통 규칙으로 정하며 일부 retinal 좌표는 connectivity 추정 또는 검증되지 않은 deterministic proxy다.

**영향:** 실제 connectome 노드/간선을 쓴 사실과 생물학적 기능을 보존한 사실을 혼동하면 실패 원인과 기대 효과를 잘못 해석한다.

**권고 조치:** 입력 mapping confidence별 구성과 동역학 민감도, output 활성/분산/감각 정보량을 보고한다.

**완료 조건:** proxy 변경에 대한 결과 민감성이 드러나며 생물학적 재현 수준을 과장하지 않는다.

**근거 파일:** `code/v7/blackout_rl/v7_2/dynamics.py`, `code/v7/blackout_rl/connectome/extract_circuit.py`

#### F25 · P1 · dev·학습 seed·세대별 조건을 분리한 최종 검증 미완료

**상태:** 최종 test 미실행은 등록된 절차상 상태

**원인:** v6는 10경기 dev/별도 confirmation gate, v7은 30맵×양 진영×여러 학습 seed로 조건이 다르다. evaluator의 bootstrap은 한 학습 seed 안의 paired map 범위다. 본실험 큐에는 confirmation/test가 없다.

**영향:** v6 50%와 v7 50%를 직접 세대간 동률로 보거나 [0.5,0.5] CI를 일반화 불확실성 0으로 해석할 수 없다. 현재 결과로 제출 성능을 확정할 수 없다.

**권고 조치:** 같은 맵·상대의 paired 차이를 계산하고 학습 seed와 map cluster를 고려한다. 선택 규칙을 잠근 뒤 미사용 confirmation/test를 실행한다.

**완료 조건:** 진영별·맵별·학습 seed별 결과, 구조별 효과와 불확실성, test 사용 이력이 함께 보존된다.

**근거 파일:** `code/v7/scripts/evaluate_v7.py`, `docs/common/history.md`, `docs/v7/reports/main_study_preregistration.md`, `docs/issues/1st_issues_v0-v7.md`

#### F26 · P2 · 통신 직렬화·planner·CSR 비용: 일부는 이미 개선됨

**상태:** 프로파일/가속 결과가 저장되어 있음; 이번에 재측정하지 않음

**원인:** 기존 128-step profile에서 collect 6.430s 중 env.step 누적 4.085s, prepare 1.414s가 큰 부분이다. 순수 Python Protobuf와 Pipe pickle, 전뇌 CSR 연산이 주요 비용이었다. 누적 시간 항목들은 서로 겹치므로 합산하면 안 된다.

**영향:** time_scale만 올려서는 개선되지 않는다. upb/queue, BLAS thread 제한, 전뇌 CSR MPS가 이미 적용되어 초기 병목의 일부는 해결됐다.

**권고 조치:** 기존 최적화를 유지하고 최신 전체 실행을 다시 프로파일한다. 작은 encoder를 무조건 MPS로 옮기거나 환경-step 정의를 바꾸지 않는다.

**완료 조건:** collect/update/checkpoint/evaluation/reset별 wall time과 단일 정책 지연을 분리 측정한다.

**근거 파일:** `logs/v7/reports/acceleration_baseline_profile.json`, `docs/v7/reports/acceleration_and_resume.md`

#### F27 · P1 · 짧은 aggregate 처리량은 학습 총시간·제출 지연 보장이 아님

**상태:** 측정 범위와 운영 계약의 공백

**원인:** 327.68 step/s는 약 20초의 12개 독립 실행 합산 수집 처리량이며 PPO update를 제외한다. 공식 inference 제한, 플랫폼/라이브러리 허용 범위, p95/p99는 확인되지 않았다.

**영향:** 좋은 aggregate 수치라도 한 팀의 deadline·메모리 제한을 넘을 수 있다. Mac Metal 가속을 Linux/CPU 제출에서 동일하게 사용할 수 있다고 가정하면 안 된다.

**권고 조치:** 제출 대상 하드웨어에서 planner+전처리+brain+readout 전체를 측정하고 update/checkpoint를 포함한 긴 실행을 별도로 보고한다.

**완료 조건:** 전체 정책 cold/warm p50/p95/p99, peak RSS/VRAM, 장기 throughput과 제한 충족 여부가 기록된다.

**근거 파일:** `docs/v7/reports/acceleration_and_resume.md`, `docs/issues/1st_issues_v0-v7.md`, `docs/pre_v1/reports/prep13_evaluation_contract_checklist.md`

#### F28 · P0 · v7 export가 의도적으로 비활성화됨

**상태:** 현재 명시적으로 차단

**원인:** code/shared/scripts/export_v7.py는 인자를 받은 뒤 research checkpoint가 submission-certified가 아니라고 오류 종료한다. tracked artifacts/submission/policy.py는 구형 독립 neural policy이며 v7 전체 실행물이 아니다.

**영향:** 학습 완료 checkpoint를 곧바로 제출물로 사용할 수 없다. export 차단을 제거하는 것만으로 호환성 문제가 해결되지는 않는다.

**권고 조치:** 실제 공식 계약에 맞춰 planner/brain/graph/state/의존성을 포함하는 export 설계를 검증한 후 차단을 해제한다.

**완료 조건:** 격리 환경에서 허용 파일만으로 load→반복 inference→여러 경기 reset을 통과하고 원래 정책과 행동이 일치한다.

**근거 파일:** `code/v7/scripts/export_v7.py`, `artifacts/submission/policy.py`, `code/v7/blackout_rl/v7_2/policy.py`

#### F29 · P0 · agent 순서·episode/step/reset·패키지 계약 미확정

**상태:** 현 interface 요구와 운영 loader 검증 간 공백

**원인:** DirectPolicy는 명시적 episode_id/env_step_id/team_id와 상태 reset을 요구한다. V6Policy도 FSM/cache를 가진다. 기존 감사의 upstream loader는 dict 순서와 public stateless 호출 형태에 의존한다.

**영향:** 같은 모델이라도 다른 행 순서로 유닛이 바뀌거나 경기 사이 신경/FSM 상태가 섞일 수 있다. 규정상 planner/teacher 허용도 최신 공식 답변으로 별도 확정해야 한다.

**권고 조치:** canonical slot, state life-cycle, 중복 호출, step 누락, device/패키지/파일 제한을 contract test로 고정한다.

**완료 조건:** 입력 순서 섞기, 동일 step 중복 호출, 연속 3경기, 새 경기 ID, 비정상 step에서 기대대로 작동하거나 fail closed한다.

**근거 파일:** `code/v7/blackout_rl/v7_2/policy.py`, `code/v6/blackout_rl/mappo_v6.py`, `artifacts/submission/policy.py`, `docs/issues/1st_issues_v0-v7.md`

#### F30 · P1 · 핵심 로그·가중치·그래프·Unity build가 저장소에 없음

**상태:** tracked tree와 gitignore에서 확인

**원인:** `logs/`, `artifacts/checkpoints/`, `artifacts/data/connectomes/`, `artifacts/builds/`는 이 snapshot의 tracked tree에 없다. 문서의 주요 근거는 로컬 파일/절대 경로를 참조한다. 해시 기록은 있지만 bytes를 제공하는 버전 고정 artifact bundle은 이번 접근 범위에서 확인되지 않았다.

**영향:** 새 담당자가 clone만으로 1,380경기를 재집계하거나 학습된 A2/A3의 행동을 비교하거나 Unity 통합 시험을 실행할 수 없다.

**권고 조치:** 대용량 파일을 Git에 넣는 대신 접근 가능한 immutable artifact store/release와 retrieval manifest, 라이선스·해시·run ID를 제공한다. Unity는 실행 파일뿐 아니라 관련 resources도 묶어 고정한다.

**완료 조건:** 다른 호스트에서 manifest로 필요한 artifacts를 받아 hash 검증 후 같은 평가를 재현할 수 있다.

**근거 파일:** `.gitignore`, `docs/common/history.md`, `docs/issues/1st_issues_v0-v7.md`, `code/v7/blackout_rl/v7_registry.py`

#### F31 · P2 · resume는 물리 상태의 bitwise 연속 실행이 아님

**상태:** 의도되고 기록된 동작

**원인:** checkpoint는 model/optimizer/RNG 등을 복구하지만 진행 중 Unity episode는 폐기하고 reset한다. 가속 실행은 별도 runtime overlay와 lineage를 추가한다.

**영향:** 중단 없는 run과 완전히 같은 trajectory라고 주장할 수 없으며 중단 빈도·backend가 모델 간 다르면 비교 교란이 가능하다. 이 설계 자체를 데이터 유출이나 PPO 오류로 단정하지 않는다.

**권고 조치:** 중단 시점·폐기 step/episode·backend/hash를 비교 표에 남기고 frozen 원본과 가속 lineage를 함께 보관한다.

**완료 조건:** 결과표에서 uninterrupted/resumed run이 식별되고 보고된 재현성 수준과 실제 계약이 일치한다.

**근거 파일:** `code/v7/blackout_rl/v7_training.py`, `docs/v7/reports/acceleration_and_resume.md`, `docs/v7/reports/main_study_preregistration.md`

#### F32 · P1 · 테스트 통과가 실제 경기 산출물의 정확성을 보장하지 못함

**상태:** 현재 검증 공백

**원인:** 기존 감사는 관련 62개 테스트 통과와 v7 점수 오류가 동시에 존재함을 기록한다. 테스트는 synthetic 계산/shape/ratio/상태 계약에 강점이 있지만 실제 evaluator 출력 회귀를 충분히 포괄하지 못한다. snapshot에는 .github workflow가 보이지 않는다.

**영향:** 버전별 새 evaluator/entrypoint를 추가할 때 과거 보정이 빠져도 일반 단위 테스트가 모두 통과할 수 있다.

**권고 조치:** 공통 evaluator fixture, 실제 Unity nightly contract suite, 결과 schema validator와 최소 CI를 분리 도입한다.

**완료 조건:** 현재 terminal-score 회귀를 반드시 실패시키는 테스트가 생기고 모든 공식 entrypoint가 같은 평가 계약을 통과한다.

**근거 파일:** `code/v7/tests/v7/test_contracts.py`, `code/v7/scripts/evaluate_v7.py`, `code/shared/eval/evaluator.py`, `docs/issues/1st_issues_v0-v7.md`

#### F33 · P2 · README·history·등록 문서의 상태가 어긋남

**상태:** 현재 문서 드리프트

**원인:** README는 9월21일 중단 상태인데 history/최신 감사는 9월22일 23개 본실험 완료를 기록한다. 사전 등록 순차 실행 문구와 이후 가속 오버레이도 별도 문서에 있다. 최신 PR #21은 이슈 문서 추가다.

**영향:** 중복 학습, 잘못된 재개 명령, 완료/검증/제출 상태 혼동과 이미 해결된 결함의 재보고가 발생하기 쉽다.

**권고 조치:** run manifest에서 status summary를 생성하고 계획/실행완료/성능통과/test/제출 상태를 별도 필드로 관리한다. 과거 문서는 날짜와 superseded 표시를 유지한다.

**완료 조건:** README·history·실행 summary가 같은 run IDs와 상태를 가리키며 각 열린 이슈에 근거·owner 역할·완료 조건이 있다.

**근거 파일:** `README.md`, `docs/common/history.md`, `docs/v7/reports/main_study_preregistration.md`, `docs/v7/reports/acceleration_and_resume.md`, `docs/issues/1st_issues_v0-v7.md`

#### F34 · P2 · v7 평가가 direct intervention 설정을 전달하지 않음

**상태:** 정적 코드상 잠재 결함; 현재 normal 본실험에는 영향 없음

**원인:** 학습 Collector는 cfg.controller.intervention을 DirectPolicy에 전달하지만 evaluate_v7.py는 전달하지 않아 기본 normal로 평가한다. 현재 본실험 설정은 normal이라 일치하지만 향후 sensory/output-block 조건은 달라질 수 있다.

**영향:** 같은 resolved config라도 학습과 평가에서 다른 인과성 intervention이 적용될 수 있어 ablation 결과를 오해할 위험이 있다.

**권고 조치:** 학습/평가의 DirectPolicy 생성 경로를 공통 factory로 통합하고 지원하는 config 필드가 동일하게 적용되는지 검사한다.

**완료 조건:** normal/sensory_block/frozen_frame/output_block별 train/eval policy 인스턴스가 같은 설정을 가진다.

**근거 파일:** `code/v7/blackout_rl/v7_training.py`, `code/v7/scripts/evaluate_v7.py`, `code/v7/blackout_rl/v7_2/policy.py`

#### F35 · P1 · 점수 감소에도 득점 shaping이 발생하는 upstream 경로

**상태:** upstream 결함 보고; 현재 Python score-delta 경로는 완화됨

**원인:** 기존 감사의 BlackOutAgent.cs는 score 변화량이 아니라 새 점수 > 0 조건으로 득점 reward/상대 penalty를 준다. 약탈로 점수가 줄어도 해당 handler의 보상 부호가 잘못될 수 있다.

**영향:** Unity shaping을 사용한 과거 학습과 reward-sum 기반 판정에 영향을 줄 수 있다. 별도 약탈 penalty도 있으므로 해당 이벤트의 총보상까지 반드시 양수라고 단정하지 않는다. v6/v7 Python score-delta는 감소 부호를 올바르게 처리한다.

**권고 조치:** upstream handler를 명시적인 score delta/event type 기반으로 고치고 reward 성분과 총합을 구분한다.

**완료 조건:** 증가·감소·0 도달·동시 약탈 fixture에서 raw shaping, Python reward, 경기 판정이 명시한 계약과 일치한다.

**근거 파일:** `docs/issues/1st_issues_v0-v7.md`, `code/shared/blackout_rl/reward.py`

#### F36 · P2 · 특수 아이템 비활성 선택의 근거가 약한 상대·소표본에 제한됨

**상태:** 기능 고장 증거가 아니라 일반화 검증 공백

**원인:** 기존 BASE-S15는 random 상대 10경기에서 양 조건 모두 10승이고, 특수 아이템 사용 시 점수 차 -1.7/길이 +74.1 step이라는 결과를 근거로 기본 기능을 껐다.

**영향:** 쉬운 상대에서 승률이 포화된 비교는 강한 target 상대에서 해당 전략의 가치가 없다는 근거가 되지 않는다.

**권고 조치:** 평가 점수를 먼저 고친 뒤 동일 target·paired map·충분한 seed로 특수 아이템 정책을 비교한다.

**완료 조건:** 득점·승률·아이템 기회비용과 진영별 결과가 보고되며 현재 기본 선택의 적용 범위가 명시된다.

**근거 파일:** `docs/issues/1st_issues_v0-v7.md`, `docs/pre_v1/reports/base_s15_special_item_ablation.md`

## 6. 다음 실행 순서: 새 대규모 학습 전에 통과할 게이트

### Gate A — 판정과 환경이 신뢰 가능한가

F01~F04를 먼저 처리한다. explicit terminal snapshot으로 final score, winner, termination reason의 source of truth를 정한다. 기존 평가 파일은 보존하고 수정된 스키마로 별도 버전을 만든다. 목표 점수 종료·timeout·점수 감소·동점+shaping·연속 경기 reset을 실제 고정 빌드에서 검사한다. 현재 공식 운영 runner와 로컬 판정의 차이는 공식 계약을 확인한 후 해결한다.

### Gate B — checkpoint가 실제로 planner와 다른 유용한 행동을 하는가

동일 관측과 동일 map seed에서 pure planner, 초기 checkpoint, final checkpoint, A0~A3를 비교한다. 가장 먼저 볼 지표는 loss가 아니라 p_KEEP, legal correction count, sampled/greedy action, planner action, requested action, 실제 변위다. 동일 결과가 동일 행동 때문인지, 다른 행동이 같은 결과를 낸 것인지부터 구분한다.

교정이 필요하면 gate/slot/direction을 분리하는 분포나 persistent option을 새 실험으로 정의한다. 선택 규칙만 사후에 바꾸고 기존 실험과 섞으면 안 된다. 실제 행동을 생성한 분포의 log_prob를 PPO에 사용해야 한다.

### Gate C — 전뇌가 필요한 정보와 제어 능력을 갖는가

자기 시점→상대 행동 decoder의 회전 일관성, 합법적인 내부 상태 입력, fixed output feature의 분리 능력을 검사한다. 학습된 F1에도 동적 시각/출력 차단을 적용한다. 동일 정보·슬롯·보상·budget의 작은 MLP/GRU 및 재배선 대조군과 비교한 뒤에만 더 큰 전뇌 실험 예산을 배정한다.

### Gate D — 비교가 통계적으로 유효하고 제출에서 실행 가능한가

고정 paired map, 여러 training seed, 진영별 breakdown, 사전 모델 선택 규칙과 최종 미사용 test를 적용한다. graph와 pure planner의 행동 변화도 함께 보고한다. 마지막으로 공식 loader, 허용 의존성, state reset과 전체 정책 지연을 격리 환경에서 시험한다.

## 7. 최소 관측/로그 설계

아래 필드는 제안 스키마다. 현재 레포가 모두 기록한다는 뜻이 아니다.

```text
run_id / experiment_id / commit / config_hash / runtime_overlay_hash
build_bundle_hash / opponent_hash / graph_hash / checkpoint_hash
training_seed / map_seed / physical_team / episode_id / env_step
termination_reason / authoritative_winner / authoritative_final_scores
score_source / score_units / preterminal_scores / raw_terminal_scores
planner_target / planner_action / legal_mask / mask_rejection_reason
p_KEEP / max_correction_logit / sampled_index / greedy_index
requested_action / executed_displacement / stuck / effective_override
actor_grad_norm / critic_grad_norm / KL / entropy / explained_variance
sensor_step / heading / output_pool_rates / intervention / brain_state_reset
collect_time / update_time / checkpoint_time / inference_latency / peak_memory
```

F01 같은 score 회귀는 per-episode validator로, F09/F15 같은 무개입은 behavior validator로, F22 같은 약한 인과성 검증은 학습된 정책에 대한 paired intervention으로 잡아야 한다. 단순 shape/NaN 테스트 하나로 세 문제를 모두 검증할 수는 없다.

## 8. 이번 산출물과 재현 안내

- `blackout_issue_register_2026-09-27.json`: 36개 항목의 상태, 원인, 영향, 권고, 완료 조건, 고정 commit permalink.
- `audit_sanity_checks.py`: 실제 Unity/가중치 없이 실행 가능한 네 가지 산술/인터페이스 검사.
- `audit_sanity_results.json`: 이번 검사 결과. 실제 정책 성능 재현 자료가 아님.

합성 검사는 Python 표준 라이브러리만 사용한다.

```bash
python audit_sanity_checks.py
```

실제 검증은 누락 artifacts를 가져온 뒤 별도 수행해야 한다. 기존 감사의 unittest 명령은 다음과 같으나, 이번 감사에서 이 명령을 실행했다고 주장하지 않는다.

```bash
.venv/bin/python code/run.py -m unittest \
  tests.test_env tests.test_contract tests.test_mappo_v6 \
  tests.v7.test_contracts tests.v7.test_teammates tests.v7.test_main_study
```

**종합 판단:** 새로운 모델 구조나 학습 예산을 먼저 확대하기보다, 판정 계약을 정리하고 학습된 residual의 실제 개입을 입증하며 전뇌의 감각/행동 병목을 분리해야 한다. 현재 자료는 “학습 실행을 완료했다”는 근거는 제공하지만 “planner보다 강해졌고 생물학적 배선 덕분이며 제출에서도 작동한다”는 세 주장을 아직 함께 뒷받침하지 않는다.
