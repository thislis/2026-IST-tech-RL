# v9 구현 계획 — 원본 환경을 유지하는 Attention 기반 경쟁·협동 강화학습

작성일: **2026-10-04 (KST)**. 기준 프로젝트 commit: `9ec07ef75e69373961bf87bbcf0d675d9826d364`.

이 문서는 v1~v8 코드·기존 로그 감사와 웹 연구 조사를 바탕으로 작성한 **구현 및 실험 계획**이다. 이번 작업에서는 학습·Unity 경기·새 처리량 벤치마크를 실행하지 않았다. 아래 예산·하이퍼파라미터·속도 목표는 v9에서 검증할 설계값이며 이미 달성한 결과가 아니다.

## 1. v9의 핵심 결정

1. **원본 게임, 제공 API, observation 생성·전처리 경로를 수정하지 않는다.** 보상 가공은 새 학습기 내부의 별도 값으로만 수행한다. 타이머 수정, 엔진 이벤트 추가, side channel 추가, 기존 환경 adapter의 monkey patch는 사용하지 않는다.
2. **주 모델은 CNN 없는 작은 Attention 정책으로 새로 학습한다.** 96×96×11 전체 graphic을 4×4 pixel patch로 펼치고, vector의 10개 유닛 entity와 결합한다. 작은 latent cross-attention을 사용해 연산량을 제한한다. 기존 CNN·플래너 가중치를 그대로 붙여 성능이 보존된다고 가정하지 않는다.
3. **종료·승패·self-ID 계약을 먼저 검증한다.** 현재 두 입력에는 자기 유닛 ID가 없다. Attention도 없는 정보를 만들어내지 못한다. 현 계약에서 가능한 정책과 공식 계약 확인 후 가능한 유닛별 협동 정책을 분리한다.
4. **긴 경기에서 계속 쌓이는 이벤트 보상을 주 목적에서 제거한다.** 검증된 자연 종료의 승/패/무 보상과 bounded potential shaping을 기본으로 한다. 무한 진행·오염 episode는 학습 반영 전에 격리한다.
5. **수집·전투·입구 견제·약탈·호위·흡수 직전 방어를 서로 다른 평가 과제로 만든다.** 고정 target 한 개에 대한 승률만으로 전략 다양성을 판단하지 않는다.
6. **Attention-MAPPO + 작은 상대 pool을 우선 구현한다.** 최신 world model·diffusion·대규모 league를 한꺼번에 붙이지 않는다. 각각의 도입 조건과 대조 실험을 둔다.
7. **현재 M4 Pro에서 실제 병목을 측정해 가속한다.** 여러 원본 Unity 프로세스의 병렬 수집, CPU 소배치 추론, MPS 학습 배치, 관측 중복 제거, 이진 버퍼, 제한된 로그, 학습 종료까지의 wall-clock을 함께 최적화한다.
8. **첫 정책부터 제출용 인터페이스를 사용한다.** 최종 산출물은 `policy.py`와 `checkpoint.pt`이며, 창을 띄우지 않는 백그라운드 실행·상태 확인·중단·재개까지 구현 범위에 포함한다.

## 2. 조사 범위와 증거의 강도

### 2.1 로컬 자료 전수 목록화와 분석 방법

| 범위 | 이번에 실제 수행한 확인 |
|---|---|
| `logs/`, `reports/` | 3,984개 파일, 68,428,610,345 bytes(약 63.7 GiB)를 목록화 |
| JSON | 2,292개 파싱, 파싱 오류 0개 |
| JSONL | 381개, 19,230,152개 행 전체를 순차 스캔. 349개 파일은 모든 행 파싱; 대용량 보관 로그 32개는 4,096행 간격 및 마지막 행을 파싱. 파싱한 행 총 6,446,598개에서 오류 0개 |
| 최종 v8 학습 | 6개 run의 `exposure.jsonl` **6,291,456행 전부** 재집계. `training.jsonl` 3,072개 update, 종료 기록, 제출 검증 자료 대조 |
| v7 본실험 | 최종 manager가 참조하는 23개 run의 **1,380경기** 재집계. progress 중복 파일은 경기 수에 추가하지 않음 |
| v8 두 계열 | 원본 환경 `provided_competition_v1`과 철회된 수정 환경 `accelerated_pilot_v1`의 평가 schema를 구분하여 각각 집계 |
| 코드·설정 | `blackout_rl/`, `scripts/`, `tools/`, `eval/`, `tests/`, `configs/`의 코드·설정 275개를 내용/해시/관련 기능으로 색인. 정책·collector·보상·GAE·평가기·제출·실행기 핵심 경로를 직접 검토 |
| 원본 소스 | `../blackout`과 `../blackout-env`의 종료·보상·전투·이동·관측·loader·runner 확인. 두 저장소 모두 기존 commit, clean 상태 확인 |
| 기록된 관측 | v8 `policy_inputs.pt` 6개에 저장된 48개 배치를 로드해 팀 내 관측 동일성 확인 |
| 하드웨어 | 현재 CPU·RAM·PyTorch·실제 MPS 가용성·디스크 여유를 읽기 전용 확인 |

세부 집계는 [감사 자료](../reports/v9/local_audit_2026-10-04.json)에 저장했다. 이 파일은 기존 `reports/` 정책대로 Git 추적 대상이 아니다. **대용량 보관 로그 모든 행의 의미나 모든 checkpoint tensor를 수작업으로 검증했다는 뜻은 아니다.** 기존의 전뇌·그래프 가중치 감사는 [내부 분석 결과](v8_research_requests_2026-09-27/internal_result.md) 및 동봉 JSON을 재확인했다. 텍스트의 `NaN`, `watchdog` 출현에는 테스트 이름·설명도 포함되므로 출현 횟수를 실제 장애 건수로 해석하지 않았다.

### 2.2 증거 분류

- **확인:** 현재 소스, 원본 실행 로그 또는 저장 관측에서 직접 확인했다.
- **기존 감사 확인:** 상세 수치 실험은 이전 내부 감사에서 수행됐고 이번에는 보고서·원자료·관련 소스를 대조했다.
- **가설:** 설명 가능한 기전이 있으나 실제 경기에서 인과 관계를 입증하지 못했다.
- **계약 미확정:** 제공 문서와 로컬 구현이 다르거나 필요한 정보가 없다. 임의의 규칙으로 채우지 않는다.

## 3. v1~v8 버전별 문제

버전별 숫자는 당시 환경·상대·평가 방식의 조건부 결과다. 표를 동일 대회 조건의 성능 순위로 사용하지 않는다.

| 버전 | 확인된 결과 | 문제와 v9에서의 대응 |
|---|---|---|
| **v1** | 2,000,384 env step, 학습 중 완료 경기 0개, 최종 target 0/10 | rollout 경계마다 reset하여 terminal 신호를 학습하지 못했다. 새 collector는 rollout과 episode 생명주기를 분리하고 자연 종료 수를 필수 지표로 둔다. [로그](../logs/mappo_vs_win70/run_summary.json), [이력](../history.md) |
| **v2** | 2,000,896 step, 학습 1,643경기 전패, target 0/10 | 수집은 정상화했으나 강한 상대 checkpoint의 성능은 neural actor만이 아니라 planner override에서 나왔다. 동일 checkpoint를 불러도 실행 정책이 같지 않았다. v9는 모델 hash뿐 아니라 실제 action 경로·decoder·상대 구현 hash를 고정한다. [로그](../logs/mappo_vs_win70_v2/run_summary.json), [정책](../blackout_rl/policy.py) |
| **v3** | 3,000,320 step, target 0/10, 마지막 관측 기반 점수차 −94.1 | BC/DAgger의 단일 step 정확도가 closed-loop 성능을 보장하지 않았다. replay guard label의 NoOp가 93~95%였고 이동 label 정확도는 매우 낮았다. 실패한 curriculum을 강제 승격한 것도 문제였다. v9는 행동·역할별 label 분포, 성공 궤적, 관측 충돌, teacher 없이 수행하는 과제를 검증한다. [로그](../logs/mappo_teacher_curriculum_v3/run_summary.json), [replay 감사](v8_research_requests_2026-09-27/internal_analysis/v3_replay_audit.json) |
| **v4** | 100,352 step, scripted 8/10·target 5/10, 첫 단계 중단 | KEEP warm-up BC와 꺼진 PPO 때문에 개선 신호가 없었다. v9는 첫 실제 update의 actor gradient·parameter delta·행동 변화를 함께 검사한다. [로그](../logs/mappo_planner_residual_v4/run_summary.json) |
| **v5** | 317,440 step, target 5/10, rollback 1회 | PPO override 평균 0.184%로 사실상 planner를 유지했다. 선택되지 않는 residual을 더 오래 학습하는 구조를 반복하지 않는다. [로그](../logs/mappo_planner_residual_v5/run_summary.json), [curriculum](../blackout_rl/mappo_curriculum_v5.py) |
| **v6** | 1,159,168 step, target 5/10, rollback 2회, best step=0 | 41-way 팀 residual은 한 번에 한 유닛의 한 frame만 수정한다. 탐색 증가가 성능 증가로 이어지지 않았다. 여러 유닛의 지속적 전투·방어·약탈 역할 전환에 제약이 있다. v9는 5개 행동을 모두 학습 가능한 정책과 원래 행동의 실제 효과를 계측한다. [로그](../logs/mappo_planner_residual_v6/run_summary.json), [행동 적용](../blackout_rl/mappo_v6.py) |
| **v7-1 A0/A1** | 각 5 seeds×2M; target 46/300, 65/300 | 학습 중 개입이 많아도 planner를 망가뜨릴 수 있다. 학습량·네트워크 용량만으로 개선을 판단할 수 없다. 중간 평가·가중치 부족으로 회귀 시점을 좁히기 어렵다. [manager](../logs/v7/main_study/accelerated/summary.json) |
| **v7-1 A2/A3** | 각 150/300. 10개 run의 60조건 경기 배열 동일 | 이전 수치 감사에서 최종 correction logit 상한이 전부 KEEP보다 낮았다. 배선·가중치는 달라도 greedy 실행에서 실제 개입이 0이었다. FlyWire 구조의 성능 이득을 입증하지 못했다. v9는 학습 분포와 제출 행동의 일치가 선행 조건이다. [상한·가중치 감사](v8_research_requests_2026-09-27/internal_analysis/behavior_audit.json) |
| **v7-2 F1** | 3 seeds×2M, 32/180; 한 유닛만 직접 제어 | 감각→전뇌→readout의 계산은 실행됐지만 작은 동일 조건 대조군이 없었다. 평균 entropy 약 2.19≈log(9), KL 약 1.8×10⁻⁸로 변화가 매우 작았고 일부 seed는 argmax가 편중됐다. 4 scripted teammate와 전뇌 비용도 분리 평가해야 한다. [훈련 감사](v8_research_requests_2026-09-27/internal_analysis/training_summary.json) |
| **v8 철회 계열** | 수정 환경 6×1,048,576 step. 최종 C1 26/180, flat 25/180, planner 30/60 | 사용자 요구와 다른 게임/환경 수정 경로가 포함되어 최종 경로에서 철회됐다. 이 수치·추가 엔진 outcome을 원본 환경 성능·관측으로 재사용하지 않는다. [별도 summary](../logs/v8/accelerated_pilot_v1/summary.json) |
| **v8 현재 제출 계열** | 원본 환경 6×1,048,576 step; 중간·최종·planner 합계 780경기 전패 | 제출 shape는 맞췄지만 자기 ID 없는 행별 정책, 평균 slot embedding·아군 centroid crop으로 바꾼 frozen encoder, full planner 제거, raw reward 평균 학습이 결합됐다. 초기 관측도 반복됐다. 제출 호환성과 실력은 별개다. [summary](../logs/v8/provided_competition_v1/summary.json), [실제 정책](../blackout_rl/v8/competition/policy.py) |

### 3.1 이번에 추가로 정량 확인한 v8 문제

| run | 자연 종료 transition 수 | 완료 episode 최대 길이 | 팀 5개 greedy 행동 코드 동일 비율 | 마지막 2,048 step의 대표 편중 |
|---|---:|---:|---:|---|
| C1 seed 11 | 967 | 1,880 | 96.12% | code 2: 5,725/10,240, code 8: 4,515/10,240 |
| C1 seed 22 | 966 | 1,892 | 96.64% | code 3: 8,555/10,240 |
| C1 seed 33 | 973 | 1,786 | 98.79% | code 8: 10,150/10,240 |
| flat seed 11 | 962 | 1,807 | 96.82% | code 8: 7,023/10,240 |
| flat seed 22 | 969 | 1,786 | 95.11% | code 7: 5,210/10,240, code 3: 5,030/10,240 |
| flat seed 33 | 970 | 2,182 | 96.91% | code 5: 9,980/10,240 |

총 5,807개의 종료 transition을 확인했다. `greedy`는 **확률적으로 행동한 학습 상태에서 계산한 코드**이고 실제 평가 경기의 이동 trace는 아니다. code 0은 KEEP이므로 코드 동일성이 언제나 최종 이동 벡터 동일성을 뜻하지도 않는다. 그러나 구조적인 동기화 위험을 보여주는 직접적인 진단이다.

- 저장된 6×8=48개 초기 입력 배치에서 모두 아군 5개의 `(vector, graphic)`이 완전히 같았다. 같은 입력에 같은 결정론적 함수를 적용하면 행동도 같아진다.
- 완료 episode의 학습용 팀 평균 raw return은 전체 run에서 약 **−34.88~−2.44**였다. 단일 step reward는 **−1.60~+0.60**였다. 이 기록에서 positive reward 폭주나 무한 학습 episode가 확인된 것은 아니다.
- 최종 run별 수집 처리량은 `sum(steps)/sum(collection.seconds)` 기준 **37.28~37.49 env step/s**였다. 초기화·상대 추론·로그가 포함된 수집 시간이고 PPO·전체 평가 시간을 포함한 종단 처리량은 아니다.
- 마지막 update의 팀 entropy는 약 **9.30~10.02**, 최대값 `5 log(9)≈10.99`에 가깝다. **높은 확률적 entropy와 편중된 greedy 행동은 동시에 발생할 수 있다.** entropy가 높다는 이유로 제출 행동 다양성이 충분하다고 판정하지 않는다.
- 원본 v8 평가에서 run·endpoint·side별 30 replicate의 initial observation hash는 각각 하나였다. side를 합친 hash 2개는 팀 관점 차이가 포함되어 **서로 다른 맵 2개라는 의미도 아니다**.

## 4. 전체 버전에 걸친 우선순위 문제

| ID / 우선순위 | 문제 | 확인 근거·영향 | v9 처리 |
|---|---|---|---|
| C01 / P0 | 타이머 재진입 | `TimerManager.Tick()` 완료 callback이 episode 재시작→Clear/Add를 실행한 뒤 기존 `RemoveAt(i)`가 새 timer를 삭제할 수 있음 | 엔진은 보존. fresh 원본 환경, 종료 감사, 독립 watchdog, 오염 episode commit 금지 |
| C02 / P0 | 서로 다른 승패 정의 | 게임은 점수/시간, `infos.winner`는 첫 terminal reward 부호, 제공 `run_match()`는 전 경기 누적 reward 합으로 winner 결정 | 세 지표를 서로 다른 이름으로 저장. 공식 계약 확인 전 점수 승률 인증 금지 |
| C03 / P0 | terminal 점수 reset | v7 1,380경기 중 1,211개가 0:0. 마지막 nonterminal 점수에도 마지막 득점은 누락 가능 | `last_visible_score`, `terminal_observed_score`, `outcome_source`를 구분. final score 임의 복원 금지 |
| C04 / P0 | 자기 유닛 식별 불가 | 96-vector에서 raw `unitIndex`가 제거됨. loader도 ID를 `forward`에 전달하지 않음 | §7의 strict 정책 및 식별 계약 gate. batch row 번호를 몰래 자기 ID로 사용하지 않음 |
| C05 / P0 | 학습과 실제 실행 불일치 | v2 neural/planner 차이, v7 sample/greedy 차이, v8 slot/centroid adaptation | 동일 제출 모듈로 rollout·평가. decoder·log probability·mask parity 검사 |
| C06 / P0 | 누적 shaping과 목적 불일치 | raw pickup/deposit/kill/absorb 보상이 최종 승리보다 커질 수 있음. v8는 raw 팀 평균을 그대로 사용 | 승리 기반 bounded objective, raw reward는 별도 보존. 반복 획득·교환·kill loop 회귀 검사 |
| C07 / P1 | 점수 감소도 양의 handler 호출 | `score > 0`만 검사하므로 예: 10→5에도 자기팀 +0.1 handler 실행. 다른 패널티가 있어 순보상이 반드시 양수는 아님 | nonterminal score 변화 직접 계산, 원본 reward 수정 금지 |
| C08 / P1 | 장기 credit 짧음 | 50 Hz 가정에서 v8 `γ=.9995, λ=.99`의 직접 GAE trace 길이 약 95 step=1.9초, 최대 경기는 약 420초 | 초 단위 horizon 기록, terminal을 포함하는 trajectory, γ/λ 대조. Attention을 장기 credit의 대체물로 보지 않음 |
| C09 / P1 | seed 요청과 적용 혼동 | reset/cache/side-channel 순서 문제. 원본 v8 초기 상태 반복 | requested seed와 initial hash·실제 변화 검증 분리. 순수 반복을 독립 맵 표본으로 세지 않음 |
| C10 / P1 | 관측 누락·가림 | 배터리 stack 수량, 적 class, self-ID 누락. renderer가 tile→item→unit 단일 ID로 덮어씀 | count/속도/class를 사실처럼 만들지 않음. 관측 가능/추정/미상 feature 구분 |
| C11 / P1 | semantic 검사의 한계 | 보간으로 다른 정수 ID가 되어도 one-hot 검사는 통과 가능. Xvfb 문제는 로컬 macOS에서 미재현 | 플랫폼별 경계·가림 fixture, 위치 대응 검사. Python에서 임의로 맵을 보정하지 않음 |
| C12 / P1 | 경기 수와 데이터 노출 비중 다름 | B 60% episode 배정이 B 60% step을 보장하지 않음. v7 F1 일부 run은 B step 약 72% | side/opponent별 episode·env step·update 기여량 모두 표시 |
| C13 / P1 | frozen encoder·teacher에 대한 과신 | 기존 표현이 새 실행 조건·전략에 충분한지 미검증. v3 label 불균형·관측 충돌 | 작은 Attention encoder를 end-to-end 학습, teacher는 선택적 분포 모방으로 제한 |
| C14 / P1 | 1-frame·1-slot residual의 전략 제약 | 역할 전환·동시 약탈/호위와 같은 장기 팀 행동을 직접 표현하기 어려움 | 5개 유닛 직접 정책. 시간적 option은 식별/reset 계약이 있을 때만 추가 |
| C15 / P1 | 고정 상대 과적합 | scripted/weak/target 혼합도 사실상 유사 planner family. 상대 다양성·상성 matrix 부족 | 수집형·rush·견제·약탈·수비 상대 및 과거 snapshot pool |
| C16 / P1 | 통계적 과신 | 10경기, 동일 맵 반복, 0승 bootstrap [0,0], 여러 seed 중 best 선택 | 독립 train seed·환경 조건 단위 CI, frozen test, 0승의 불확실성 명시 |
| C17 / P1 | 보호 경로가 이전 구간 오염을 취소하지 못함 | watchdog 발생 전 rollout이 이미 update에 사용되면 마지막 예외만으로 회수 불가 | 기본은 자연 종료 검증 후 episode commit. fast streaming은 별도 rollback 설계 없으면 금지 |
| C18 / P2 | 로그·저장 비용 | 철회 v8의 `window_rows.jsonl` 6개만 약 51.47 GiB. 중복 관측·JSON 실수 직렬화 비용 | bounded binary ring, 한 번 저장한 map 참조, per-step JSON 중단 |
| C19 / P2 | 환경·통신과 모델 비용 혼동 | v7 인코더는 CPU 1.73ms, MPS 3.89ms였음. 통신 변경으로 크게 가속한 이력 | 구성 요소별 p50/p95/p99 및 end-to-end 측정; 원본 API transport 변경은 재도입하지 않음 |
| C20 / P2 | 운영·배포 결함 | `blackout_rl` import 실패, executable bit 누락, 백그라운드 manager와 Unity GUI의 혼동 | 독립 import 검사, 실행 권한, detached session 및 실제 창 없음 검사 |

관련 기존 문서: [1차 이슈](../issues/1st_issues_v0-v7.md), [내부 원인 분석](v8_research_requests_2026-09-27/internal_result.md), [game spec](../game_spec.md). 과거 문서가 권고한 Unity 수정이나 `ContractBlackOutEnv` 사용은 **현재 사용자의 원본 보존 요구보다 우선하지 않는다**.

## 5. 피드백을 현재 게임 규칙에 맞게 해석

### 5.1 게임이 끝나지 않음과 보상 과다

다음 세 경우를 구별해야 한다.

1. **정상적으로 긴 경기:** 양 팀이 100점에 도달하지 못해 420초까지 진행. 보유 배터리는 사망/변신으로 파괴될 수 있어 두 팀 모두 100점을 못 채우는 상태가 가능하다. 이것만으로 게임 버그라고 부르지 않는다.
2. **종료 실패:** 타이머 재진입·진행 정지·통신 실패 등으로 원래 종료가 오지 않음. 소스 결함은 확인했지만 모든 과거 실험에서 무한 경기가 발생했다고 할 수 없다. v7 기록의 최대 21,003 step 및 원본 v8의 짧은 완료 경기들은 이를 구별할 근거다.
3. **보상 목적 오류:** 제한 시간 안에서도 pickup/deposit/반복 kill/흡수 shaping을 많이 받는 것이 승리와 다를 수 있음. 무한 경기 여부와 독립적으로 고쳐야 한다.

원본 Unity의 `reward_config.json`, C# `AddReward`, 종료 규칙은 수정하지 않는다. 새 learner의 `raw_reward`와 `train_reward`를 분리하고 §8의 보상을 사용한다. watchdog은 실패를 보고하는 장치이지 임의 승리·무승부·학습 terminal을 만드는 장치가 아니다.

### 5.2 올킬·입구 막기·약탈의 실제 의미

- **올킬:** 현재 [MatchManager](../../blackout/Assets/Project/Runtime/Scripts/MatchManager.cs)는 100점/시간만 종료 조건으로 쓰고 사망 즉시 Collector로 부활시킨다. 따라서 “5명이 동시에 죽어 경기 종료”가 아니라 **짧은 시간 동안 서로 다른 적 5명을 처치/귀환시켜 수집을 방해하는 전술**로 정의한다. 숨은 kill ID가 없으므로 관측 기반 탐지에는 추정 표시를 붙인다.
- **입구 막기:** [UnitMovementSystem](../../blackout/Assets/Project/Runtime/Scripts/Object/UnitMovementSystem.cs)는 지형 충돌을, [UnitInteractionSystem](../../blackout/Assets/Project/Runtime/Scripts/Object/UnitInteractionSystem.cs)는 유닛 overlap 전투를 처리한다. 유닛을 벽처럼 취급하는 물리적 body blocking은 확인되지 않았다. 목표는 **Hunter 접촉 위협·진로 차단을 통한 통과 억제**다. 아군 Collector를 입구에 세워두기만 하면 물리적으로 막힌다는 전제로 보상하지 않는다.
- **배터리 약탈:** 적 공개 창고의 배터리를 가져오면 원래 팀 점수가 줄고, 자기 창고 적재 시 자기 점수가 오른다. 보호 본진에는 적이 들어갈 수 없다. 20초 흡수는 추가 득점이 아니라 약탈 불가능한 확정이다. stack 수량을 graphic 면적으로 추정하지 않는다.
- **다양한 전략:** 특정 전술을 모든 경기에서 강제하지 않는다. 점수·남은 시간·운반 상태·적 배치에 따라 수집, 교전, 수비, 약탈, 호위가 승리에 기여하는지를 평가한다.

### 5.3 Attention 전환의 목표

Attention을 사용하는 방향은 채택한다. 다만 **CNN이면 느리고 Attention이면 빠르다는 명제는 검증되지 않았다.** Hide-and-Seek 연구에서도 self-attention이 sample efficiency를 높였지만 연산량 때문에 wall-clock 수렴은 느려진 사례를 보고했다. [원문 Appendix A.3](https://arxiv.org/html/1909.07528v2)

v9의 속도 개선 근거는 작은 token/latent 수, 팀 공통 입력 재사용, 배치화, 유효 trajectory당 비용 절감이어야 한다. 입력 pixel 9,216개에 full self-attention을 직접 적용하는 대형 ViT는 기본안에서 제외한다.

## 6. 관련 연구 조사와 선택

검색 기준일은 2026-10-04다. 논문 원문·학회 proceedings·저자/프로젝트 공식 자료를 우선 확인했다. 검색에 노출된 LLM self-play 연구는 실시간 연속 이동 게임과 조건이 달라 핵심 근거에서 제외했다. 오래된 직접 관련 성공 사례와 2025~2026 연구를 함께 검토하며, 새 논문이라는 이유만으로 BlackOut에서 우월하다고 가정하지 않는다.

### 6.1 유사 게임에서 검증된 방법

| 연구 | 방법·reward 설계·개선 원리 | v9에 적용할 부분 / 제한 |
|---|---|---|
| **FTW, Science 2019 — Capture the Flag** | population 간 대전, 내부 이벤트 reward를 승리 기준으로 상위 최적화, 빠른/느린 두 시간축 기억. flag 획득·상대 tagging 등 이벤트 가중치가 학습 과정에서 변함 | 수집·공격·수비가 함께 필요한 가장 가까운 사례. 승리를 상위 목적, 전술을 보조 신호로 분리한다. 대규모 PBT와 recurrent memory를 현 계약에 그대로 복제하지 않는다. [공식 연구 설명](https://deepmind.google/blog/capture-the-flag-the-emergence-of-complex-cooperative-agents/) |
| **OpenAI Five, 2019 — Dota 2** | PPO/self-play, 승리 외 자원·전투 shaping, 상대 보상을 빼는 대칭화, 개인 보상을 팀 평균과 섞는 team spirit, 긴 credit horizon | 이벤트 보상과 최종 목적의 정렬, 팀 평균 중복 계산 방지, 학습/추론 정책 일치를 가져온다. 원 논문도 self identity를 가진다. BlackOut의 ID 누락을 해결해주지는 않는다. [원문 §3, Appendix G/O](https://arxiv.org/html/1912.06680v1) |
| **AlphaStar, Nature 2019** | 인간 replay 초기화, league와 exploiters, PFSP로 약점 상대를 선택. 단일 상대를 이기는 순환적 과적합 방지 | 작은 역사 상대 pool·상성 matrix·최악 상대 성능. 인간 데이터·대규모 계산·전용 state API는 전제하지 않는다. [논문](https://www.nature.com/articles/s41586-019-1724-z) |
| **Hide-and-Seek, ICLR 2020** | entity self-attention, PPO, 단순 팀 경쟁 목적에서 상대와의 공진화로 방어·도구 사용 전략 출현 | 다양한 전략이 반드시 각각 큰 행동 보상에서 나오는 것은 아니다. 상대 다양성과 관측 관계 학습이 핵심. 원 게임의 도구·환경 수정은 가져오지 않는다. [논문](https://arxiv.org/abs/1909.07528) |
| **MAPPO, NeurIPS 2022** | centralized critic, value normalization, PPO clipping·epoch·batch 관리로 SMAC/Football 등에서 강한 기준선 | 기존 프로젝트 경험을 활용할 가장 작은 알고리즘 변경. CTDE의 입력은 허용된 관측만 사용한다. cooperative benchmark 성공을 경쟁 게임 승리 보장으로 해석하지 않는다. [논문 §5](https://arxiv.org/html/2103.01955v4) |
| **UPDeT, ICLR 2021 / MAT, NeurIPS 2022** | entity/agent 관계를 Transformer로 표현. MAT는 agent action sequence를 autoregressive하게 생성하고 online policy optimization 사용 | 작은 Attention 인코더를 채택한다. MAT의 팀 전체 순서·식별 전제가 현재 제출 계약과 맞는지 별도 gate. 임의 B축을 한 팀으로 간주하지 않는다. [UPDeT](https://arxiv.org/abs/2101.08001), [MAT](https://arxiv.org/abs/2205.14953) |
| **TiZero, AAMAS 2023 — Football 11v11** | 공동 정책 최적화, curriculum, self-play. goal ±1, 소유 +0.0001/step, 득점 전 성공 pass +0.05, 밀집/영역 이탈 −0.001의 팀 보상 및 대칭화 | goal 이전 협동 행동 평가와 상대 curriculum을 참고. BlackOut에서는 소유/생존의 매 step 양의 보상을 그대로 도입하지 않는다. ID conditioning·초기 random rollout은 제출 계약·step 회계 검증 없이는 사용하지 않는다. [원문 §4](https://arxiv.org/html/2302.07515v2) |
| **RODE, ICLR 2021** | 행동 효과로 역할을 묶고 낮은 시간 해상도의 role selector와 역할별 행동 정책을 학습 | 수집/전투/호위 역할의 표현과 평가. 기본은 stateless mixture; 유지되는 option은 reset·self-ID가 보장된 경우에만 허용. [논문](https://arxiv.org/abs/2010.01523) |
| **MADRID, 2024** | 다양한 adversarial 상황을 만들어 강한 Football 정책의 전략적 약점을 탐색 | 평균 승률 외 약탈 대응·입구 견제·상대 rush의 최악 조건을 측정. 엔진 state 삽입 없이 정상 플레이로 도달한 상황만 사용한다. [논문](https://arxiv.org/abs/2401.13460) |

### 6.2 2025~2026 연구에서 가져올 것과 보류할 것

| 연구·확인 상태 | 핵심 | v9 결정 |
|---|---|---|
| **Sable, ICML 2025** — proceedings/초록 확인 | retention으로 긴 multi-agent sequence를 효율적으로 처리. 많은 agent에서 메모리 증가를 억제 | 작은 공간 Attention은 즉시 적용. 시간 memory는 actor state/reset 계약 확인 후 비교한다. 5-agent 소배치가 자동으로 빨라진다는 근거는 아니다. [학회 원문](https://proceedings.mlr.press/v267/mahjoub25a.html) |
| **MARIE, 2024 초고·2025-09 개정, TMLR accepted 표기** — 본문 확인 | decentralized Transformer world model + centralized Perceiver, 관측·reward·discount 예측과 imagination으로 환경 sample 절감 | Unity 비용이 병목이고 신뢰할 trajectory가 충분해진 뒤의 확장안. 오류가 있는 종료·reward를 그대로 world model에 학습하면 오류도 복제한다. 최초 v9 범위에서는 제외. OpenReview 검색 결과의 ICLR 표기보다 현재 arXiv metadata를 기준으로 기록. [v2](https://arxiv.org/html/2406.15836v2) |
| **Reidentify/CAID, ICML 2025** — proceedings/초록 확인 | causal Transformer로 context-dependent identity를 표현해 다른 팀 구성에 일반화 | identity가 왜 중요한지 참고. 동일 관측만으로 실제 제어 유닛을 알 수 있다는 결과는 아니다. self-ID 계약 대체용으로 사용하지 않는다. [논문](https://proceedings.mlr.press/v267/xu25j.html) |
| **HPS, ICML 2026** — proceedings/초록 확인 | 공유 parameter의 방향/scale gradient 충돌 분리, agent별 scale modulation | 역할·클래스별 gradient 충돌을 측정한 후 선택적 실험. 현재 입력으로 구별 불가능한 유닛을 임의 ID로 나누지 않는다. [논문](https://proceedings.mlr.press/v306/fu26k.html) |
| **Dspic, ICML 2026** — proceedings/초록 확인 | role embedding과 maximum entropy, diffusion policy로 heterogeneous/multimodal 행동 표현 | 공통 정책이 역할 다양성을 잃는 문제의 최신 참고. diffusion 여러 denoising step은 현재 속도 목표와 비용이 맞지 않아 먼저 작은 mixture head로 검증한다. [논문](https://proceedings.mlr.press/v306/li26it.html) |
| **Adaptive TD-λ, 2026-05 preprint** — 초록·본문 확인 | 두 replay 분포의 density ratio 추정으로 state-action별 λ를 조절; QMIX/MAPPO와 Football academy 실험 | 고정 λ 대조 후 장기 credit가 실제 병목일 때만 추가. estimator/replay 비용과 PPO 데이터 조건을 재검증. 확립된 기본값으로 취급하지 않는다. [논문](https://arxiv.org/abs/2605.11880) |
| **Territory Paint Wars, 2026-04 preprint** — 본문 확인 | Unity 경쟁 PPO에서 누적량 보상 폭주, terminal/승리 검출 오류, self-play 과적합을 분석. 누적 lock 보상을 새 lock의 증가분으로 바꾸고 고정 상대를 혼합 | 우리 문제와 가까운 실패 사례지만 소규모 2-agent 환경·일부 단일 seed ablation이다. 논문의 GAE 설명을 장기 credit가 자동 해결된다는 증거로 쓰지 않는다. 고정 상대 평가와 이벤트 중복 검사를 채택. [논문](https://arxiv.org/html/2604.04983v1) |
| **PufferLib 2.0, RLC 2025** — 학회 자료 확인 | 빠른 vectorization과 simulation 데이터 경로 개선 | 공유 버퍼·작은 Python overhead·배치 inference 설계 참고. native 환경의 1M step/s를 원본 Unity/Mac의 예상 속도로 제시하지 않는다. 제공 환경을 PufferLib용으로 다시 쓰지 않는다. [학회 자료](https://rlj.cs.umass.edu/2025/papers/Paper151.html) |

보상 안정화는 [potential-based shaping 원 논문](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf), [stochastic game 확장](https://arxiv.org/abs/1401.3907), [PopArt](https://arxiv.org/abs/1602.07714)를 참고한다. 평가 불확실성은 [Statistical Precipice/rliable](https://arxiv.org/abs/2108.13264)를 따른다. 이 문서의 BlackOut용 수치·과제·예산은 논문에서 가져온 정답이 아니라 **프로젝트에 맞춘 설계 제안**이다.

## 7. observation·제출 계약과 Attention 정책

### 7.1 수정 금지 범위와 허용되는 새 코드

| 보존 | v9에서 새로 작성/변경 가능 |
|---|---|
| `../blackout`, `../blackout-env`, 설치된 `blackout_env`/`mlagents_envs`, `artifacts/builds/BlackOut.app`, reward config, semantic renderer·전처리 | 새 모델, 모델 내부 feature/token 계산, PPO/BC, 학습용 reward transform, 평가 기록, 외부 supervisor·exec launcher |
| reset/step 순서·반환 의미, 원본 observation 96 및 11×96×96 | 반환값을 읽고 복사하여 버퍼에 저장, 학습 입력으로 가공, 원본과 별도인 감사 기록 |
| 제공 승패 runner | 제공 runner 결과를 그대로 저장하는 호환 평가 및 별도로 명명한 연구용 지표 |

현재 고정 원본: Unity commit `d2220a7d01be88d413f551efd529f4758833be8b`, API commit `6ba7d9993cf1bdefe1ed480c8efbcabcb923f539`, player SHA-256 `49172f88b1ba2429677e7ec4a7876c4589876090be4888aeaef39269e29f4a69`. 새 배포를 공식적으로 받으면 새 manifest/실험 ID로 구분한다.

### 7.2 self-ID 문제는 모델 구조보다 먼저 해결해야 함

[제출 문서](../blackout_last_4_pages.md)의 계약은 다음뿐이다.

```python
class MyPolicy(nn.Module):
    def __init__(self, vector_size=96, n_channels=11): ...
    def forward(self, vector, graphic):
        # vector:  float32[B, 96]
        # graphic: float32[B, 11, 96, 96]
        # result:  float32[B, 2], each coordinate in [-1, 1]
        ...
```

원본 `model/loader.py`는 `agents=list(obs.keys())`로 stack하고 반환 action을 같은 순서의 이름에 연결한다. 이름을 모델에 전달하지 않는다. 과거 raw 행 순서도 A팀 `3,2,4,1,0`, B팀 `6,7,5,8,9`로 canonical slot 순서와 달랐다. vector의 10개 block 순서는 알 수 있지만 **현재 row가 그중 누구를 제어하는지는 알 수 없다**.

따라서 `o_i=o_j`이면 stateless deterministic `f(o_i)=f(o_j)`다. B축 self-attention이나 context-generated identity를 넣어도 permutation-equivariant 결정론적 모델은 동일 row를 구분하지 못한다. 다음 두 경로를 명시한다.

| 경로 | 현 상태에서 가능한 구현 | 성능 주장 한계 |
|---|---|---|
| **V9-S: 엄격한 두 입력 정책 — 기본 구현** | row 내부의 map/entity Attention, 자기 class와 전체 팀 상황 기반 공통 행동 분포. ID·숨은 history·고정 row 매핑 없음. 로컬 PPO·평가·제출 모듈은 같은 categorical sampling 사용 | 클래스가 같으면 분포가 같지만 sample은 달라질 수 있음. 개별 위치를 아는 5개 역할 할당·정밀 호위까지 구현됐다고 주장하지 않음 |
| **V9-S greedy 대조** | 같은 학습 weight에서 argmax decoder를 명시적으로 선택한 별도 평가·export | 다른 실행 정책이다. sample 결과를 greedy의 성능으로 쓰지 않으며, 별도 경기 검증 없이 최종 제출 decoder를 바꾸지 않음 |
| **V9-I: 식별 가능한 협동 정책 — 조건부 확장** | 주최 측이 허용·보장한 self 식별/팀 batch 계약이 제공된 경우 self query·5역할·필요 시 memory 추가 | 우리 쪽에서 관측 차원을 늘리거나 loader를 patch하여 조건을 충족한 척하지 않음. 공식 제공 방식과 두 파일 loader parity를 확인한 뒤 별도 실험 |

현 계약으로도 V9-S 모델·보상·collector·속도 측정·제출 검사는 구현 가능하다. **유닛별 다전략 협동을 완성했다고 판정하는 gate는 self-ID 해결 전 통과시킬 수 없다.** 문서에 없는 reset callback, batch=5의 영구 보장, 이전 action 입력을 요구하는 recurrent actor는 기본안에서 제외한다.

제공 문서에는 확률적 `forward` 금지 규정이 없다. 따라서 로컬 구현·연구는 sampling으로 진행할 수 있으며, 확인되지 않은 규정을 이유로 학습을 막지 않는다. 다만 공식 서버의 RNG·재현성 제한은 제출 전 확인 항목으로 남긴다. sampling은 동일 관측의 행동 대칭을 깨는 방법이며 실제 self 위치를 복구하는 방법은 아니다.

### 7.3 CNN 없는 모델 설계

주 모델 이름은 `SemanticEntityAttentionPolicy`로 한다. 구조는 [Perceiver의 latent bottleneck](https://arxiv.org/abs/2103.03206)에서 착안하되 BlackOut용 소형 모델로 작성한다.

| 단계 | 초기 설계 |
|---|---|
| graphic tokenization | `(B,11,96,96)`을 reshape/permute로 **576개 4×4 pixel patch**, patch당 176값으로 펼침. 보간·평균 downsample 없이 모든 채널과 타일 내부 위치를 입력에 유지 |
| map embedding | patch별 `Linear(176,64)` + 2D 좌표/type embedding. Conv1d/Conv2d 없음 |
| entity embedding | vector 첫 90값을 10×9로 분리, 공유 MLP→64. team sign, 절대 위치, holding type 사용. 없는 stack·적 class·self flag를 생성하지 않음 |
| context | self class 3, 두 점수, time_left 3을 별도 context token으로 표현. 흡수 phase는 420초/20초 계약이 검증된 경우에만 time_left에서 파생 |
| attention | 입력 587 tokens → 32개 latent cross-attention, latent self-attention 2층, width 64, heads 4, FFN 128, pre-LayerNorm, dropout=0 |
| actor readout | self-class/context query로 latent를 읽고 9-action logits 출력. 기본 1개 head; 전략 대조군은 §9의 작은 4~6 expert mixture |
| critic | 허용된 아군 관측을 집계하는 Attention/MLP value head. 상대 측 self-class 등 추가 정보를 학습 actor에 흘리지 않음. 기본은 상대 관측을 critic에도 추가하지 않음 |
| 출력 | 학습 내부 9개 행동(정지+8방향)을 선택하여 `(dx,dy)`로 변환. Unity가 nonzero 방향을 normalize하므로 연속 magnitude를 속도 제어로 학습하지 않음 |

계산 규모: 9,216 pixel full attention은 head당 약 8,493만 attention pair지만, 위 cross-attention은 `32×587=18,784` pair이고 latent self-attention은 층당 `32²=1,024` pair다. 이는 attention score 행렬 크기 비교이지 실제 wall-clock 배속 보장은 아니다. projection·전처리·메모리·상대 추론 비용을 포함해 측정한다.

- 모든 11 graphic 채널을 사용한다. 적은 token을 위해 벽/창고/아이템 채널을 삭제하거나 처음부터 `argmax` 맵만 actor 입력으로 강제하지 않는다.
- 작은 unit/item 위치도 4×4 patch 내부 패턴으로 유지한다. 96→24의 단순 평균 축소와 비교할 수 있으나 기본안에서는 하지 않는다.
- Attention은 **각 row 안의 token 축**에 적용한다. 임의의 inference batch B를 한 팀으로 연결하지 않는다.
- v8의 `vector[:49]` 부분 추가 입력, 평균 slot embedding, centroid local crop을 가져오지 않는다. vector 96값 전체를 사용한다.
- end-to-end encoder를 학습하므로 v8처럼 rollout의 frozen latent만 저장해 PPO를 수행하면 안 된다. 원 관측의 lossless 압축본을 보관하고 update마다 현재 encoder로 재계산한다.
- game position으로 만드는 행동 mask는 self identity가 없으면 적용하지 않는다. “막힌 것 같음”만으로 행동을 영구 제거하지 않는다.

## 8. 종료·보상·trajectory 안전성

### 8.1 먼저 objective profile을 고정

제출 문서의 “reward는 학습용”과 로컬 `run_match()`의 누적 reward 승패 판정이 충돌한다. v9는 이를 임의로 하나로 합치지 않는다.

| profile | terminal 기본 보상 `z` | 사용 범위 |
|---|---|---|
| `provided_runner_v1` | 원본 자연 종료 후 `sign(sum(raw_reward_own) - sum(raw_reward_opp))` | 현재 로컬 runner와 정확히 맞는 **임시 기본 연구 목적**. raw 합 자체를 dense reward로 학습하지 않음 |
| `verified_game_outcome` | 공식적으로 보장된 게임 winner의 승 +1 / 패 −1 / 무 0 | 주최 측 제공 outcome 계약 또는 수정 없이 얻는 신뢰 가능한 공개 결과가 확보된 경우에만 활성화 |
| `wrapper_terminal_proxy` | 현재 `infos.winner` | 감사용 지표. true winner로 인증하지 않고 주 학습 objective로 자동 승격하지 않음 |

`provided_runner_v1`은 수치 폭주를 막아도 runner가 좋아하는 reward-farming 전략과 진짜 점수 승리의 불일치까지 해결하지는 못한다. 이 profile에서 성능이 올라가도 “게임 규칙 기준 승리 개선”으로 쓰지 않는다. 최종 대회 제출 성능 승인에는 실제 server objective 확인이 필요하다.

### 8.2 기본 보상 공식

완료된 유효 episode 길이를 T, 현재 허용 관측을 `o_t`라고 할 때:

```text
r_base_t = 0                         (자연 종료 전)
r_base_terminal = z ∈ {-1, 0, +1}   (검증한 objective profile, 정확히 1회)

Φ(o) = 0.25 × clip(0.7 × score_gap(o) + 0.3 × progress(o), -1, +1)
Φ(terminal) = 0
F_t = γ Φ(o_(t+1)) - Φ(o_t)
r_train_t = r_base_t + F_t
```

- `score_gap = own_score_norm - opp_score_norm`; terminal/reset 점수로 delta를 만들지 않는다.
- 첫 reward 대조에서는 `progress=0`으로 시작한다. 이후 관측 기반 team-to-visible-target 거리와 배달 가능성의 bounded potential을 추가한다. 벽 거리 계산은 학습 reward에만 쓸 수 있으며 full planner의 행동 override는 하지 않는다.
- progress는 배터리 **종류·존재·경로 거리**의 proxy다. 숨은 stack 수량·적 class·kill event의 정답을 가정하지 않는다. 목표 선택 함수와 가중치는 episode 도중 변경하지 않는다.
- 초기 주 설정은 **γ=1.0, 유한 시간 episode, time_left 입력**이다. 이때 완결 episode shaping 합은 `−Φ(o_0)`이고 return은 절댓값 1.25 이내다. 반복 왕복·대기·약탈 교환을 길게 이어도 같은 초기 상태에서 추가 terminal reward를 벌 수 없다.
- `γ=.99995`는 별도 대조로만 둔다. γ<1에서 terminal 시점 선호가 생기므로 무할인 승률 목적과 같다고 하지 않는다. `F_t`와 GAE에 같은 γ를 사용한다.
- terminal 점수가 reset되어도 `Φ(terminal)=0`으로 처리할 수 있다. 마지막 nonterminal 점수를 final score라고 가정해 terminal 보너스를 만들어내지 않는다.
- **raw reward, clipped raw reward, kill 보너스를 이 식에 다시 더하지 않는다.** per-step clipping, 양의 shaping만 남기기, episode 내부 weight 변경은 telescoping 성질을 깨므로 하지 않는다.
- 정상화를 추가한다면 critic의 ValueNorm/PopArt와 advantage normalization에서 처리한다. reward를 크게 자른 뒤 목적이 같다고 주장하지 않는다. raw return은 비교·감사를 위해 그대로 별도 저장한다.

PBRS의 telescoping은 동일한 γ·potential·boundary 처리에서 성립한다. 부분관측, 함수 근사, 변하는 상대가 있는 MARL에서 이 성질을 곧바로 수렴·성능 보장으로 확대하지 않는다. [PBRS](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf), [게임 확장](https://arxiv.org/abs/1401.3907)

### 8.3 반복 행동 보상은 우선 reward가 아닌 진단·보조학습으로 사용

| 행동 | 기본 reward 처리 | 별도 지표/보조학습 |
|---|---|---|
| 배터리 pickup/deposit | raw 이벤트 보너스 재사용 안 함. score/거리 potential | 관측 holding 변화, 점수 변화, delivery latency |
| 적 처치 | 무제한 +kill 없음 | 고신뢰 respawn proxy·운반 손실 proxy. 실제 kill 정답 없으면 masked label |
| 올킬성 rush | 별도 큰 bonus 없음 | 10 game-sec 내 서로 다른 적 5개 respawn 추정과 그 뒤의 점수 우위; 확정/추정 구분 |
| 입구 지키기 | 위치 점유·대기 매 step 양의 reward 없음 | 상대 통과/배달 감소, 아군 수집 유지, 과도한 진지 고착 여부 |
| 약탈 | 상대 점수 감소 포함 potential, 추가로 양의 감소분만 취하지 않음 | 공개 창고 출발·holding 변화·자기 적재 연결. 왕복 교환과 실제 순이득 구별 |
| 생존/진행 | 무조건 생존 보너스 없음 | 사망 proxy, stuck, 거리 감소, class 전환 |

기술 warm-up에서 별도 dense 이벤트 보상을 실험하려면 독립 ablation으로 표시하고, event 중복 제거·episode당 절댓값 예산·최종 승리와의 상관을 검증한다. 이 변형에는 PBRS의 정책 불변성을 주장하지 않는다.

### 8.4 정상 종료·truncation·고장 분리

| 경계 | bootstrap / GAE | outcome 처리 |
|---|---|---|
| 제공 환경의 정상 게임 종료 | next value=0, trace 종료 | profile에 따른 terminal 보상 1회 |
| 게임 규칙의 420초 정상 timeout | finite-horizon 게임의 termination으로 처리 | 점수/runner의 해당 계약에 따른 결과; 단순 truncation으로 바꾸지 않음 |
| 학습 rollout 저장 chunk 경계 | 같은 episode 연속성을 유지. next value는 실제 다음 관측 | reset하지 않음 |
| 명시적 정상 수집 절단을 채택한 별도 실험 | 실제 final 관측으로 bootstrap, reset을 가로지르는 trace 차단 | 임의 승/패 생성 금지 |
| timer 소실·통신 정지·불완전 terminal·비정상 관측 | **고장**으로 격리. stale 관측/자동 reset 관측으로 bootstrap하지 않음 | invalid, 승/패/무 분모에 섞지 않으며 invalid rate 별도 보고 |

일반적인 termination/truncation 구분과 게임 내 시간 제한은 [Farama 공식 설명](https://farama.org/Gymnasium-Terminated-Truncated-Step-API)을 따른다.

### 8.5 오염이 update에 들어가지 않게 하는 수집 계약

1. 새 원본 환경 인스턴스를 생성해 episode 하나를 진행한다. reset 내부 cache/seed 전달을 수정하지 않는다.
2. 기본 collector는 **완전 episode를 임시 이진 버퍼에 staging**한다. 중간 2,048-step chunk는 저장 단위일 뿐 PPO commit 단위가 아니다.
3. 한 수집 wave에서는 행동 policy version을 고정한다. 여러 환경의 각 episode가 자연 종료·감사를 통과한 후 해당 wave의 유효 episode로 PPO를 수행한다. 긴 episode가 남은 동안 짧은 게임만 새 version으로 바꿔 섞지 않는다.
4. episode step 상한은 기존 계약에 맞춘 **22,000**을 초기 안전값으로 두고, time_left 진행·벽시계 heartbeat·10-agent termination 동시성을 함께 검사한다. 게임시간을 time_scale×wall-clock으로 대신하지 않는다.
5. wall watchdog은 blocking `env.step()` 바깥의 supervisor가 소유 PID를 종료할 수 있어야 한다. 최초 runtime 값은 정상 20경기에서 보수적으로 정하고 v8의 1,800초 stall 값을 그대로 최적값이라고 가정하지 않는다.
6. invalid episode의 모든 chunk는 학습에 반영하지 않고 종료 원인과 마지막 관측을 보존한다. 재시도는 새로운 attempt ID로 기록하며 성공한 시도만 남겨 실패율을 숨기지 않는다.
7. invalid rate가 1%를 넘거나 동일 종료 결함이 연속 3회 발생하면 wave 이후 학습을 중단한다. 문제 있는 상황을 전부 제외해 쉬운 경기만 학습하는 selection bias도 경고한다.
8. resume은 원본 Unity 물리 snapshot 복구를 주장하지 않는다. 저장된 모델·optimizer·RNG·정규화 상태를 복원하고 미완료 episode는 폐기·회계 후 새로 시작한다.

완전 episode commit은 업데이트 지연을 늘리므로 §12에서 비용을 측정한다. streaming PPO를 나중에 도입하려면 고장 탐지 이후 **그 episode prefix로 갱신된 모든 모델·optimizer·normalizer를 되돌릴 수 있는** 설계가 필요하다. 마지막 rollout만 버리는 방식은 충분하지 않다.

## 9. 전략 다양성과 curriculum

### 9.1 기본 정책과 전략 표현

1차 기준선은 single-head Attention 정책이다. 그다음 동일 encoder에 `collect`, `intercept`, `raid`, `defend/escort`, `recover`의 작은 expert head를 붙여 비교한다.

```text
π(a | o) = Σ_z g(z | o) π_z(a | o)
PPO log_prob = log π(실제로 실행한 a | o)
```

전략 선택을 `forward` 외부 인자로 요구하지 않는다. soft gate는 현재 관측으로 계산한다. latent z를 따로 sample한다면 marginal action log probability 또는 joint likelihood 중 어떤 objective를 쓰는지 고정하고 rollout/update에서 일치시킨다. 기본은 marginal mixture다. hard role switching, top-k pruning, sampled-training→greedy-export 변경은 별도 검증 없이 적용하지 않는다.

V9-S에서는 같은 class의 row가 같은 전략 분포를 공유할 수 있다. 여러 head가 있다고 실제 유닛별 역할 분담이 생겼다고 보고하지 않는다. V9-I 조건이 충족되면 self query와 다른 아군 목표와의 관계를 이용해 역할을 할당한다. 지속 option/retention은 그 이후 ablation이다.

### 9.2 원본 게임 안에서의 전략 과제

| 과제 | 원본 환경에서 만드는 방법 | 성공 지표 / 반례 |
|---|---|---|
| 수집·배달 | no-op/약한 상대와 정상 초기 상태부터 플레이 | 첫 배달 시간, 순점수 증가, 정체 비율. pickup만 반복은 성공 아님 |
| 전투·rush | 정상 이동으로 중앙 성소와 적 접촉에 도달하는 상대 | 처치/respawn 추정, 적 운반 손실, rush 뒤 점수 이익. 자해·무한 교환 구분 |
| 입구 견제 | 적이 공개 창고/중앙 통로를 반복 이용하는 상대 | 통과/배달 억제와 아군 득점 유지. 지형을 바꾸거나 blocker object 추가하지 않음 |
| 약탈·회수 | 원본 플레이로 적이 공개 창고에 적재하는 상황을 유도 | 상대 감소→자기 운반→자기 적재 연결, 흡수 전 성공. 보호 본진 침입은 목표에서 제외 |
| 수비·호위 | 상대 rush/약탈 bot을 투입 | 아군 delivery 성공률, 적 순득점 억제, 공격 상대 전환에 대한 회복 |
| 흡수 phase 대응 | `time_left`로 20초 cycle을 검증하고 정상 플레이 유지 | 흡수 직전 적재·견제와 직후 재수집. 흡수를 추가 득점으로 오인하지 않음 |

모든 시나리오는 시작 이후 수행한 실제 action과 step을 기록한다. teleport, 강제 class/score 변경, 아이템 생성, timeout 축소, 숨긴 random warm-up으로 쉽게 만든 환경은 사용하지 않는다. 원본 reset seed가 실제로 적용되지 않으면 seed별 curriculum을 구현했다고 하지 않는다.

### 9.3 기존 모델을 활용하는 범위

- v6·v7-1 A3·scripted를 **teacher/상대 후보**로 사용한다. 모델 ID만으로 실력을 가정하지 않고 현재 원본 환경·동일 평가 metric에서 다시 기준선을 측정한다.
- 기존 v3 replay는 episode/map grouping 부재·NoOp 편향 때문에 본 학습 데이터로 바로 합치지 않는다. 새 teacher 궤적에는 episode/step/build/관측 hash/실행 action/teacher source를 저장한다.
- V9-S에서 동일 관측에 teacher의 서로 다른 유닛 행동이 붙으면 contradictory label이다. 관측·class별 action histogram을 만들거나 해당 supervision을 제외하고 충돌률을 보고한다. ID를 필요로 하는 teacher 행동을 완벽히 증류할 수 있다고 약속하지 않는다.
- BC 정확도 외에 teacher 없이 최초 배달·이동·득점까지 수행하는지 확인한다. PPO에 teacher가 강제 실행한 행동을 정책 sample로 기록하지 않는다.
- 모델에는 full planner override를 남기지 않는다. 필요하면 관측으로 계산한 거리·geometry feature와 teacher 데이터를 사용하고 제출 시에도 같은 가공을 구현한다.

### 9.4 상대 pool과 승격

기본 상대 family는 `random/no-op`, `collector`, `rush/interceptor`, `raider`, `defender`, 기존 target, 과거 v9 snapshot이다. script는 모델 관련 새 코드로 구현하고 제공 게임/환경은 보존한다.

초기 혼합 제안은 쉬운 상대 20%, 전략별 상대 30%, 기존 target 20%, frozen history 30%다. history가 비어 있으면 해당 비중을 쉬운/전략 상대에 사전 지정 비율로 배분한다. 안정화 후 history 내부에 `p(j) ∝ ε + (1-win_rate_j)^2`의 PFSP를 사용하되 어려운 상대 한 개가 표본을 독점하지 않도록 family 최소 비중을 유지한다. 상대는 episode 동안 고정한다.

한 대의 Mac에서 처음부터 여러 learner를 동시 진화시키지 않는다. **활성 learner 하나 + 작은 frozen pool(초기 최대 8~12개)**로 시작한다. snapshot은 timestamp뿐 아니라 관측 가능한 전투/약탈/수비 behavior descriptor와 상성 차이로 보존한다. exploitability를 정확히 계산했다고 하지 않고 worst-family win rate와 교차 대전 matrix를 사용한다.

curriculum 승격은 사전 고정한 개발 조건을 통과할 때만 한다. 실패 시 쉬운 단계를 무기한 반복하거나 일정 step 후 강제로 최종 상대 90%로 바꾸지 않는다. 정해진 진단 예산 이후 실패 원인을 보고하고 해당 arm을 중단한다.

## 10. 학습 알고리즘과 초기 하이퍼파라미터

### 10.1 초기 구현

- **Attention encoder + shared actor + centralized critic의 MAPPO**를 기준으로 한다. 5개 agent log probability의 무조건 곱을 기본 clipping 단위로 하지 않고, team advantage를 사용하는 agent별 ratio/clip 평균으로 시작한다. joint-ratio는 별도 ablation으로만 비교한다.
- 상대 policy는 rollout wave 내 frozen. 과거 rollout을 새 PPO batch로 무제한 재사용하지 않는다. policy version/hash를 모든 row에 기록한다.
- actor의 시작 logits는 작은 값으로 초기화한다. KEEP/override 특별 gate를 기본 행동 공간에서 제거한다. 정지 포함 9개 action 모두 직접 선택한다.
- PPO rollout은 categorical 분포에서 행동을 sample하고 그 행동의 정확한 old log probability를 저장한다. argmax로 모은 행동에 sample policy의 likelihood를 붙여 on-policy PPO라고 처리하지 않는다. `forward`의 decoder는 저장된 설정으로 정하며 `train()`/`eval()` 호출만으로 바뀌지 않는다.
- GAE는 `(environment, episode, timestep)`별로 먼저 계산한 뒤 minibatch로 섞는다. 다른 환경·episode·reset을 넘어 trace를 연결하지 않는다. padding을 loss에서 제외하고, old value·return은 일관된 원래 단위로 유지한다. rollout 중 actor와 normalizer를 고정하고 ValueNorm/PopArt 갱신 시 denormalization 순서를 검사한다.
- critic target 단위와 denormalization을 맞추고 advantage·value·reward scale을 별도 로그로 기록한다. PopArt를 쓰면 output-preserving 가중치 보정까지 구현한다.
- score 변화, holding 변화, 다음 관측에서의 이동 가능성 등을 예측하는 보조 loss는 valid trajectory의 관측에서만 만든다. 적 class/배터리 수량을 가짜 정답으로 사용하지 않는다.

| 항목 | 초기 후보 | 선택 기준 |
|---|---|---|
| optimizer / lr | Adam, actor 1e-4 / critic 3e-4 | 실제 KL·gradient·held-out 성능; 대규모 자동 sweep 안 함 |
| PPO clip / target KL | 0.15 / 0.02 | minibatch KL·clip fraction·실행 policy 변화를 함께 확인 |
| epochs | 2, 후보 4 | 데이터 재사용과 비정상성 억제, 유효 step당 비용 |
| mini-batch | env timestep 128 또는 256(×5 agent) | 메모리·MPS 전송 포함 처리량. 단위 명시 |
| γ | 기본 1.0; 대조 .99995 | finite-horizon 승률 목적 및 시간 선호 차이 |
| GAE λ | .99 vs .9975 작은 대조 | 50 Hz일 때 대략 2초 vs 8초 trace. actual dt로 재계산 |
| entropy coefficient | .01 시작, 과도 탐색 여부로 .003 후보 | sample/greedy 각각의 유효 이동·득점·행동분포 |
| value loss / gradient clip | Huber 또는 clipped value loss, max norm .5 | 폭주 감지가 우선. clip 전·후 norm 저장 |
| 환경 수 | 4 시작, 2/4/6/8 후보 | §12 프로파일로 한 대 전체 최적 조합 선택 |
| trajectory | 완료 episode wave; 중간 저장 chunk 2,048 | invalid prefix의 optimizer 유입 차단 |

γ=.9995일 때 21,000-step 후 terminal의 직접 할인 계수는 약 2.7×10⁻⁵다. 이를 근거로 긴 지연 신호를 점검하되 GAE가 낮다는 이유만으로 과거 실패 원인을 확정하지 않는다. 정확한 critic이면 bootstrapping이 더 긴 정보를 전달할 수도 있다. **시간 horizon과 critic 정확도를 함께** 비교해야 한다.

### 10.2 필수 학습 계측

`raw_return`, `train_return`, potential telescope residual, `outcome_source`, valid/invalid/timeout 수, natural episode length, side/opponent별 steps, requested/applied seed 상태, initial hash, actual action/log_prob, sample/greedy histogram, class별 동일 행동 비율, 실제 displacement, score 변화, expert 사용량, entropy, ratio min/max, KL, clip fraction, critic EV, 모듈별 gradient·parameter delta, collector/update/logging/save 시간, 메모리, policy lag를 기록한다.

분포가 거의 uniform인데 argmax만 일정한 현상, 행동이 달라졌지만 벽에 막혀 실제 위치는 같은 현상, raw return만 오르고 승률은 악화되는 현상을 각각 별도 경보로 둔다.

## 11. 실험·평가 계획과 성능 승인

### 11.1 gate를 통과한 만큼만 예산을 확장

| 단계 | 예산 제안 | 통과 조건 |
|---|---|---|
| P0 계약·종료 | offline 검사 + 실행 단계에서 정상/timeout/연속 lifecycle 최소 20경기 | 원본 hash, 공개 API, 결과 provenance, self-ID 한계, 같은 입력→행동 일치, invalid 처리 |
| P1 제출·연산 | 학습 없는 녹화 관측 inference/1-update fixture, microbenchmark | 두 파일 clean-room load, gradient 도달, all-channel 입력, shape·batch·device parity |
| P2 기능 pilot | 조건당 최대 131,072 env step, 초기 seed 11 | 최초 배달·수집/전투 전개, reward bound, 종료 정상, decoder 불일치 없음. 실패 원인 확인 전 장기 run 금지 |
| P3 핵심 대조 | 3 arms×3 seeds(11/22/33)×1,048,576 = **9,437,184 env step** 목표 | 아래 ablation, frozen dev 평가, 동일 환경·상대 계약 |
| P4 확인 | 선택 arm의 추가 seeds 44/55 또는 사전 지정 독립 test | 성능·전략·유효 표본·속도 조건 통과 |
| P5 제출 | frozen checkpoint 하나 export·검증 | 형식/의존성/loader/행동/실경기 평가 일치 |

완료 episode를 commit하므로 예산 경계에서는 마지막 episode가 끝날 때까지 overrun이 생길 수 있다. run마다 실제 env step과 마지막 wave overrun을 공개하고 최대 `N_env×22,000` 이내로 제한한다. 목표 숫자에 맞추려고 terminal 직전 게임을 버리지 않는다. pilot, teacher 수집, invalid 시도, 평가 step은 본 학습 예산과 별도 항목으로 합산한다.

### 11.2 핵심 대조군

| arm | 차이 | 검증할 질문 |
|---|---|---|
| A | 작은 single-head Attention + bounded reward + 고정 opponent mixture | 원본 입력·제출 계약에서 최소 학습이 가능한가? |
| B | A + frozen-history PFSP | 상대 다양성이 고정 target 과적합을 줄이는가? |
| C | B + 전략 mixture 및 관측 기반 보조 loss | 전투/약탈/수비 다양성과 최악 상대 성능이 함께 개선되는가? |

P0/P2에서 필요한 γ/λ·stochastic/greedy·potential 유무는 작은 진단 대조로 먼저 고정한다. 구조·reward·상대·decoder를 한 번에 바꾼 결과를 Attention의 효과로 주장하지 않는다. 기존 CNN은 **동일한 새 reward/collector/계약을 사용한 matched 진단 기준선** 및 속도 비교 대상으로만 유지한다. v8 최종 수치와 새 v9 수치 차이만으로 CNN 대비 우월성을 증명하지 않는다.

V9-I를 사용할 수 있게 되면 V9-S와 별도 contract ID로 등록하고 각 군 안에서 동일 조건 대조를 다시 수행한다. 정보량이 다른 모델을 네트워크 구조만의 비교로 제시하지 않는다.

### 11.3 평가 데이터와 선택 규칙

- 단계별 개발 확인은 양 진영을 모두 사용한다. 환경 seed가 검증되면 train/dev/test map seed를 분리한다. 검증되지 않으면 initial hash 중복률과 `map_independence_unverified`를 표시하고 새로운 맵 일반화를 주장하지 않는다.
- final dev는 **checkpoint당 최소 30개 환경 조건×2 sides×4 상대 family=240경기**를 초기 목표로 한다. 실제로 독립 환경 조건을 만들 수 없는 경우 “30개 맵”으로 기록하지 않는다. 동일 초기 맵에서 확률적 policy 반복은 action-RNG 반복이라고 부른다.
- 현재 loader로 로드한 정확한 제출 actor를 주 평가 대상으로 사용한다. teacher나 fallback을 평가에서만 추가하지 않는다.
- strict stochastic actor라면 배치 permutation/split에 대해 **분포의 일치**를 검사한다. RNG 소비 순서가 바뀐 결과까지 byte-identical일 것을 요구하지 않는다. 재현 평가의 RNG 상태와 decoder는 등록한다.
- 개발 성능 선택은 family macro-average와 worst-family, side gap, invalid rate, inference p95를 함께 본다. 최종 test는 선택에 사용하지 않는다.
- 승/패/무 개수와 `W/N`, `(W+.5D)/N`을 별도 표시한다. invalid를 무승부로 넣지 않으며 valid 수·attempt 수를 모두 보고한다.
- seed/run과 실제 독립 환경 조건을 계층적으로 고려한 interval을 사용한다. 0/N의 bootstrap [0,0]을 일반화 오차가 0이라는 의미로 쓰지 않는다. Wilson/이항 구간도 독립 시행 전제가 맞을 때만 해석한다.
- A2/A3처럼 같은 60조건을 여러 번 평가한 것을 600개의 독립 맵으로 세지 않는다. paired seed 요청만으로 paired map을 주장하지 않는다.

### 11.4 v9 성능 승인 조건

아래는 프로젝트 제안 gate이며 대회 공식 기준이 아니다.

1. baseline 대비 macro win-rate 차이의 불확실성을 보고하며, 독립 확인에서 개선 방향이 유지될 것. 과거 목표 85%는 장기 목표로 두고 작은 개발 표본 9/10으로 달성 선언하지 않을 것.
2. 적어도 수집·전투/견제·약탈 세 family에서 정상 전술 수행 증거가 있고, 한 family 개선이 다른 family의 큰 붕괴와 맞바뀌지 않을 것.
3. 입구 제자리 유지·교환 kill·배터리 왕복으로 raw reward만 늘어나는 counterexample을 통과할 것.
4. 불확실한 self-ID/공식 winner 계약을 해결하지 못하면 승인 상태를 `format_ready`, `local_runner_improved`, `game_outcome_unverified`처럼 구분할 것.
5. 현재 하드웨어의 inference/전체 유효 학습 처리량·메모리·백그라운드 안정성 검사를 통과할 것.

## 12. 현재 하드웨어에서의 학습 가속

### 12.1 실제 하드웨어와 과거 측정값

| 항목 | 2026-10-04 확인 |
|---|---|
| 장비 | Mac mini `Mac16,11`, Apple **M4 Pro** |
| CPU | 12 cores = 8 performance + 4 efficiency |
| 통합 메모리 | **48 GiB** |
| OS / PyTorch | macOS 26.5 arm64 / torch 2.13.0 |
| MPS | 실제 권한이 있는 프로세스에서 사용 가능. 제한된 sandbox의 `False`와 구분 |
| MPS 권고 작업 메모리 | API 반환 40,200,896,512 bytes, 약 37.44 GiB. 전체 RAM과 별도인 독립 VRAM이 아님 |
| 디스크 여유 | 조사 시 약 144.9 GiB. 학습 시작 때 재측정 |

과거 [v7 가속 보고서](../reports/v7/acceleration_and_resume.md)에는 인코더 B=5 CPU 1.73ms/MPS 3.89ms, 전뇌 CSR CPU 19.10ms/MPS 5.15ms가 기록되어 있다. 따라서 작은 Attention 추론의 MPS 사용은 실측으로 결정한다. 같은 보고서의 12-worker 327.68 step/s는 짧은 수집 benchmark이며 **현재 금지된 API/통신 overlay가 포함되어** v9 기대치로 그대로 쓰지 않는다.

원본 최종 v8 수집의 6개 run은 각 약 37.3~37.5 step/s였다. 동시 실행이면 단순 합 약 224 step/s이지만, 이것도 모든 실행 구간이 완전히 겹쳤다는 보장이나 v9의 실제 처리량은 아니다. v9는 완전 episode commit·학습 가능한 encoder 때문에 연산 구성이 달라진다.

### 12.2 프로세스·배치 설계

```text
background supervisor
  ├─ original Unity worker 0..N-1 + 가벼운 수집 프로세스
  ├─ CPU actor 추론 또는 제한된 공용 inference batcher
  ├─ learner 1개: CPU 또는 MPS의 큰 학습 batch
  ├─ frozen opponent cache
  └─ bounded telemetry/checkpoint writer
```

- 초기에는 한 learner에 4개 원본 환경을 연결한다. 학습 seed 간 병렬화와 한 seed 내부 vectorization을 구분한다. 예: 6개 환경을 이미 쓰는 learner 3개를 무심코 띄워 18개 Unity를 실행하지 않는다.
- **전체 machine worker cap**을 supervisor 한 곳에서 관리한다. 후보는 2/4/6/8개이고 10/12개는 RAM·열·처리량이 실제로 좋아질 때만 추가한다.
- actor는 CPU 1 thread부터 측정한다. `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, torch inter-op 1을 자식별로 명시하고 2 thread 대조를 한다. 사용자의 shell 전역 설정은 바꾸지 않는다.
- learner는 MPS 한 프로세스에서 mini-batch를 크게 묶어 전송·동기화를 줄인다. Unity rendering과 같은 GPU/통합 메모리를 쓰므로 CPU-only update와 비교한다.
- 한 forward에 B=5, 여러 환경을 합친 B=20/40, 학습 B=640/1280을 구분한다. 공용 inference batcher를 쓰면 `max_wait_ms`를 1~2ms 후보로 측정하고 기다림 때문에 느려지면 제거한다.
- macOS multiprocessing은 `spawn`, spawn 전에 MPS/Unity를 생성하지 않는다. 서로 다른 환경이 port/worker_id·RNG·episode 기록을 공유하지 않게 한다.
- 각 rollout wave는 고정 actor version을 사용한다. MPS 학습과 CPU collection의 무제한 overlap으로 stale PPO를 만드는 가속은 기본안에 넣지 않는다. 비동기 확장은 policy lag를 계측하고 정합한 보정이 있을 때만 도입한다.

### 12.3 적용 우선순위와 금지하는 지름길

| 순서 | 가속 방안 | 검증 / 한계 |
|---|---|---|
| 1 | 같은 호출의 공통 map/entity 연산 재사용 | 정확히 같은 graphic/vector 부분만 묶고 inverse index로 row 복원. class는 별도 처리. `.tolist()`/CPU round-trip 비용까지 비교 |
| 2 | 모든 agent를 한 번에 추론 | actor 및 같은 checkpoint 상대별 batch. 제출은 arbitrary B에서도 같은 분포 |
| 3 | 원본 환경 다중 프로세스 | 2/4/6/8 sweep, 각 3회·warm-up 제외·최소 60초, 최종 30분 안정성 확인. env/agent step 단위 구분 |
| 4 | `inference_mode`, contiguous layout, buffer 재사용 | 우리 모델/collector만 변경. 원본 반환 배열 in-place 변경 없음 |
| 5 | MPS 큰 update batch + CPU 작은 추론 | 데이터 전송 포함 wall-clock, `torch.mps.synchronize()`를 benchmark 경계에만 사용. production per-step 불필요 sync 금지 |
| 6 | SDPA 사용, latent/token 폭 제한 | PyTorch backend가 선택하는 커널 확인. CUDA FlashAttention/Triton 성능을 M4에 약속하지 않음 |
| 7 | 학습용 lossless 압축·중복 map 저장 방지 | one-hot 정확성 확인 후 semantic ID uint8로 보관하고 update 시 11채널 원복. 검사를 통과하지 않은 입력은 argmax로 정보 손실시키지 않음 |
| 8 | end-to-end train encoder의 안전한 캐시 | 같은 forward 내 공유만 기본. optimizer step을 넘어 trainable latent 캐시를 재사용하지 않음 |
| 9 | checkpoint/eval 빈도 조정 | 최종/중간 endpoint·장애 복구는 유지. warm-up 매 step export/전체 모델 reload 금지 |
| 10 | 제한적 compile·mixed precision | 설치 torch 2.13/실제 MPS에서 지원·손익·gradient parity가 확인된 경우 별도 run. 기본 fp32. 최종 제출은 eager CPU를 포함 |

제외: 게임의 C++/JAX 재구현, timestep/DecisionPeriod 변경, 관측 해상도·채널 삭제, private cache flush/추가 hidden step, 설치 API transport monkey patch, `-nographics`, 무조건 긴 action repeat, 시스템 memory watermark 무제한 해제. 기존 protobuf/upb 가속은 원본 API 그대로 사용한다는 요구와 충돌하므로 재사용하지 않는다.

공식 참고: [PyTorch performance guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html), [SDPA](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html), [MPS memory API](https://docs.pytorch.org/docs/stable/mps.html). 웹 문서의 현재 stable 버전과 설치된 2.13이 다를 수 있으므로 사용 API는 로컬 버전으로 다시 검증한다.

### 12.4 저장·메모리 예산

- 96×96×11 fp32 graphic은 agent당 405,504 bytes다. 동일 팀 5개의 중복 graphic을 매 step JSON으로 기록하지 않는다.
- one-hot 조건이 성립하면 96×96 uint8 ID map은 9,216 bytes로 원본 한 graphic의 1/44이다. 이는 **저장 표현만** 바꾸는 것으로 모델에 전달하는 11채널은 완전히 복원한다.
- 4환경×최대 22,000-step staging의 공통 ID map 한 벌은 약 774 MiB다. 5개 vector fp32를 모두 저장하면 약 161 MiB가 추가된다. 양 팀·메타데이터·trace·버퍼 복제·학습 activation은 별도다. 실제 peak RSS와 MPS driver memory를 측정한다.
- 초기 total process memory 목표는 32 GiB 이하, OS/Unity 여유를 두고 sustained pressure·swap이 증가하면 worker 수를 줄인다. MPS 권고 37.44 GiB를 Unity 외 별도 예산처럼 더하지 않는다.
- 일반 로그는 episode 1행, update 1행, health 5~10초 1행. per-step 상세는 최근 ±250 step ring과 실패/중요 event 주변 구간에 한정한다. raw observation sample은 재현에 필요한 제한된 개수만 영구 보존한다.
- run당 telemetry soft budget 2 GiB, 전체 pilot/main log budget은 단계별 등록한다. 보존할 evidence와 재생성 가능한 cache를 구분한다. 기존 로그를 자동 삭제하지 않는다.
- checkpoint는 `latest`, 개발 `best`, 등록 endpoint, 실패 직전 진단본을 immutable hash로 관리한다. 임시 파일→atomic rename, checksum, resume log offset을 함께 저장한다.
- 디스크 여유 25 GiB 미만이면 새 job 시작 중단, 15 GiB 미만이면 정상 save 후 중단한다. 현재 여유가 있다고 실험 전체 공간이 확보됐다고 단정하지 않는다.

### 12.5 속도 gate와 시간 예측

속도는 다음을 분리한다: 원본 `env.step`, 전처리/tokenization, actor, 상대, episode 재생성, PPO forward/backward, logging, checkpoint, 평가, 전체 종료까지.

초기 목표는 **현재 원본 v8과 같은 하드웨어·worker 부하에서 전체 유효 학습 처리량 개선**, B=5 추론 p95의 악화 방지다. “2배 이상”을 선보장하지 않는다. 모델 연산만 2배 빨라도 나머지가 병목이면 총 시간이 거의 줄지 않을 수 있다.

대략 시간은 `총 training env steps / 실제 유효 end-to-end steps/s + 별도 평가/초기화 시간`으로 계산한다. 예를 들어 9,437,184 step을 150/250 aggregate step/s로 처리하면 순수 step 환산 약 **17.5/10.5시간**이다. 이는 예시이고 완전 episode commit·PPO·평가·invalid 재시도를 포함한 실측 ETA로 교체해야 한다.

## 13. 백그라운드 학습·창 숨김·운영 계약

### 13.1 실행 방식

새 `code/v9/scripts/run_v9_fast.sh`는 프로젝트 위치를 계산하고 `.venv/bin/python`의 새 v9 entrypoint를 실행한다. 내부 supervisor가 `Popen(start_new_session=True, stdin=DEVNULL, stdout=log, stderr=log)`로 분리된다. shell의 `&`에만 의존하지 않는다. 작업 디렉터리와 package import root는 절대 경로로 고정한다.

원본 binary는 v8에서 검증한 방식과 같은 **exec-only launcher**로 `-batchmode`를 전달하고 `no_graphics=False`를 유지한다. `-nographics`, `-force-gfx-null`은 금지한다. 원본 app bundle·렌더러·게임 C#을 수정하지 않는다. v9 전용 launcher를 사용해 과거 v8 실행물의 hash를 바꾸지 않는다.

Unity 공식 문서는 `-batchmode`와 `-nographics`를 별도 옵션으로 설명한다. 실제 graphic 유지·창 없음은 현재 Mac에서 확인해야 한다. [Unity Player arguments](https://docs.unity3d.com/6000.0/Documentation/Manual/PlayerCommandLineArguments.html), [기존 로컬 검증](../reports/v8/background_window_fix/live_rendered_check.json)

### 13.2 운영 기능과 완료 검사

- `--check`: 파일/원본 hash/import/config/export fixture 검사만 수행. **Unity·학습을 시작하지 않음**.
- `--benchmark`: 명시적으로 요청한 짧은 benchmark만 실행, gradient update 없음. 본 실험과 별도 ID.
- 기본 실행: detached supervisor 준비 후 PID·로그 경로를 출력하고 terminal prompt로 복귀. 터미널 종료에도 유지.
- `--status`: manager/worker 상태, 유효·폐기 step, natural episode 수, invalid 원인, worker별 속도, 현재 phase, ETA.
- `--stop`: 소유 process에만 정상 중단 요청. 모델·optimizer·RNG 저장, 미완료 episode 폐기 회계, Unity close. timeout 시 기록된 PID와 birth time이 맞는 자식만 종료.
- `--resume`: source/config/build/contract hash가 같은 체크포인트만 재개. 새 물리 episode에서 시작했음을 표시.
- 중복 실행 lock, port 할당, worker ownership, 충분한 disk reserve, 실패한 attempt 보존, 부분 평가 재시도 회계를 유지.
- 학습 중 `caffeinate -i -w <supervisor_pid>`로 idle sleep을 막되 화면 잠금/사용자 logout/전원 종료 후까지 실행을 보장한다고 하지 않는다.
- shell 파일은 executable bit와 `bash script` 두 방식 모두 검사한다. 프로젝트 외 디렉터리에서 실행해도 import가 되는지 검사한다.
- 실제 game window enumeration으로 Unity 창이 안 뜨는지 확인하고, nonzero graphic·위치 대응·11채널 유효성을 동시에 검증한다. 프로세스가 background라는 것만으로 창 숨김을 통과시키지 않는다.

구현 완료 후 제공할 명령 형태는 아래와 같다. **현재 문서 작성으로 이 스크립트가 구현·준비됐다는 뜻은 아니다.** 기존 v8 명령은 변경하지 않는다.

```bash
bash /Users/safeailab_macmini/Desktop/2026-IST-tech-RL/code/v9/scripts/run_v9_fast.sh
```

같은 명령 뒤에 `--check`, `--status`, `--stop`, `--resume`를 붙인다. 학습을 실제 시작할지는 이후 사용자의 실행 요청에 따른다.

## 14. 두 파일 제출물과 실제 loader 검증

최종 포맷은 [blackout_last_4_pages.md](../blackout_last_4_pages.md)를 따른다.

```text
artifacts/submission/v9<checkpoint_sha256>/
  policy.py
  checkpoint.pt
```

```python
# checkpoint 저장 형식
torch.save({"policy_state": model.state_dict()}, "checkpoint.pt")

# 제공 loader 경로 그대로 검증
loaded = load_checkpoint(
    MyPolicy, "checkpoint.pt", state_dict_key="policy_state",
    device="cpu", vector_size=96, n_channels=11,
)
```

`policy.py`는 torch와 Python 표준 라이브러리만 사용한다. 프로젝트 package, 실행 directory, 체크포인트 외 외부 파일, network, Unity, NumPy, custom Metal extension을 runtime dependency로 요구하지 않는다. 모델 hyperparameter는 고정 architecture 또는 등록 buffer로 재구성하고 모든 학습 weight를 `policy_state`에 포함한다. CPU/MPS 학습을 했더라도 CPU load 가능해야 한다.

| 검사 | 합격 조건 |
|---|---|
| 격리 import | 빈 임시 디렉터리에 두 파일만 복사, `python -I`에서 import/load 성공. `blackout_rl`/`blackout_v9` 미설치 상태 |
| 입력/출력 | B=0/1/3/5/10, float32 `(B,96)`·`(B,11,96,96)` → finite float32 `(B,2)` in `[-1,1]` |
| 제공 전처리 | `load_checkpoint().act(obs)`의 실제 HWC→CHW 및 name-action mapping 경로 검사. 이미 CHW인 입력을 다시 transpose하지 않음 |
| deterministic 정책 | row permutation/split/repeated call parity. 동일 입력에 임의 row ID를 삽입하지 않음 |
| stochastic 정책 | logits/distribution permutation·split parity, 고정 RNG 경로 재현, sampled action의 log_prob 일치. 허용 규정과 선택 decoder를 export manifest에 기록 |
| 학습/제출 parity | 실제 저장 관측에서 training actor와 두 파일 actor의 logits 및 동일 decoder 출력 일치 |
| 시간·메모리 | cold load, warm forward p50/p95/p99, B=5 전체 정책 시간, peak memory. CPU 필수·MPS 추가, 서버 GPU는 접근 가능할 때 별도 검사 |
| lifecycle | 여러 경기 연속 호출, 동일 시간값·부분 batch에서 이전 경기 hidden state 유출 없음 |
| 행동 검증 | 원본 runner에서 export된 모델로 평가. 결과는 같은 checkpoint hash/정확한 두 파일 hash에 연결 |

제출 **형식 통과**, 로컬 runner 성능 통과, 공식 server 규정 확인, 실제 외부 제출은 서로 다른 상태다. 공식 하드웨어·추론 제한·RNG 허용·승패 계약이 제공되지 않은 상태에서 `official_server_certified=true`로 표시하지 않는다. 외부 업로드는 이번 계획/구현의 자동 동작으로 넣지 않는다.

## 15. 구현 파일·순서·검증 산출물

기존 v8 fingerprint가 `blackout_rl` 소스 범위를 넓게 포함하므로 새 코드는 **별도 top-level `blackout_v9/`**에 둔다. 과거 실험 source를 수정해 resume 검사를 통과시키지 않는다. 재사용이 필요하면 API가 순수한 utility인지 확인하고 hash를 고정한다.

| 순서 | 새 파일/산출물 제안 | 완료 조건 |
|---|---|---|
| 0 | `code/v9/blackout_v9/contracts.py`, `code/v9/configs/contract.json` | 입력·objective·원본 artifact·seed 한계·decoder 계약 고정 |
| 1 | `code/v9/blackout_v9/policy.py`, `tokens.py`, `export.py` | CNN 없는 Attention, 두 파일 clean-room fixture, 모든 graphic 채널 |
| 2 | `code/v9/blackout_v9/rewards.py`, `trajectory.py`, `collector.py` | 원본 env 반환값만 사용, bounded reward, episode staging/commit/invalid 구분 |
| 3 | `code/v9/blackout_v9/ppo.py`, `value.py`, `diagnostics.py` | ratio/GAE 경계·ValueNorm·모듈별 gradient·encoder 재계산 검증 |
| 4 | `code/v9/blackout_v9/opponents.py`, `curriculum.py`, `league.py` | 허용 관측 기반 전략 상대, frozen snapshot, 실패 단계 강제 승격 없음 |
| 5 | `code/v9/blackout_v9/evaluation.py`, `statistics.py` | metric provenance, initial hash, 독립 표본/CI, frozen test |
| 6 | `code/v9/blackout_v9/runtime.py`, `benchmark.py`, `telemetry.py` | 원본 API 무수정, binary buffer, CPU/MPS/worker sweep, end-to-end 보고 |
| 7 | `code/v9/blackout_v9/runner.py`, `code/v9/scripts/run_v9_fast.sh`, v9 launcher | 실제 detached 실행, 창 없음/graphic 있음, stop/resume/중복 실행 검사 |
| 8 | `code/shared/tests/v9`, `logs/v9/reports`, `docs/v9.md` | 아래 critical 검증 및 사용자가 실행할 한 줄 명령 안내 |

필수 회귀 검증:

1. 종료 보상 정확히 1회, 자연 timeout과 watchdog 구분, timer 소실 합성 fixture, blocked step을 외부 supervisor가 회수.
2. invalid episode 이전 chunk가 optimizer·normalizer에 반영되지 않음; wave 안 valid/invalid 혼합과 resume에서 중복 update 없음.
3. potential telescope, 정지·왕복·반복 약탈/kill loop, terminal reset 점수, 긴 episode를 붙였을 때 누적 크기 bound 유지.
4. 원본 obs/reward 배열 불변, 11채널 복원 exact, pixel 위치/방향·팀 관점 확인, occlusion을 pickup으로 오판하지 않는 진단.
5. 동일 입력 row, arbitrary B, batch 순서/분할, logits/실행 action/log_prob 일치. trainable encoder까지 gradient 도달.
6. 실제 exporter를 처음부터 사용하고, 프로젝트 import 없이 load. PyTorch mode 변경으로 sampling/argmax가 바뀌지 않음.
7. source/config 변경 시 resume 거부, immutable checkpoint·로그 offset 회복, 상태 파일만 남은 crash·port 충돌·디스크 부족에서 안전하게 종료.
8. benchmark는 학습/평가 예산으로 섞지 않고 source/environment hash를 남김. 제출 CPU fallback은 정확성·속도까지 측정.

문서만 작성하는 현재 단계에서는 이러한 검증을 이미 통과했다고 표시하지 않는다. 구현 후 offline critical checks→원본 short lifecycle 검증→pilot→본실험 순서로 진행한다.

## 16. 선결 정보와 중단 기준

외부 계약에 대해 확인할 항목은 다음과 같다. 이 계획서 작성 자체를 멈추는 질문이 아니라 구현의 acceptance gate다.

| 미확정 정보 | 없을 때 가능한 범위 | 승인할 수 없는 것 |
|---|---|---|
| 게임 점수 winner vs 누적 reward winner | 제공 runner 호환 평가와 proxy/visible-score 감사 | 공식 점수 승률·대회 최적화 완료 |
| 자기 유닛 식별·팀 batch 의미 | V9-S Attention, stochastic 대조의 기술 검증 | 안정된 유닛별 역할 할당·self 위치 기반 정밀 제어 |
| stochastic forward 허용·RNG 규칙 | 현 문서의 두 입력 계약으로 sampling PPO·로컬 제출 평가 진행, greedy는 별도 대조 | 확인되지 않은 서버 RNG 제한까지 검증 완료했다는 주장 |
| episode reset 통지·stateful policy 허용 | stateless 정책 | 임의 hidden state·호출 횟수로 경기/유닛 추적 |
| 실제 map seed 적용·서버 seed 분포 | 초기 관측 hash 기록, 동일 조건의 반복 평가 | 검증하지 않은 다중 맵 일반화 |
| 서버 time/memory/package 제한 | 현재 CPU/MPS p95와 두 파일 의존성 보고 | 공식 하드웨어 추론시간 인증 |

**학습 중단/설계 재검토 조건:** 관측/보상 nonfinite, 원본 hash 변경, actor gradient 단절, invalid rate gate 초과, 초기 상태 반복을 독립 표본으로 처리한 평가, teacher 없이 기본 이동·배달에 실패한 pilot, raw reward만 개선되고 승인 metric이 악화되는 경우다. 더 큰 Attention·더 긴 학습을 자동 처방하지 않는다.

v9의 완료 기준은 새로운 아키텍처 파일의 존재가 아니라, **원본 환경·실제 제출 인터페이스에서 유효한 학습 신호를 받고, 기록으로 확인 가능한 전략과 성능을 보이며, 현재 Mac에서 관리 가능한 속도로 백그라운드 학습·재개·제출 검증까지 연결되는 것**이다.
