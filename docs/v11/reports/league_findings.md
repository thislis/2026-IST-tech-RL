# v1–v10 리그 분석과 v11에 넘길 근거

분석일: 2026-10-24. 대상은 [version_best_ties_league](../../../logs/shared/agent_league/version_best_ties_league/report.md)이며, 새 경기나 학습은 실행하지 않았다. 이 문서는 분석이고, 제안한 모델의 성능 달성 보고가 아니다.

## 1. 자료의 범위와 재검증

21개 참가자 = v1–v10 대표 12개(v7·v8 각 2개) + scripted 9개. 210개 대진 × 5회 × 양 진영 = 2,100경기, 참가자당 200경기다. [manifest](../../../logs/shared/agent_league/version_best_ties_league/manifest.json), [ranking](../../../logs/shared/agent_league/version_best_ties_league/ranking.csv), [분석 자료](../../../logs/shared/agent_league/version_best_ties_league/analysis_data.json), 2,100개 `cells/*.json`을 읽었다.

[독립 재집계 스크립트](../../../code/v11/analysis/audit_league.py)로 각 cell과 원본 attempt result, 초기 맵 snapshot의 SHA, 마지막 trace chunk의 SHA·누적 reward·자연 종료·관측 점수를 대조했다. 21개 모델 성적·행동 집계와 210개 대진이 기존 분석과 일치했다. 재실행 명령은 다음과 같다.

```bash
.venv/bin/python code/v11/analysis/audit_league.py
```

자동 결과: [logs/v11/reports/league_audit.json](../../../logs/v11/reports/league_audit.json). 마지막 전이를 포함한 총 경기 길이는 17,695,695 step이다. 중간 trace 전체를 다시 읽지는 않았으며, 행동 요약은 각 result의 계산값과 [trace 구현](../../../code/shared/agent_league/trace.py)을 대조했다. 원본 리그 파일은 변경하지 않았다.

| 확인 항목 | 결과 | 해석 |
|---|---:|---|
| 유효 자연 종료 | 2,100/2,100 | 실패 셀 0; 중단 경기를 무승부로 센 결과가 아님 |
| 양 진영 쌍의 초기 맵 동일 | 1,050/1,050 | 이 리그의 side pair가 실제 같은 배치를 공유함 |
| 서로 다른 초기 전체 맵·배터리 배치 | 각각 1,050 | 이전 v9/v10의 동일 배치 반복 평가와 구별됨 |
| 서로 다른 벽 / 창고 배치 | 1 / 5 | 임의의 새로운 벽 구조까지 일반화했다는 증거는 없음 |
| raw reward 무승부 | 100 | 단순 승률과 승점률을 구별해야 함 |
| reward와 마지막 관측 점수의 승자 반대 | 95 | 둘 다 우열이 있는 경기에서의 불일치 |
| 관측 점수 동점, reward는 승패 존재 | 119 | reward 승리가 배터리 경쟁의 승리를 뜻하지 않음 |
| 마지막 전이에서 관측 점수 우열 변경 | 5 | 마지막 행동 전 점수만으로 최종 결과를 대신하면 오류 |

초기 맵의 실제 다양성은 확인했지만 요청 seed 적용의 일반적인 인과 계약까지 검증한 것은 아니다. 맵 해시와 양 진영 pairing을 향후에도 보관한다. 기존 [report](../../../logs/shared/agent_league/version_best_ties_league/report.md)의 seed 미검증 설명과 이후 [analysis](../../../logs/shared/agent_league/version_best_ties_league/analysis.html)의 실제 다양성 확인은 구분해서 읽어야 한다.

## 2. 버전별 결과

`승점률=(승+0.5×무)/경기 수`. 표의 승패는 **로컬 runner 누적 raw reward** 기준이다. 버전 이름은 모델 가족의 우월성을 뜻하지 않으며, 아래에 지정한 대표 checkpoint의 관측 결과다.

| 버전 / 참가 ID 뒷부분 | 승/무/패 | 승점률 | 진영 A / B 승점률 | 반납 proxy / 경기 | 방향 전환 비율 |
|---|---:|---:|---:|---:|---:|
| v1 `d4c8e00e` | 11/29/160 | 12.75% | 15 / 10.5% | 0.240 | 1.58% |
| v2 `80375a72` | 20/27/153 | 16.75% | 23 / 10.5% | 0.245 | 28.63% |
| v3 `80375a72` | 8/35/157 | 12.75% | 14.5 / 11% | 0.285 | 47.40% |
| v4 `29794421` | 180/0/20 | 90.00% | 87 / 93% | 20.715 | 1.81% |
| v5 `9c6b5f28` | 179/0/21 | 89.50% | 89 / 90% | 21.135 | 1.76% |
| v6 `5c651487` | 180/0/20 | 90.00% | 86 / 94% | 20.835 | 1.69% |
| v7 Rewired `23262e36b5` | 180/0/20 | 90.00% | 89 / 91% | 20.795 | 1.77% |
| v7 FlyWire `27d97f725e` | 179/0/21 | 89.50% | 89 / 90% | 21.025 | 1.92% |
| v8 C1 | 9/34/157 | 13.00% | 13.5 / 12.5% | 0.000 | 2.19% |
| v8 Flat | 20/22/158 | 15.50% | 22 / 9% | 0.525 | 0.00% |
| v9 B | 70/6/124 | 36.50% | 30 / 43% | 1.405 | 47.16% |
| v10 C1 | 73/2/125 | 37.00% | 31 / 43% | 2.070 | 45.08% |

반납 proxy는 보유 Battery→빈손 전이와 팀 점수 증가가 함께 보인 **step 사건 수**다. 유닛별 정확한 반납 횟수나 수량이 아니다. 방향 전환은 연속 nonzero action 사이 cosine이 `0.70710678`보다 작은 유닛 비율을 경기별로 평균했다. 길이가 긴 경기의 모든 step을 합쳐 가중 평균한 비율과 다르다. 정체·죽음·역할 판단에도 관측 proxy를 확정 이벤트처럼 사용하지 않는다.

대표 선택은 모두 같은 방식의 최적 checkpoint 탐색이 아니다. manifest는 완료 여부·주 실험·낮은 seed·최종 step 등의 규칙을 기록한다. v9는 B seed 22, v10은 C1 seed 11이고, v4는 100,352 step, v6는 1,159,168 step, v7 두 모델은 각각 seed 11의 2,000,000 step이다. 따라서 학습량이나 학습 seed가 통제된 아키텍처 비교가 아니다. v2/v3의 짧은 ID가 같아도 학습된 가중치가 같다는 뜻은 아니다. 정확한 경로와 asset SHA는 manifest가 기준이다.

### 직접 맞대결과 상대군

- 상위 다섯 모델(v4/v5/v6/v7 두 개)의 10개 상호 대진은 **모두 5승 5패**다. 상위권 내부 우열이나 학습 residual의 개선을 입증하지 못한다.
- v4/v6/v7 Rewired는 상위권 밖 160경기 전승. v5와 v7 FlyWire는 각각 159/160승이다. v5의 예외는 scripted-weak, FlyWire의 예외는 scripted-default다.
- v9/v10은 상위 다섯 모델 각각에 0/10, 각자 합계 0/50이다. 서로는 5승 5패다. 전체 승점률 0.5%p 차이로 v10의 우위를 확정하지 않는다.
- scripted-random은 전체 승점률 41.5%로 v9/v10보다 높다. 직접 맞대결에서도 v9는 random에 5승 5패, v10은 4승 6패다.
- v8 C1의 팀 전체 동일 행동 비율은 100%, Flat은 96.99%다. v9/v10의 해당 비율은 약 0.02~0.03%로 낮지만 빈번한 방향 변경과 적은 반납이 남는다. 무작위성을 늘리는 것만으로 목표 수행이 생기지 않는다.
- 관측된 scripted 순환은 default→rush→special→default다. 그러나 raw reward 결과에 한정하며, 5개 맵 쌍의 표본이다. 모든 상대를 하나의 순위나 Elo로만 압축하지 않는다.

## 3. 결과를 바꾸는 실행 계약

### 식별 정보와 상태

[league adapter](../../../code/shared/agent_league/adapters.py)는 v1–v7에 이름·팀·상태를 사용하는 과거 실행 경로를 제공하고, v8–v10에는 두 tensor만 전달한다. 상위권과 신형 모델의 격차에는 구조·학습뿐 아니라 입력 계약 차이가 섞여 있다.

현재 로컬 Unity의 `BlackOutAgent.Setup()`은 `matchManager.Units[unitIndex]`를 자기 유닛으로 잡고 `CollectObservations()`는 같은 `Units` 배열을 순회한다. 전처리기는 routing용 raw index를 모델용 vector에서 제거한다. 공개 `act(obs)`의 이름을 보존하면 로컬 구현에서 자기 위치를 연결할 수 있지만, 두 tensor만 받은 기존 모델에는 그 연결이 없다. [PREP-06](../../pre_v1/reports/prep06_agent_identity_order.md)의 초기 동일 관측 기록과도 일치한다.

같은 팀·같은 class의 유닛이 완전히 같은 입력을 받는 결정론적 함수라면 같은 행동을 낸다. GRU를 넣어도 초기 상태와 입력 이력까지 같으면 식별 신호를 만들어내지 못한다. v11에서는 이름 기반 routing과 ONNX 상태 연결을 처음부터 구현하며, row 번호를 ID라고 추측하지 않는다. 최종 실행 규격의 차이는 adapter에서 처리한다.

### 승패와 점수

[match.py](../../../code/shared/agent_league/match.py)는 환경 `infos`를 버리고 누적 reward를 비교한다. 따라서 이번 trace만으로 공식 winner 필드 전체를 복원했다고 말할 수 없다. 마지막 score는 유용한 독립 진단일 뿐이다.

대표 사례는 다음과 같다. 점수는 normalized 값을 100배 했다.

| 경기 ID | 확인된 결과 | v11 설계에 주는 의미 |
|---|---|---|
| `v10-C1__v6-5c651487__r02_s1` | 1,045 step, v6:v10 점수 100:0, reward 39.70:-19.00 | 직접 실행되는 수집·귀환 경로를 먼저 확보 |
| `v4-29794421__v6-5c651487__r03_s1` | 1,242 step, v6:v4 점수 102:86 | 5:5 대진의 한 사례; 최강 판정 근거로 쓰지 않음 |
| `scripted-collector__scripted-rush__r04_s0` | 21,004 step, collector:rush 점수 9:0, reward -12,192.20:3,654.60 | reward만 높이는 전투 반복과 경제 능력을 별도로 검사 |
| `scripted-noop__v8-v8_comp-c1__r02_s1` | 21,004 step, 0:0이지만 C1 reward 0.20으로 승리 | noop 상대 승리만으로 행동 능력 통과 처리 금지 |
| `v10-C1__v9-B__r02_s1` | v9:v10 점수 4:12, 21,004 step | 이 사례의 승리와 대진 전체 5:5를 구별 |
| `scripted-weak__v5-9c6b5f28__r02_s1` | 655 step, v5:weak 점수 53:102 | 짧은 경기의 패배도 포함한 회귀 시나리오 필요 |

각 사례는 [기존 인터랙티브 분석](../../../logs/shared/agent_league/version_best_ties_league/analysis.html) 및 [재검증 JSON](../../../logs/v11/reports/league_audit.json)의 `cases`에서 확인할 수 있다. 이 사례 선택은 원인에 대한 인과 실험이 아니다.

## 4. 각 버전에서 계승할 것과 고칠 것

| 버전 | 계승 가치 | 확인된 약점 / 제한 | v11의 구체적 대응 |
|---|---|---|---|
| v1 | 초기 self-play·공통 관측/평가 기반과 실패 기록 | 512-step마다 collector를 버려 장기 학습에서 terminal 0 | 자연 종료까지 상태 보존, episode 단위 회수 및 예산 집계 |
| v2 | persistent collector, 진영 교대·완료 경기 계측 | 강한 checkpoint의 neural core만 복사해 planner 성능 상실 | checkpoint뿐 아니라 실행 policy·planner·decoder hash까지 초기화 계약에 포함 |
| v3 | teacher label·DAgger·단계적 상대 학습이라는 도구 | NoOp 편향, BC 82.3%에도 0/10; 강제 승급; 기억·입력 격차 | 강한 tensor planner를 우선 유지, 선택적 증류와 closed-loop 평가, 행동 gate 미달 시 승급 금지 |
| v4 | 3 worker+2 guard, 목표 할당·경로·FSM, 강한 초기 실력 | 보존 위주 학습으로 planner 초과 개선 없음 | 동일 planner baseline 동결, 전략/목표 선택과 실제 예외 행동을 학습 |
| v5 | 상대 방향 residual, 한 유닛만 수정하는 초기 안정화 | 약 0.184% agent-action override, train 선택과 eval 선택 차이 | gate와 correction의 결합 확률을 정확히 저장; 후기 여러 유닛 개입 실험 |
| v6 | 팀 확률 일치, 실제 planner 문맥, 정체·위상, rollback·재개·독립 확인 | 탐색이 늘어도 target 개선 없음, 한 step 한 유닛의 제한 | 41-way를 비교군으로 보존; 주 모델은 지속 목표와 역할 변경까지 학습 |
| v7 | 다중 seed·동일 예산·재배선 대조 실험 | 실제 FlyWire 고유 이득 미확인; 복잡도/지연; 전뇌 성적 낮음 | 대조 실험 방법을 계승, connectome은 기본에서 제외하고 이득이 입증될 때만 추가 |
| v8 | 원본 게임/API 보존, 제출 격리 검증, 실행 인프라 | self-ID 손실·동일 행동 집중, 목표 능력 부족 | 공개 routing 보존, tensor-only planner, 상태 생명주기·ONNX 초기 검증 |
| v9 | 작은 map/entity Attention, 독립 sampling·상대 다양성·bounded return | 순식간에 바뀌는 방향, 수집·반납 부족; history/PFSP 도입의 실력 향상 미입증 | Attention을 목표/역할 문맥의 선택기로 사용, 옵션 단위 탐색·완료 행동 평가 |
| v10 | actor와 독립 map critic, PopArt·batched target·source lock | critic 변경만으로 행동 개선 미입증; 강한 상대 step 노출 부족 | critic은 재사용·대조, step 및 경기 수를 함께 제어하는 curriculum |

위 과거 실패 수치의 근거는 [실험 이력](../../common/history.md), [v2](../../v2/reports/mappo_vs_win70_v2_plan_changes.md), [v4](../../v4/reports/mappo_planner_residual_v4_plan_changes.md), [v6](../../v6/reports/mappo_planner_residual_v6_plan_changes.md) 문서다. 현재 리그로 해당 과거 원인을 새로 실험 검증한 것은 아니다. “각 버전의 장점”에는 좋은 실행·검증 방법도 포함하며, 모든 구조가 실력 향상을 입증했다는 의미는 아니다.

### 현재 v10 결과의 추가 확인

[mapcritic_v1/summary.json](../../../logs/v10/mapcritic_v1/summary.json)은 `complete=true`, `local_runner_improved=false`, `map_information_causality_verified=false`다. frozen test는 C1 60/240승, C0 47/240승, initial 58/240승이다. C1 25%라는 점 추정만으로 개선 판정을 뒤집지 않는다.

본학습 6개 run의 target 상대 step 비율은 0.892~1.656%였다. C1 seed 11은 1,096,887 step 중 9,789 step이다. 상대를 episode 단위로 고르게 선택해도 약한 상대와의 긴 경기가 update를 지배한다. 따라서 v11에서는 상대군별 **경기 수·환경 step·전략 결정 수·학습 가중치**를 모두 기록한다. 선택 모델의 성적만 보지 않고 학습 seed별 분산을 보존한다.

## 5. 설계로 넘기는 결론

강점의 중심은 현재 증거상 **개체 식별을 가진 목표·경로·역할 제어**다. Attention과 지도 critic은 이를 개선할 학습 기반으로 채택하고, 장기 행동을 매 frame 방향 샘플링만으로 다시 배우게 하지 않는다. “planner의 기존 능력 보존 → 목표·전략의 유효한 개선 → 같은 행동을 ONNX로 재현 → 강한 상대·양 진영·두 결과 기준에서 검증” 순서로 진행한다.

상위 5개가 모두 5:5인 현재 결과를 뛰어넘는 것은 앞으로 증명할 목표다. 구체적인 모델은 [설계](../plans/v11_design.md), 작업·예산·통과 기준은 [구현 계획](../plans/v11_implementation_plan.md)에 명시한다.
