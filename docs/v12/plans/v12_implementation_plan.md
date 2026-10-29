# v12 구현·검증 계획

작성일: 2026-10-26. [설계](v12_design.md)를 구현하기 위해 작성한 사전 계획이다. 이후 Python 정책·학습·실행기를 구현했으며 현재 구현 범위와 가벼운 검사 결과는 [구현 기록](../reports/implementation.md)에 정리했다. 아래 장기 학습·대규모 평가·전체 상태 ONNX 검증은 완료 결과가 아니라 실행 후 확인할 계획으로 유지한다.

## 1. 구현 위치와 재사용 범위

| 예정 경로 | 책임 |
|---|---|
| `code/v12/blackout_v12/policy.py` | 현재 두 tensor I/O, stream 상태, task event, 최종 action |
| `code/v12/blackout_v12/tasks.py` | task identity·용량·commitment·후보 선택 |
| `code/v12/blackout_v12/features.py` | 64차원 schema·ETA·왕복 비용·unknown 처리 |
| `code/v12/blackout_v12/assignment.py` | log-partition, 정확한 sample, MAP, marginal |
| `code/v12/blackout_v12/navigation.py` | semantic decode, 거리장 캐시, waypoint·정체 회복 |
| `code/v12/blackout_v12/trips.py` | terminal 포함 trip ledger·ambiguous 사건 |
| `code/v12/blackout_v12/training.py` | event PPO·분리 critic·frozen episode wave |
| `code/v12/configs/`, `tests/`, `analysis/` | 동결 설정·필수 검증·비교 분석 |
| `logs/v12/<name>/` | 등록·실행·자동 보고서 |
| `artifacts/data/v12/`, `artifacts/submission/v12/` | 수집 데이터·최종 제출물 |
| `docs/v12/` | 사람이 작성한 설계·결과 해석 |

v11의 public I/O 검증, 반사 좌표, environment contract, 원자적 저장, 상대 adapter, seed/map 검증을 가능한 한 재사용한다. recurrent collector와 Attention encoder를 무조건 복사하지 않는다. 기존 버전 소스·checkpoint·등록 hash는 수정하지 않고 v12에서 새로 등록한다. frozen v4/v6는 해당 checkpoint의 실제 과거 planner/decoder까지 묶는다.

## 2. 단계와 중단 조건

### P0 — 데이터·계약·공동 배정의 정확성

구현 전에 task schema, feature 단위, prior, event 규칙, logprob 계약을 고정한다. 필요한 검사:

- 2~5유닛 작은 fixture에서 모든 배정 열거와 DP logZ/MAP/marginal 대조. 설계 문서의 29배정 예제를 포함한다.
- capacity 0/1/2/5, private hold, 모든 일반 edge 불법, 잠긴 작업, Carrier 예약, Hunter 변신, 동일 source 중복을 검사한다.
- 실제 Carrier 0·locked 변신 예약 1인 fixture에서 총 capacity 1과 기존 예약을 유지하고 신규 예약만 차단하는지 확인한다. 예약을 이중 차감해 유효 배정을 없애면 실패다.
- finite-difference/autograd gradient, old=new ratio=1, backward sample 확률, 강제 배정 logprob=0, all-invalid 내부 상태의 NaN 부재를 확인한다.
- 입력 유닛을 함께 permutation한 고유해의 배정 대응, 완전 동률 tie-break를 분리한다. 관측 행을 임의 순서로 넣는 공개 API 지원 시험과 혼동하지 않는다.
- 자연 terminal, 시계/진영 reset, Collector→Hunter/Carrier, respawn, 운반 중 성소 회피, 반납 실패·모호 사건을 검사한다.
- 24×24 경로 비용과 도달 여부를 같은 규칙의 reference에 대조한다. 문턱/대각선/성소/적 본진/크기·특수 효과/임시 blocked가 달라지면 캐시가 무효화되어야 한다.

입출력 오류·확률 불일치·잘못된 reset·NaN은 학습 전에 해결한다. 성능이 낮다는 이유와 구현 오류를 같은 gate로 처리하지 않는다.

### P1 — ONNX와 전체 비용을 먼저 확인

MLP+MAP DP부터 export한 뒤 semantic decoder·FSM·경로와 state를 연결한다. **실제 event가 없는 호출에서 후보/DP/거리장 재계산이 빠지는지** profiler와 graph로 확인한다. 단순 Where 마스크로 계산 결과를 버린 경우는 실패다.

같은 공개 관측 sequence를 PyTorch와 ONNX Runtime에 넣어 action·task·state·캐시 버전을 비교한다. FP32 연속값 초기 허용오차는 `atol=1e-5, rtol=1e-4`이며, 최종 task/action 불일치는 별도 재현·해결한다. 1회 forward뿐 아니라 reset·진영 전환·goal 완료·창고 실패를 포함한 10,000 호출 이상을 확인한다. 정적 입력 replay의 일치와 실제 closed-loop 경기의 일치는 따로 보고한다.

최종 지원 opset/provider와 package version을 고정한다. 공개 두 파일 로딩, 명시 state ONNX core를 각각 확인한다. sampled decoder가 선택되면 같은 RNG 계약까지 검증하며 deterministic MAP 그래프의 성공으로 대체하지 않는다.

경량 기준은 actor 6,273개(기본안 상한 10,000), checkpoint 목표 256 KiB, ONNX 목표 2 MiB, 팀 state/cache 목표 256 KiB다. 초과 시 원인을 공개하고 단순화를 먼저 시도한다. 정책이 느린 상태에서 장기 학습을 시작하지 않는다.

### P2 — 학습 없는 기반 제어 비교

같은 공개 계약·경로 제어·작업 점수식·고정 3+2 역할에서 아래 세 모델을 비교한다. `Q0`는 v4 checkpoint 자체가 아니라 **같은 v12 기반 코드에서 원인을 분리하는 대조군**이다.

| arm | 후보 | 배정 | learned score |
|---|---|---|---|
| Q0 | 편도 우선 제한 | 고정 slot 순차 | 없음 |
| Q1 | 왕복·경합 후보 | 고정 slot 순차 | 없음 |
| Q2 | Q1과 동일 | 공동 DP MAP | 없음 |

Q0→Q1은 후보 생성 효과, Q1→Q2는 동일 비용의 배정 효과다. Q0/Q1/Q2의 scorer 공식·실행 FSM·표집 상대·map pair를 맞춘다. 별도로 **실제 frozen v4/v6 실행 정책**을 측정해 port 자체의 손실을 확인한다. v11의 과거 league 성적은 존재하지 않으므로 만들어 넣지 않는다.

noop/random/collector에서 기본 수집·반납을 못하면 먼저 경로·목표·class·창고 동작을 수정한다. Q2가 Q1보다 실제 경기에서 낫지 않을 수 있으며, 이 결과를 감추고 neural 학습 이득으로 포장하지 않는다. 수정은 새 config/source hash로 등록한다.

### P3 — 작은 scorer의 본 비교

| arm | 공통 조건 | 다른 부분 |
|---|---|---|
| S | 왕복 후보, 6,273개 MLP, 고정 3+2, 같은 학습 자료/예산 | 순차 조건부 배정과 그 올바른 logprob |
| J | S와 같은 feature·후보·초기화·학습 예산 | 정확한 공동 배정 분포 |
| F | J와 같은 구조 | 고정 3+2 대신 변신 task·기회비용 |

S/J의 차이로 learned scorer 위의 공동 배정 이득을 확인한다. J/F로 역할 변경의 추가 가치를 확인한다. Q2/J로 학습된 점수의 기여를 확인한다. PPO update 뒤 후처리로 과거 행동을 바꾸지 않는다.

주변 대조는 필요할 때 하나씩 추가한다: 후보 16/24/32, ETA·위험 feature 제거, task 주기 10/25/50, Hunter 상한 1/3, critic 확장. 한번에 모든 조합을 실행하지 않는다. 후보 상한을 바꾸면 task padding과 DP graph도 함께 버전화한다.

### P4 — 후보 동결·추가 seed·최종 평가

dev로 arm·checkpoint·decoder·후보 상한·목표 주기를 한 번 선택한다. 아래 confirmation/test 결과로 seed나 decoder를 다시 고르지 않는다. 추가 seed는 재현성 진단이며 특정 seed의 성적을 골라 제출 모델을 바꾸는 단계가 아니다.

최종 후보가 no-learning Q2이면 학습 seed 진단을 억지로 만들지 않고 실제 난수·맵 반복으로 평가한다. 신경망의 기여가 없으면 Q2가 정식 경량 후보가 될 수 있다.

## 3. 단계별 학습 예산

모델 경량성과 별개로 불필요한 장기 학습을 줄인다. 아래는 P0–P2를 통과한 뒤의 **최대 기본 계획량**이다. 학습을 시작했다는 뜻이 아니다.

| 단계 | 구성 | env step |
|---|---|---:|
| 작업 데이터·초기화 | 공통 공개 관측 수집 | 262,144 |
| pilot | S/J/F × seed 11 × 131,072 | 393,216 |
| 본 비교 | S/J/F × seeds 11/22/33 × 1,048,576 | 9,437,184 |
| 추가 seed | 선택 arm × seeds 44/55 × 1,048,576 | 2,097,152 |
| 합계 | 평가·벤치마크 제외 | **12,189,696** |

본 비교는 공통 초기화에서 독립적으로 시작한다. pilot을 이어받는다면 실제 누적 노출이 달라지므로 예산/등록을 수정해야 한다. 자연 종료 wave의 초과량·실패/폐기 episode·평가 step은 별도 합산한다. no-learning 후보가 선택되면 불필요한 추가 학습을 생략하고 실제 사용량을 기록한다. 이 계획량을 v11보다 학습 효율이 좋다는 실증 결과로 해석하지 않는다.

optimizer 초기값은 actor lr `1e-4`, critic lr `3e-4`, PPO clip `.2`, 4 epochs, grad norm `.5`, target KL `.02`로 v11 비교값을 사용한다. minibatch는 유효 event 수로 정하고, 긴 경기/특정 상대가 loss를 과점하는지 집계한다. joint entropy 구현 전에는 계수 0으로 시작하고 실제 sampling으로 탐색한다. event마다 temperature와 legal support를 기록한다.

## 4. 평가 집합과 규모

### 공통 원칙

- train/dev/confirmation/test는 별도 초기 map hash·seed manifest를 사용한다. 같은 명목 seed만으로 다른 초기 배치가 같다고 가정하지 않는다.
- 모든 비교 모델은 같은 opponent·초기 배치·양 진영 pair를 받는다. wall hash 하나만 보고 seed 누출로 판단하지 않고 배터리/창고/전체 초기 관측을 함께 본다.
- 주 평가 대상은 기존 고정 벽과 새로운 배터리/창고 배치다. 5종 창고 조합 중 일부 보류 진단을 별도로 둔다. 새 벽에 대한 일반화는 경로 단위 synthetic 검증과 구별한다.
- matchup별 score `(W+.5D)/N`, 진영별 성적, 전략 가족 macro 평균, raw-reward/관측-score 각각의 결과를 보고한다.
- 신뢰구간은 두 진영을 함께 묶은 map-pair bootstrap으로 만든다. 같은 맵을 여러 상대에 재사용하면 그 map ID를 공동 cluster로 묶는다. 여러 학습 seed 결과는 seed와 map cluster를 함께 보거나 seed별 결과를 병기한다. 200경기를 독립 학습 반복 200개로 세지 않는다.

### 기본 평가량

dev 핵심 상대 8개는 noop, random, collector, rush, scripted-weak, v4, v6, v10으로 시작한다. 정확한 registry ID/checkpoint/source/decoder hash를 동결한다. v7 계열은 최종 전체 리그와 선택적 강자 진단에서 확인한다.

| 평가 | 구성 | 경기 수 |
|---|---|---:|
| Q0/Q1/Q2 dev | 3 × 8상대 × 6 map pairs × 2진영 | 288 |
| frozen v4/v6 dev 기준 | 각자 자신 제외 7상대 × 6 pairs × 2진영 × 2모델 | 168 |
| S/J/F 본학습 dev | 3 arms × 3 seeds × 8상대 × 6 pairs × 2진영 | 864 |
| confirmation | 선택 arm의 5 seeds × 핵심 5상대 × 10 pairs × 2진영 | 500 |
| 선택 checkpoint 최종 | 기존 21상대; v4/v6는 각각 30 pairs, 나머지는 10 | 500 |
| 최종 frozen v4 기준 | 같은 공통 상대; 자기 자신 제외, v6만 30 pairs | 440 |
| 합계 | pilot·기능 smoke·선택 실험 제외 | **2,760** |

confirmation 핵심 상대는 noop/random/collector/v4/v6다. test 비교의 공통 상대는 v4 자기 자신을 제외한 20개로 맞추고, 별도로 v12 대 v4 결과를 보고한다. 기존 21상대의 raw 평균과 이 공통 상대 평균의 분모를 섞지 않는다. v11 checkpoint가 사용 가능하면 추가 pair로 비교하되 위 기존 21개 리그 분모에 몰래 추가하지 않는다.

**비퇴행의 주 추정량은 공통 20상대 각각의 paired 승점률 차이를 동일 가중 평균한 opponent-macro 차이**다. v6 경기 수가 많아도 가중치는 1/20이다. bootstrap마다 map cluster를 재표집하고 이 추정량을 다시 계산한다. 경기 수로 가중한 pooled 평균과 전략 가족 macro는 보조 지표로 구분한다. 강자 개선의 v4/v6 합산은 두 상대에 같은 가중치를 주고, confirmation 기본 능력 기준은 상대별·학습 seed별로 계산한다.

후보가 Q2이면 confirmation의 ‘5 training seeds’ 대신 평가 action RNG가 있는지에 맞춰 별도 manifest를 만든다. 결정론적 동일 모델을 5개의 독립 학습 모델로 표기하지 않는다. 추가 경기량은 별도 공개한다.

## 5. 사전 채택 기준

아래 문턱은 이번 설계의 제안값이다. 실제 구현을 시작할 때 등록하고 결과를 본 뒤 낮추지 않는다. 경기 수가 적어 신뢰구간이 넓으면 미확정으로 보고하며, 사전 등록한 별도 확장 평가 없이 유리한 결과가 나올 때까지 반복하지 않는다.

| 구분 | 기준 |
|---|---|
| 계약 | 지원 입력에서 오류/NaN/잘못된 capacity/상태 오염 0, logprob·ONNX 핵심 검증 통과 |
| 기본 능력 | confirmation의 noop/random/collector 각각 승점률 ≥90%, 각 진영 ≥80%; seed별 실패 공개 |
| 동급 경량 후보 | 최종 공통 상대에서 v4 대비 raw 승점률 차이의 95% 하한 ≥−3%p |
| 경제 비퇴행 | 같은 공통 상대 관측 score 승점률 차이의 95% 하한 ≥−3%p; terminal 유효성/누락 별도 보고 |
| 강자 개선 주장 | v4/v6 대상 합산 승점률 ≥55%이고 map-pair 95% 하한 >50%; 두 상대·두 진영 모두 별도 공개 |
| seed 재현성 | 선택 arm 5개 학습 seed의 분포/최악값 공개; 적어도 4개가 기본 능력 기준 통과 |
| 경량성 | 파라미터/파일/상태 예산 및 같은 호스트 v4·v11 대비 p50/p95 감소; 경기 처리량 악화 ≤5% |

v11 checkpoint/동작 환경이 없어 지연 비교를 할 수 없다면 미측정으로 남긴다. v4 대비 경량 여부는 확인할 수 있어도 v11보다 실제 빠르다고 주장하지 않는다. 각 비퇴행 기준은 별도로 확인하며 raw reward 이득으로 score 퇴행을 상쇄하지 않는다.

강자 개선 기준을 못 넘고 비퇴행·경량 조건만 넘으면 **‘동급 경량 후보’**다. 개선을 확정하지 못했지만 구간이 넓은 경우는 **‘미확정’**이다. 실패한 arm도 보존한다. score 또는 실행 계약이 잘못된 경기를 삭제해 분모를 줄이지 않고 실패율/원인 및 보수적 결과를 함께 낸다.

## 6. 새로 남길 계측

| 영역 | 필수 필드/질문 |
|---|---|
| 후보 | 전체 합법 수·선택 수·pruning 사유·넓은 후보 reference 대비 누락/점수차 |
| 배정 | task ID, eligibility/capacity, baseline/learned score, 전체 배정, joint logprob, DP logZ, 순차 대비 효용 차 |
| 학습 기여 | learned scorer 0일 때와 배정이 다른 비율, 그 상태의 실제 후속 trip/승패 |
| trip | 시작·pickup·귀환·완료/실패·중단 시각, 목적 창고, 실제 소요 step, 보유 상태, ambiguous 사유 |
| 경제·역할 | 점수차 시간계열, class별 인원, 변신/예약, 수집 포기 시간, 보유물 손실 proxy |
| 위협 | ETA 범위, 실제 경쟁 도착 관측, class unknown, 예측 신뢰도와 선점 실패 |
| 실행 비용 | decode/feature/path/MLP/DP 시간, solve한 field 수, relax iteration, cache hit, event 비율 |
| 결과 | runner raw 승자, terminal 직후 관측 score 및 유효 flag, 마지막 비종료 score, 실제 종료 원인 |

trip 수는 경기당과 1,000 simulation step당 모두 낸다. terminal 직후가 reset된 경우 마지막 비종료 점수를 최종값으로 확정하지 않고 reset/unknown으로 표시한다. Battery→빈손과 점수 상승만으로 여러 유닛의 반납을 분배하지 않는다. 명확/모호 사건을 나누고 score 유효 경기만의 결과를 전체 경기의 결과처럼 쓰지 않는다.

프로파일링은 같은 호스트, batch=팀5, threads=1 기본에서 warm-up 제외 10,000 호출 이상으로 하고, 2 threads는 별도 표로 낸다. no-event, decision, cache miss, reset/cold-start의 p50/p95/p99/max를 나눈다. 실제 경기에서는 worker 수를 고정해 v4/v11/v12를 비교한다. 입력 map decode와 host↔device 이동도 포함한다.

## 7. 최종 산출물

- 등록 manifest: source/config/feature schema/teacher·opponent hash, objective ID, train/dev/test map 집합, 선택 규칙.
- 비교 보고서: 후보 생성·배정·학습·변신 효과를 분리한 표, 모든 seed와 양 진영 결과, raw reward와 score 진단.
- 비용 보고서: 실제 parameters, checkpoint/ONNX/state 크기, event별 지연·전체 처리량·경로 solve 수.
- 제출 후보: `policy.py`, `checkpoint.pt`, 별도 개발 ONNX와 state manifest, 전체 sequence parity 기록.
- 결정 기록: ‘개선’, ‘동급 경량’, ‘미확정’, ‘기각’ 중 결과에 맞는 판정과 남은 한계.

위 채택 기준의 완료는 파일 생성과 구분한다. 이번 구현 작업에서는 사용자의 요청에 따라 짧은 합성 검사와 자원 측정만 수행하고 실제 학습·Unity 경기는 실행하지 않았다. 전체 상태 ONNX, 장기 경기 성능 및 정식 동결 상대 비교는 미검증이며 현재 구현 기록을 기준으로 판단한다.
