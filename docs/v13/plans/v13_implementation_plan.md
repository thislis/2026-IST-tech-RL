# v13 구현·검증 계획

작성일: 2026-11-02. [설계](v13_design.md)의 구현 순서와 채택 기준이다. **2026-11-03 구현 반영:** 코드·실행기와 게임 없는 검사를 추가했다. 실제 파일·계획 대비 차이·남은 실측 항목은 [구현 보고서](../reports/implementation.md), 실행 명령은 [v13 안내](../README.md)를 따른다. 아래 표는 원 설계의 연구 단계이며 실제 학습·리그·Unity 실행은 하지 않았다.

## 1. 파일 배치

[루트 README](../../../README.md), [배치 규칙](../../common/project_layout.md)을 따른다. 실제 파일을 넣을 때 디렉터리를 만들며 빈 v13 코드/로그 디렉터리를 미리 생성하지 않는다.

| 경로 | 계획한 내용 |
|---|---|
| `code/v13/blackout_v13/contracts.py` | 공개 입력, feature/state schema, 시간·진영·slot 계약 |
| `code/v13/blackout_v13/history.py` | 64 step 버퍼, lag read, mask, 시간 gap, reset |
| `code/v13/blackout_v13/geometry.py` | semantic decode, 합법 경로 거리, ONNX tensor 경로 |
| `code/v13/blackout_v13/tasks.py` | 왕복 작업, 후보, source ID와 capacity |
| `code/v13/blackout_v13/features.py` | 왕복 ETA·경합·전환 비용·실행 통계 |
| `code/v13/blackout_v13/assignment.py` | 공동 배정 분포·MAP·sampling |
| `code/v13/blackout_v13/manager.py` | 배정 actor와 temporal encoder |
| `code/v13/blackout_v13/worker.py` | 목표 조건부 이동 actor, optional GRU |
| `code/v13/blackout_v13/policy.py` | 두 actor, task/FSM, state port 조합 |
| `code/v13/blackout_v13/critic.py`, `ppo.py`, `rollout.py` | 팀 critic, event/sequence PPO, 버전별 rollout |
| `code/v13/blackout_v13/collector.py`, `runtime.py` | 완료 slot 즉시 재배정, chunk 경계 동기화 |
| `code/v13/blackout_v13/export.py` | 관측부터 행동까지 full ONNX와 parity 검사 |
| `code/v13/configs/` | 기본 설정, ablation, 상대군, seed/map split |
| `code/v13/scripts/`, `code/v13/tests/` | 수집·학습·평가·export 진입점과 의미 있는 계약 검사 |
| `code/v13/run_v13.ps1`, `run_v13.sh` | 구현 후 추가할 실행기; check/dry-run/status/stop/resume |
| `logs/v13/<run>/` | 등록 manifest, 진행·평가·자동 보고서, profiler 결과 |
| `artifacts/data/v13/<dataset>/` | teacher/task/관측 데이터, source와 schema 해시 |
| `artifacts/checkpoints/v13/<run>/` | 분리된 Manager·Worker·critic·optimizer 및 선택 checkpoint |
| `artifacts/submission/v13/<run>/` | 최종 `meta.json`, `model.onnx`, 검증 결과 |
| `docs/v13/reports/` | 사람이 작성한 실험 해석과 구현 차이 기록 |

v12 구현은 reference로 읽고 고정 baseline을 보존한다. 공통 `project_paths.py`에는 실제 v13 패키지가 생길 때 검색 경로를 등록한다. 이전 실험의 source/config hash를 고치거나 기존 run 이름으로 다른 모델을 이어서 학습하지 않는다.

## 2. 구현 순서와 통과 조건

| 단계 | 작업 | 다음 단계로 가는 조건 |
|---|---|---|
| P0 계약·기준선 | 최신 게임/API/teacher 해시, 점수 판정, 관측 시각, task schema 고정 | v12 bundle을 재해석 없이 baseline으로 읽고 새 실험 split을 등록 |
| P1 제출 가능성 | 최소 history+dummy 두 actor+후보/거리/DP/상태의 end-to-end ONNX | Python/ONNX parity, 상태 전달, 캐시 miss 포함 지연 측정; 해결되지 않은 100ms 초과가 있으면 구조 재검토 |
| P2 기억과 제어 | lag MLP, no-memory MLP, GRU, 양자화 Worker 구현 | 정확한 lag/리셋, 교대 진영, 목표 crop/거리장, 행동 확률 검사 |
| P3 모방 초기화 | 새 teacher 실행 데이터와 task/action label 수집 | episode 단위 분리, teacher 재생 확인, 새 상태에서의 Worker 실패 진단 |
| P4 이동 학습 | Manager 동결, Worker 학습 | 실제 왕복 완주·정체·점수 성능이 사전 등록한 dev 기준을 충족 |
| P5 배정 학습 | Worker 동결, ETA 통계 보정, Manager 학습 | old log-prob 재현, 실제 Worker 비용과 feature 의미 일치 |
| P6 교대 학습·대조 | actor 하나씩 고정해서 교대 update | 기억/GRU 효과와 train seed 분산 분리, dev에서 모델 선택 |
| P7 확정 평가·제출 | 선택 모델 동결 후 held-out 대결 및 공식 실행기 검증 | 성능·실행 한도·오류 복구 조건을 모두 통과 |

P1은 모델 성능 검증이 아니라 **비용 계산을 포함한 제출 경로의 실현 가능성 확인**이다. 경로 근사를 바꾸면 dataset/feature schema도 바뀌므로 P3 전에 결정한다. full ONNX가 없으면 산출물 이름에 `research_only` 또는 `partial`을 표시하며 공식 제출 완료로 기록하지 않는다.

## 3. 핵심 검증

### 기억·상태

- synthetic 증가 sequence의 t=0/4/8/16/32/64/65에서 정확한 lag와 startup mask를 확인한다. 이후 200 step 이상에서 shift/ring 구현 결과를 비교한다.
- 두 episode, 두 팀, 여러 환경을 섞어도 이력이 교차하지 않는다. slot 순서 변화는 ID로 정렬하고 유효 행을 복원한다.
- 개별 respawn proxy, class 변신, 목표 변경, 누락 slot 복귀, timeout gap, 서버 reset을 서로 다른 사건으로 처리한다.
- 그래프 호출 간에는 state 포트만 전달해 Python 객체 없이 다음 행동이 일치하는지 확인한다. 누락 slot의 stale 예약으로 용량이 초과하지 않는지 검사한다.
- token 128차원과 상태 메모리 크기는 실제 export shape로 다시 산출한다. 상태 포트가 큰 경우 전송 시간도 profiler에 포함한다.

### 배정·제어·학습

- 작은 문제에서 전수 열거와 DP의 partition/MAP/sample 분포를 비교한다. capacity=2 task 복제에 따른 중복 counting이 없고 all-invalid 상태에서 NaN이 발생하지 않아야 한다.
- 후보 순서가 바뀌어도 semantic task ID와 귀환지가 유지된다. 신규 예약과 기존 변신 예약의 double subtraction을 검사한다.
- Team B 대칭을 관측·history·goal·거리장·action 모두에 적용해 동치성을 확인한다. 불완전한 지도·unreachable·diagonal corner의 reference 결과를 비교한다.
- 32방향 양자화만 적용한 planner를 따로 평가하여 신경망 학습 전의 방향 근사 손실을 분리한다.
- frozen 파라미터로 수집 rollout을 재생하면 old/new log-prob가 일치한다. Manager 비결정 frame에는 policy loss가 없고, Worker는 실제 실행 action index로 학습한다.
- event 길이가 다르거나 terminal/chunk가 겹치는 예제에서 duration-aware return과 bootstrap을 수작업 계산과 비교한다. score 승자와 raw reward 승자가 반대인 사례를 포함한다.
- recurrent sequence의 burn-in, loss mask, hidden detach, 현재 raw-history 재인코딩을 확인한다. 지도 모방 데이터와 PPO on-policy rollout을 섞어 같은 ratio로 계산하지 않는다.

### 제출 경로

표준 ONNX 연산과 공식 loader의 I/O 검사를 통과하고, 같은 공개 입력·초기 상태·여러 step 재생에서 reference와 행동/상태를 비교한다. argmax 동률이나 float 오차는 tolerance와 tie-break를 문서화한다. 지도 갱신·동시 재배정·첫 호출·재시작을 포함한 지연 분포를 측정한다. 권장 여유 목표는 팀 전체 p99 80ms 이내이나, 이는 공식 100ms 제한을 위한 개발 목표이며 보장값이 아니다. 관측된 최대값과 timeout 비율도 반드시 보고한다.

## 4. 경기 수집과 PPO 정책 버전

[공통 학습 실행 원칙](../../common/training_runtime.md)에 따라 한 경기 종료 시 해당 slot에 다음 경기를 즉시 배정한다. 관전용 느린 경기나 긴 episode를 기다려 전체 wave를 시작하지 않는다.

초기 구현은 **고정 길이 chunk 경계에서만 동기화하는 on-policy PPO**로 제한한다.

1. collector는 동결한 `(manager_version, worker_version, cost_schema_version)`으로 경험을 모은다. 한 episode가 끝나면 같은 버전으로 다음 경기를 시작한다.
2. 수집 목표에 도달하면 진행 중 slot은 최대 현재 chunk 끝까지만 모은다. 자연 episode 종료까지 기다리지 않는다. 이 대기는 `ppo_update_boundary`로 기록한다.
3. 같은 버전의 rollout만 update에 사용한다. 종료되지 않은 episode의 마지막 상태는 bootstrap하며 게임 자체를 임의 종료하지 않는다.
4. 갱신 actor를 배포할 때 temporal raw history와 burn-in prefix를 이용해 hidden을 재구성하고, task/FSM의 의미 상태는 유지한다. feature 보정기는 다음 수집 버전부터 바꾼다.
5. 새 버전 수집을 재개한다. 버전이 늦은 응답은 별도 보관하거나 폐기하며 새 PPO batch에 조용히 섞지 않는다.

이 방식은 learner update 때 짧게 멈추지만 ‘모든 경기 종료를 기다리는 수집’은 아니다. 이후 비동기 learner를 도입하려면 policy lag 보정과 별도 알고리즘 검증이 필요하다. CPU/GPU·worker 수는 새 모델로 실측하며, v12의 16 worker와 468.95 env steps/s를 v13 처리량으로 재사용하지 않는다. Windows helper는 숨김 창으로 실행하고 관전이 명시된 slot만 표시한다.

재개 checkpoint에는 actor/critic/optimizer, RNG, 버전, schema/config/build/API 해시, 수집량, split을 저장한다. Unity 중간 상태를 완전히 복원할 수 없다면 미완료 rollout을 폐기하고 새 episode로 시작한다. Python hidden만 복원해 게임도 복원됐다고 간주하지 않는다.

## 5. 대조 실험

기억 ablation은 같은 입력 정보·teacher·후보·학습량·train seed·평가 seed를 사용한다. 가능한 범위에서 파라미터 수를 맞추고, 관측 기억 범위와 계산량을 별도로 보고한다.

| arm | 배정 | 이동 | 시간 입력 | 답할 질문 |
|---|---|---|---|---|
| B0 | 동결 v12 그대로 | v12 규칙 제어 | v12 기존 상태 | 실제 강한 기준선 |
| B1 | v13, 현재 feature만 | 동일 규칙 제어 | 현재 | 새 배정 구현·학습 효과 |
| B2 | v13 | MLP/CNN Worker | 현재 | B1 대비 이동망 효과 |
| B3 | v13 | GRU Worker | 현재 + recurrent hidden | GRU만의 효과 |
| B4 | v13 | MLP/CNN Worker | `{0,4,8,16,32,64}` | GRU 없는 지수 간격 기억 효과 |
| B5 | v13 | GRU Worker | `{0,4,8,16,32,64}` + hidden | 추가 GRU의 순효과; 최종 후보 |
| B6 | B4와 같음 | B4와 같음 | `{0,13,26,38,51,64}` | 같은 6개 시점·64 step 범위의 근사 균등 간격 대조 |

B0와 B1은 architecture·학습 조건이 달라 단일 원인 비교가 아니다. B1~B6는 동일 파이프라인에서 바뀐 축을 기록한다. B3는 Manager에 lag 기억이 없고 Worker만 GRU다. B4/B5에서는 두 actor 모두 lag 입력을 받는다. 개선이 확인되면 **Manager만 기억 / Worker만 기억**을 추가하여 어느 쪽에 필요한지 분리한다.

추가 dev 실험은 `{0,1,2,4,8,16,32,64}`, 후보 폭 24/32/전체, 배정 주기 8/16/25, 방향 수 16/32다. 한꺼번에 조합을 늘리기보다 기본 비교가 통과한 축에만 예산을 쓴다. 최종 test를 보고 lag나 후보 규칙을 다시 고르지 않는다.

## 6. 초기 예산과 평가

아래는 실행 승인이나 측정치가 아닌 **초기 연구 예산 제안**이다. 처리량을 확인한 뒤 실행 전 resolved config에 동결한다. env step은 아군 5명이 함께 행동한 팀 환경 step으로 세며, 유닛 transition 수 5배와 구분한다.

- P0~P2: Unity 없는 계약/export 검사 후 소규모 실제 경기와 profiler. 공식 컨테이너 자원이 없으면 그 검증 항목은 미완료로 남긴다.
- teacher: 우선 최대 1,048,576 env steps. 서로 다른 맵·상대·진영을 포함하고 dataset 용량을 먼저 산출한다. 자기 플레이 양쪽 자료를 저장해도 env step을 이중 계산하지 않는다.
- pilot: arm당 262,144 env steps, 우선 train seed 1개. 학습 가능성과 명백한 회귀를 찾는 용도이며 승률 결론을 내리지 않는다.
- 본 비교 후보: train seed 3개. seed당 Worker 2,097,152 + Manager 2,097,152 + 교대 4,194,304 = 8,388,608 env steps를 잠정 상한으로 둔다. pilot/부모 학습량과 새 학습량을 따로 기록하고 전체 비용에 합산한다.
- dev: 상대 22개(기존 21개 + v12), seed 5개 × 진영 2 = checkpoint당 220경기. 평가 주기는 우선 1,048,576 training steps이며 train throughput과 평가 비용에 따라 사전에 조정한다.
- 최종 확인: 선택된 v13 모델 1개와 동결 v12를 같은 기존 상대 21개·새 seed 20개·진영 2에서 비교한다. 각각 840경기, 합계 1,680경기다. 별도 v13-v12 직접 대결 20 seed × 2 = 40경기를 더해 총 **1,720경기**다. v12-v12 mirror를 평가에 넣지 않는다.

학습 상대의 초기 구성은 scripted 9종, 확보된 상위 v4/v5/v6/v7, 동결 v12, 과거 v13 snapshot의 혼합이다. 상대 ID는 샘플링·분석 메타데이터일 뿐 actor 입력이 아니다. 고정 비중과 snapshot 갱신 규칙을 등록하고 weak 상대에서 진 결과를 계기로 특정 상대 전용 행동 규칙을 추가하지 않는다.

train/dev/confirmation seed 영역을 분리하고 실제 map hash 중복도 점검한다. 이번 풀리그 seed는 이미 설계에 참고했으므로 최종 test로 재사용하지 않는다. 진영 교대 쌍은 같은 환경 seed·초기 맵을 확인하고 하나의 통계 단위로 묶는다.

### 채택 기준

1. 기본 지표는 상대별 `(승 + 0.5×무)/완료경기`의 동일 가중 평균이다. v12와 같은 opponent-seed-side에서 계산한 차이를 비교하고, 진영 쌍을 유지한 seed-block bootstrap으로 95% 구간을 보고한다. 미완료 경기를 승리·무승부로 채우지 않으며 두 모델의 공통 완료 집합과 누락 편향을 함께 공개한다.
2. 최종 후보는 기존 상대군 macro 승점률 차이의 95% 구간 하한이 0보다 높고, 직접 대결 승점률의 구간 하한이 50%보다 높은 것을 채택 목표로 한다. 40경기는 큰 불확실성이 있을 수 있어 조건 미달은 ‘우위 미확정’으로 기록한다. 더 평가하려면 새로운 확인 세트를 사전 등록하며 유리해질 때까지 임의 반복하지 않는다.
3. scripted weak/default/special, 상위 학습 모델군, 진영별 성적을 별도로 보고한다. 최약 상대 승점률과 운반 완주율이 악화되면 macro 개선만으로 자동 채택하지 않는다. 실무상 허용할 회귀 폭은 확인 평가 전에 dev에서 등록한다.
4. proxy 진단은 실제 완료시간, 반납 성공, 운반 중 손실, 정체 시간, task 전환 수, 중복 목표, ETA 오차, 후보 누락률이다. 관측 proxy와 엔진에서 확인한 사건을 구분한다. 이것들을 승패 대신 최적화한 성능 주장으로 사용하지 않는다.
5. 공식 입력·상태·행동 계약, full ONNX, 팀 전체 시간/메모리 한도를 통과해야 제출 후보가 된다. B4와 B5가 비슷하면 지연·분산이 작은 쪽을 선택한다.

## 7. 아직 확인할 사항

관측 step의 실제 게임 시간 간격, 누락/timeout의 빈도, ONNX 경로 계산 비용, 새 Worker의 귀환 안정성, GRU의 순효과는 아직 측정하지 않았다. 사용자에게 전달된 발언의 원 구현도 없으므로 저장 token이 관측인지 embedding인지 hidden인지는 확정할 수 없다. 이 미확인 항목은 위의 구현·대조 순서로 해결한다.
