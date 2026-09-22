# MAPPO 및 connectome v1–v7 발전 과정

> 2026-09-22 갱신: v7 본실험 23개 실행이 각 2,000,000 step을 완료했고,
> 고정 target 상대 dev 평가 1,380경기도 완료됐다. 실제 FlyWire와 재배선 대조군은
> 모두 승률 50%로, 생물학적 배선의 우월성은 확인되지 않았다. 최종 test는 미실행이다.
> 실행 근거는 [`logs/v7/main_study/accelerated/summary.json`](logs/v7/main_study/accelerated/summary.json),
> 사전 등록 범위는 [`reports/v7/main_study_preregistration.md`](reports/v7/main_study_preregistration.md)에 있다.

## 목표와 공통 평가 기준

최종 목표는 고정 상대 `checkpoints/win_70_vs_scripted.pt`를 상대로 MAPPO 정책을
학습해, dev seed 5개를 양 진영으로 바꿔 치르는 10경기에서 승률 85% 이상을
달성하는 것이다. 10경기에서는 최소 9승이 필요하며 무승부는 승리로 계산하지
않는다. v1~v5는 decentralized actor와 centralized critic을 사용하는 MAPPO
CTDE를 기반으로 한다. v6는 centralized critic을 유지하되 팀 전체의 수정 행동을
하나의 분포에서 선택하므로 엄밀한 decentralized actor 구조와는 구분한다.

v6에서는 dev 9/10 외에 별도 confirmation seed 15개를 양 진영으로 평가한
30경기에서 최소 26승, 진영별 최소 11/15승을 최종 승급 조건으로 사용한다.
train/dev/confirmation/test seed는 분리하며 test 결과는 모델 선택에 사용하지 않는다.

v7은 같은 고정 target 상대를 사용하되, 구조 비교를 위해 고정 예산과 새 dev seed
41000–41029의 양 진영 60경기를 사용한다. 각 실행의 2,000,000-step latest를 평가하며
dev 성적으로 중간 checkpoint를 선택하지 않는다. v7의 승률은 v1~v6의 10경기와
평가 맵·표본 수가 다르므로 직접적인 세대 간 성능 향상 수치로 해석하지 않는다.

핵심 발전 흐름은 다음과 같다.

> 직접 PPO(v1) → 에피소드 수집 정상화(v2) → planner 모방(v3) → planner 보존형
> residual(v4) → planner-conditioned 안전 탐색과 롤백(v5) → 팀 단위 행동 선택과
> 탐색 확률 보장·실패 seed 재표집(v6) → 고정 예산의 connectome residual 대조 실험과
> 전뇌 특징 기반 PPO 직접 제어(v7)

## 한눈에 보는 결과

| 세대 | 핵심 접근 | 이전 문제에 대한 해결 | 실제 장기 실행 결과 | 다음 세대로 넘긴 문제 |
| --- | --- | --- | --- | --- |
| v1 | 기존 self-play actor를 `win70` 상대에게 직접 MAPPO 학습 | 최초의 고정 상대 fine-tuning 기준선 구축 | 2,000,384 step, 완료된 학습 경기 0개, 평가 0/10 | rollout마다 환경을 초기화해 terminal 보상을 전혀 학습하지 못함 |
| v2 | 진영별 persistent collector와 2,048-step rollout | rollout 경계를 넘어 경기 상태와 terminal 보상을 보존 | 2,000,896 step, 1,643경기 전패, 평가 0/10 | 상대의 실력은 neural actor가 아니라 checkpoint의 planner override에서 나왔음 |
| v3 | DAgger·teacher forcing·BC와 단계별 상대 curriculum | planner 행동을 neural actor에 증류한 뒤 PPO로 개선 시도 | 3,000,320 step, 최종 평가 0/10, 점수 차 -94.1 | 단일-step 모방 오차 누적, NoOp 편향, 실패 단계 강제 승격, 성능 회귀 |
| v4 | action 0은 planner 유지, 1~8은 방향 override인 planner residual | 불완전한 planner 복제 대신 검증된 planner 성능을 그대로 보존 | 100,352 step, scripted 8/10·target 5/10 그대로, 첫 단계에서 안전 중단 | warm-up이 fallback 정답만 BC하고 PPO도 꺼져 있어 개선 신호가 없었음 |
| v5 | planner 문맥 기반 상대 방향 residual, 한 step당 override 1개, PPO·확인 평가·롤백 | planner를 보존하면서 제한된 수정 행동을 실제 보상으로 학습 | 317,440 step, scripted 8/10·target 5/10, 롤백 1회 후 단계 상한에서 중단 | PPO override가 평균 0.184%에 그쳐 정책이 planner에서 벗어나지 못함 |
| v6 | 팀 단위 41-way residual, 명시적 탐색 하한, 실패 seed 재표집 | 실행 행동과 PPO 확률을 일치시키고 B 진영·실패 경기의 학습 비중 확대 | 1,159,168 step, 최종 target 5/10, 롤백 2회 후 full_win70 단계 상한에서 중단 | 탐색은 증가했지만 best는 초기 baseline 그대로이며 성능 악화가 반복됨 |
| v7-1 | 실제 FlyWire 부분회로, 재배선, MLP, 파라미터 수 대응 MLP 비교 | 동일 planner·encoder·학습 예산에서 residual 구조의 영향을 분리 | 4모델 × 5시드 × 200만 step 완료; dev 승률 A0 15.33%, A1 21.67%, A2/A3 각 50% | 실제 배선과 재배선의 경기 기록이 같아 배선 고유의 이득이나 planner 대비 개선을 입증하지 못함 |
| v7-2 | 고정 MaleCNS 전뇌 특징 + PPO 출력층, 한 유닛 직접 제어 | 제어 슬롯의 planner 우회 없이 감각→전뇌→행동 경로를 실행 | 3시드 × 200만 step 완료; dev 32승 148패, 승률 17.78% | 전뇌 재배선·CNN/GRU 대조군과 최종 test가 없으며 전뇌 배선의 우월성을 판정할 수 없음 |

## v1 — 직접 MAPPO fine-tuning

### 구현과 동작

- `phase3_selfplay_gen_08.pt`의 actor를 초기화해 고정된 full `win70` 상대와 직접
  MAPPO로 학습했다.
- 학습 진영을 update마다 교대하고, 512 environment step마다 rollout을 닫아 PPO를
  업데이트했다.
- reward는 점수 변화, terminal 승패, Unity shaping을 결합했다.
- 구현 당시 학습률은 `3e-4`, entropy coefficient는 `0.01`이었다.

### 결과와 문제

- 3,907회 update, 총 2,000,384 step을 수행했지만 학습 로그의 완료 에피소드는
  끝까지 0개였다.
- 한 경기가 보통 770~1,133 step 이상 걸리는데 512 step마다 환경 자체를 폐기해,
  경기 종료 전에 상태가 계속 초기화됐다. 따라서 승패와 terminal reward가 PPO
  데이터에 한 번도 들어가지 않았다.
- 최종 target 평가는 0승 10패, 평균 점수 차 `-96.2`였다.

**v2로 이어진 결론:** 모델 구조나 하이퍼파라미터보다 먼저 episode continuity를
보장해야 했다.

## v2 — persistent collector로 수집 계약 복구

### v1 문제를 해결한 구현

- A/B 진영별 Unity 환경과 collector를 update 사이에 유지해 rollout 경계에서도
  진행 중인 경기를 보존했다.
- rollout을 512에서 2,048 step으로 늘리고, seed도 rollout이 아니라 완료된 경기
  단위로 순환시켰다.
- 8,192 step까지 terminal episode가 없으면 중단하는 watchdog을 추가했다.
- 초기 actor를 `win_70_vs_scripted.pt`의 neural core로 바꾸고 학습률을 `5e-5`,
  entropy를 `0.001`로 낮춰 pretrained policy 훼손을 줄였다.
- 체크포인트에 critic·optimizer뿐 아니라 진영별 seed cursor와 episode 통계까지
  저장해 정확히 재개할 수 있게 했다.

### 결과와 새로 드러난 문제

- 수집 문제는 해결되어 2,000,896 step 동안 terminal episode 1,643개를 정상적으로
  관측했다.
- 그러나 학습 경기는 1,643전 1,643패였고, 최종 target 평가도 0승 10패,
  평균 점수 차 `-95.4`였다.
- 원인은 `win_70_vs_scripted.pt`의 강한 행동이 neural actor가 아니라 추론 시 적용되는
  `planner_override`에서 나왔기 때문이다. v2는 제출 가능한 순수 neural actor를
  만들기 위해 이 override를 복사하지 않았으므로, 실제로는 매우 약한 neural core만
  초기화한 셈이었다.

**v3로 이어진 결론:** planner를 제외한 채 결과 정책만 fine-tuning해서는 출발 성능을
재현할 수 없으므로 planner의 행동 지식을 actor에 전달해야 했다.

## v3 — planner DAgger 증류와 opponent curriculum

### v2 문제를 해결한 구현

- learner가 실제 방문한 상태에서 `3 worker + 2 guard`, chase radius 48 planner의
  행동 label을 수집하는 on-policy DAgger를 도입했다.
- 첫 약 100k step은 PPO를 끄고 teacher forcing과 behavior cloning으로 warm-up한 뒤,
  이후에도 replay buffer의 planner label을 BC 보조 손실로 계속 사용했다.
- 상대 난도를 `base scripted → weak win70 → full win70 혼합 → full win70 →
  historical mixture` 순으로 올렸다.
- 양 진영 평가, 상대 혼합 RNG, replay buffer, historical snapshot까지 checkpoint에
  저장했다.

### 결과와 새로 드러난 문제

- 3,000,320 step을 완주했지만 best와 final target 평가가 모두 0승 10패였다.
  평균 점수 차도 best `-86.7`에서 final `-94.1`로 악화됐다.
- replay label의 41.4%가 NoOp이었고 guard 슬롯의 NoOp은 각각 93.3%, 95.1%였다.
  warm-up replay 정확도가 82.3%여도 실제 closed-loop 행동 평가는 0/10이었다.
- planner의 한 step 행동을 높은 정확도로 모방해도 작은 분류 오류가 다음 상태를
  바꾸며 누적됐다. PPO 병행 후 replay 정확도도 49.7%까지 하락했다.
- gate를 통과하지 못한 단계도 최대 budget에 도달하면 다음 난도로 강제 전환했고,
  성능이 악화되어도 장기 실행을 계속했다.

**v4로 이어진 결론:** planner 전체를 neural actor가 다시 배우게 하지 말고, 검증된
planner를 기본 행동으로 고정한 채 더 나은 예외 행동만 학습해야 했다.

## v4 — planner-preserving residual과 fail-closed curriculum

### v3 문제를 해결한 구현

- 정책의 categorical action 의미를 바꿨다. action 0은 planner 행동을 그대로
  실행하고, action 1~8은 actor가 고른 절대 방향으로 planner를 override한다.
- residual head의 weight를 0, fallback bias를 4.0으로 초기화해 deterministic 정책이
  처음부터 planner와 동일하게 동작하도록 했다.
- 승률·평균 점수 차·양 진영 조건을 모두 만족해야만 승급하고, 단계 최대 budget까지
  실패하면 더 강한 상대에게 넘어가지 않고 `stage_blocked`로 중단하도록 바꿨다.
- 학습과 평가 Unity를 `background=True` 및 `-batchmode`로 실행해 게임 창이 앞으로
  튀어나오지 않게 했다. 시각 관측을 유지하기 위해 `-nographics`는 사용하지 않았다.
- 실측상 time scale 50 이후 처리량 증가가 없어 학습은 50, 병렬 평가는 100으로
  설정했다.

### 결과와 새로 드러난 문제

- 시작부터 planner baseline을 재현해 scripted 상대 8승 2패, target 상대 5승 5패를
  기록했다. v3의 0/10 붕괴는 해결했다.
- 하지만 51,200 및 100,352 step에서도 두 평가가 정확히 같았다. 첫 단계의 scripted
  승급 기준 9/10을 넘지 못해 100,352 step에서 안전 중단됐다.
- warm-up은 PPO를 사용하지 않았고 모든 BC 정답이 fallback action 0이었다. 즉,
  planner를 유지하는 법만 반복 학습했으며 planner보다 나아질 신호가 없었다.
- scripted baseline 자체가 8/10인데 gate가 9/10으로 고정돼 있던 점도 병목이었다.

**v5로 이어진 결론:** residual이 언제, 누구의, 어느 방향 행동을 수정해야 하는지
판단할 문맥과 제한적인 PPO 탐색이 필요했다.

## v5 — planner-conditioned residual, 제한 탐색, 확인 평가와 롤백

### v4 문제를 해결한 구현

- action 1~8을 절대 방향 대신 planner 방향 기준 `±45°`, `±90°`, `±135°`, 정지,
  반대 방향으로 정의해 planner 행동에 대한 상대적 수정으로 바꿨다.
- residual head에 planner action·방향, worker/guard 역할, path 유효성, 가장 가까운
  작업 대상 또는 적의 상대 방향·거리를 제공했다.
- 한 environment step에서 최대 한 agent만 override할 수 있게 하고, 여러 요청이
  있으면 fallback 대비 logit margin이 가장 큰 agent만 선택했다.
- 16,384-step planner 보존 단계 뒤에는 all-zero BC를 제거하고 PPO 보상으로
  override를 학습했다. fallback bias는 6.5, override penalty는 단계 진행에 따라
  줄어들도록 했다.
- 평상시 10게임 평가 외에 승급과 회귀 판단은 policy RNG가 다른 3개 replica,
  총 30게임으로 재확인했다.
- target-best와 stage-best를 분리했고, target 승률이 초기 baseline보다 10%p 이상
  낮아지면 재확인 후 model과 optimizer를 target-best로 롤백하도록 했다.
- 전용 background launcher를 추가하고 v4와 동일하게 학습 time scale 50,
  평가 time scale 100을 유지했다.

### 결과와 새로 드러난 문제

- planner 보존 확인 평가는 30게임 24승 6패로 통과했다.
- 116,736 step에서 target 성능이 10게임 2승 8패, 30게임 재확인 6승 24패로
  떨어지자 자동 롤백이 한 번 작동했고, 이후 초기 target 승률 50%를 복구했다.
- 최종 317,440 step에서 scripted 평가는 8승 2패, 평균 점수 차 `+21.5`였고 target
  평가는 5승 5패, 점수 차 `0.0`이었다. scripted 단계의 최대 budget을 소진했지만
  전체 90% 및 진영별 최소 70% gate를 못 넘어 `stage_blocked`로 정상 종료했다.
- 총 1,587,200 agent action 중 override는 2,764회였다. PPO 구간 평균 override율이
  약 0.184%에 불과해 deterministic 정책은 대부분 초기 planner와 같은 행동을 했다.
- scripted 평가에서 model-side A는 100%, B는 60%였고 물리적 진영 승률도 A 70%,
  B 30%로 편향되어 B 진영이 명확한 병목으로 남았다.

**v6로 이어진 결론:** 독립적으로 행동을 뽑은 뒤 한 agent만 남기는 선택 과정과
PPO의 확률 계산을 일치시키고, 탐색 하한과 B 진영·실패 seed 표집을 통해 실제로
수정 행동을 학습할 기회를 늘려야 했다. 평가와 승급 기준도 baseline에 맞게
재보정하고 서로 다른 map seed로 확인해야 했다.

## v6 — 팀 단위 residual, 탐색 확률 보장과 실패 seed 재표집

### v5 문제를 해결한 구현

- 행동을 `KEEP + 5 agent × 8 correction`의 팀 단위 41-way categorical로 바꿨다.
  한 step에 최대 한 agent만 수정하며, 실제로 표집한 혼합 분포의 log probability를
  저장하고 PPO 업데이트에서도 같은 분포로 다시 계산한다.
- 4,096-step planner 보존 단계 이후에는 BC 없이 PPO를 사용한다. 유효한 수정
  행동에 대한 명시적 탐색 하한을 단계별 15%에서 2%까지 낮추고 override penalty를
  제거했다. 이 비율은 팀 step 기준이며 v5의 agent action 기준 override율과 다르다.
- planner의 실제 목표·경로, 이동·정체 상태, 역할·아이템·적의 상대 위치와 시간
  문맥을 입력한다. 연속 방향을 기준으로 correction을 회전하고, 정지 중에도
  8개 방향을 구별하며 B 진영의 좌표와 행동 방향을 함께 변환한다.
- 관측의 시간 기준을 실제 420초 경기와 일치시키고 `GlobalLocalMapEncoder`의
  local crop Y축 오류를 수정했다. 기존 IPPO encoder의 crop 동작은 변경하지 않았다.
- 진영별 persistent collector를 유지하면서 update 비중을 A 40%, B 60%로
  조정했다. 학습 seed는 실패 점수의 이동 평균과 40% 균등 표집을 혼합해 선택한다.
  과거 PPO transition을 재사용하는 대신 실패한 seed의 새 경기를 수집한다.
- curriculum을 planner 보존 → balanced residual → target 혼합 → full target →
  historical 혼합으로 구성했다. 초기 gate는 측정한 baseline 대비 승률·점수 차
  허용 범위를 사용하고, 단계 budget 소진만으로 강제 승급하지 않는다.
- confirmation은 동일 map의 RNG replica 대신 별도 map seed 15개를 사용한다.
  target-best와 stage-best를 분리하고, dev와 confirmation 모두 같은 기준점보다
  악화된 경우에만 롤백한다. 최종 test 20경기는 승급한 모델의 보고용으로만 사용한다.
- score delta와 terminal 중심 보상, 긴 경기용 discount, value/gradient clipping,
  KL 조기 종료와 비정상 수치 검사를 적용했다. 고정 encoder의 feature와 압축된
  critic 입력을 저장해 rollout 메모리 사용도 줄였다.
- 기존 JSONL·summary·평가 JSON·checkpoint 방식을 v6 전용 경로로 유지한다.
  source/config snapshot과 hash, checkpoint별 불변 best 사본, 로그 offset 및
  crash 이후 로그 복구를 추가했다. 재개 시 Unity 경기는 새로 시작하며 중단된
  경기의 폐기는 기록한다. 따라서 중단 전 궤적의 bitwise 재현을 의미하지 않는다.
- background launcher와 사전 점검·재개 옵션을 제공하고, planner와 residual을
  모두 포함하는 두 파일 제출 export를 구현했다. 실행 명령은
  [README의 v6 학습 실행 방법](README.md#v6-training)에 정리했다.

### 결과와 현재 문제

- 전체 테스트 200개가 통과했으며 이 중 v6 테스트는 17개다. 합성 환경 기반으로
  PPO·재개·로그 복구·export 등을 확인했고 shell 문법 및 background `--check`도
  통과했다. 이 검증은 실제 Unity 경기나 학습 성능 검증을 대체하지 않는다.
- 구현 당시에는 학습을 실행하지 않았으며, 이후 사용자가 백그라운드 학습을 시작했다.
  실제 실행은 한국 시간 2026-09-08 16:21부터 09-09 11:52까지 약 19시간 31분 동안
  진행됐다. 실행 시간에는 baseline 평가 80경기와 이후 평가가 포함된다. 학습량은
  1,159,168 environment step, 566 update이며 이 중 PPO update는 564회였다.
- 4,096 step에서 planner 보존 단계를, 55,296 step에서 balanced residual 단계를,
  157,696 step에서 target 혼합 단계를 통과했다. 앞의 두 단계 scripted 확인 평가는
  각각 23/30, target 혼합 단계의 full win70 확인 평가는 15/30이었다.
- `full_win70` 단계에서 1,001,472 step을 사용했지만 승률 70% 및 진영별 최소 50%
  gate를 충족하지 못했다. `maximum_stage_budget_without_confirmed_gate` 사유로
  `stage_blocked` 종료했으며, 전체 4,000,000-step 상한 소진이나 목표 달성 종료는
  아니다. historical 혼합 단계에는 진입하지 못했다.
- 최종 target dev 평가는 5승 5패, 평균 점수 차 `0.0`이었다. model-side A는 3/5,
  B는 2/5로 B 진영의 열세가 남았다. target-best도 global step 0의 초기 baseline
  5/10으로 유지되어, planner 대비 평가 성능 향상은 확인되지 않았다.
- 567,296 step에서 target 확인 평가가 0승 30패, 평균 점수 차 `-44.8`로 떨어져
  첫 롤백이 작동했다. 925,696 step에서는 2승 28패, 점수 차 약 `-32.97`로 두 번째
  롤백이 작동했다. 두 번 모두 초기 target-best를 복원했고 최종 dev 50%를 회복했다.
  이 확인 평가들은 회귀 판단용이며 최종 모델의 confirmation 결과는 아니다.
- PPO 구간 1,155,072 team step에서 수정 행동 540,856회를 실행했다. team step 기준
  override율은 46.82%, agent action 기준은 9.36%로 v5의 약 0.184%보다 증가했다.
  `full_win70` 단계의 team override율은 50.28%였다. 탐색 부족은 완화됐지만 수정
  빈도 증가가 유효한 전략 개선으로 이어지지 않았고, 성능 악화도 두 차례 발생했다.
- 완료된 학습 경기는 226경기, 37승 1무 188패였다. 상대별로 scripted 34경기 중 6승,
  weak win70 8경기 중 4승, full win70 184경기 중 27승이었다. 학습 중 탐색과 상대
  혼합을 포함하므로 이 수치는 deterministic dev 평가 승률과 직접 비교하지 않는다.
- latest·target-best·stage-best checkpoint와 실행 로그는 저장됐다. 최종 dev 9/10 및
  confirmation 조건을 통과하지 못해 `mappo_win_85_vs_win70_v6.pt`와 `submission/v6/`
  산출물은 생성되지 않았고, 최종 test 평가도 실행되지 않았다.
- 제출 export의 독립 실행은 테스트했지만 stateful planner와 canonical batch 순서를
  공식 평가 환경이 허용하는지는 별도 확인이 필요하다. 순수 stateless Torch actor만
  허용된다면 추가 증류 또는 제출 구조 변경이 필요하다.

**현재 결론:** v6는 초기 단계 정체와 탐색 부족을 완화했지만 planner보다 좋은 수정
행동을 학습하지 못했다. 다음 개선에서는 탐색 빈도를 더 높이는 것만으로 해결된다고
가정하지 말고, 수정 행동의 장기 보상 기여와 학습·평가 행동 차이, B 진영 실패 및
롤백 직전의 정책 변화를 분석해야 한다. 세부 원인은 아직 확정하지 않았다.

## v7 — connectome residual 대조 실험과 전뇌 특징 기반 직접 제어

### v6 문제를 검증하기 위한 구현

- v6에서 수정 행동을 더 자주 탐색해도 성능이 개선되지 않아, residual의 연결 구조를
  바꾸는 실험과 planner를 거치지 않는 직접 제어 실험을 분리했다. v7-1은 기존
  팀 단위 `KEEP + 5 agent × 8 correction`, 고정 encoder와 planner를 공통으로 유지한다.
- v7-1은 A0 기존 MLP, A1 파라미터 수 대응 MLP, A2 차수 보존 재배선, A3 실제
  FlyWire FAFB v783 부분회로를 비교한다. 실제 부분회로는 205뉴런·7,820방향성 연결이며,
  그래프 residual의 가중치를 학습한다. A1/A3의 전체 학습 파라미터는 각각
  127,131/127,209개이고 residual head의 파라미터 차이도 등록 허용 범위 ±5% 이내다.
- v7-2는 MaleCNS v1.0의 166,700뉴런·25,582,938방향성 연결을 유지한 고정 전뇌
  시뮬레이터의 출력 특징을 사용한다. F1은 9-way PPO 출력층과 critic을 학습하며
  slot 0 한 유닛만 직접 제어하고 나머지 네 유닛은 고정 scripted 정책으로 움직인다.
  `fly64_proxy_v1` 동역학, action repeat 1, 가소성 비활성 조건을 사용한다.
- 게임 상태 입력은 `blackout-env`의 96차원 `vector`와 96×96×11 `graphic`에서
  가져온다. v7-1은 관측에서 만든 planner 문맥·이동 이력과 고정 제단 좌표 등 사전
  지식을 사용하며, v7-2는 semantic map과 자기 위치를 인공 망막 입력으로 변환한다.
  학습 critic은 상대를 포함한 10개 유닛의 관측을 모으므로 actor의 입력 범위와 다르다.
- 파일럿에서 전뇌 담당 유닛까지 scripted planner가 계획하던 경로 탐색 오류를 수정했다.
  본실험은 `fixed_scripted_active_slots_v2`로 전뇌 슬롯을 scripted 할당·이동 계산에서
  제외한다. 제어 슬롯에 planner fallback을 넣지 않았으며, 오류 수정과 복구 기록은
  별도 `teammate_v2` 실험으로 보존했다.
- 본실험은 파일럿 학습 가중치를 이어받지 않고 새로 시작했다. v7-1 시드
  11/22/33/44/55, v7-2 시드 11/22/33에서 각각 2,000,000 environment step을 고정했다.
  상대 혼합은 20만까지 scripted/weak/target 65/25/10%, 60만까지 20/30/50%,
  이후 10/0/90%다. 성적 기반 조기 종료·실패 seed 우선 표집을 사용하지 않고,
  v6의 명시적 탐색 하한도 0으로 설정했다. 따라서 구조 외에 학습 프로토콜도 v6와 다르다.
- 9월 21일 순차 실행을 저장 후 중단하고 독립 모델·시드를 최대 12개 병렬 실행하는
  가속 경로로 재개했다. 격리 Protobuf native 파서, 프로세스 내부 queue, 전뇌 CSR의
  MPS 연산을 적용했으며 전뇌 작업은 최대 2개로 제한했다. 등록 모델·예산·PPO 조건은
  유지했고, 짧은 수집 구간의 원본/가속 동등성 검사와 실행 해시·재개 이력을 남겼다.

### 결과와 현재 문제

- 한국 시간 2026-09-22 17:39에 가속 관리자 완료 상태가 기록됐다. 개별 실행 23개
  모두 `budget_complete`, 각 2,000,000 step으로 총 46,000,000 step을 완료했다.
  관리자 상태는 `complete`, `evaluated=23`, `failed=null`, `active=[]`다.
  기존 순차 관리자의 `stopped` 기록은 가속 재개 전 상태이며 최종 상태가 아니다.
- 각 실행의 예산 종료 checkpoint로 고정 target 상대 dev 30맵 × 양 진영을 평가했다.
  23개 결과 파일 모두 60경기씩, 총 1,380경기를 기록했다. 아래는 독립 학습 시드별
  승수와 전체 경기 합산이다. 모든 시드의 평가량이 같으므로 합산 승률은 시드 평균과 같다.

| 모델 | 시드별 승수 / 60경기 (11, 22, 33, 44, 55 순) | 합산 승·무·패 | dev 승률 |
| --- | --- | --- | --- |
| v7-1 A0 MLP | 10, 12, 14, 5, 5 | 46승 1무 253패 | 15.33% |
| v7-1 A1 파라미터 수 대응 MLP | 6, 13, 11, 15, 20 | 65승 6무 229패 | 21.67% |
| v7-1 A2 차수 보존 재배선 | 30, 30, 30, 30, 30 | 150승 0무 150패 | 50.00% |
| v7-1 A3 실제 FlyWire | 30, 30, 30, 30, 30 | 150승 0무 150패 | 50.00% |
| v7-2 F1 전뇌 특징 + PPO 출력층 | 8, 12, 12 (시드 11/22/33) | 32승 0무 148패 | 17.78% |

- A2와 A3는 다섯 시드 모두 승률뿐 아니라 각 맵·진영의 winner, 점수, 종료 step을
  포함한 `episodes` 기록이 동일했다. 같은 맵의 A/B 평가도 물리적 경기 결과가 같아
  모델 관점에서 1승 1패씩이었고, 단일 시드 paired-map bootstrap 구간은 모두
  `[0.5, 0.5]`였다. 이는 기록된 쌍별 승률이 일정하다는 뜻이며 일반화 불확실성이
  없다는 뜻이 아니다. 행동 궤적이나 KEEP 선택률은 이 결과 파일만으로 확정할 수 없다.
- A2/A3의 합산 model-side A 승률은 60/150=40%, B는 90/150=60%였다. 이 dev 맵
  집합에서는 B가 더 높아, v6의 다른 평가 맵에서 관측한 B 진영 열세를 그대로
  적용할 수 없다. A0/A1은 합산 승률이 낮고 시드별 변동도 있어 정책 변화 분석이 필요하다.
- 현재 수치에서는 실제 배선이 재배선보다 좋지 않으며 planner 대비 개선도 입증되지
  않았다. MLP보다 높은 dev 승률만으로 생물학적 배선 고유의 이득을 주장할 수 없다.
  학습 중 탐색과 deterministic 평가의 차이, 그래프 head의 KEEP 선호·수정 행동 및
  보상 기여는 추가 확인 대상이다. 시드 간 변동을 포함한 구조 비교 검정은 미완료다.
- v7-2는 세 시드 모두 학습·평가를 완주했지만 합산 승률이 17.78%였다. 이는 한 유닛
  전뇌 제어와 네 유닛 scripted 정책으로 구성된 팀의 성적이며 5유닛 전뇌 제어 성과가
  아니다. 전뇌 재배선 B2, CNN/GRU B3 대조군과 F2 가소성 실험은 수행하지 않았다.
- 본실험의 confirmation, 최종 모델 잠금, held-out test 결과는 없다. 제출 export도
  검증되지 않았으며, 학습·dev 큐 완료를 최종 성능 검증이나 제출 준비 완료로 보지 않는다.

**현재 결론:** v7은 동일 예산의 다중 시드 구조 비교와 전뇌 특징 기반 직접 제어의
장기 실행을 완료했다. 그러나 실제 FlyWire와 재배선의 평가 결과가 같고 전뇌 직접
제어도 낮은 승률에 머물러, connectome 도입에 따른 전략 개선은 확인되지 않았다.
다음에는 그래프 residual이 평가에서 실제로 어떤 수정을 선택했는지와 MLP 성능
저하 원인을 분석하고, 전뇌 대조군 및 별도 최종 평가를 통해 해석 범위를 넓혀야 한다.

## 세대 전체에서 얻은 핵심 결론

1. **v1→v2:** 긴 게임에서는 rollout과 episode의 수명을 분리해야 terminal 보상을
   학습할 수 있다.
2. **v2→v3:** checkpoint의 neural weight만 보고 초기 성능을 가정하면 안 된다.
   실제 추론 경로의 planner/guardrail까지 성능 계약에 포함해야 한다.
3. **v3→v4:** 강한 장기 의사결정 planner를 단일-step BC로 완전히 복제하는 것은
   closed-loop 오류에 취약하다. 강한 baseline을 직접 보존하는 편이 안정적이다.
4. **v4→v5:** 안전한 residual만으로는 부족하다. planner를 이길 만큼의 탐색 빈도와
   실패 상태에 집중된 학습 신호가 있어야 한다.
5. **v5→v6:** 성능 붕괴 방지와 성능 향상은 별개다. 실행 행동과 학습 확률을
   일치시키고, 탐색 하한·B 진영 비중·실패 seed 재표집을 구현했으며 초기 gate는
   baseline 기준으로, 최종 gate는 분리된 dev/confirmation 기준으로 구성했다.
6. **v6→v7:** v6에서 탐색률이 증가하고 초기 단계 승급과 롤백이 작동해도 최종 target
   성능은 5/10에 머물렀다. 탐색량이나 안전장치의 동작을 전략 개선으로 간주할 수
   없어, v7에서는 고정 예산의 구조 대조와 planner 없는 제어 슬롯을 별도로 비교했다.
7. **현재:** v7 본실험 23개를 완주했지만 실제 FlyWire와 재배선은 모두 dev 50%로
   경기 기록까지 같았다. MLP 대조군보다 높은 승률이나 실행 완료만으로 실제 배선의
   이득을 주장할 수 없으며, 행동 변화·시드 변동 분석과 전뇌 대조군·최종 test가 남았다.

또한 v4/v5 checkpoint는 Python evaluator가 planner와 residual head를 함께 실행한다.
v6는 planner 실행 코드까지 포함하는 제출 export를 구현했지만 공식 제출 계약과의
호환성은 아직 검증하지 않았다. 순수 stateless Torch actor만 허용한다면 별도 증류가
필요하다는 제약은 남아 있다.
v7 역시 공식 제출 계약과 자원·의존성 검증이 남아 `export_v7.py`가 차단된 상태다.

## 구현 및 근거 위치

| 세대 | 구현/설계 | 실행 근거 |
| --- | --- | --- |
| v1 | 초기 구현은 현재 v2로 교체되었으며 스키마와 실패 분석으로 보존 | `logs/mappo_vs_win70/`, `reports/mappo_vs_win70_v2_plan_changes.md` |
| v2 | `scripts/train_mappo_vs_win70.py` | `logs/mappo_vs_win70_v2/`, `reports/mappo_vs_win70_training.md` |
| v3 | 당시 구현은 v4 entry point로 발전했으며 설계 문서로 보존 | `logs/mappo_teacher_curriculum_v3/`, `reports/mappo_teacher_curriculum_v3_plan.md` |
| v4 | `scripts/train_mappo_planner_residual_v4.py` 및 v4 호환 학습기 | `logs/mappo_planner_residual_v4/`, `reports/mappo_planner_residual_v4_plan_changes.md` |
| v5 | `scripts/train_mappo_planner_residual_v5.py`, `blackout_rl/mappo_curriculum_v5.py` | `logs/mappo_planner_residual_v5/`, `reports/mappo_planner_residual_v5_plan_changes.md` |
| v6 | `scripts/train_mappo_planner_residual_v6.py`, `blackout_rl/mappo_v6.py`, `blackout_rl/mappo_v6_training.py`, `blackout_rl/mappo_curriculum_v6.py` | `logs/mappo_planner_residual_v6/`의 `run_summary.json`, `training.jsonl`, `training_episodes.jsonl`, `target_eval_step_1159168.json`, `confirmation_rollback_eval_step_*.json`; 구현 검증: `tests/test_mappo_v6.py`, `reports/mappo_planner_residual_v6_plan_changes.md` |
| v7-1 | `blackout_rl/v7_1/`, `blackout_rl/v7_training.py`, `configs/v7/main_study/v7_1_*.yaml`, `reports/v7/main_study_preregistration.md` | `logs/v7/v7_1_*_main_2m_v1/<seed>/`의 `status.json`, `training.jsonl`, `eval_dev_target_*.json`; 집계: `logs/v7/main_study/accelerated/summary.json` |
| v7-2 | `blackout_rl/v7_2/`, `configs/v7/main_study/v7_2_readout_ppo.yaml`, `reports/v7/teammate_v2_recovery.md` | `logs/v7/v7_2_*_teammate_v2_main_2m_v1/<seed>/`의 `status.json`, `training.jsonl`, `eval_dev_target_*.json`; 개입 검증: `reports/v7/unity_causality_teammate_v2.json` |
| v7 공통 실행 | `scripts/connectome_main.py`, `scripts/accelerated_connectome.py`, `scripts/evaluate_v7.py`, `reports/v7/acceleration_and_resume.md` | `logs/v7/main_study/accelerated/status.json`, `summary.json`; 등록·재개 근거: `reports/v7/main_study_registration.json`, `acceleration_registration.json`, `acceleration_resume_points.json` |
