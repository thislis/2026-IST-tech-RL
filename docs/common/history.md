# MAPPO·connectome·Attention v1–v9 발전 과정

> 2026-10-09 갱신: 원본 환경을 유지한 v9 `attention_original_v1`이 10월 8일
> 23:45 KST에 종료됐다. 예비·본·추가 확인 학습 총 12,515,327 step과 중간·기준선·
> 최종 평가 4,016경기를 완료했다. 선택 모델 B_s22의 최종 승률은 58/240=24.17%로,
> 학습 전 Attention 기준선 60/240=25%를 넘지 못했다. 제출 형식 검증은 통과했지만
> 학습에 따른 성능 개선·전략적 협동·공식 서버 성능은 입증하지 못했다.
> 실행 근거: [`v9 최종 집계`](../../logs/v9/attention_original_v1/summary.json),
> 실행·제출 계약: [`v9 안내`](../v9/README.md). 로그·reports·체크포인트·submission은
> 로컬 산출물로 보존하며 Git 추적 대상에서 제외한다.

## 목표와 공통 평가 기준

최종 목표는 고정 상대 `artifacts/checkpoints/pre_v1/win_70_vs_scripted.pt`를 상대로 MAPPO 정책을
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

v8의 최종 실행 경로는 제공 `run_match()`의 경기 전체 누적 reward 비교를 승패 기준으로
사용한다. 이전 평가기의 terminal `info['winner']` 기준과 같다고 가정하지 않는다.
원본 게임·obs 생성 경로를 변경하지 않는 조건에서 30회 × 양 진영을 평가했으며,
실제 map seed 적용이 검증되지 않아 30개의 서로 다른 맵을 평가했다고 표현하지 않는다.

v9도 원본 게임·obs와 제공 runner의 승패 기준을 유지한다. collector/rush/raider/target
4종 상대별 30회 × 양 진영, run당 240경기를 평가하고 3개 학습 시드의 평균으로
설정을 선택한다. 추가 학습 시드 44/55와 별도 action RNG의 최종 test는 선택에
사용하지 않는다. 이는 새로운 맵을 보장하는 held-out map 평가가 아니며, 학습 전
Attention 정책 및 기존 v8 제출물과의 기준선 비교를 함께 기록한다.

핵심 발전 흐름은 다음과 같다.

> 직접 PPO(v1) → 에피소드 수집 정상화(v2) → planner 모방(v3) → planner 보존형
> residual(v4) → planner-conditioned 안전 탐색과 롤백(v5) → 팀 단위 행동 선택과
> 탐색 확률 보장·실패 seed 재표집(v6) → 고정 예산의 connectome residual 대조 실험과
> 전뇌 특징 기반 PPO 직접 제어(v7) → 명시적 개입 gate와 동일 용량 대조군,
> 원본 환경·두 입력 제출 인터페이스에 맞춘 관측 기반 정책(v8) → CNN 없는 소형
> Attention·완료 경기 기반 bounded reward·상대 pool과 전략 출력 대조(v9)

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
| v8 | C1 개입 gate / flat 대조군, 원본 환경용 stateless C1-obs9·flat9와 두 파일 export | agent ID·reset callback 없는 제출 계약에 맞추고 원본 게임·obs 생성 경로 보존 | 최종 경로 6개 × 1,048,576 step 완료; 중간·최종 및 planner 평가 총 780경기 전패 | 초기 관측 반복, greedy 행동 편중, planner 제거 후 약한 baseline, 로컬/공식 승패 계약 차이 검증 필요 |
| v9 | 소형 Attention-MAPPO, bounded reward, 과거 상대 pool·전략 mixture 대조 | encoder 전체 학습, 완료 경기 검증, 확률적 행동과 제출 경로 일치, 학습 전 기준선 추가 | 본 실험 9개·추가 확인 2개 및 pilot 총 12,515,327 step; 선택 B_s22 최종 58/240=24.17%, 미학습 기준선 60/240=25% | 자기 유닛 식별 제약, 높은 행동 entropy, 희소한 학습 신호와 상대별 step 불균형, 전략 개선 미입증 |

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
  confirmation 조건을 통과하지 못해 `mappo_win_85_vs_win70_v6.pt`와 `artifacts/submission/v6`
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

## v8 — 개입 gate 실험과 원본 환경·제출 인터페이스 대응

### v7 문제를 해결하기 위한 구현

- 기존 `KEEP + 5유닛 × 8방향` flat 분포에서 수정 logit이 KEEP보다 낮으면 학습 중
  탐색을 하더라도 greedy 실행은 KEEP에 머물 수 있어, KEEP/개입 gate와 조건부
  수정 분포를 분리한 C1을 구현했다. 동일 용량 flat 대조군과 frozen encoder,
  작은 64×64 trunk, centralized critic, 공동 행동 log probability 기반 PPO를 사용했다.
- 초기 연구 경로에는 별도 Unity 결과·타이머 코드와 추가 side-channel을 구현했다.
  이후 사용자가 제공 게임·강화학습 환경·obs 생성 코드 변경을 금지하여 해당 경로를
  철회했다. 원본 앱·upstream 저장소·설치된 API를 복원했고, 수정 빌드·로그는 별도
  보관했다. 수정 게임 checkpoint를 원본 환경에서 학습한 것으로 바꾸어 사용하지 않았다.
- 제공 제출 안내는 `MyPolicy.forward(vector, graphic)`와 두 파일
  `policy.py`, `checkpoint.pt`를 요구한다. 입력은 `float32[B,96]`와
  `float32[B,11,96,96]`, 출력은 `float32[B,2]`, 범위 [-1,1]이다.
  이 인터페이스에는 agent ID·canonical slot·episode reset callback이 없다.
- 최종 경로는 이에 맞춰 배치 순서·분할 호출에 의존하지 않는 stateless
  **C1-obs9 / flat9**로 변경했다. 각 입력 행이 KEEP 또는 8방향 수정을 선택한다.
  기존 팀당 최대 한 유닛 개입 조건은 유지되지 않으므로 joint41과 같은 정책으로
  부르지 않는다. PPO는 팀원 5명의 log probability 합을 사용한다.
- 제출 모델의 KEEP baseline은 기존 target checkpoint의 동결 신경망을 관측 기반으로
  적응시킨 것이다. self slot 대신 평균 slot embedding과 관측된 아군 중심 crop을
  사용한다. 과거 target의 완전한 planner override와 동일한 강도를 보장하지 않는다.
  기존 scripted planner는 별도 평가 기준선, 완전한 target 정책은 고정 상대로 유지했다.
- 원본 API의 reward 평균을 학습에 사용하고, 에피소드마다 원본 Unity를 새로 실행했다.
  reset/step·seed 전송·obs cache·점수·타이머를 패치하거나 숨은 step을 넣지 않았다.
  요청 seed가 실제 맵에 적용됐다는 보장은 없으며 평가에는 seed 인자가 없는 제공
  `run_match()`를 그대로 사용했다.
- 실험은 C1/flat × seed 11·22·33, run당 1,048,576 step으로 고정했다.
  rollout 2,048, PPO epochs 4, minibatch 128, LR 1e-4, gamma .9995, lambda .99,
  기존 상대 혼합 일정과 진영 순환 `[A,A,B,B,B]`를 유지했다. 여섯 worker, CPU/BLAS
  1 thread, 백그라운드 관리자, 불변 checkpoint·등록 검사·중지 후 재개를 연결했다.

### 수정 환경 실험과 원본 환경 실험의 구분

| 경로 | 최종 C1 | 최종 flat | planner 기준선 | 지위 |
| --- | --- | --- | --- | --- |
| 과거 `accelerated_pilot_v1` | 26/180, 14.44% | 25/180, 13.89% | 30/60, 50% | 수정 Unity·연구용 결과 계약, 이후 철회. 원본 환경 성능으로 사용하지 않음 |
| 현재 `provided_competition_v1` | 0/180, 0% | 0/180, 0% | 0/60, 0% | 원본 게임·제공 API, 누적 reward 기반 로컬 runner 결과 |

두 경로는 각각 총 6,291,456 학습 step과 중간·최종·기준선 dev 780경기를 수행했다.
환경·정책 입력 및 실행 구조·reward·승패 기준이 바뀌었으므로 두 승률의 차이를
같은 조건에서의 성능 증감으로 해석하지 않는다.

### 결과와 현재 문제

- 2026-09-29 첫 원본 환경 실행에서 터미널 프로세스는 분리됐지만 Unity 창은 표시됐다.
  중지 요청에 따라 6개 run 모두 2,048 step에서 저장·종료했다. 이후 원본 실행 파일에
  `-batchmode`만 추가하는 별도 실행 스크립트를 연결했다. `-nographics`는 사용하지
  않아 렌더링을 유지하며 원본 게임·환경 코드는 변경하지 않았다.
- 창 숨김 기술 검사에서 10개 에이전트의 `96×96×11` graphic, 픽셀별 one-hot과
  벽·양 팀 유닛 채널을 확인했고, 32 step 동안 화면에 표시된 Unity 창은 감지되지
  않았다. 기존 checkpoint·등록을 보존하고 tensor·optimizer·RNG 상태가 같은 자식
  checkpoint로 실행 방식 변경을 기록한 뒤 사용자가 같은 명령으로 재개했다.
- **2026-09-30 02:49 KST**에 원본 환경 실험의 최종 완료 상태가 기록됐다.
  6개 run 각각 1,048,576 step과 PPO 업데이트 512회, 총 6,291,456 step을 완료했다.
  학습 6개와 평가 shard 78개로 84/84 작업이 완료됐고 실패 기록은 없었다.
- 원시 평가 파일의 해시·모델 lock·진영/반복 번호를 재검사했다. 중간 262,144 step에서
  C1 0승/180경기, flat 0승/180경기, 최종 1,048,576 step에서도 각각 0승/180경기였다.
  planner는 0승/60경기였다. 합계 **780경기 전패, 무승부 0경기**다.
- 각 모델·진영의 30회 평가에서 초기 관측 해시는 한 종류뿐이었다. 반복된 초기
  상태에서 얻은 결과이므로 서로 다른 30개 맵의 일반화 성능으로 해석할 수 없다.
  승률 차이와 bootstrap 구간이 `[0,0]`이어도 두 정책의 동등성이나 불확실성 부재를
  입증하는 것은 아니다.
- 모든 run에서 정책의 학습 대상 tensor 10개가 초기값과 달라졌고 encoder는 동결된
  상태였다. 학습 미실행은 아니다. 마지막 2,048 step의 greedy 기록에서는 C1 seed 33이
  수정 방향 8을 10,240개 행 중 10,150회, flat seed 33은 방향 5를 9,980회 선택했다.
  이는 학습 구간에서 확인한 행동 편중이며 전체 평가 궤적이나 전패의 단일 원인으로
  확정하지 않는다.
- 자동 선택 모델은 `c1_s11`이다. 최종 arm 평균 동률이면 C1, arm 내 seed별 동률이면
  먼저 등록한 seed를 선택하는 규칙에 따른 결과다. 성능 우위로 선정된 모델이 아니다.
  `artifacts/submission/v8/37907297bcf0234e823b12bfdb95433b0490e7e7f2672b119c9929b98cd1e493`
  에 두 제출 파일을 생성했고, 제공 loader의 격리 CPU 로드·입력 형식·행동 범위·배치
  순서/분할·실제 수집 관측 8개 배치의 행동 일치 검사를 통과했다.
- 제공 안내는 보상이 대회 점수와 무관하다고 설명하지만, 설치된 로컬 `run_match()`는
  누적 reward로 승자를 정한다. 과거 terminal reward 기반 `info['winner']`와도 다르다.
  공식 서버 승패·자원 조건 검증, confirmation·held-out test와 외부 제출은 수행하지 않았다.

**현재 결론:** v8은 원본 환경을 유지하는 학습·관전 없는 백그라운드 실행·제출 파일 생성
경로를 완성했지만, 실험한 정책은 고정 상대를 이기지 못했다. 제출 형식 통과를 전략적
성능 검증으로 보지 않는다. 다음 개선은 평가 초기 상태의 반복과 승패 기준을 먼저
분리해 확인하고, 식별 정보가 없는 입력에서 planner를 대체한 baseline의 강도 및
greedy 행동 편중을 분석하는 데서 시작해야 한다. 게임·obs 생성 코드는 임의 수정하지 않는다.

## v9 — 소형 Attention 정책과 완료 경기 기반 보상·상대 다양성 대조

### v8 문제를 해결하기 위한 구현

- v1~v8 로그·코드 감사와 관련 연구를 바탕으로 `docs/v9/plans/v9_plan.md`를 작성하고,
  새 구현을 `blackout_v9/`에 분리했다. 원본 Unity·제공 API·obs 생성 및 전처리와
  v1~v8 코드는 유지한다. 게임 내부의 타이머 문제를 수정하지 않고 경기마다 원본
  환경을 새로 실행하며, 정상 종료되지 않은 경기는 학습에 반영하지 않는다.
- CNN과 고정 encoder를 소형 Attention으로 교체했다. 96×96×11 graphic 전체를
  4×4 patch 576개로 펼치고 유닛 entity 10개·상황 token 1개를 합친다. width 64,
  heads 4, latent 32개, latent self-attention 2층, FFN 128을 사용한다. 이는 속도를
  고려한 초기 설계값이며 모델 크기별 대조로 최적성을 검증한 값은 아니다.
- 제출 정책은 전체 123,765개 파라미터로, v8 제출 정책 249,891개보다 약 50.5%
  작다. 다만 v8의 학습 대상은 20,297개였고 v9는 encoder까지 학습하므로 가중치 수
  감소를 학습 연산·메모리의 같은 비율 감소로 해석하지 않는다. A/B는 하나의 expert를
  활성화하고 C는 5개를 사용하되, 동일한 제출 모듈 구조를 유지한다.
- 기존 두 입력·두 파일 계약을 따르는 stateless V9-S를 구현했다. 학습·평가·제출은
  같은 9-action categorical sampling을 사용하며 agent 이름·batch row 번호를 자기
  ID로 주입하지 않는다. 동일 클래스·동일 관측의 팀원은 같은 행동 분포를 공유한다.
  self-ID를 요구하는 V9-I, 시간 memory, 유닛별 역할 할당은 구현된 것으로 보지 않는다.
- 학습 보상은 제공 runner의 팀별 누적 raw reward 차이의 최종 부호 ±1/0과 점수차
  기반 potential shaping이다. 원본 reward는 수정하지 않으며 gamma=1에서 완료 경기
  return의 절댓값을 1.25 이하로 제한한다. 22,000-step·wall/heartbeat watchdog으로
  무효 경기를 폐기하고 완료 episode 안에서 GAE를 계산한다. GAE lambda는 .9975다.
- 팀 advantage를 사용하는 agent별 PPO ratio/clip, PopArt critic, PPO epochs 2,
  minibatch 128 scene, actor/critic LR 1e-4/3e-4, clip .15, entropy .01을 사용한다.
  A는 고정 상대 혼합, B는 과거 모델 pool·PFSP 추가, C는 B에 5-expert mixture와
  관측 기반 보조 loss를 추가한다. expert 개수가 실제 전략 분화의 증거는 아니다.
- 공통 장면의 forward 내 계산 재사용, graphic의 lossless 저장, CPU actor·MPS learner,
  최대 8개 수집 worker 자동 조정을 연결했다. 평가 worker는 4개이며 API 통신이나
  obs를 패치하지 않는다. 원본 앱에 `-batchmode`를 전달해 graphic 렌더링은 유지하고
  창을 숨긴다. `code/v9/run_v9_fast.sh` 한 줄로 분리된 백그라운드 실행·상태·중지·재개를 제공한다.
- 환경 lifecycle 20경기와 기능 pilot 후 A/B/C × seed 11/22/33을 각각 목표
  1,048,576 step까지 새로 학습했다. 완료 경기 wave 단위로 예산을 마감해 실제 step은
  목표를 초과할 수 있다. 설정 선택 후 seed 44/55를 추가 학습하고 선택 모델·미학습
  기준선에 별도 action RNG 최종 테스트를 수행한다. 최종 테스트로 모델을 다시 고르지 않는다.

### 결과와 현재 문제

- **2026-10-04 11:21부터 2026-10-08 23:45 KST까지 약 108시간 25분** 실행했다.
  최종 상태는 `complete`, 활성 supervisor는 없다. 본 실험 10,085,650 step,
  추가 확인 2,277,514 step, pilot 152,163 step으로 총 **12,515,327 step**이다.
  학습 838경기에서 무효 에피소드는 0건이었다. lifecycle 20경기는 이 학습량과 별도다.
- 기준선·중간·최종·추가 확인·최종 test를 합쳐 평가 보고서 41개, **4,016경기**를
  완료했고 평가 무효 기록은 0건이었다. 아래 본 실험 dev는 run당 240경기이며
  모든 seed의 평가량이 같아 합산 승률과 seed 평균이 같다.

| 설정 | 시드별 승수 / 240경기 (11, 22, 33 순) | 합산 승·무·패 | dev 평균 승률 |
| --- | --- | --- | --- |
| A 기본 Attention + 고정 상대 혼합 | 4, 1, 60 | 65승 0무 655패 | 9.03% |
| B 과거 모델 pool·PFSP 추가 | 23, 55, 20 | 98승 0무 622패 | 13.61% |
| C 전략 mixture·보조학습 추가 | 32, 47, 15 | 94승 1무 625패 | 13.06% |

- 설정 평균으로 B를 선택하고 B 내부 dev 최고인 **B_s22**를 제출 후보로 고정했다.
  A_s33은 단일 run dev 60승으로 B_s22의 55승보다 높지만, 등록 규칙은 전체 run 중
  최고 하나가 아니라 설정 평균을 먼저 비교한다. B의 추가 확인은 seed 44에서
  61/240=25.42%, seed 55에서 6/240=2.50%로 시드 변동이 컸다. 확인 결과로
  제출 후보를 바꾸지 않았으며 B가 다른 설정보다 확실히 우월하다고 단정하지 않는다.

| 비교 대상 | 평가 구분 | 승·무·패 | 승률 |
| --- | --- | --- | --- |
| 선택 B_s22 | 별도 action RNG 최종 test | 58승 0무 182패 | 24.17% |
| 미학습 Attention 기준선 | 같은 최종 test 조건 | 60승 0무 180패 | 25.00% |
| 기존 v8 제출 모델 | dev 기준선 평가 | 0승 0무 240패 | 0.00% |

- B_s22 최종 test는 collector 0/60, rush 0/60, raider 58/60, target 0/60이었다.
  미학습 Attention도 raider 60/60, 나머지 세 상대는 0/60이었다. v8보다 높은 수치를
  Attention 학습의 효과로 귀속할 수 없으며 `local_runner_improved=false`다.
  상대 종류가 적고 승리가 약탈형에 집중돼 올킬·입구 봉쇄·탈취·협동 습득은 입증되지 않았다.
- B_s22의 마지막 PPO 구간 행동 entropy는 약 2.181로 9-action 균등 분포의 최대값
  log(9)≈2.197에 가깝다. encoder gradient와 가중치 변화는 확인돼 학습 미실행은
  아니지만, 뚜렷한 행동 선호를 배우지 못했다. 이 평균만으로 모든 상태에서 완전히
  무작위였다고 단정하거나 Attention 용량 부족을 원인으로 확정하지 않는다.
- B_s22는 1,137,886 step을 학습했지만 완료 경기는 76개, 수집→PPO 갱신 주기는
  14회였다(PPO 내부 미니배치 업데이트는 별도). 이 중 50경기가 21,004 step으로
  종료됐다. terminal 보상의 직접 GAE 계수는 1,000 step 전에 약 .082, 5,000 step
  전에 약 3.7e-6이다. 가치함수를 통한 전달 가능성은 있으나 희소한 승패·점수 변화가
  초기 행동에 충분한 학습 신호를 주었는지는 검증되지 않았다. 이 run의 실제 return은
  부동소수점 오차 범위에서 [-1,1]로, 관측된 실패를 보상 폭증 때문이라고 보지는 않는다.
- 같은 run에서 target 19경기는 전패했고 해당 경험은 22,510 step으로 전체의 약 2%,
  history 22경기는 462,088 step으로 약 41%였다. 경기 수로 상대를 섞어도 빠르게
  패배하는 강한 상대의 step 비중은 작아진다. 이는 로그로 확인한 데이터 불균형이며
  전패의 독립적인 인과 효과를 검증한 대조 실험은 없다.
- 자기 유닛 ID가 없는 제출 actor와 달리 scripted 상대는 환경이 넘긴 `unit_번호`로
  자기 위치를 식별하고 길찾기한다. 같은 클래스의 아군 위치별 제어·역할 분담에는
  이 정보 차이가 제약이다. 관측을 임의 확장하거나 Attention만으로 없는 식별 정보를
  복원할 수 있다고 가정하지 않는다. pilot의 이동·획득·배달 발생 gate 또한 미학습
  기준선 대비 개선을 요구하지 않아, 기능 정상화와 실제 학습 성공을 구별하기에 부족했다.
- 평가 초기 관측 해시는 진영별 한 종류였다. 별도 action RNG 반복을 새로운 맵
  일반화 검증으로 해석하지 않는다. 로컬 누적 reward 승패와 공식 게임 승패의 일치,
  self-ID 계약, 전략적 협동·공식 서버 성능은 검증되지 않았다.
- `artifacts/submission/v9/4b6e8bae8713413fb65a557883ab1b1b9353f02273d4854b174654e6556cf18c`
  에 `policy.py`와 `checkpoint.pt`를 생성했다. 제공 loader의 격리 CPU 로드,
  B=0/1/3/5/10 입력, 출력·분포 일치 검증을 통과했고 저장 파일 해시도 일치했다.
  `format_ready=true`지만 `official_server_certified=false`, 외부 제출은 하지 않았다.

**현재 결론:** v9는 원본 환경·제출 계약을 유지한 Attention 전체 학습, 보상 상한,
완료 경기 검증과 최종 기준선 비교를 완주했다. 그러나 선택 모델이 미학습 정책보다
좋지 않아 학습에 따른 성능 향상은 확인되지 않았다. 다음 개선은 허용 관측의 자기
식별 가능성, 긴 경기의 보상 전달과 상대별 실제 학습량, 미학습 기준선 대비 개선을
요구하는 pilot을 먼저 검증해야 한다. 모델 확대나 추가 학습 시간만으로 해결된다고
가정하지 않으며 원본 게임·obs 생성 코드는 계속 보존한다.

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
7. **v7:** v7 본실험 23개를 완주했지만 실제 FlyWire와 재배선은 모두 dev 50%로
   경기 기록까지 같았다. MLP 대조군보다 높은 승률이나 실행 완료만으로 실제 배선의
   이득을 주장할 수 없으며, 행동 변화·시드 변동 분석과 전뇌 대조군·최종 test가 남았다.
8. **v8:** 제출 인터페이스에 맞는 형식과 강한 행동 정책은 별개다. 입력에서 사라진
   agent ID·reset 계약을 임의로 가정할 수 없고, planner를 제거하면 과거 checkpoint의
   강도를 잃을 수 있다. 실제 11채널 수신·checkpoint 갱신·실험 완주가 확인돼도
   780경기 전패와 초기 상태 반복을 성능 향상의 증거로 바꿀 수 없다.
9. **v9:** 작은 Attention·전체 encoder 학습·bounded reward·완전 경기 수집이 정상
   동작해도 정책 개선은 별도로 확인해야 한다. 최종 24.17%는 미학습 기준선 25%보다
   높지 않았다. 기능 gate와 성능 gate를 분리하고, 자기 식별 제약·희소 보상 전달·
   상대별 step 불균형을 검증해야 한다. 모델 경량화와 학습 성공은 같은 주장이 아니다.

또한 v4/v5 checkpoint는 Python evaluator가 planner와 residual head를 함께 실행한다.
v6는 planner 실행 코드까지 포함하는 제출 export를 구현했지만 공식 제출 계약과의
호환성은 아직 검증하지 않았다. 순수 stateless Torch actor만 허용한다면 별도 증류가
필요하다는 제약은 남아 있다.
v7 역시 공식 제출 계약과 자원·의존성 검증이 남아 `export_v7.py`가 차단된 상태다.
v8은 두 입력 stateless 인터페이스와 제공 loader 검증을 통과했지만 성능 및 공식 서버
검증을 통과한 것은 아니다. v6·v7을 현재 인터페이스로 그대로 제출할 수 있다는 뜻도 아니다.
v9 역시 두 파일 제출 검증과 최종 test를 완료했지만 미학습 기준선 대비 개선은 없었으며,
형식 준비 완료를 공식 서버 검증이나 경쟁력 있는 제출물이라는 인증으로 해석하지 않는다.

## 구현 및 근거 위치

| 세대 | 구현/설계 | 실행 근거 |
| --- | --- | --- |
| v1 | 초기 구현은 현재 v2로 교체되었으며 스키마와 실패 분석으로 보존 | `logs/mappo_vs_win70/`, `docs/v2/reports/mappo_vs_win70_v2_plan_changes.md` |
| v2 | `code/v2/scripts/train_mappo_vs_win70.py` | `logs/mappo_vs_win70_v2/`, `docs/v2/reports/mappo_vs_win70_training.md` |
| v3 | 당시 구현은 v4 entry point로 발전했으며 설계 문서로 보존 | `logs/mappo_teacher_curriculum_v3/`, `docs/v3/reports/mappo_teacher_curriculum_v3_plan.md` |
| v4 | `code/v4/scripts/train_mappo_planner_residual_v4.py` 및 v4 호환 학습기 | `logs/mappo_planner_residual_v4/`, `docs/v4/reports/mappo_planner_residual_v4_plan_changes.md` |
| v5 | `code/v5/scripts/train_mappo_planner_residual_v5.py`, `code/v5/blackout_rl/mappo_curriculum_v5.py` | `logs/mappo_planner_residual_v5/`, `docs/v5/reports/mappo_planner_residual_v5_plan_changes.md` |
| v6 | `code/v6/scripts/train_mappo_planner_residual_v6.py`, `code/v6/blackout_rl/mappo_v6.py`, `code/v6/blackout_rl/mappo_v6_training.py`, `code/v6/blackout_rl/mappo_curriculum_v6.py` | `logs/mappo_planner_residual_v6/`의 `run_summary.json`, `training.jsonl`, `training_episodes.jsonl`, `target_eval_step_1159168.json`, `confirmation_rollback_eval_step_*.json`; 구현 검증: `code/v6/tests/test_mappo_v6.py`, `docs/v6/reports/mappo_planner_residual_v6_plan_changes.md` |
| v7-1 | `code/v7/blackout_rl/v7_1`, `code/v7/blackout_rl/v7_training.py`, `code/v7/configs/main_study/v7_1_*.yaml`, `docs/v7/reports/main_study_preregistration.md` | `logs/v7/v7_1_*_main_2m_v1/<seed>/`의 `status.json`, `training.jsonl`, `eval_dev_target_*.json`; 집계: `logs/v7/main_study/accelerated/summary.json` |
| v7-2 | `code/v7/blackout_rl/v7_2`, `code/v7/configs/main_study/v7_2_readout_ppo.yaml`, `docs/v7/reports/teammate_v2_recovery.md` | `logs/v7/v7_2_*_teammate_v2_main_2m_v1/<seed>/`의 `status.json`, `training.jsonl`, `eval_dev_target_*.json`; 개입 검증: `logs/v7/reports/unity_causality_teammate_v2.json` |
| v7 공통 실행 | `code/v7/scripts/connectome_main.py`, `code/v7/scripts/accelerated_connectome.py`, `code/v7/scripts/evaluate_v7.py`, `docs/v7/reports/acceleration_and_resume.md` | `logs/v7/main_study/accelerated/status.json`, `summary.json`; 등록·재개 근거: `logs/v7/reports/main_study_registration.json`, `acceleration_registration.json`, `acceleration_resume_points.json` |
| v8 초기 연구 경로(철회) | `code/v8/blackout_rl/v8`, `docs/v8/environment_restoration.md` | `logs/v8/accelerated_pilot_v1/summary.json`, `logs/v8/reports/environment_restoration`, `build/retired_v8_environment_2026-09-29/` |
| v8 원본 환경·제출 | `code/v8/blackout_rl/v8/competition`, `code/v8/configs/competition/study.json`, `code/v8/contracts/submission_contract.json`, `docs/v8/competition.md` | `logs/v8/provided_competition_v1/{status,summary,submission_verification}.json`, `logs/v8/reports/competition/registration.json`, `logs/v8/reports/background_window_fix` |
| v9 Attention·원본 환경·제출 | `blackout_v9/`, `code/v9/configs/default.json`, `code/v9/contracts/original.json`, `docs/v9/plans/v9_plan.md`, `docs/v9.md`, `code/v9/run_v9_fast.sh` | `logs/v9/attention_original_v1/{status,summary,registration,selection}.json` 및 `runs/`, `confirmation/`, `frozen_test/`, `frozen_baseline_test/`; 구현 검증: `code/v9/tests/v9/test_v9.py`, `logs/v9/reports/implementation_validation.json` |
| v6·v7 관전 | `code/shared/watch_best_models.sh`, `tools/watch_best_models.py` | v6 latest(1,159,168 step), v7-1 A3 seed 11 평가 당시 불변 checkpoint(2,000,000 step); `--check`로 실행 전 검사 |

## 선정 모델 관전과 로컬 산출물

관전 대상으로 v6 최종 모델과 v7-1 A3(FlyWire) seed 11을 선택했다. v4·v5도 과거 target
5/10이며 v7 A2/A3·시드들도 동률이므로 절대적인 상위 두 모델이라는 인증은 아니다.
v6는 개선 기반, v7-1 A3는 다중 시드 비교 후보라는 기준과 동률 내 명시적 선택이다.

```bash
bash /Users/safeailab_macmini/Desktop/2026-IST-tech-RL/code/shared/watch_best_models.sh
```

두 모델이 원본 게임 창에서 정상 속도로 두 경기를 치르며 두 번째 경기는 진영을 바꾼다.
Ctrl+C로 종료하며 학습이나 가중치 변경은 하지 않는다. `--check`는 모델·원본 게임만
검사하고 창을 열지 않으며, `--games 1 --speed 2`처럼 경기 수·배속을 지정할 수 있다.
제공 API를 그대로 사용하므로 과거 seed 보정 환경의 평가를 재현한 공식 성적이 아니다.

`logs/`, `reports/`, `submission/`, checkpoint·빌드·대용량 데이터는 로컬에 보존하고
Git에서는 제외한다. 이 문서의 해당 근거 링크와 관전 실행에 필요한 모델·그래프·Unity
빌드는 현재 작업 공간에 있으며 새 clone에는 자동 포함되지 않는다. 재사용할 Phase 3
정책 소스는 `templates/phase3_submission/`에 분리했다.
