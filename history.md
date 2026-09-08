# MAPPO v1–v5 발전 과정

## 목표와 공통 평가 기준

최종 목표는 고정 상대 `checkpoints/win_70_vs_scripted.pt`를 상대로 MAPPO 정책을
학습해, dev seed 5개를 양 진영으로 바꿔 치르는 10경기에서 승률 85% 이상을
달성하는 것이다. 10경기에서는 최소 9승이 필요하며 무승부는 승리로 계산하지
않는다. 모든 세대는 decentralized actor와 centralized critic을 사용하는 MAPPO
CTDE를 기반으로 한다.

핵심 발전 흐름은 다음과 같다.

> 직접 PPO(v1) → 에피소드 수집 정상화(v2) → planner 모방(v3) → planner 보존형
> residual(v4) → planner-conditioned 안전 탐색과 롤백(v5)

## 한눈에 보는 결과

| 세대 | 핵심 접근 | 이전 문제에 대한 해결 | 실제 장기 실행 결과 | 다음 세대로 넘긴 문제 |
| --- | --- | --- | --- | --- |
| v1 | 기존 self-play actor를 `win70` 상대에게 직접 MAPPO 학습 | 최초의 고정 상대 fine-tuning 기준선 구축 | 2,000,384 step, 완료된 학습 경기 0개, 평가 0/10 | rollout마다 환경을 초기화해 terminal 보상을 전혀 학습하지 못함 |
| v2 | 진영별 persistent collector와 2,048-step rollout | rollout 경계를 넘어 경기 상태와 terminal 보상을 보존 | 2,000,896 step, 1,643경기 전패, 평가 0/10 | 상대의 실력은 neural actor가 아니라 checkpoint의 planner override에서 나왔음 |
| v3 | DAgger·teacher forcing·BC와 단계별 상대 curriculum | planner 행동을 neural actor에 증류한 뒤 PPO로 개선 시도 | 3,000,320 step, 최종 평가 0/10, 점수 차 -94.1 | 단일-step 모방 오차 누적, NoOp 편향, 실패 단계 강제 승격, 성능 회귀 |
| v4 | action 0은 planner 유지, 1~8은 방향 override인 planner residual | 불완전한 planner 복제 대신 검증된 planner 성능을 그대로 보존 | 100,352 step, scripted 8/10·target 5/10 그대로, 첫 단계에서 안전 중단 | warm-up이 fallback 정답만 BC하고 PPO도 꺼져 있어 개선 신호가 없었음 |
| v5 | planner 문맥 기반 상대 방향 residual, 한 step당 override 1개, PPO·확인 평가·롤백 | planner를 보존하면서 제한된 수정 행동을 실제 보상으로 학습 | 317,440 step, scripted 8/10·target 5/10, 롤백 1회 후 단계 상한에서 중단 | PPO override가 평균 0.184%에 그쳐 정책이 planner에서 벗어나지 못함 |

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

### 결과와 현재 문제

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

## 세대 전체에서 얻은 핵심 결론

1. **v1→v2:** 긴 게임에서는 rollout과 episode의 수명을 분리해야 terminal 보상을
   학습할 수 있다.
2. **v2→v3:** checkpoint의 neural weight만 보고 초기 성능을 가정하면 안 된다.
   실제 추론 경로의 planner/guardrail까지 성능 계약에 포함해야 한다.
3. **v3→v4:** 강한 장기 의사결정 planner를 단일-step BC로 완전히 복제하는 것은
   closed-loop 오류에 취약하다. 강한 baseline을 직접 보존하는 편이 안정적이다.
4. **v4→v5:** 안전한 residual만으로는 부족하다. planner를 이길 만큼의 탐색 빈도와
   실패 상태에 집중된 학습 신호가 있어야 한다.
5. **현재:** v5는 성능 붕괴 방지에는 성공했지만 성능 향상에는 실패했다. 다음 버전은
   B 진영 및 planner 실패 상태를 별도로 표집하고, 안전 범위 안에서 override 빈도를
   높이며, 고정 10게임 gate와 학습 난이도를 baseline에 맞게 재보정해야 한다.

또한 v4/v5 checkpoint는 Python evaluator가 planner와 residual head를 함께 실행한다.
최종 제출이 순수 Torch actor만 허용한다면 planner+residual을 단일 actor로 증류하거나
planner 실행 코드를 제출 패키지에 포함하는 별도 작업이 필요하다.

## 구현 및 근거 위치

| 세대 | 구현/설계 | 실행 근거 |
| --- | --- | --- |
| v1 | 초기 구현은 현재 v2로 교체되었으며 스키마와 실패 분석으로 보존 | `logs/mappo_vs_win70/`, `reports/mappo_vs_win70_v2_plan_changes.md` |
| v2 | `scripts/train_mappo_vs_win70.py` | `logs/mappo_vs_win70_v2/`, `reports/mappo_vs_win70_training.md` |
| v3 | 당시 구현은 v4 entry point로 발전했으며 설계 문서로 보존 | `logs/mappo_teacher_curriculum_v3/`, `reports/mappo_teacher_curriculum_v3_plan.md` |
| v4 | `scripts/train_mappo_planner_residual_v4.py` 및 v4 호환 학습기 | `logs/mappo_planner_residual_v4/`, `reports/mappo_planner_residual_v4_plan_changes.md` |
| v5 | `scripts/train_mappo_planner_residual_v5.py`, `blackout_rl/mappo_curriculum_v5.py` | `logs/mappo_planner_residual_v5/`, `reports/mappo_planner_residual_v5_plan_changes.md` |
