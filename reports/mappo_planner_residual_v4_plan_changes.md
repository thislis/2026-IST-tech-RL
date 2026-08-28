# MAPPO planner-residual v4 계획 변경

## 변경 이유

2026-08-25~26에 실행한 teacher curriculum v3는 3,000,320 environment step을
완주했지만 `win_70_vs_scripted.pt` 상대 dev 평가 10경기에서 전패했다. 실패는
일시적인 평가 분산이 아니라 다음 계약 문제에서 발생했다.

1. `win_70_vs_scripted.pt`의 실제 성능은 neural actor가 아니라
   `inference_guardrail.mode=planner_override`가 만든다. v3 learner는 neural core만
   복사했기 때문에 step 0부터 base scripted와 full target에 모두 0/10이었다.
2. v3는 승격 gate를 한 번도 통과하지 못한 단계도 `maximum_stage_budget`으로
   강제 승격했다. base scripted를 이기지 못한 정책이 weak/full 상대까지 진행했다.
3. teacher replay는 마지막 10만 label 중 NoOp이 41.4%였고 guard 두 슬롯은
   NoOp 비율이 각각 93.3%, 95.1%였다. warm-up best의 replay 정확도는 82.3%였지만
   행동 평가는 0/10이었고, PPO 병행 뒤 final replay 정확도는 49.7%로 하락했다.
   전체 planner를 단일-step 분류로 복제하는 방식은 작은 오차가 누적되는
   closed-loop covariate shift를 해결하지 못했다.
4. v3 latest는 best보다 full-target 평균 점수 차가 `-86.7 → -94.1`로 악화됐다.
   회귀를 감지하면서도 장기 실행을 계속했다.

v3 산출물과 로그는 실패 증거로 그대로 보존하며 v4에서 재개하지 않는다.

## 새 정책 표현: planner residual

v4의 categorical action 의미는 다음과 같다.

| action index | 실행 의미 |
| ---: | --- |
| 0 | 검증된 `3 worker + 2 guard`, chase radius 48 planner action을 실행 |
| 1~8 | neural actor가 선택한 절대 방향으로 planner action을 override |

초기 actor head의 weight를 0으로 초기화하고 index 0 bias를 4.0으로 둔다.
deterministic 평가는 처음부터 planner와 정확히 같은 행동을 하며, stochastic PPO는
작은 비율의 방향 override를 탐색한다. PPO log-probability는 실제 high-level 선택인
`planner fallback` 또는 `direction override`에 대응하므로 on-policy 계약을 유지한다.

저장 checkpoint에는 `planner_residual_v1` guardrail metadata가 포함된다. 평가와
frozen historical opponent도 같은 실행 의미를 사용한다. 이 방식은 planner 전체를
neural actor가 다시 모사하게 하지 않고 planner보다 나은 예외 행동만 학습한다.

## Fail-closed curriculum

- fresh run은 학습 전에 committed dev seed의 base-scripted side-swap 평가를 한다.
  기본 최소 승률 60%를 보존하지 못하면 environment training을 시작하지 않는다.
- 단계 승격은 win rate, 평균 점수 차, 양 진영 최소 1승 조건을 모두 통과할 때만
  `evaluation_gate`로 수행한다.
- 최대 stage budget에 도달했는데 gate를 통과하지 못하면
  `stage_blocked/maximum_stage_budget_without_gate`를 기록하고 exit code 3으로
  중단한다. 더 강한 상대에게 자동으로 넘어가지 않는다.
- 실패 v3 checkpoint는 schema가 달라 `--resume-latest` 대상으로 허용되지 않는다.
- rollout log에는 `residual_fallback_actions`와 `residual_override_actions`를 기록해
  정책이 planner에만 머무는지, 학습된 override를 실제로 쓰는지 확인한다.

첫 단계는 planner residual 의미를 안정화하는 50k minimum/100k maximum warm-up이다.
base scripted 승격 gate는 90% 및 평균 점수 차 +10, weak 상대 gate는 70% 및
평균 점수 차 0이다. 이후 full-opponent 혼합 gate는 기존 기준을 유지하되 모든
단계가 fail-closed로 동작한다.

## 실행 성능과 창 동작

모든 training/evaluation Unity 환경은 `background=True`, `no_graphics=False`로
실행한다. Unity에는 `-batchmode`만 전달하므로 게임 창이 foreground로 활성화되지
않으면서 96×96×11 semantic visual observation은 유지된다.

동일 macOS 호스트에서 background 렌더링으로 400 step씩 재측정한 결과는 다음과
같다.

| Unity time scale | steps/s | visual nonzero cells |
| ---: | ---: | ---: |
| 50 | 53.901 | 9,216 |
| 100 | 53.635 | 9,216 |
| 200 | 53.044 | 9,216 |

50 이상에서는 Python↔Unity 동기 통신이 병목이라 추가 배속 이득이 없었다. 따라서
학습 기본값은 실효 처리량이 가장 높은 50, 병렬 평가는 기존 검증값 100을 사용한다.
두 값은 각각 `--time-scale`, `--eval-time-scale`로 변경할 수 있다.

초기 residual checkpoint를 실제 Unity에서 seed 7171, 양 진영으로 base scripted와
평가한 smoke 결과는 2승 0패, 평균 점수 차 +49.5였다. 동일 neural core를 직접
평가한 v3의 0승과 달리 planner baseline 보존 계약이 실제 환경에서도 확인됐다.
추가 64-step live rollout에서는 fallback 292회, override 28회, residual teacher
label/replay 320개를 기록했고 320-sample MAPPO update의 KL은 약 `0.000002`였다.

## 실행 및 산출물

새 학습 시작:

```bash
./scripts/train_mappo_planner_residual_v4.sh
```

기존 v3 wrapper도 안전을 위해 위 v4 wrapper로 전달된다. 명시적인 v4 명령을
권장한다.

터미널 프로세스까지 백그라운드로 실행:

```bash
mkdir -p logs/mappo_planner_residual_v4
nohup ./scripts/train_mappo_planner_residual_v4.sh \
  > logs/mappo_planner_residual_v4/console.log 2>&1 &
```

재개:

```bash
./scripts/train_mappo_planner_residual_v4.sh --resume-latest
```

| 용도 | 경로 |
| --- | --- |
| latest/resume | `checkpoints/mappo_planner_residual_v4_latest.pt` |
| best dev | `checkpoints/mappo_planner_residual_v4_best.pt` |
| 85% target | `checkpoints/mappo_win_85_vs_win70.pt` |
| snapshots | `checkpoints/mappo_v4_snapshots/` |
| logs | `logs/mappo_planner_residual_v4/` |

## 제한 사항

v4 checkpoint의 action 0은 planner fallback이므로 현재 torch-only
`submission/policy.py`만으로는 동일 행동을 재현할 수 없다. 내부 evaluator와 학습
상대 계약에서는 완전하게 재현되지만, 최종 제출이 두 파일의 torch-only 정책만
허용한다면 target 달성 뒤 planner를 submission package에 포함하거나 residual
정책을 별도로 증류하는 단계가 필요하다. 이 제한을 숨기지 않고 checkpoint
metadata와 본 문서에 명시한다.
