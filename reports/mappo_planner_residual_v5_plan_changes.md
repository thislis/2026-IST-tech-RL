# MAPPO planner-conditioned residual v5 변경 계획 및 구현 결과

## 변경 배경

v4는 planner baseline을 보존했지만 첫 warmup에서 실제 성능이 변하지 않았다.

- 초기·51,200·100,352 step의 scripted 평가가 모두 `8승 2패, +21.5`
- full win70 평가는 모두 `5승 5패, 0.0`
- warmup은 PPO를 사용하지 않고 모든 teacher label을 fallback action 0으로 지정
- 승급 기준은 90%였지만 고정 dev baseline은 80%

따라서 v4는 안전하지만 planner보다 강해질 학습 신호가 없는 구조였다. v5는
planner를 보존하면서도 실패 상태에서 제한적인 방향 수정을 PPO로 학습하도록
목적함수와 curriculum을 분리한다.

## v5 residual 계약

v5 체크포인트의 guardrail mode는 `planner_residual_v2`다.

- action 0: planner 행동 유지
- action 1/2: planner 기준 좌/우 45도
- action 3/4: planner 기준 좌/우 90도
- action 5: 정지
- action 6: planner 반대 방향
- action 7/8: planner 기준 좌/우 135도
- 한 environment step에서 최대 한 agent만 override
- 여러 agent가 deterministic override를 요청하면 fallback 대비 logit margin이 가장 큰
  한 agent만 허용

Residual head에는 기존 local vector/map/slot latent와 함께 다음 파생 입력을 제공한다.

- planner action one-hot 및 방향 벡터
- worker/guard 역할
- planner path 유효성
- worker의 battery/own-storage 또는 guard의 enemy까지 가장 가까운 상대 방향·거리

## 탐색과 손실

- 초기 fallback logit bias: `6.5`
- 매 step 무작위로 한 agent만 override 탐색 가능
- warmup 이후 all-zero behavior cloning을 사용하지 않음
- PPO에 단계별 override probability penalty를 추가하고 난이도가 올라가면서 감쇠
- warmup은 planner forcing 100%이며 actor/critic update를 하지 않음

이 구조는 여러 agent가 동시에 planner를 망가뜨리는 초기 탐색을 차단하면서, 보상으로
확인된 override에는 PPO gradient가 흐르게 한다.

## v5 curriculum

| Stage | 최소/최대 step | 평가 상대 | 승급 승률 | PPO | Override penalty |
|---|---:|---|---:|---|---:|
| planner_preservation | 16,384 / 16,384 | base scripted | 70% | off | 0.02 |
| scripted_residual_ppo | 50k / 300k | base scripted | 90% | on | .010 → .002 |
| weak_win70_mix | 100k / 400k | weak win70 | 70% | on | .006 → .001 |
| full_win70_mix10 | 100k / 250k | full win70 | 30% | on | .004 → .0008 |
| full_win70_mix25 | 100k / 300k | full win70 | 40% | on | .003 → .0005 |
| full_win70_mix50 | 150k / 400k | full win70 | 50% | on | .002 → .0003 |
| full_win70 | 200k / 1M | full win70 | 70% | on | .001 → .0001 |
| robust_historical_mix | 0 / 2M | full win70 | 85% | on | .0005 → 0 |

각 gate는 평균 점수 차와 양 진영 최소 1승도 함께 요구한다. 최대 budget만 소진해서는
승급하지 않고 fail-closed 중단한다.

## 평가, checkpoint, rollback

- 평상시 평가는 committed dev 5 seed × side swap = 10게임
- 빠른 gate 통과 시 policy RNG를 바꾼 3 replica, 총 30게임으로 재확인
- target 최고와 stage 최고 checkpoint를 분리 저장
- 초기 full-win70 승률을 baseline으로 기록
- 10게임 target 평가가 baseline보다 10%p 이상 낮으면 30게임 재확인
- 재확인도 낮으면 target-best model/optimizer로 자동 rollback
- test seed는 학습 및 model selection에 사용하지 않음

## 백그라운드와 배속

모든 학습·평가 환경은 `background=True`로 Unity `-batchmode`를 사용한다. 시각 관측을
필요로 하므로 `-nographics`는 사용하지 않는다. 실제 측정에서 time scale 50 이후 처리량
향상이 없었기 때문에 학습 50, 병렬 평가 100을 유지한다.

터미널까지 백그라운드에서 시작하는 권장 명령:

```bash
./scripts/start_mappo_planner_residual_v5_background.sh
```

로그 확인:

```bash
tail -f logs/mappo_planner_residual_v5/console.log
```

중단된 학습 재개:

```bash
./scripts/start_mappo_planner_residual_v5_background.sh --resume-latest
```

## 산출물

- latest: `checkpoints/mappo_planner_residual_v5_latest.pt`
- target best: `checkpoints/mappo_planner_residual_v5_target_best.pt`
- stage best: `checkpoints/mappo_planner_residual_v5_stage_best.pt`
- 85% target: `checkpoints/mappo_win_85_vs_win70_v5.pt`
- training log: `logs/mappo_planner_residual_v5/training.jsonl`
- summary: `logs/mappo_planner_residual_v5/run_summary.json`

## 검증

- planner context, 제한적 탐색, PPO penalty, curriculum gate, v5 checkpoint round-trip 테스트
- 실제 Unity `-batchmode` fresh run: planner preservation rollout과 평가 통과
- 실제 Unity resume: `scripted_residual_ppo` PPO update 성공
- 축소 smoke curriculum에서 preservation → scripted PPO → weak mix 승급 성공
- smoke 산출물은 `/private/tmp`만 사용했으며 작업공간 학습 산출물을 만들지 않음

Unity 프로세스가 종료 요청에 늦게 응답해 wrapper가 강제 종료하는 경고가 발생할 수
있지만 checkpoint 저장과 학습 결과에는 영향을 주지 않는다.

## 남은 제출 단계

v5는 Python evaluator가 checkpoint metadata의 planner와 residual head를 함께 실행한다.
최종 제출 형식이 순수 Torch 두 파일만 허용한다면 학습 완료 후 planner+residual을 단일
actor로 distillation하거나 planner 실행 코드를 제출 패키지에 포함하는 별도 단계가 필요하다.
