# MAPPO planner residual v6 — 구현 및 실행 안내

작성: 2026-09-08. **실제 Unity 학습·승률 평가는 실행하지 않았다.** 아래 내용은 구현,
기존 자료와의 대조, 합성 환경 테스트 결과이며 v6 성능 달성 주장이 아니다.

## 문제별 변경

| 기존 문제 | v6 변경 | 확인 범위 |
|---|---|---|
| 표현 모듈의 300초 시계와 실제 420초 경기 불일치 | `strategy.DEFAULT_EPISODE_SECONDS`를 공유하고 20초 실제 경계 테스트 | 단위 테스트 |
| 별도 `GlobalLocalMapEncoder` crop의 Y축 반전 누락 | world Y를 top-down pixel row로 변환 | 위치가 알려진 픽셀로 검증; 기존 IPPO encoder와 별도 클래스 |
| v1 rollout마다 episode 초기화 | 진영별 persistent collector 유지, 22,000-step episode watchdog | 여러 호출에 걸친 terminal·GAE 테스트 |
| neural core만 복사하면 강한 초기 성능이 사라짐 | 원래 counter planner를 기본 행동으로 실행. 초기 guardrail 설정을 preflight에서 검사 | 양 진영에서 초기 greedy 행동이 planner와 일치 |
| BC 정확도/NoOp 비율이 실전 성능을 대변하지 못함 | v6는 teacher-label BC를 사용하지 않고 완료 경기·실제 winner로 판단 | 강제 planner 보존 단계는 PPO도 끔 |
| v4 보존만 반복하고 개선 신호 없음 | 짧은 보존 단계 다음부터 PPO. scripted/weak/target 혼합 상대 | 합성 환경에서 실제 gradient와 파라미터 변화 |
| v5 PPO 수정 행동 약 0.184% | 보정 가능한 상태에 명시적 탐색 확률 질량 15%→2%; 초기 learned mass 약 10%; override penalty 제거 | 실제 분포의 확률 하한·마스크·finite gradient 테스트 |
| 학습의 무작위 유닛 선택과 평가의 top-margin 선택이 다름 | KEEP 또는 (slot, correction) 하나를 선택하는 **41-way team categorical**로 통일 | 샘플링·PPO 재평가·추론이 같은 logits/마스크 사용 |
| 수집 행동 확률과 PPO log-prob 불일치 위험 | rollout의 exploration 값을 함께 저장하고 동일 mixture distribution으로 likelihood ratio 계산 | 업데이트 전 ratio=1 검증 |
| nearest target과 path-valid 추정값이 실제 planner 상태와 다름 | 실제 planner target/path, 이동량·정체, 역할·class·보유 item·적 상대 좌표·흡수 위상 제공 | 실제 FSM 컨텍스트로 shape/동작 검증 |
| 상대 방향을 8방향으로 양자화하면서 planner 방향 손실 | 실제 연속 planner vector를 회전. planner 정지 시 8개 고유 방향 | reverse·stop·정지 중 8방향 coverage 테스트 |
| B 진영 병목 | B에 60%, A에 40% update 노출; y=x 대칭에 맞춘 좌표·수정 방향 표현 | 진영별 별도 집계, 양 진영 추론 테스트 |
| 실패 상태를 드물게 재방문 | train seed별 승패 EMA로 다음 episode seed 표집; 40% uniform mass 유지 | seed 분포·재개 RNG 테스트; 과거 transition을 PPO에 재사용하지 않음 |
| 8/10 baseline에 고정 9/10 초기 gate | baseline을 같은 seed에서 측정하고 초기 readiness gate를 상대적으로 설정 | 최종 목표는 여전히 dev 9/10 및 confirmation ≥85% |
| 같은 5개 맵을 반복해 30경기로 취급 | confirmation용 **별도 15개 맵 seed × 양 진영** | train/dev/confirmation/test 전부 disjoint 검증 |
| 전승/전패 bootstrap 구간의 과도한 확신 | 기존 paired bootstrap과 함께 seed-pair Hoeffding bound도 기록 | 독립 seed 표본 가정 명시, 성능 보증으로 사용하지 않음 |
| dev 점수 개선 뒤 target-best 성능 하락 | 새 target-best는 confirmation에서도 이전 best 이상일 때만 저장 | stage-best와 target-best 분리 |
| 성능 붕괴 후 계속 학습 | 동일 split baseline보다 10%p 넘게 하락 시 confirmation 후 rollback. 최대 3회에서 중단 | rollback 이벤트·복원 checkpoint hash 기록 |
| 저장·평가가 torch RNG를 바꿈 | checkpoint facade/loader 생성 시 RNG 격리; sampler/optimizer/CPU·MPS RNG 저장 | 저장 전후 RNG 동일성 테스트 |
| 재개 시 모델보다 로그가 앞서거나 best 파일이 덮어써짐 | checkpoint에 로그 byte offset 저장; crash-ahead suffix를 recovery 파일로 보존. best의 immutable 사본 사용 | recovery·재개 테스트 |
| Git에 없는 실행 자료로 재현 어려움 | 실행 때 코드/config SHA manifest, source ZIP, run_config, 초기/상대/build SHA 저장 | 실행 전 hash 검사; 이전 artifact 변경하지 않음 |
| planner가 두 파일 제출에 빠짐 | planner+encoder+residual 소스를 `policy.py`에 포함하는 exporter와 `checkpoint.pt` | workspace/blackout-env 없는 별도 interpreter에서 로딩, residual 활성 상태의 행동 일치 |

## 알고리즘과 범위

v6는 5개 독립 action을 샘플링한 뒤 일부를 취소하는 방식 대신 하나의 team decision을
직접 최적화한다. 각 유닛의 score head는 공유되지만 팀 coordinator가 41가지 선택 중
하나를 고르므로, 정확한 명칭은 **centralized critic을 사용하는 coordinated team residual
PPO**다. 다섯 actor가 독립적으로 행동하는 엄격한 decentralized MAPPO라고 주장하지 않는다.
actor/coordinator는 제출 시 제공되는 팀 관측과 planner 상태만 사용한다.

Neural encoder는 frozen으로 유지해 v3의 표현 회귀를 방지한다. PPO buffer에는 frozen
latent와 planner context, scalar team log-prob, compact centralized feature를 저장한다.
critic은 123-field 전체 유닛/class state와 11채널 4×4 pooled map을 사용한다. 원본 맵을
유닛마다 중복 저장하지 않아 긴 rollout의 메모리 부담을 줄인다. PPO에는 advantage 정규화,
value clipping, gradient clipping, KL 조기 중단과 NaN 검사가 적용된다.

기하학 mask는 짧은 이동의 벽·맵 경계와 역할별 성소 진입을 검사한다. 모든 Unity 물리 충돌을
재현하거나 안전을 보장하는 장치는 아니다. KEEP은 항상 허용한다. 한 step에 수정하는
유닛은 최대 1개다. 보정 가능한 상태에서 초기 team override 확률은 최대 약 23.5%
(`0.15 + 0.85 × 0.10`)이며, 실제 mask에 따라 달라진다. 이것은 모든 유닛 행동의 23.5%가
수정된다는 뜻이 아니다. 로그에 team-step과 agent-action 기준 비율을 둘 다 기록한다.

최종 default reward는 score-delta+terminal이며 Unity shaping은 0이다. gamma=.9995,
GAE lambda=.99, learning rate=1e-4, entropy=.003, KL 한도=.02를 사용한다.
설정 변경은 새 run으로 기록하며, 중단 재개에서는 의미가 바뀌는 config 변경을 거부한다.

## Curriculum 및 평가

| 단계 | 기본 최대 step | 학습 상대 | 평가 gate |
|---|---:|---|---|
| planner_preservation | 4,096 | scripted | 동일 seed baseline 보존; PPO off |
| balanced_residual | 200,704 | scripted 65% / weak 25% / full 10% | scripted baseline−5%p 이상, 점수차 baseline−5 이상, 각 진영 ≥40% |
| target_mix | 600,064 | scripted 20% / weak 30% / full 50% | target ≥50%, 각 진영 ≥30%, 점수차 ≥−5 |
| full_win70 | 1,001,472 | scripted 10% / full 90% | target ≥70%, 각 진영 ≥50%, 점수차 ≥−5 |
| robust_historical_mix | 2,000,896 | full 80% / 과거 stage snapshot 20% | target ≥85%, 각 진영 ≥70%, 점수차 ≥−5 |

각 단계에는 별도 최소 step이 있다. budget 소진으로 강제 승급하지 않는다. 초기 readiness
승급은 실력 향상 주장이 아니라 다음 상대 혼합을 시도할 조건이다. 최종 산출물은 원래
dev 3101~3105 양 진영 **9/10 이상**, 별도 confirmation 3401~3415 **26/30 이상**,
confirmation의 각 진영 **11/15 이상**을 함께 만족해야 생성한다.

confirmation도 반복적인 선택에 사용하므로 최종 test가 아니다. test 3201~3210은 최종
모델 선택 완료 후 한 번 보고하고 그 결과로 모델을 다시 선택하지 않는다. 최초 실행은
기준 정책을 scripted/full 상대의 dev/confirmation 모두에서 측정하므로 학습 update 전
80경기의 baseline 평가 시간이 필요하다.

## 실행 (이번 작업에서는 실행하지 않음)

사전 점검만 실행 — Unity 시작·학습·v6 실행 산출물 생성 없음:

```bash
./scripts/start_mappo_planner_residual_v6_background.sh --check
```

실제 백그라운드 시작:

```bash
./scripts/start_mappo_planner_residual_v6_background.sh
```

중단한 실행 재개:

```bash
./scripts/start_mappo_planner_residual_v6_background.sh --resume-latest
```

확인:

```bash
tail -f logs/mappo_planner_residual_v6/console.log
```

Unity는 `-batchmode`, graphics 유지, 학습 time scale 50 / 평가 100으로 실행한다.
작은 head와 planner의 CPU 경로를 기본으로 선택했다. 지원되는 호스트에서는
`--device mps`로 새 실행을 시작할 수 있다. 독립 실행은 같은 checkpoint 경로를 공유하지
않도록 모든 output 경로를 바꿔야 한다. 실행기는 log directory lock과 checkpoint 경로
검사를 사용한다. PID 재사용 때문에 다른 프로세스를 죽이는 동작은 하지 않는다.

SIGINT/SIGTERM은 요청을 기록하고 진행 중인 rollout/평가가 끝나는 경계에서 저장한다.
`stage_blocked`, `rollback_limit`, `target_reached`는 이미 종료된 run이므로 같은 설정의
resume를 거부한다. `budget_exhausted`는 `--max-env-steps`를 늘려 재개할 수 있다.

Unity 내부 물리 상태는 저장하지 않으므로 resume는 새 episode에서 시작한다.
이때 폐기한 진행 중 seed/step을 로그로 남긴다. optimizer/RNG/seed 분포 복원은 지원하지만
중단하지 않은 게임과 bitwise 동일한 trajectory 재개를 주장하지 않는다.

## 기존과 같은 산출물 배치

| 경로 | 내용 |
|---|---|
| `logs/mappo_planner_residual_v6/training.jsonl` | update별 PPO, rollout W/D/L, seed 분포, 두 가지 override율; start/resume/승급/rollback/중단 이벤트 |
| `logs/mappo_planner_residual_v6/run_summary.json` | 현재 stage/step/status, best/latest 평가, config와 runtime 상태 |
| `logs/mappo_planner_residual_v6/*_eval_step_*.json` | 기존 `blackout.paired_series.v1` 및 `blackout.episode.v1` 원자료, terminal winner와 양 진영 통계 |
| `logs/mappo_planner_residual_v6/console.log`, `training.pid` | 백그라운드 출력과 실행 중 PID |
| `logs/mappo_planner_residual_v6/training_episodes.jsonl` | 완료된 학습 경기의 seed/진영/상대/승패/점수 |
| `logs/mappo_planner_residual_v6/run_config.json`, `source_snapshot.zip` | 실행 config와 실제 사용 코드/hash 보존 |
| `logs/mappo_planner_residual_v6/recovery_*.jsonl` | crash 시 saved model보다 앞선 로그를 복구용으로 보존 |
| `checkpoints/mappo_planner_residual_v6_latest.pt` | 재개 기준 |
| `checkpoints/mappo_planner_residual_v6_target_best.pt` | 확인 평가를 통과한 target best alias |
| `checkpoints/mappo_planner_residual_v6_stage_best.pt` | 현재/최근 stage best alias |
| `checkpoints/mappo_v6_snapshots/` | 변경되지 않는 best 사본 및 historical opponents |
| `checkpoints/mappo_win_85_vs_win70_v6.pt` | 최종 gate 통과 시에만 저장 |
| `submission/v6/policy.py`, `checkpoint.pt` | 목표 통과 후 자동 export |
| `logs/mappo_planner_residual_v6/submission_export.json` | export hash, 원본 checkpoint hash, 제출 계약의 남은 제한 |

checkpoint 외형은 기존 `blackout.checkpoint.v1`을 유지하고 v6 model/optimizer/runtime과
실제 실행 action 계약을 추가한다. 학습 이벤트 schema는 `blackout.mappo_planner_residual.v6`.
v1~v5 checkpoint/log는 변경하지 않는다. `override_rate`는 기존과 같이 유닛 행동 기준,
새 `team_override_rate`는 환경 step 기준이다. PPO `samples`는 5개 독립 row가 아니라
complete team decision 수다.

수동 export:

```bash
.venv/bin/python scripts/export_mappo_v6.py \
  --checkpoint checkpoints/mappo_planner_residual_v6_target_best.pt \
  --output /private/tmp/blackout-v6-export
```

두 파일 export는 numpy+torch만 필요하고 추론에서 파일·네트워크·subprocess를 사용하지
않는다. 이름 기반 `act(obs)`는 dict 순서를 canonicalize한다. 공식 torch
`forward(vector, graphic)` 경로에는 canonical batch row가 필요하며 planner는 비학습
episode state를 유지한다. **이 두 요구사항의 공식 대회 허용 여부는 로컬 코드로 확정할 수
없다.** 단독 로딩과 실제 행동 누락 문제는 해결했지만 공식 규정 승인이나 순수 stateless
신경망 제출로 바뀌었다고 주장하지 않는다.

## 검증 방법과 근거

`tests/test_mappo_v6.py`는 Unity를 시작하지 않는 합성 환경을 사용한다. 실제 v6 학습,
승률 향상, 최적 탐색률은 아직 검증되지 않았다. 구현된 대책이 성능 목표를 보장하지 않으며
실행 후 기존과 같은 평가 로그로 판단해야 한다.

이번 작업의 검증 결과: 전체 **200개 테스트 통과**(v6 테스트 17개 포함), shell 구문 검사,
`git diff --check`, background launcher의 `--check` 통과. 사전 점검은
`training_started=false`, train/dev/confirmation/test seed 수 `40/5/15/10`을 반환했다.
v6 Unity 실행, 학습 로그 및 학습 checkpoint는 생성하지 않았다.

참고한 1차 자료:

- [Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201): 기존 제어기를 보존한 채 residual을 학습하는 설계 근거. 이 게임에서의 성능 보장은 아님.
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347): 실제 수집 policy의 likelihood ratio와 clipped surrogate. v6 mixture 확률도 동일하게 재평가.
- [Prioritized Level Replay 공식 구현](https://github.com/facebookresearch/level-replay): 훈련 환경 난도에 따라 seed 표집을 조정하는 근거. v6는 단순 승패 EMA 방식이며 논문의 PLR 전체 구현이라고 주장하지 않음.
- [Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264): 적은 평가 표본의 불확실성과 점 추정의 한계. v6는 독립 confirmation seed와 기존 paired CI를 함께 사용.
