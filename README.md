# 2026-IST-tech-RL

BlackOut 5대5 환경에서 scripted planner, IPPO/MAPPO 및 planner residual PPO를
구현하고 평가하는 연구 작업 공간입니다. 현재 목표는 고정 상대
`checkpoints/win_70_vs_scripted.pt`를 상대로 dev 10경기 중 9승 이상을 달성하는 것입니다.

## 현재 상태 — v7 파일럿 완료 (2026-09-19 확인)

v7-1 파일럿 9개와 오류 수정 후 v7-2 실행 4개가 등록된 step 예산을 완료했고,
최종 체크포인트 저장까지 확인했습니다. 다음 단계는 두 버전의 dev 성능 평가입니다.
**[v7 파일럿 실행 방법](#파일럿-한-번에-실행)**의 한 줄 명령으로 완료된 학습을 건너뛰고
평가까지 백그라운드에서 실행할 수 있습니다. 별도로 **[4,600만 스텝 본실험 실행](#대규모-본실험-실행)**도 준비했습니다.
본실험은 2026-09-21에 사용자 요청으로 정상 중단했습니다. A0·A1 시드 11은 각각 200만
스텝 완료, A2 시드 11은 **147,456스텝**에서 저장했습니다. 아래 본실험 명령은
가속 설정으로 이 지점부터 재개합니다. 본실험 완료나 승률 개선을 의미하지 않습니다.

## v6 실행 기록 — 목표 미달 (2026-09-10 갱신)

v6는 **centralized critic을 사용하는 coordinated team residual PPO**입니다.
장기 실행은 2026-09-09에 `stage_blocked`로 종료됐으며, 최종 target 승률은 **50%**입니다.
단계별 승급 기준을 충족하지 못한 안전 중단으로, 목표 달성이나 전체 학습 예산 완주는 아닙니다.

| 항목 | v6 실제 실행 결과 |
| --- | --- |
| 실행 시간 (한국 시간) | 09-08 16:21 → 09-09 11:52, 약 19시간 31분 (baseline 및 중간 평가 포함) |
| 학습량 | 1,159,168 environment step, 566 update (PPO 564회) |
| 진행 단계 | planner 보존 → balanced residual → target 혼합 통과, `full_win70`에서 종료 |
| 종료 사유 | `full_win70`의 1,001,472-step 예산 소진, 승률 70%·각 진영 50% gate 미달 |
| 최종 target dev | 5승 5패, 평균 점수 차 0.0; A 3/5, B 2/5 |
| target-best | 학습 시작 시점(global step 0)의 baseline 5/10 유지 |
| 성능 회귀·복구 | 확인 평가 0/30 및 2/30 이후 롤백 2회, 최종 dev 5/10 회복 |
| PPO 수정 행동 비율 | team step 기준 46.82%, agent action 기준 9.36% |
| 완료된 학습 경기 | 226경기, 37승 1무 188패 (탐색·상대 혼합 포함, dev 평가와 구분) |
| 목표 모델·제출·test | 목표 미달로 최종 모델 및 `submission/v6/` 미생성, 최종 test 미실행 |

근거: [실행 요약](logs/mappo_planner_residual_v6/run_summary.json),
[최종 target 평가](logs/mappo_planner_residual_v6/target_eval_step_1159168.json),
[학습·승급·롤백 기록](logs/mappo_planner_residual_v6/training.jsonl),
[완료 경기 기록](logs/mappo_planner_residual_v6/training_episodes.jsonl).

탐색 부족과 초기 단계 정체는 완화됐지만 planner보다 좋은 수정 행동의 학습에는
성공하지 못했습니다. 다음 분석 과제는 수정 행동의 장기 보상 기여, 탐색 학습과
deterministic 평가의 차이, B 진영 실패 및 롤백 직전 정책 변화입니다. 세부 원인은
아직 확정하지 않았습니다. **단순한 step 상한 증가는 종료된 이 run의 재개 방법이 아닙니다.**

### 완료된 MAPPO 장기 실험

| 세대 | 학습 step | 최종 target dev | 핵심 관찰 | 실행 근거 |
| --- | ---: | --- | --- | --- |
| v1 | 2,000,384 | 0/10 | rollout마다 환경 초기화, 완료된 학습 경기 0개 | [요약](logs/mappo_vs_win70/run_summary.json) |
| v2 | 2,000,896 | 0/10 | episode 수집은 복구했지만 neural core가 상대 planner 실력을 재현하지 못함 | [요약](logs/mappo_vs_win70_v2/run_summary.json) |
| v3 | 3,000,320 | 0/10 | planner 모방 정확도가 closed-loop 성능으로 이어지지 않음 | [요약](logs/mappo_teacher_curriculum_v3/run_summary.json) |
| v4 | 100,352 | 5/10 | planner 보존, 개선 신호 부족으로 첫 단계 중단 | [요약](logs/mappo_planner_residual_v4/run_summary.json) |
| v5 | 317,440 | 5/10 | 탐색 부족, 롤백 1회 후 단계 상한 중단 | [요약](logs/mappo_planner_residual_v5/run_summary.json) |
| v6 | 1,159,168 | 5/10 | 탐색 증가에도 best 개선 없음, 롤백 2회 후 full_win70 단계 중단 | [요약](logs/mappo_planner_residual_v6/run_summary.json) |

위 수치는 각 실행의 최종 target 평가이며 동일 budget의 통제 실험은 아닙니다.
이전 scripted 상대 승률과 현재 win70 상대 승률, 서로 다른 seed 집합의 결과도
구분해야 합니다. 세대별 문제·해결 과정·남은 한계는 [history.md](history.md)를 참고하세요.

### 구현과 문서 위치

| 역할 | 주요 코드·자료 |
| --- | --- |
| 환경·관측·행동·게임 계약 | [env.py](blackout_rl/env.py), [observation.py](blackout_rl/observation.py), [game_spec.md](game_spec.md), [versions.md](versions.md) |
| scripted planner와 팀 전략 | [navigation.py](blackout_rl/navigation.py), [scripted_fsm.py](blackout_rl/scripted_fsm.py), [strategy.py](blackout_rl/strategy.py) |
| IPPO·기존 MAPPO | [ippo_model.py](blackout_rl/ippo_model.py), [ppo.py](blackout_rl/ppo.py), [mappo.py](blackout_rl/mappo.py) |
| v6 actor·collector·PPO | [mappo_v6.py](blackout_rl/mappo_v6.py), [mappo_v6_training.py](blackout_rl/mappo_v6_training.py) |
| v6 승급·seed 분리 | [mappo_curriculum_v6.py](blackout_rl/mappo_curriculum_v6.py), [seed_splits_v6.json](configs/seed_splits_v6.json) |
| v6 학습·백그라운드 실행 | [학습기](scripts/train_mappo_planner_residual_v6.py), [launcher](scripts/launch_mappo_v6_background.py), [실행 안내](#v6-training) |
| 평가·저장·export | [evaluator.py](eval/evaluator.py), [mappo_v6_artifacts.py](blackout_rl/mappo_v6_artifacts.py), [export CLI](scripts/export_mappo_v6.py) |
| 변경 근거·검증 기록 | [v6 구현 보고서](reports/mappo_planner_residual_v6_plan_changes.md), [v6 테스트](tests/test_mappo_v6.py), [실행 이력](history.md) |

v6는 frozen encoder와 원래 planner를 보존하고 `KEEP + 5유닛 × 8수정`의 41-way 팀
분포를 최적화합니다. 실제 planner 문맥, 실패 seed 재표집, B 진영 60% update 노출,
별도 confirmation 15개 맵, baseline 기준 초기 승급과 롤백을 구현했습니다.
420초 시간 기준 및 별도 `GlobalLocalMapEncoder`의 crop Y축 계약도 수정했습니다.

구현 당시 전체 테스트 200개(v6 17개 포함), shell 문법 및 사전 점검이 통과했습니다.
이는 구현 검증 기록이며 목표 승률 달성의 증거는 아닙니다. v6 구현 보고서의
“미실행” 표기는 09-08 작성 당시 상태이고, 이후 실측 결과는 위 로그와 history에 반영돼 있습니다.
Phase 1–3의 제출 테스트 역시 당시 모델의 검증이며 v6의 공식 제출 승인을 의미하지 않습니다.
v6 export는 planner와 residual을 함께 포함하지만 canonical batch 순서와 stateful
planner의 공식 허용 여부는 미확인입니다. 엄격한 독립 decentralized actor나 순수
stateless neural policy로 표현하지 않습니다.

## 구현·실험 체크리스트 — Phase 1–3 기록

아래 체크 표시는 해당 구현·테스트·실험 수행의 완료를 뜻합니다. 성능 향상에 실패한
ablation도 포함하며, 목표 승률 달성이나 최종 제출 준비 완료를 뜻하지 않습니다.
당시 구조·설계 원칙을 보존한 기록이므로 최신 v6의 팀 단위 actor와 B 진영 60% 표집은
위 구현 설명 및 v6 실행 설정을 기준으로 합니다.

- [x]  Phase 1

| ID | Todo | 산출물 및 완료 조건 |
| --- | --- | --- |
| PREP-01 | ☑ 게임/Python API commit, executable hash, Python·PyTorch·CUDA·Unity 버전 고정 | [`versions.md`](versions.md), [`requirements.lock`](requirements.lock), [`docker-image.env`](docker-image.env) |
| PREP-02 | ☑ 로컬 또는 Docker에서 random policy 1경기 실행 | [`logs/prep04_07_contract.json`](logs/prep04_07_contract.json): corrected seed contract로 reset부터 terminal까지 실행, seed·점수·winner 저장 및 검증 |
| PREP-03 | ☑ 게임 규칙을 RL state/action/event 관점으로 문서화 | [`game_spec.md`](game_spec.md): 유닛, 아이템, 창고, 성소, 전투 상성, 20초 이벤트 |
| PREP-04 | ☑ 관측값 96개 vector field와 11개 map channel 파서 작성 | [`blackout_rl/observation.py`](blackout_rl/observation.py), [`reports/prep04_observation_contract.md`](reports/prep04_observation_contract.md): field shape·범위·team perspective 테스트 통과 |
| PREP-05 | ☑ PettingZoo API contract test 작성 | [`tests/test_contract.py`](tests/test_contract.py), [`reports/prep05_pettingzoo_contract.md`](reports/prep05_pettingzoo_contract.md): reset/step/termination, agent 수, dtype, action 범위, seed 재현성 통과 |
| PREP-06 | ☑ self-ID 및 batch-order 실험 | [`reports/prep06_agent_identity_order.md`](reports/prep06_agent_identity_order.md): 5+5 canonical batch·side perspective·terminal 순서 검증, slot ID 보완 확정 |
| PREP-07 | ☑ 행동 의미 테스트 | [`reports/prep07_action_semantics.md`](reports/prep07_action_semantics.md): `(0,0)`, 작은/큰/초과 vector와 8방향 displacement 측정 |
| PREP-08 | ☑ 실제 terminal winner 기반 evaluator 작성 | [`eval/paired_series.py`](eval/paired_series.py), [`reports/prep08_terminal_evaluator.md`](reports/prep08_terminal_evaluator.md): terminal winner만 사용하고 side-swapped model 결과 검증 |
| PREP-09 | ☑ Python score-delta team reward 구현 | [`blackout_rl/reward.py`](blackout_rl/reward.py), [`reports/prep09_score_delta_reward.md`](reports/prep09_score_delta_reward.md): 증가·감소·약탈·terminal 부호 테스트 통과 |
| PREP-10 | ☑ 통합 logging schema 작성 | [`schemas/episode_v1.schema.json`](schemas/episode_v1.schema.json), [`reports/prep10_logging_schema.md`](reports/prep10_logging_schema.md), [`logs/prep08_10_paired_seed_810.json`](logs/prep08_10_paired_seed_810.json): seed·side·opponent·score·winner·length·checkpoint SHA 검증 |
| PREP-11 | ☑ 처리량 benchmark | [`scripts/benchmark_env.py`](scripts/benchmark_env.py), [`logs/prep11_benchmark.json`](logs/prep11_benchmark.json), [`reports/prep11_throughput_benchmark.md`](reports/prep11_throughput_benchmark.md): 환경 1/2개의 steps/sec, host CPU/RSS, Apple GPU utilization, full-episode 비용 측정 |
| PREP-12 | ☑ 첫 actor/critic 인터페이스 설계 | [`blackout_rl/model_contract.py`](blackout_rl/model_contract.py), [`schemas/checkpoint_v1.schema.json`](schemas/checkpoint_v1.schema.json), [`reports/prep12_actor_critic_interface.md`](reports/prep12_actor_critic_interface.md): tensor/slot/action/checkpoint 계약과 실제 torch round-trip 검증 |
| PREP-13 | ☑ 평가 계약에 없는 항목 확인 | [`reports/prep13_evaluation_contract_checklist.md`](reports/prep13_evaluation_contract_checklist.md): 확인된 계약, 공식 미정 항목, 확인 전 fail-closed 결정을 분리 관리 |
| PREP-14 | ☑ random-vs-random paired-seed 평가 | [`logs/prep14_random_paired_5seeds.json`](logs/prep14_random_paired_5seeds.json), [`reports/prep14_random_paired_evaluation.md`](reports/prep14_random_paired_evaluation.md): 5 paired seeds·10경기의 side별 승률/점수 차 측정, 균형 side 배정과 evaluator attribution 검증 |

#### 준비 단계의 필수 테스트

`tests/test_contract.py`에는 적어도 다음이 포함되어야 합니다.

```
test_observation_shape_and_dtype
test_model_input_layout_hwc_to_chw
test_seed_reproduces_initial_map
test_nonzero_action_is_normalized
test_zero_action_stops_unit
test_agent_batch_order
test_team_side_swap
test_score_delta_sign
test_terminal_winner_matches_game_result
test_paired_seed_evaluation
test_checkpoint_round_trip
```

#### 준비 단계 완료 기준

- 같은 seed의 초기 맵과 아이템 위치가 재현
- 실제 winner와 최종 점수 기반 판정이 일치
- score 감소가 음의 학습 보상으로 반영
- 제출 모델이 5개 유닛을 구분할 수 있는 방법이 확정
- random policy를 다수 에피소드 실행해도 agent 누락, NaN, deadlock이 없음
- paired-seed evaluator가 side 기준이 아닌 모델 기준 결과를 반환

#### 학습 시 사용 예정인 지표

```jsx
win / draw / loss, 평균 최종 점수 차
100점 조기 종료 비율, 평균 episode 길이
팀 전체 배터리 적재량, 약탈로 잃은 점수
유닛별 역할·사망·변신 횟수, A/B side별 성능
상대 checkpoint별 성능
```

- [x]  Phase 2
    - [x]  2-1
    
    | ID | Todo | 산출물 및 완료 조건 |
    | --- | --- | --- |
    | BASE-S01 | ☑ `Random`, `NoOp`, `FixedDirection` policy 구현 | [`blackout_rl/policy.py`](blackout_rl/policy.py), [`reports/base_s01_policy_primitives.md`](reports/base_s01_policy_primitives.md): 공통 evaluator protocol, deterministic/범위/누락-agent 회귀 테스트 통과 |
    | BASE-S02 | ☑ semantic map decoder 구현 | [`blackout_rl/semantic_map.py`](blackout_rl/semantic_map.py), [`reports/base_s02_semantic_decoder.md`](reports/base_s02_semantic_decoder.md): 좌표 반전 계약과 벽·창고·유닛·Battery·특수 아이템 위치 추출 검증 |
    | BASE-S03 | ☑ 이동 가능 영역과 path planner 구현 | [`blackout_rl/navigation.py`](blackout_rl/navigation.py), [`scripts/verify_base_s01_s03.py`](scripts/verify_base_s01_s03.py), [`logs/base_s01_s03_live_navigation.json`](logs/base_s01_s03_live_navigation.json), [`reports/base_s03_single_unit_navigation.md`](reports/base_s03_single_unit_navigation.md): A*·waypoint·replan 신호 및 실제 단일 유닛 wall 우회 Battery pickup 통과 |
    | BASE-S04 | ☑ stuck detector 구현 | [`blackout_rl/coordination.py`](blackout_rl/coordination.py), [`reports/base_s04_stuck_detector.md`](reports/base_s04_stuck_detector.md): 위치 window 판정, 임시 장애물 우회 및 목표 재선정 연결 검증 |
    | BASE-S05 | ☑ team-local slot과 unit state 추적 | [`blackout_rl/team_state.py`](blackout_rl/team_state.py), [`reports/base_s05_team_state.md`](reports/base_s05_team_state.md): 위치·보유 item·class·역할·목표를 canonical team slot으로 추적 |
    | BASE-S06 | ☑ task assignment 구현 | [`blackout_rl/coordination.py`](blackout_rl/coordination.py), [`reports/base_s06_s07_assignment_roles.md`](reports/base_s06_s07_assignment_roles.md): A* path distance 기반 deterministic greedy 고유 할당 및 도달 불가/예약 목표 테스트 |
    | BASE-S07 | ☑ 기본 역할 구성 | [`blackout_rl/team_state.py`](blackout_rl/team_state.py), [`logs/base_s04_s07_s13_coordination.json`](logs/base_s04_s07_s13_coordination.json): worker 3·guard 1·carrier 1 역할과 실제 5유닛 동시 Battery 회수 검증 |
    | BASE-S08 | ☑ 노동자 FSM 구현 | [`blackout_rl/scripted_fsm.py`](blackout_rl/scripted_fsm.py), [`reports/base_s08_worker_fsm.md`](reports/base_s08_worker_fsm.md): 노동자 3명의 고유 Battery 회수→창고 적재→재탐색 및 포화 창고 재배정 검증 |
    | BASE-S09 | ☑ 경비원 FSM 구현 | [`reports/base_s09_guard_fsm.md`](reports/base_s09_guard_fsm.md), [`logs/base_s08_s10_role_fsm.json`](logs/base_s08_s10_role_fsm.json): 중앙 성소 Hunter 변신→창고 순찰과 적 추격 전환 검증 |
    | BASE-S10 | ☑ 전달자 FSM 구현 | [`reports/base_s10_carrier_fsm.md`](reports/base_s10_carrier_fsm.md), [`logs/base_s08_s10_role_fsm_trajectory.jsonl`](logs/base_s08_s10_role_fsm_trajectory.jsonl): 본진 Carrier 변신→원거리 Battery 회수·적재 및 적 접근 회피 검증 |
    | BASE-S11 | ☑ 20초 흡수 주기 전략 구현 | [`blackout_rl/strategy.py`](blackout_rl/strategy.py), [`reports/base_s11_absorption_strategy.md`](reports/base_s11_absorption_strategy.md): `time_left` 기반 20초 phase 복원, 직전 적재·직후 수집·조건부 약탈과 실제 첫 흡수 통과 검증 |
    | BASE-S12 | ☑ 위험 지도 구현 | [`reports/base_s12_danger_map.md`](reports/base_s12_danger_map.md), [`logs/base_s11_s12_strategy.json`](logs/base_s11_s12_strategy.json): 적 위치 거리 비용과 역할별 weighted A*를 실제 중앙 적 위치에서 경로 비교 검증 |
    | BASE-S13 | ☑ scripted trajectory recorder 구현 | [`blackout_rl/trajectory.py`](blackout_rl/trajectory.py), [`logs/base_s13_coordination_trajectory.jsonl`](logs/base_s13_coordination_trajectory.jsonl), [`reports/base_s13_trajectory_recorder.md`](reports/base_s13_trajectory_recorder.md): obs/action/role/target/reward/score/seed JSONL 기록과 hash·round-trip 검증 |
    | BASE-S14 | ☑ scripted-vs-random 평가 | [`eval/scripted_series.py`](eval/scripted_series.py), [`logs/base_s14_scripted_vs_random.json`](logs/base_s14_scripted_vs_random.json), [`reports/base_s14_scripted_vs_random.md`](reports/base_s14_scripted_vs_random.md): held-out 5 paired seeds·10경기 전승, 평균 점수 차 +97.7, model A/B 모두 전승 및 물리 side 50/50 |
    | BASE-S15 | ☑ 특수 아이템 정책을 선택적으로 추가 | [`logs/base_s15_special_item_ablation.json`](logs/base_s15_special_item_ablation.json), [`reports/base_s15_special_item_ablation.md`](reports/base_s15_special_item_ablation.md): 옵션 동작 검증 후 평균 점수 차 −1.7·평균 +74.1 step으로 개선 없어 기본 battery-only 유지 |
    
    첫 scripted agent- 특수 아이템과 적극적인 약탈을 우선 제외. 아래 동작 확인 우선
    
    ```
    3명: 서로 다른 배터리 수집 → 가까운 아군 창고에 전달
    1명: 중앙 성소 → 경비원 변신 → 열린 창고 또는 운반 경로 방어
    1명: 본진 성소 → 전달자 변신 → 먼 배터리 운반
    ```
    
    - [x]  2-2
    
    | ID | Todo | 산출물 및 완료 조건 |
    | --- | --- | --- |
    | BASE-R01 | ☑ 9-way categorical action distribution 구현 | [`blackout_rl/action_distribution.py`](blackout_rl/action_distribution.py), [`reports/base_r01_r05_r14_policy_model.md`](reports/base_r01_r05_r14_policy_model.md): sampling·argmax·log-prob·entropy와 `(dx,dy)` 변환 검증 |
    | BASE-R02 | ☑ vector/entity encoder 구현 | [`blackout_rl/ippo_model.py`](blackout_rl/ippo_model.py): 10개 unit block과 score·time·class context 분리 인코딩 검증 |
    | BASE-R03 | ☑ semantic CNN encoder 구현 | `11×96×96` 입력의 고정 길이 latent 변환 검증 |
    | BASE-R04 | ☑ slot embedding 또는 specialized head 구현 | 동일 observation에서 5개 slot의 출력 분화와 embedding gradient 검증 |
    | BASE-R05 | ☑ shared actor와 local critic 구현 | 단일 actor parameter 공유, logits `(B,9)`와 local value `(B,)` 검증 |
    | BASE-R06 | ☑ PettingZoo parallel rollout collector 구현 | [`blackout_rl/rollout.py`](blackout_rl/rollout.py), [`reports/base_r06_r09_ppo_engine.md`](reports/base_r06_r09_ppo_engine.md): agent별 obs/action/logprob/value/reward/mask와 10-agent parallel step 검증 |
    | BASE-R07 | ☑ GAE와 episodic buffer 구현 | 여러 episode를 가로지르는 rollout, truncation bootstrap과 termination 차단 검증 |
    | BASE-R08 | ☑ PPO update 구현 | [`blackout_rl/ppo.py`](blackout_rl/ppo.py): clipped policy/value loss, entropy, minibatch epoch, gradient clipping과 parameter update 검증 |
    | BASE-R09 | ☑ PPO 진단 logging | [`tests/test_ppo_training.py`](tests/test_ppo_training.py): KL, clip fraction, entropy, explained variance, gradient norm의 strict JSONL round-trip 검증 |
    | BASE-R10 | ☑ score-delta team reward와 terminal reward 연결 | [`blackout_rl/training_reward.py`](blackout_rl/training_reward.py), [`reports/base_r10_r12_fixed_opponent_bc.md`](reports/base_r10_r12_fixed_opponent_bc.md): 팀 공유 score-delta·winner bonus와 Unity shaping off/on/combined 검증 |
    | BASE-R11 | ☑ 고정 scripted opponent 연결 | [`blackout_rl/frozen_opponent.py`](blackout_rl/frozen_opponent.py): trainable state 없는 battery-only scripted opponent와 episode reset·fingerprint 불변 검증 |
    | BASE-R12 | ☑ scripted trajectory 기반 BC pretraining 실험 | [`blackout_rl/behavior_cloning.py`](blackout_rl/behavior_cloning.py), [`logs/base_r12_bc_warm_start.json`](logs/base_r12_bc_warm_start.json): 별도 held-out trajectory에서 scratch 대비 NLL `2.1738→2.1145`, 동일 environment-step budget 기록 |
    | BASE-R13 | ☑ checkpoint와 experiment registry 구현 | [`blackout_rl/experiment_registry.py`](blackout_rl/experiment_registry.py), [`experiments/registry.jsonl`](experiments/registry.jsonl), [`reports/base_r13_r15_registry_evaluation.md`](reports/base_r13_r15_registry_evaluation.md): config·seed·git SHA·opponent ID와 artifact/config SHA를 checkpoint/registry에 저장·검증 |
    | BASE-R14 | ☑ 제출용 deterministic policy wrapper 구현 | [`blackout_rl/model_contract.py`](blackout_rl/model_contract.py), [`tests/test_ippo_model.py`](tests/test_ippo_model.py): `forward(vector, graphic) → (B,2)`, argmax와 `[-1,1]` 계약 검증 |
    | BASE-R15 | ☑ random/scripted/IPPO 평가 matrix 작성 | [`eval/policy_matrix.py`](eval/policy_matrix.py), [`logs/base_r15_policy_matrix_seed1401.json`](logs/base_r15_policy_matrix_seed1401.json): 공통 environment seed·opponent artifact/RNG seed·side swap으로 3개 policy 비교, scripted > random > PPO-step-0 IPPO baseline |
    
    #### IPPO actor 권장 구조
    
    ```
    semantic map ── CNN ─────────────┐
                                     ├─ fusion MLP ─ actor logits[9]
    10 unit entities ─ entity MLP ──┤
                                     └────────────── local value[1]
    self class / score / time ───────┤
    slot embedding ──────────────────┘
    ```
    
    Vector 전체를 그대로 한 번에 MLP에 넣는 모델도 smoke baseline으로는 가능하지만, 10개 유닛 block을 entity 단위로 분리하면 이후 attention이나 centralized critic으로 확장하기 편함.
    
    #### IPPO 학습 시 권장 원칙
    
    - 하나의 학습 팀만 update하고 상대 scripted agent는 고정
    - 팀원 5명에게 동일한 팀 reward를 주되 actor의 slot은 구분
    - PPO epoch 수를 지나치게 높이지 않고 KL과 clip fraction을 감시
    - advantage와 value normalization을 사용
    - rollout이 최소한 한 번의 20초 흡수 주기를 포함하도록 구성
    - gamma는 “step 수”가 아니라 실제 초 단위 horizon을 기준으로 결정
    - 특수 아이템과 적극적 약탈은 배터리 수집이 먼저 학습된 후 curriculum으로 추가
    
    MAPPO는 원래 협력형 multi-agent 환경에서 검증된 방법이지만, BlackOut에서는 상대를 고정한 상태에서 한 팀의 5명을 협력 집단으로 보면 자연스럽게 적용가능. 이후 self-play에서 경쟁적 비정상성을 추가 예정. PPO/MAPPO 구현 세부사항은 PPO·GAE, CleanRL, MAPPO 공식 구현을 기준으로 삼는 것이 좋을 듯
    
- [x]  Phase 3
    - [x]  MAPPO와 CTDE
    
    | ID | Todo | 산출물 및 완료 조건 |
    | --- | --- | --- |
| AGENT-01 | ☑ centralized state builder 구현 | [`blackout_rl/mappo.py`](blackout_rl/mappo.py), [`reports/agent01_05_mappo_ctde.md`](reports/agent01_05_mappo_ctde.md): 10개 유닛·score·map·전체 class의 123-field critic state 검증 |
| AGENT-02 | ☑ decentralized actor + centralized critic 구현 | 제출 경로는 local actor만 받고 centralized critic은 joint 학습에서만 사용 |
| AGENT-03 | ☑ joint rollout과 team mask 검증 | [`tests/test_phase3_mappo.py`](tests/test_phase3_mappo.py): death mask·respawn·termination GAE와 실제 Unity joint rollout/update 통과 |
| AGENT-04 | ☑ IPPO checkpoint에서 MAPPO 초기화 | 이전 전후 actor logits bit-identical, centralized critic만 신규 초기화 |
| AGENT-05 | ☑ IPPO 대비 MAPPO ablation | [`logs/phase3_mappo_ablation_128.json`](logs/phase3_mappo_ablation_128.json): 동일 초기 actor·seed·scripted opponent·128-step budget·5 paired dev seeds 비교(단기 결과 동률) |
    
    Centralized critic은 학습 중 전체 팀 또는 전체 게임 정보를 사용할 수 있지만, actor에는 제출 환경에서 실제로 제공되는 관측만 전달. 
    
    - [x]  표현력과 전략 개선
    
    | ID | Todo | 산출물 및 완료 조건 |
    | --- | --- | --- |
    | AGENT-06 | ☑ unit entity attention 실험 | [`reports/agent06_08_11_representation_ablation.md`](reports/agent06_08_11_representation_ablation.md), [`logs/phase3_representation_ablation.json`](logs/phase3_representation_ablation.json): 동일 budget dev 10경기에서 replay 정확도 `89.84→90.25%`지만 점수차 `-90.3→-93.4`로 악화되어 미승격 |
| AGENT-07 | ☑ self-relative position feature 추가 | [`blackout_rl/representation.py`](blackout_rl/representation.py), [`tests/test_phase3_representation.py`](tests/test_phase3_representation.py): team-local self slot 기준 10개 유닛 상대 좌표 검증 |
    | AGENT-08 | ☑ global map + local crop 구조 실험 | global/local replay 정확도 `91.04%`였으나 dev 점수차 `-95.4`로 legacy 대비 5.1점 악화되어 미승격 |
| AGENT-09 | ☑ absorption phase feature 추가 | normalized `time_left`에서 반복되는 20초 sin/cos 경계 테스트 통과 |
| AGENT-10 | ☑ auxiliary target logging | 역할·보유 item·다음 흡수 시간·score delta head/loss 및 strict JSONL logger 검증 |
    | AGENT-11 | ☑ auxiliary loss 선택 실험 | role·holding item·absorption time·score delta를 독립 평가했으나 모두 0승, baseline 대비 점수차 `-0.1~-1.0`; 선택 target 없음 |
| AGENT-12 | ☑ 역할 분화 시각화 | [`reports/phase3_role_differentiation.svg`](reports/phase3_role_differentiation.svg), [`logs/phase3_role_metrics.json`](logs/phase3_role_metrics.json), [`reports/agent07_12_representation_features.md`](reports/agent07_12_representation_features.md): 실제 325 slot 관측 분석 |
    
    현재 제출 인터페이스는 명시적인 hidden state와 reset API를 제공하지 않으므로, stateful model 허용 여부와 episode reset 감지가 검증된 후에만 RNN 도입.
    
    - [x]  Curriculum과 reward 개선
    
    | ID | Todo | 산출물 및 완료 조건 |
    | --- | --- | --- |
| AGENT-13 | ☑ curriculum 단계 정의 | [`blackout_rl/curriculum.py`](blackout_rl/curriculum.py), [`reports/agent13_17_curriculum_contract.md`](reports/agent13_17_curriculum_contract.md): random→weak scripted→full scripted→frozen RL 순서와 승격 gate 검증 |
    | AGENT-14 | ☑ 초기 navigation shaping 실험 | [`reports/agent14_navigation_shaping_ablation.md`](reports/agent14_navigation_shaping_ablation.md), [`logs/phase3_agent14_navigation.json`](logs/phase3_agent14_navigation.json): 동일 512-step budget에서 navigation arm 점수차 `-94.3`, score-only `-89.7`로 미승격 |
| AGENT-15 | ☑ shaping annealing 구현 | linear schedule 종료 후 navigation weight 0, score+terminal reward만 잔존 |
| AGENT-16 | ☑ curriculum별 별도 evaluator 구성 | stage/opponent별 결과 분리와 중복 stage fail-closed 검증 |
| AGENT-17 | ☑ 약탈·특수 아이템 curriculum 추가 | common-seed/common-budget 무회귀 승격 gate 구현; 기존 BASE-S15 성능 저하 증거에 따라 기본 비활성 유지 |
    
    게임 규칙 자체를 단순화한 별도 Unity 빌드를 curriculum으로 쓰면 최종 환경과 dynamics가 달라질 수 있음. 우선은 동일 환경에서 opponent 난이도, 초기 policy, reward weight를 조절.
    
    - [x]  Self-play
    
    | ID | Todo | 산출물 및 완료 조건 |
    | --- | --- | --- |
| AGENT-18 | ☑ frozen checkpoint opponent loader 구현 | [`blackout_rl/self_play.py`](blackout_rl/self_play.py): immutable SHA 검증, inference-only opponent, learner SHA 재사용 차단 |
| AGENT-19 | ☑ snapshot pool 구현 | bounded immutable pool과 configurable latest/history sampling 검증 |
| AGENT-20 | ☑ side 균형 sampling | 모든 prefix에서 A/B 노출 차이가 최대 1인 alternating sampler 검증 |
    | AGENT-21 | ☑ opponent mixture에 따른 PPO 안정화 | [`reports/agent21_24_self_play_empirical.md`](reports/agent21_24_self_play_empirical.md): 8세대 frozen mixture에서 entropy `1.699~1.929`, KL 최대 `0.0081`, value loss 최대 `0.00132`로 안정성 gate 통과 |
    | AGENT-22 | ☑ checkpoint evaluation matrix 자동화 | [`logs/phase3_agent21_24_self_play.json`](logs/phase3_agent21_24_self_play.json): 주요 세대 0/2/4/8의 모든 12 directed cell을 dev seed 양 진영 5,000-step score horizon으로 생성 |
    | AGENT-23 | ☑ exploiter/과거 상대 회귀 검사 | gen 4 기준 past gen 0/2 및 `win_70_vs_scripted.pt` 상대 5% tolerance 회귀 없음; 강도 향상 증거는 아님 |
    | AGENT-24 | ☑ 필요 시 population/PSRO 확장 판단 | 8-slot pool 포화·6세대 plateau 뒤 cyclic regression 0으로 PSRO 확장 보류 |
    
    두 최신 정책을 동시에 계속 update하는 방식은 피할 것. 상대 정책이 매 update마다 변하면 PPO가 보는 환경도 계속 변하기 때문. 학습 팀 하나와 frozen opponent를 두고, 일정 시점마다 snapshot을 pool에 추가하는 방식이 안정적. Unity ML-Agents의 self-play 문서도 snapshot window와 최신 모델 sampling 비율을 통해 안정성과 상대 다양성의 균형을 잡도록 설명.
    
    - [x]  최종 평가와 제출
    
    | ID | Todo | 산출물 및 완료 조건 |
    | --- | --- | --- |
| AGENT-25 | ☑ train/dev/test seed 집합 고정 | [`configs/seed_splits_v1.json`](configs/seed_splits_v1.json), [`reports/agent25_26_evaluation_protocol.md`](reports/agent25_26_evaluation_protocol.md): split 중복과 test 기반 모델 선택을 fail-closed로 차단 |
| AGENT-26 | ☑ paired-seed bootstrap CI 구현 | [`blackout_rl/evaluation_protocol.py`](blackout_rl/evaluation_protocol.py), [`tests/test_phase3_evaluation.py`](tests/test_phase3_evaluation.py): 개별 경기가 아닌 side-swapped seed pair 단위 재표본 검증 |
| AGENT-27 | ☑ opponent suite 구성 | [`configs/opponent_suite_v1.json`](configs/opponent_suite_v1.json), [`reports/agent18_27_self_play.md`](reports/agent18_27_self_play.md): random·scripted·IPPO·historical MAPPO·latest candidate 고유 ID 계약 |
| AGENT-28 | ☑ deterministic inference mode 확정 | [`submission/policy.py`](submission/policy.py), [`tests/test_phase3_submission.py`](tests/test_phase3_submission.py), [`reports/agent28_31_submission_validation.md`](reports/agent28_31_submission_validation.md): argmax 반복 bit-identical |
    | AGENT-29 | ☑ 모델 경량화 | [`reports/agent29_32_final_submission_review.md`](reports/agent29_32_final_submission_review.md): legacy encoder 선택으로 parameter 31.3%, CPU median latency 39.8% 감소, dev 점수 회귀 없음 |
| AGENT-30 | ☑ clean-room 제출 테스트 | 임시 빈 디렉터리에 `policy.py`·`checkpoint.pt`만 복사해 `(5,2)` 추론 통과 |
    | AGENT-31 | ☑ CPU와 GPU 양쪽 smoke test | [`logs/phase3_agent29_32_submission.json`](logs/phase3_agent29_32_submission.json): CPU·Apple MPS 실제 deterministic inference와 dtype·batch·device mismatch 검사 통과 |
    | AGENT-32 | ☑ 최종 모델 승격 회의 | gen 8이 incumbent에 0/10, 점수차 `-94.9`로 미승격; guarded incumbent도 two-file clean-room 비호환이므로 final-submission-ready 모델 없음 |

### MAPPO 장기 학습: `win_70_vs_scripted.pt` 상대 85%

[`scripts/train_mappo_vs_win70.py`](scripts/train_mappo_vs_win70.py)는 frozen
`win_70_vs_scripted.pt`를 상대로 진영을 번갈아 MAPPO를 학습한다. dev seed
5개를 side-swap한 10경기 중 최소 9승일 때만
`checkpoints/mappo_win_85_vs_win70.pt`를 저장한다. 기본 실행과 resume 방법,
체크포인트·로그 계약은
[`reports/mappo_vs_win70_training.md`](reports/mappo_vs_win70_training.md)에 정리했다.
v1 실패 원인과 persistent collector, 초기 actor, 하이퍼파라미터 및 산출물
경로 변경은
[`reports/mappo_vs_win70_v2_plan_changes.md`](reports/mappo_vs_win70_v2_plan_changes.md)에
별도로 기록했다.

v2 장기 실행에서 planner를 사용하는 상대 전략이 neural learner에 전달되지 않는
문제가 확인되어, 이후 v3에서 DAgger teacher 증류와 단계별 상대 혼합을 적용했다.
당시 구현과 변경 근거는
[`reports/mappo_teacher_curriculum_v3_plan.md`](reports/mappo_teacher_curriculum_v3_plan.md)에
정리했다.

## Reference

## 4. 우선 읽을 자료

### 필독

| 우선 | 자료 | 유형 | BlackOut에서 볼 부분 |
| --- | --- | --- | --- |
| P0 | blackout-env README, 게임 규칙, test_env.py | 직접, 공식 repo | API contract와 현재 shape |
| P0 | PettingZoo Parallel API, 환경 테스트 | 직접, 공식 문서 | simultaneous step, `parallel_api_test`, seed test. ParallelEnv “steps every live agent at once” and is based on “Partially Observable Stochastic Games.” |
| P0 | Sutton & Barto, Reinforcement Learning | 2018, 교재 | MDP/POMDP, policy gradient, actor-critic |
| P0 | PPO, Spinning Up PPO, GAE | 2016-2017 | PPO objective, rollout, GAE, KL monitoring. PPO-Clip limits incentives for the new policy to move far from the old policy, and the implementation uses GAE. |
| P0 | CleanRL PPO | 공식 구현 문서 | PPO의 코드 수준 세부사항과 logging. CleanRL’s PPO implementations include the commonly overlooked code-level implementation details. |
| P0 | MAPPO 논문, 공식 구현 | NeurIPS 2022 | IPPO/MAPPO, value normalization, 적은 PPO epoch, shared policy. MAPPO uses a centralized value function and highlights value normalization, data usage, action masking and agent-specific state as important implementation choices. |
| P0 | TorchRL Multi-Agent PPO tutorial | 공식 튜토리얼 | IPPO와 centralized critic 전환, multi-agent rollout 구조. 연속 행동을 유지한다면 Tanh-Normal도 참고. The tutorial covers both independent and centralized critics and uses a Tanh-Normal distribution for bounded continuous actions. |
| P0 | Revisiting Parameter Sharing in MARL | 2020, arXiv | 현재 self-ID 누락 문제를 이해하는 데 직접적 |

MAPPO 연구는 주로 협력형 benchmark를 대상으로 하므로 BlackOut 전체와 정확히 같지는 않음. 다만 **한 팀의 5개 유닛을 협력 집단으로 보고 상대 정책을 환경의 일부로 고정**하면 가장 자연스러운 baseline.

### 평가와 self-play 단계

| 우선 | 자료 | 활용 |
| --- | --- | --- |
| P1 | Unity ML-Agents self-play configuration | checkpoint window와 최신 상대 혼합 비율 설계. Unity recommends balancing training stability against opponent diversity through saved snapshots, a snapshot window and a latest-model sampling ratio. |
| P1 | Empirical Design in Reinforcement Learning | train seed와 environment seed 분리, 개별 run과 신뢰구간 보고. Agent behavior can differ substantially across random seeds, and the paper recommends reporting variability and setting environment and agent seeds separately. |
| P1 | SMACv2 | 절차 생성 맵의 train/eval seed 분리와 closed-loop 정책 검증. SMACv2 procedurally generates scenarios and evaluates generalization to previously unseen settings. |
| P2 | PSRO Survey | 단일 self-play가 특정 상대에 과적합한 뒤 population training으로 확장할 때 참고. PSRO focuses learning on a sufficient subset of strategies for large games. |

### 유사 게임 참고자료

- Lux AI Challenge S3: 1대1 자원 수집, 유닛 할당, opponent adaptation. Scripted baseline과 평가 도구 참고. Lux S3 is a 1v1 resource gathering and allocation competition with partial observability and randomized parameters.
- Pommerman: 팀 경쟁, sparse reward, rule-based 안전 계층과 curriculum 참고. Pommerman combines cooperative and competitive agents on randomly generated grid maps.
- Melting Pot: unfamiliar opponent와 held-out scenario 평가 설계 참고.
- SMACv2: 5대5 coordination과 procedural generalization 참고.

<a id="v6-training"></a>

## MAPPO coordinated team residual v6 실행·재개·로그 안내

v6는 planner 유지 또는 한 유닛의 방향 수정을 41가지 선택으로 통일하고, 명시적 탐색
확률과 실패 seed 재표집을 적용합니다. 현재 기본 경로의 run은 `stage_blocked`로
종료됐습니다. 아래 기본 경로 명령은 최초 실행 당시 방법을 보존한 것으로, 현재는
새 실행 시 산출물 충돌, 재개 시 terminal status 검사로 거부됩니다. 로그와 checkpoint를
삭제해 우회하지 마세요. 새 실험은 아래 별도 출력 경로 예시를 사용합니다.

전제 조건은 프로젝트의 `.venv/bin/python`, 설치된 의존성, `builds/BlackOut.app`,
`checkpoints/win_70_vs_scripted.pt`입니다. launcher는 이 로컬 Python 환경을 사용하며,
입력 checkpoint의 planner 설정과 Unity executable SHA를 검사합니다.

### 사전 점검 — 학습 미실행

```bash
cd /Users/safeailab_macmini/Desktop/2026-IST-tech-RL
./scripts/start_mappo_planner_residual_v6_background.sh --check
```

초기/상대 체크포인트, Unity executable hash, seed 분리, 출력 경로 및 실행 설정을
검사합니다. `preflight_passed`, `training_started=false`를 반환하며 Unity나 학습을
시작하지 않습니다. 검사를 통과한 경우에만 위 결과를 반환합니다. 기존 산출물이 있는
기본 경로는 현재 충돌 오류가 나며, `--resume-latest --check`도 종료된 run이므로
재개를 거부하는 것이 정상입니다.

### 최초 백그라운드 실행 명령 (기록용)

```bash
cd /Users/safeailab_macmini/Desktop/2026-IST-tech-RL
./scripts/start_mappo_planner_residual_v6_background.sh
```

터미널과 분리된 프로세스로 실행하고 PID 및 `console.log` 경로를 출력합니다. Unity는
`-batchmode`로 실행하되 시각 관측을 위해 graphics를 유지합니다. 최초 실행에서는
scripted/full win70 상대의 dev·confirmation **baseline 80경기 평가를 먼저 수행**하므로,
그동안 PPO update 로그가 없는 것은 정상입니다.

### 재개 가능한 중단과 종료된 run 구분

다음 명령은 정상 진행 중 사용자 중단·프로세스 장애 등으로 멈춘 run의 재개용입니다.
현재 저장된 `stage_blocked` run에는 적용할 수 없습니다.

```bash
cd /Users/safeailab_macmini/Desktop/2026-IST-tech-RL
./scripts/start_mappo_planner_residual_v6_background.sh --resume-latest
```

model·optimizer·난수 상태·seed 표집 분포를 복원합니다. Unity 내부 물리 상태는 복원하지
않으므로 진행 중이던 경기는 폐기 내역을 기록하고 새 episode에서 시작합니다. 저장된
checkpoint보다 앞선 로그는 `recovery_*.jsonl`로 보존합니다.

`budget_exhausted`로 종료된 실행은 전체 step 상한을 늘려 재개할 수 있습니다.
이는 현재 v6의 단계별 예산 소진(`stage_blocked`)과 다른 경우입니다.

```bash
./scripts/start_mappo_planner_residual_v6_background.sh --resume-latest --max-env-steps 5000000
```

`stage_blocked`, `rollback_limit`, `target_reached`는 종료 조건이 확정된 run이므로 동일
설정의 재개를 거부합니다. 새 실험은 log directory뿐 아니라 latest/best/target checkpoint,
snapshot 및 export 경로도 분리해야 합니다.

### 기존 결과를 보존하는 새 실험 경로

아래는 **사전 점검만** 수행하는 예시입니다. `v6_trial_02`는 아직 사용하지 않은 실험명으로
바꾸세요. 코드·학습 설정은 기본값 그대로이며, 경로 분리가 성능 개선을 뜻하지는 않습니다.

```bash
cd /Users/safeailab_macmini/Desktop/2026-IST-tech-RL
./scripts/start_mappo_planner_residual_v6_background.sh \
  --log-dir logs/mappo_v6_trial_02 \
  --latest-checkpoint checkpoints/mappo_v6_trial_02_latest.pt \
  --target-best-checkpoint checkpoints/mappo_v6_trial_02_target_best.pt \
  --stage-best-checkpoint checkpoints/mappo_v6_trial_02_stage_best.pt \
  --target-checkpoint checkpoints/mappo_v6_trial_02_target.pt \
  --snapshot-dir checkpoints/mappo_v6_trial_02_snapshots \
  --export-dir submission/v6_trial_02 \
  --check
```

사용자가 실제 학습을 시작하려면 동일 명령에서 마지막 `--check`를 제거합니다.
해당 새 run을 재개할 때는 같은 출력 경로와 학습 옵션을 유지하고 `--resume-latest`를
붙입니다. source/config/hash 일치 검사도 통과해야 합니다. 기존 v1~v6 산출물은 보존합니다.

### 로그와 결과 확인

```bash
tail -n 20 logs/mappo_planner_residual_v6/console.log
jq '{status, global_step, update, stage, target_reached,
     latest_target_evaluation: (.latest_target_evaluation |
       {wins, draws, losses, win_rate, mean_model_score_diff})}' \
  logs/mappo_planner_residual_v6/run_summary.json
```

위 명령은 완료된 기본 run의 기록을 읽습니다. 실행 중인 새 실험을 추적할 때는
해당 log directory로 바꾸고 `tail -f .../console.log`를 사용합니다.

| 산출물 | 경로 |
| --- | --- |
| update별 PPO·rollout·탐색률 및 승급/rollback 기록 | `logs/mappo_planner_residual_v6/training.jsonl` |
| 현재 단계·step·종료 상태·best/latest 평가 | `logs/mappo_planner_residual_v6/run_summary.json` |
| 양 진영 평가 원자료 | `logs/mappo_planner_residual_v6/*_eval_step_*.json` |
| 완료된 학습 경기의 seed·진영·상대·승패 | `logs/mappo_planner_residual_v6/training_episodes.jsonl` |
| 실행 설정·소스 보존 | `logs/mappo_planner_residual_v6/run_config.json`, `logs/mappo_planner_residual_v6/source_snapshot.zip` |
| 백그라운드 출력·실행 중 PID | `logs/mappo_planner_residual_v6/console.log`, `logs/mappo_planner_residual_v6/training.pid` (종료 후 PID 파일 제거) |
| 재개용 최신 모델 | `checkpoints/mappo_planner_residual_v6_latest.pt` |
| target-best / stage-best | `checkpoints/mappo_planner_residual_v6_target_best.pt`, `checkpoints/mappo_planner_residual_v6_stage_best.pt` |
| immutable best 사본·historical 상대 | `checkpoints/mappo_v6_snapshots/` |
| 최종 목표 통과 모델 (현재 미생성) | `checkpoints/mappo_win_85_vs_win70_v6.pt` |
| 목표 통과 후 두 파일 export (현재 미생성) | `submission/v6/policy.py`, `submission/v6/checkpoint.pt` |

최종 모델은 dev **9/10 이상**, 별도 confirmation **26/30 이상**, confirmation의 각
진영 **11/15 이상**을 모두 만족할 때 생성합니다. 이후 test seed 10개·20경기는 최종
보고에만 사용하고 모델 선택에는 사용하지 않습니다. 이번 실행은 최종 dev 5/10으로
목표 미달입니다. 회귀 확인용 30경기 결과를 최종 모델의 confirmation/test로 해석하면
안 됩니다. latest·best 저장 완료와 목표 모델 승격은 별개입니다.

### 기본 설정과 직접 실행

기본값은 CPU, 최대 4,000,000 environment step, rollout 2,048 step, 평가 간격
50,000 step, 저장 간격 25,000 step입니다. 학습 time scale은 50, 평가 time scale은
100이며 실제 평가·저장은 rollout 경계에서 이루어집니다. 옵션은 `.sh` 뒤에 그대로
전달할 수 있고, 전체 목록은 다음 명령으로 확인합니다.

```bash
./scripts/train_mappo_planner_residual_v6.sh --help
```

터미널에서 직접 실행하는 entry point는 아래와 같습니다. 역시 기존 기본 경로에서는
산출물 충돌로 거부되므로 새 실험에는 위의 출력 경로 옵션을 모두 전달해야 합니다.
`Ctrl+C`를 누르면 중단 요청을 기록하고 진행 중인 rollout/평가가 끝나는 경계에서 저장합니다.

```bash
./scripts/train_mappo_planner_residual_v6.sh
```

구현 차이, 검증 범위 및 planner를 포함한 제출 계약은
[`reports/mappo_planner_residual_v6_plan_changes.md`](reports/mappo_planner_residual_v6_plan_changes.md)를
참고하세요.

## MAPPO planner-conditioned residual v5 학습 실행 방법 (이전 실험 보존용)

v5는 planner 행동·역할·경로·가까운 목표를 residual actor 입력에 포함하고,
한 step에서 한 agent만 제한적으로 방향 override를 탐색하는 이전 버전입니다. v4의 all-zero BC
warmup은 사용하지 않습니다.

실제 실행은 317,440 step에서 `stage_blocked`, 최종 target 5/10으로 종료됐습니다.
아래는 당시 실행 방법의 기록이며 기존 로그·checkpoint 경로에 새 실행을 덮어쓰지 마세요.

### 학습 시작

```bash
cd /Users/safeailab_macmini/Desktop/2026-IST-tech-RL
./scripts/train_mappo_planner_residual_v5.sh
```

### 중단한 v5 학습 재개

```bash
cd /Users/safeailab_macmini/Desktop/2026-IST-tech-RL
./scripts/train_mappo_planner_residual_v5.sh --resume-latest
```

### 백그라운드 실행

```bash
cd /Users/safeailab_macmini/Desktop/2026-IST-tech-RL
./scripts/start_mappo_planner_residual_v5_background.sh
```

```bash
tail -f logs/mappo_planner_residual_v5/console.log
```

학습 지표는 `logs/mappo_planner_residual_v5/training.jsonl`, 실행 요약은
`logs/mappo_planner_residual_v5/run_summary.json`에 기록됩니다. latest, target-best,
stage-best checkpoint를 분리하며 빠른 10게임 gate는 30게임 확인 평가를 통과해야
승급합니다. 전체 dev 평가에서 85%를 확인하면
`checkpoints/mappo_win_85_vs_win70_v5.pt`를 저장합니다.

실패 원인, residual action 계약, fail-closed gate, 백그라운드/배속 검증은
[`reports/mappo_planner_residual_v5_plan_changes.md`](reports/mappo_planner_residual_v5_plan_changes.md)를
참조합니다.

v4 실행 파일과 산출물은 이전 실험 재현용으로 보존합니다.

## MAPPO v2 학습 실행 방법 (이전 실험 보존용)

`checkpoints/win_70_vs_scripted.pt`를 적용한 상대를 대상으로 MAPPO를 학습하고, 승률 85% 이상을 달성하면 `checkpoints/mappo_win_85_vs_win70.pt`에 체크포인트를 저장합니다.

실제 실행은 2,000,896 step, 최종 target 0/10으로 종료돼 목표를 달성하지 못했습니다.
아래 명령은 당시 실행 방법의 기록입니다. v2의 입력은 neural core이며 v6의 planner
보존형 초기화와 다릅니다. 기존 산출물을 덮어쓰는 새 실행은 피하세요.

### 학습 시작

```bash
cd /Users/safeailab_macmini/Desktop/2026-IST-tech-RL
./scripts/train_mappo_vs_win70.sh
```

학습을 중단하려면 `Ctrl+C`를 누릅니다. 중단 시점의 재개용 체크포인트가 저장됩니다.

### 중단한 학습 재개

```bash
cd /Users/safeailab_macmini/Desktop/2026-IST-tech-RL
./scripts/train_mappo_vs_win70.sh --resume-latest
```

### 백그라운드 실행

```bash
cd /Users/safeailab_macmini/Desktop/2026-IST-tech-RL
mkdir -p logs/mappo_vs_win70_v2
nohup ./scripts/train_mappo_vs_win70.sh > logs/mappo_vs_win70_v2/console.log 2>&1 &
```

실시간 콘솔 로그는 다음 명령으로 확인합니다.

```bash
tail -f logs/mappo_vs_win70_v2/console.log
```

학습 지표는 `logs/mappo_vs_win70_v2/training.jsonl`, 실행 요약은 `logs/mappo_vs_win70_v2/run_summary.json`에 기록됩니다. 재개용 최신 모델은 `checkpoints/mappo_vs_win70_v2_latest.pt`에 저장됩니다. 실패한 v1 산출물은 보존되며 v2에서 재개할 수 없습니다.

## v7 connectome 실험

v7-1은 FAFB v783 실제 부분 회로를 기존 v6 특징 뒤의 PPO 보정 모듈로 사용한다. v7-2는 MaleCNS 전체 유지 그래프로 한 유닛을 직접 제어하며, F0 고정 제어와 F1 출력층 PPO를 제공한다. 실제 데이터·체크포인트 해시 검사, 실행 잠금, 백그라운드 실행, 재개 및 paired-side 평가를 포함한다.

### 대규모 본실험 실행

아래 한 줄은 **중단한 본실험을 저장 지점부터 가속 재개**합니다. 준비 작업에서는
본실험을 다시 시작하지 않았으며, 이 명령을 사용자가 실행하면 시작됩니다.
2026-09-19에 요청한 예산으로 v7-1 **4천만** + v7-2 **6백만**, 합계 **4,600만 환경 스텝**입니다.

```bash
bash "/Users/safeailab_macmini/Desktop/2026-IST-tech-RL/scripts/start_connectome_fast.sh"
```

| 구분 | 구성 | 학습 예산 | 학습 후 dev 평가 |
| --- | --- | --- | --- |
| 1 | v7-1 A0 MLP / A1 파라미터 수 대응 MLP / A2 재배선 / A3 FlyWire × 시드 11·22·33·44·55 | 20개 × 2,000,000 = 40,000,000 step | 20개 × 60 = 1,200경기 |
| 2 | v7-2 현재 구현된 F1 출력층 PPO, teammate-v2, 한 유닛 제어 × 시드 11·22·33 | 3개 × 2,000,000 = 6,000,000 step | 3개 × 60 = 180경기 |

사전 검사 후 터미널에서 분리하여 **최대 12개 독립 실행**을 병렬 처리합니다.
전뇌 MPS 작업은 이 중 최대 2개입니다. 이 Mac의 단기 동시 수집 벤치마크에서 선택한 값입니다.
v7-1 완료를 기다리지 않고 v7-2도 함께 시작하며, 남는 작업 슬롯에 학습 완료 모델의
dev 평가를 배치합니다. 개별 모델의 rollout·PPO·난수는 다른 모델과 섞지 않습니다.
실행 중에는 `caffeinate`로 자동 유휴 잠자기를 막고, 종료 시 해제합니다.
수동 잠자기·재부팅·전원 종료까지 막는 기능은 아닙니다.

적용한 가속은 **native Protobuf 메시지 해석, 프로세스 내부 재직렬화 제거,
작은 게임 표시 창과 프레임 제한 해제, 라이브러리 스레드 과다 생성 억제,
v7-2 전뇌 희소 행렬 계산의 MPS/Metal 실행**입니다. 인코더는 측정상 MPS가 더 느려 CPU를
유지합니다. 원본 `.venv`, 학습 코드·설정과 Protobuf wire schema는 보존하고 별도 실행
오버레이의 해시·체크포인트 이력을 기록합니다. 자세한 병렬 수·측정 결과·검증 범위는
[가속 및 재개 보고서](reports/v7/acceleration_and_resume.md)를 참고하세요.

- **A2 시드 11은 147,456스텝에서 optimizer·난수 상태를 포함해 재개**합니다.
  완료된 A0·A1 시드 11은 재학습하지 않고 평가만 남깁니다. 아직 시작하지 않은 모델만
  새로 초기화하며, 파일럿 체크포인트를 이어받지 않습니다.
  기존 고정 인코더·상대 체크포인트, 그래프, 맵 분할, 보상과 상대 전환 일정은 유지합니다.
- 별도 설정은 `configs/v7/main_study/`, 학습 기록은
  `logs/v7/<experiment_id>_main_2m_v1/<seed>/`에 저장합니다. 기존 파일럿은 보존합니다.
- 중단 후 **동일 명령으로 재개**합니다. 본실험의 미완료 체크포인트는 이어서 학습하고,
  완료된 학습과 해당 체크포인트의 완전한 평가 결과는 건너뜁니다. Unity 진행 중 경기는
  복원하지 않고 새 경기로 시작하며, 부분 평가만 남았다면 그 모델의 평가를 다시 수행합니다.
- 오류가 나면 다음 실행으로 넘어가지 않고 중단합니다. 설정·소스·등록 파일이 바뀌면
  해시 검사로 중단하며, 체크포인트 없는 기존 실행 기록을 자동 덮어쓰지 않습니다.
- 평가는 **200만 스텝 최종 체크포인트**를 고정 target 상대, dev 30맵 × A/B 두 진영으로
  수행합니다. dev 성적에 따라 학습 시드를 제외하거나 예산을 변경하지 않습니다.

**v7-2 범위는 사용자가 지정한 현재 F1 모델의 장기 학습입니다.** B2 재배선 전뇌와
B3 CNN/GRU 대조군, F2 및 다섯 유닛 제어를 포함한 전체 비교 연구는 포함하지 않습니다.
이 스크립트는 학습과 dev 평가까지 수행하며, confirmation·블라인드 최종 test·제출은 별도입니다.

실행 없이 전체 설정·체크포인트·데이터 검사:

```bash
bash "/Users/safeailab_macmini/Desktop/2026-IST-tech-RL/scripts/start_connectome_fast.sh" --check
```

실행 상태 확인:

```bash
bash "/Users/safeailab_macmini/Desktop/2026-IST-tech-RL/scripts/start_connectome_fast.sh" --status
```

정상 중단 요청(현재 학습의 저장·종료를 기다리며, 재개는 최초 실행 명령과 동일):

```bash
bash "/Users/safeailab_macmini/Desktop/2026-IST-tech-RL/scripts/start_connectome_fast.sh" --stop
```

| 경로 | 내용 |
| --- | --- |
| `logs/v7/main_study/accelerated/console.log` | 가속 관리자 시작·검사 로그 |
| `logs/v7/main_study/accelerated/*_train.log`, `*_evaluate.log` | 실행별 학습·평가 콘솔 및 오류 |
| `logs/v7/main_study/accelerated/status.json` | 현재 동시 실행 목록·대기 수·실패 상태 |
| `logs/v7/main_study/accelerated/summary.json` | 두 버전의 모델·시드별 dev 평가 결과 |
| `logs/v7/main_study/pre_acceleration_backup/` | 중단 시점 원본 체크포인트·설정·상태 백업 |
| `logs/v7/<experiment_id>_main_2m_v1/<seed>/status.json` | 해당 모델의 현재 학습 스텝 |
| `logs/v7/<experiment_id>_main_2m_v1/<seed>/checkpoints/latest.pt` | 해당 모델의 저장된 재개 지점 |
| [본실험 사전 등록](reports/v7/main_study_preregistration.md) | 비교 범위, 예산, 선택·중단 규칙, 검증 범위 |
| [기계 판독 등록 파일](reports/v7/main_study_registration.json) | 코드·설정 해시 및 소스 보관본 |
| [가속 실행 등록 파일](reports/v7/acceleration_registration.json) | 통신·MPS·병렬 설정과 추가 코드 해시 |

`--status`의 통합 `status: complete`와 `summary.json`의 `complete: true`는
**23개 학습과 1,380경기 dev 평가가 모두 끝났음**을 뜻합니다. 목표 승률 달성 판정은 아닙니다.
이전 `start_connectome_main.sh`는 순차 실행 재현용으로 보존합니다. 이제 재개·상태·정지는
`start_connectome_fast.sh`를 사용하세요. 기존 순차 실행과 가속 실행은 공유 잠금으로 중복을 막습니다.

### 파일럿 한 번에 실행

아래 **한 줄만 실행**하면 v7-1과 수정된 v7-2 파일럿의 학습 확인·재개와 dev 평가를
순서대로 수행합니다. 어느 디렉터리에서 실행해도 되며, 사전 검사 후 백그라운드로
전환되므로 터미널을 닫아도 계속 실행됩니다. Mac이 잠자기 상태가 되면 실행은 진행되지 않습니다.

```bash
bash "/Users/safeailab_macmini/Desktop/2026-IST-tech-RL/scripts/start_connectome_experiments.sh"
```

| 순서 | 실행 범위 | 완료된 학습 예산 | dev 평가 |
| --- | --- | --- | --- |
| 1 | v7-1 A0 MLP / A2 rewired / A3 FlyWire × 시드 11·22·33 | 9개 × 200,000 step | 9개 × 60경기 = 540경기 |
| 2 | v7-2 teammate-v2 F1 PPO × 시드 11·22·33, F0 고정 정책 × 시드 11 | F1 3개 × 128,000 step, F0 22,000 step | 4개 × 60경기 = 240경기 |

- **현재는 13개 모두 학습이 완료되어, 남은 dev 평가 최대 780경기를 실행합니다.**
  각 실행을 고정 target 상대에 대해 dev 30맵 × A/B 두 진영으로 평가합니다.
- 학습이 미완료라면 저장된 체크포인트에서 재개하고, 새 실행이면 등록된 예산으로 시작합니다.
  완료된 학습 및 동일 체크포인트의 완전한 paired 평가 결과는 재사용합니다.
  체크포인트 없이 기존 학습 기록만 남았거나 설정·소스 해시가 달라지면 중단합니다.
- 중간 오류나 중단 후에는 **같은 한 줄을 다시 실행**하면 됩니다. 부분 평가만 남은 모델은
  해당 모델의 60경기를 처음부터 다시 평가하며, 완료된 다른 모델은 건너뜁니다.
- 동시에 같은 통합 명령을 입력해도 중복 실행하지 않습니다. 기존 개별 학습·평가가
  실행 중이면 잠금 충돌로 중단하므로 그 작업이 끝난 뒤 통합 명령을 실행하세요.
- v7-1은 학습 당시 등록된 소스 스냅샷, v7-2는 수정된 teammate-v2 소스로 검증·실행합니다.
  기존 v7-2 실패 기록은 보존하며 복구된 실행을 사용합니다.

이 명령의 범위는 **현재 등록된 파일럿 + dev 평가**입니다. 계획서의 v7-1 A1 포함
4천만 step 본실험, v7-2 B2/B3 대조군, F2, confirmation 및 최종 test는 포함하지 않습니다.
본실험 규모는 파일럿의 성능·처리량·비용을 확인한 뒤 별도로 확정합니다.

### 검사·상태·결과 확인

Unity 실행 없이 설정·데이터·체크포인트 검사:

```bash
bash "/Users/safeailab_macmini/Desktop/2026-IST-tech-RL/scripts/start_connectome_experiments.sh" --check
```

통합 실행 및 각 버전의 진행 상태 확인:

```bash
bash "/Users/safeailab_macmini/Desktop/2026-IST-tech-RL/scripts/start_connectome_experiments.sh" --status
```

| 파일 | 내용 |
| --- | --- |
| `logs/v7/pilot_experiments/console.log` | 통합 실행 로그와 오류 원인 |
| `logs/v7/pilot_experiments/status.json` | 현재 버전, 실행·완료·실패 상태 |
| `logs/v7/pilot_experiments/summary.json` | 두 버전의 실행별 승률·무승부율·맵 bootstrap 신뢰구간 |
| `logs/v7/pilot_v7_1_dev_evaluation/status.json` | v7-1 현재 모델·평가 완료 개수 |
| `logs/v7/pilot_v7_2_dev_evaluation/status.json` | v7-2 현재 모델·평가 완료 개수 |
| `logs/v7/<experiment_id>/<seed>/eval_dev_target_*.json` | 모델별 평가 원자료 (`*.progress.json`은 진행 중 기록) |

`--status`의 `active`는 통합 실행 잠금의 실제 점유 여부입니다. 통합 `status`가
`complete`이고 `summary.json`의 `complete`가 `true`이면 두 버전의 평가까지 끝난 것입니다.
실행별 신뢰구간은 맵 변동성을 나타내며, 학습 시드 전체를 합친 유의성 검정 결과는 아닙니다.

기존 `scripts/start_pilot_evaluation.sh`는 v7-1 평가만 수행합니다.
두 버전을 함께 실행할 때는 위 통합 스크립트를 사용하세요.

상세 설정·실행·정지·재개와 검증 범위는 [v7 실행 안내](reports/v7/implementation_and_runbook.md)를 참고한다. F2, 전뇌 대조군 전체 연구, 공식 제출 export는 후속 범위이며 현재 승률 개선을 주장하지 않는다.
