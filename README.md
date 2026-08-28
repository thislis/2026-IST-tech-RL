# 2026-IST-tech-RL

## To-Do

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
문제가 확인되어, 현재 권장 학습 경로는 DAgger teacher 증류와 단계별 상대 혼합을
적용한 v3다. 구현과 변경 근거는
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

## MAPPO planner-residual v4 학습 실행 방법 (권장)

현재 권장 실행은 검증된 planner를 action 0 fallback으로 보존하고 neural actor가
방향 override만 학습하는 fail-closed v4입니다. 실패한 v3 장기 실행에서는
재개하지 않습니다.

### 학습 시작

```bash
cd /Users/safeailab_macmini/Desktop/2026-IST-tech-RL
./scripts/train_mappo_planner_residual_v4.sh
```

### 중단한 v4 학습 재개

```bash
cd /Users/safeailab_macmini/Desktop/2026-IST-tech-RL
./scripts/train_mappo_planner_residual_v4.sh --resume-latest
```

### 백그라운드 실행

```bash
cd /Users/safeailab_macmini/Desktop/2026-IST-tech-RL
mkdir -p logs/mappo_planner_residual_v4
nohup ./scripts/train_mappo_planner_residual_v4.sh > logs/mappo_planner_residual_v4/console.log 2>&1 &
```

```bash
tail -f logs/mappo_planner_residual_v4/console.log
```

학습 지표는 `logs/mappo_planner_residual_v4/training.jsonl`, 실행 요약은
`logs/mappo_planner_residual_v4/run_summary.json`에 기록됩니다. 재개용 모델은
`checkpoints/mappo_planner_residual_v4_latest.pt`, dev 최고 모델은
`checkpoints/mappo_planner_residual_v4_best.pt`입니다. 전체 dev seed 양 진영
10경기에서 9승 이상이면 `checkpoints/mappo_win_85_vs_win70.pt`를 저장합니다.

실패 원인, residual action 계약, fail-closed gate, 백그라운드/배속 검증은
[`reports/mappo_planner_residual_v4_plan_changes.md`](reports/mappo_planner_residual_v4_plan_changes.md)를
참조합니다.

## MAPPO v2 학습 실행 방법 (이전 실험 보존용)

`checkpoints/win_70_vs_scripted.pt`를 적용한 상대를 대상으로 MAPPO를 학습하고, 승률 85% 이상을 달성하면 `checkpoints/mappo_win_85_vs_win70.pt`에 체크포인트를 저장합니다.

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
