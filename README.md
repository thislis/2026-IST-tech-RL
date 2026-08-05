# 2026-IST-tech-RL

## To-Do

- [ ]  Phase 1

| ID | Todo | 산출물 및 완료 조건 |
| --- | --- | --- |
| PREP-01 | ☑ 게임/Python API commit, executable hash, Python·PyTorch·CUDA·Unity 버전 고정 | [`versions.md`](versions.md), [`requirements.lock`](requirements.lock), [`docker-image.env`](docker-image.env) |
| PREP-02 | ☑ 로컬 또는 Docker에서 random policy 1경기 실행 | [`logs/prep02_random_seed_20260805.json`](logs/prep02_random_seed_20260805.json): reset부터 terminal까지 실행, seed·점수·winner 저장 및 검증 |
| PREP-03 | ☑ 게임 규칙을 RL state/action/event 관점으로 문서화 | [`game_spec.md`](game_spec.md): 유닛, 아이템, 창고, 성소, 전투 상성, 20초 이벤트 |
| PREP-04 | ☐ 관측값 96개 vector field와 11개 map channel 파서 작성 | field별 shape·범위·team perspective 단위 테스트 |
| PREP-05 | ☐ PettingZoo API contract test 작성 | reset/step/termination, agent 수, dtype, action 범위, seed 재현성 통과 |
| PREP-06 | ☐ self-ID 및 batch-order 실험 | 팀별 batch 크기와 row 순서가 reset·side swap·terminal 전후에 안정적인지 보고서 작성 |
| PREP-07 | ☐ 행동 의미 테스트 | `(0,0)`, 작은 벡터, 큰 벡터, 8방향의 실제 displacement 측정 |
| PREP-08 | ☐ 실제 terminal winner 기반 evaluator 작성 | `eval/paired_series.py`; shaping reward를 승자 판정에 사용하지 않음 |
| PREP-09 | ☐ Python score-delta team reward 구현 | 점수 증가·감소·약탈·terminal 사례에 대해 예상 부호 테스트 |
| PREP-10 | ☐ 통합 logging schema 작성 | seed, side, opponent, score, winner, episode length, checkpoint SHA 저장 |
| PREP-11 | ☐ 처리량 benchmark | 환경 1개/복수 개의 steps/sec, GPU utilization, episode 실행 비용 측정 |
| PREP-12 | ☐ 첫 actor/critic 인터페이스 설계 | 입력 tensor shape, slot embedding, action adapter, checkpoint schema 명세 |
| PREP-13 | ☐ 평가 계약에 없는 항목 확인 | batch 순서, stateful policy 허용 여부, inference 제한, 모델 크기 제한을 확인 목록으로 관리 |
| PREP-14 | ☐ random-vs-random paired-seed 평가 | side별 승률·점수 차를 측정하고 evaluator가 특정 side에 편향되지 않는지 확인 |

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

- [ ]  Phase 2
    - [ ]  2-1
    
    | ID | Todo | 산출물 및 완료 조건 |
    | --- | --- | --- |
    | BASE-S01 | ☐ `Random`, `NoOp`, `FixedDirection` policy 구현 | smoke test와 evaluator 회귀 테스트에 사용 |
    | BASE-S02 | ☐ semantic map decoder 구현 | 벽, 아군·적군 창고, 유닛, 배터리, 특수 아이템 위치 추출 |
    | BASE-S03 | ☐ 이동 가능 영역과 path planner 구현 | 우선 A* 또는 flow field, waypoint 추종, 충돌 시 재계획 |
    | BASE-S04 | ☐ stuck detector 구현 | 일정 시간 위치 변화가 없으면 경로 또는 목표 재선정 |
    | BASE-S05 | ☐ team-local slot과 unit state 추적 | 각 유닛의 위치·보유 아이템·클래스·현재 역할 관리 |
    | BASE-S06 | ☐ task assignment 구현 | 배터리와 유닛 간 path distance 기반 greedy 또는 Hungarian 할당 |
    | BASE-S07 | ☐ 기본 역할 구성 | 노동자 3명 수집, 경비원 1명, 전달자 1명을 초기 정책으로 사용 |
    | BASE-S08 | ☐ 노동자 FSM 구현 | `SEEK_BATTERY → PICKUP → DELIVER → RETARGET` |
    | BASE-S09 | ☐ 경비원 FSM 구현 | 중앙 성소 이동, 변신, 주요 창고·운반 경로 순찰, 적 추격 |
    | BASE-S10 | ☐ 전달자 FSM 구현 | 본진 성소 변신, 먼 배터리 운반, 적 접근 시 회피 |
    | BASE-S11 | ☐ 20초 흡수 주기 전략 구현 | 흡수 직전 안전한 적재, 직후 새 수집, 상황별 약탈 시도 |
    | BASE-S12 | ☐ 위험 지도 구현 | 적 위치 주변 회피 비용과 경비 경로 비용 반영 |
    | BASE-S13 | ☐ scripted trajectory recorder 구현 | obs, action, role, target, reward, score, seed를 저장 |
    | BASE-S14 | ☐ scripted-vs-random 평가 | paired held-out seeds에서 일관된 우세 및 side 편향 없음 |
    | BASE-S15 | ☐ 특수 아이템 정책을 선택적으로 추가 | 배터리-only agent보다 실제 성능이 좋아질 때만 유지 |
    
    첫 scripted agent- 특수 아이템과 적극적인 약탈을 우선 제외. 아래 동작 확인 우선
    
    ```
    3명: 서로 다른 배터리 수집 → 가까운 아군 창고에 전달
    1명: 중앙 성소 → 경비원 변신 → 열린 창고 또는 운반 경로 방어
    1명: 본진 성소 → 전달자 변신 → 먼 배터리 운반
    ```
    
    - [ ]  2-2
    
    | ID | Todo | 산출물 및 완료 조건 |
    | --- | --- | --- |
    | BASE-R01 | ☐ 9-way categorical action distribution 구현 | sampled action과 deterministic argmax를 `(dx,dy)`로 변환 |
    | BASE-R02 | ☐ vector/entity encoder 구현 | 10개 유닛 block과 score·time·class를 분리 인코딩 |
    | BASE-R03 | ☐ semantic CNN encoder 구현 | `11×96×96` 입력을 고정 길이 latent로 변환 |
    | BASE-R04 | ☐ slot embedding 또는 specialized head 구현 | 5개 유닛이 서로 다른 역할을 학습할 수 있음 |
    | BASE-R05 | ☐ shared actor와 local critic 구현 | 팀 내 5개 유닛이 actor parameter 공유 |
    | BASE-R06 | ☐ PettingZoo parallel rollout collector 구현 | agent별 obs/action/logprob/value/reward/mask 수집 |
    | BASE-R07 | ☐ GAE와 episodic buffer 구현 | 긴 에피소드와 truncation/termination을 구분 |
    | BASE-R08 | ☐ PPO update 구현 | clipped loss, value loss, entropy, gradient clipping |
    | BASE-R09 | ☐ PPO 진단 logging | KL, clip fraction, entropy, explained variance, gradient norm |
    | BASE-R10 | ☐ score-delta team reward와 terminal reward 연결 | Unity shaping reward on/off 비교 가능 |
    | BASE-R11 | ☐ 고정 scripted opponent 연결 | 학습 대상 팀만 update하고 상대는 완전히 freeze |
    | BASE-R12 | ☐ scripted trajectory 기반 BC pretraining 실험 | scratch 학습과 동일 환경 step 기준 비교 |
    | BASE-R13 | ☐ checkpoint와 experiment registry 구현 | config, seed, git SHA, 상대 ID를 checkpoint와 함께 저장 |
    | BASE-R14 | ☐ 제출용 deterministic policy wrapper 구현 | `forward(vector, graphic) → (B,2)`와 `[-1,1]` 계약 준수 |
    | BASE-R15 | ☐ random/scripted/IPPO 평가 matrix 작성 | 같은 seed와 side swap으로 성능 비교 |
    
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
    
- [ ]  Phase 3
    - [ ]  MAPPO와 CTDE
    
    | ID | Todo | 산출물 및 완료 조건 |
    | --- | --- | --- |
    | AGENT-01 | ☐ centralized state builder 구현 | 전체 유닛 관측, 팀 score, map, 전체 class 정보를 critic 입력으로 구성 |
    | AGENT-02 | ☐ decentralized actor + centralized critic 구현 | inference 시 actor만 사용하고 critic은 학습에서만 사용 |
    | AGENT-03 | ☐ joint rollout과 team mask 검증 | 사망·respawn·termination 시 critic target이 깨지지 않음 |
    | AGENT-04 | ☐ IPPO checkpoint에서 MAPPO 초기화 | 동일 actor로 critic 효과를 공정하게 비교 |
    | AGENT-05 | ☐ IPPO 대비 MAPPO ablation | 같은 seed, opponent, environment step budget으로 평가 |
    
    Centralized critic은 학습 중 전체 팀 또는 전체 게임 정보를 사용할 수 있지만, actor에는 제출 환경에서 실제로 제공되는 관측만 전달. 
    
    - [ ]  표현력과 전략 개선
    
    | ID | Todo | 산출물 및 완료 조건 |
    | --- | --- | --- |
    | AGENT-06 | ☐ unit entity attention 실험 | 단순 entity pooling 대비 held-out 성능 비교 |
    | AGENT-07 | ☐ self-relative position feature 추가 | slot이 가리키는 자기 위치 기준으로 모든 위치를 상대 좌표화 |
    | AGENT-08 | ☐ global map + local crop 구조 실험 | 전체 전략과 근거리 회피·전투를 함께 처리 |
    | AGENT-09 | ☐ absorption phase feature 추가 | `time_left`에서 20초 주기의 sin/cos feature 계산 |
    | AGENT-10 | ☐ auxiliary target logging | 역할, 아이템 보유, 다음 흡수까지 시간, score delta 예측 분석 |
    | AGENT-11 | ☐ auxiliary loss 선택 실험 | 실제 승률을 개선하는 항목만 유지 |
    | AGENT-12 | ☐ 역할 분화 시각화 | slot별 이동 경로, 수집량, 사망, 변신, 창고 방문 빈도 분석 |
    
    현재 제출 인터페이스는 명시적인 hidden state와 reset API를 제공하지 않으므로, stateful model 허용 여부와 episode reset 감지가 검증된 후에만 RNN 도입.
    
    - [ ]  Curriculum과 reward 개선
    
    | ID | Todo | 산출물 및 완료 조건 |
    | --- | --- | --- |
    | AGENT-13 | ☐ curriculum 단계 정의 | random → 약한 scripted → 완성 scripted → frozen RL opponent |
    | AGENT-14 | ☐ 초기 navigation shaping 실험 | score/terminal-only 대비 sample efficiency 비교 |
    | AGENT-15 | ☐ shaping annealing 구현 | 후반에는 실제 score와 승패가 주 보상이 되도록 감소 |
    | AGENT-16 | ☐ curriculum별 별도 evaluator 구성 | 쉬운 상대 성능과 강한 상대 성능을 분리 기록 |
    | AGENT-17 | ☐ 약탈·특수 아이템 curriculum 추가 | 단순 수집 정책의 성능을 훼손하지 않을 때 승격 |
    
    게임 규칙 자체를 단순화한 별도 Unity 빌드를 curriculum으로 쓰면 최종 환경과 dynamics가 달라질 수 있음. 우선은 동일 환경에서 opponent 난이도, 초기 policy, reward weight를 조절.
    
    - [ ]  Self-play
    
    | ID | Todo | 산출물 및 완료 조건 |
    | --- | --- | --- |
    | AGENT-18 | ☐ frozen checkpoint opponent loader 구현 | 학습 대상과 상대 optimizer/state가 완전히 분리 |
    | AGENT-19 | ☐ snapshot pool 구현 | 최신 모델 및 과거 여러 세대에서 상대 sampling |
    | AGENT-20 | ☐ side 균형 sampling | 학습 모델이 A/B side를 동일 빈도로 경험 |
    | AGENT-21 | ☐ opponent mixture에 따른 PPO 안정화 | entropy, KL, value error가 급격히 붕괴하지 않음 |
    | AGENT-22 | ☐ checkpoint evaluation matrix 자동화 | 모든 주요 세대 간 paired-seed 대전 결과 생성 |
    | AGENT-23 | ☐ exploiter/과거 상대 회귀 검사 | 최신 모델이 특정 최신 상대에만 과적합하지 않음 |
    | AGENT-24 | ☐ 필요 시 population/PSRO 확장 판단 | 단순 snapshot pool이 포화된 뒤에만 진행 |
    
    두 최신 정책을 동시에 계속 update하는 방식은 피할 것. 상대 정책이 매 update마다 변하면 PPO가 보는 환경도 계속 변하기 때문. 학습 팀 하나와 frozen opponent를 두고, 일정 시점마다 snapshot을 pool에 추가하는 방식이 안정적. Unity ML-Agents의 self-play 문서도 snapshot window와 최신 모델 sampling 비율을 통해 안정성과 상대 다양성의 균형을 잡도록 설명.
    
    - [ ]  최종 평가와 제출
    
    | ID | Todo | 산출물 및 완료 조건 |
    | --- | --- | --- |
    | AGENT-25 | ☐ train/dev/test seed 집합 고정 | 학습 과정에서 test seed 결과를 모델 선택에 사용하지 않음 |
    | AGENT-26 | ☐ paired-seed bootstrap CI 구현 | 개별 경기가 아닌 seed pair 단위로 신뢰구간 계산 |
    | AGENT-27 | ☐ opponent suite 구성 | random, scripted, IPPO, MAPPO 과거 세대, 최신 후보 포함 |
    | AGENT-28 | ☐ deterministic inference mode 확정 | sampling 없이 같은 입력에 같은 행동 출력 |
    | AGENT-29 | ☐ 모델 경량화 | 추론 latency를 측정하고 불필요한 encoder 제거 |
    | AGENT-30 | ☐ clean-room 제출 테스트 | 새 환경에서 `policy.py`와 `checkpoint.pt`만으로 로드·실행 |
    | AGENT-31 | ☐ CPU와 GPU 양쪽 smoke test | device mismatch, dtype, batch 크기 변화 처리 |
    | AGENT-32 | ☐ 최종 모델 승격 회의 | 승률뿐 아니라 score 차, side bias, 과거 상대 회귀를 함께 검토 |

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
