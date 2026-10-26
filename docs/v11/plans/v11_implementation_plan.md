# v11 구현·학습·평가 계획

작성일: 2026-10-24. [설계](v11_design.md)를 실제 작업으로 나눈다. 현재 만든 코드는 **리그 재검증 도구뿐**이다. 아래 모델·학습기·exporter·실험 명령은 앞으로 구현할 항목이며, 이번 계획 수립 중에는 새 학습/경기를 실행하지 않았다.

> 이 문서는 구현 전 계획 기록이다. 후속 구현의 실행법은 [v11 안내](../README.md)에 있다. 최종 공개 정책은 제공 runner의 canonical 팀 행 순서를 사용하는 두 입력/한 출력이며, 추가 ID·state 입력을 요구하지 않는다. ONNX 명시 상태 포트는 별도 개발 계약이다. 기본 P0–P6 예산은 유지하고, CPU/MPS는 actor·planner·learner별 실측으로 선택한다. 추가 ablation은 명시적인 `--ablation` 실행으로 분리한다.

2026-10-26 승인 변경: 아래 §5–7의 **평가량만 축소**하고, 학습량·학습 설정·기존 checkpoint·완료된 baseline/pilot 결과는 보존한다. `goal_memory_v3`의 원본 `registration.json`을 수정하지 않고 `evaluation_revision.json`으로 평가 변경과 허용 source override를 추적한다. 이 변경 과정에서는 학습이나 실제 환경 실험을 재개하지 않는다.

## 1. 작업 순서와 완료 기준

계획 완료, 구현 완료, 성능 목표 달성을 구별한다. 구현 완료에는 실제 planner·정책·collector·ONNX·평가기 검증이 필요하고, 성능 달성에는 §7의 동결 평가가 필요하다. export나 BC accuracy만으로 성능을 인증하지 않는다.

의존 순서는 `계약/기록 → 상태·ONNX 최소 core → tensor planner → 목표/잔차 정책 → 학습 → 실환경 pilot → 본 비교 → 확인 → 동결 평가·패키징`이다.

| 작업 | 구현 내용 | 완료 증거 / 실패 대응 |
|---|---|---|
| V11-00 | 과거 source/asset, 입력·objective·split·예산 고정 | [현재 리그 audit](../../../logs/v11/reports/league_audit.json)와 향후 registration; 기존 run 덮어쓰기 금지 |
| V11-01 | 공개 이름 routing·state schema·최소 ONNX 왕복 | 행 순서 변경·두 경기 interleave·진영 교체의 sequence parity; 실패 시 adapter 수정 |
| V11-02 | tensor map decode·거리장·waypoint·FSM·할당 | 독립 A*의 거리/도달성, 기존 3 worker+2 guard와 closed-loop 비교 |
| V11-03 | Attention·unit/team GRU·목표 head·조건부 gate/correction | mask·확률·STOP·결정 시점·개입 용량의 검사 |
| V11-04 | teacher rollout·목표 레이블·DAgger·replay | 입력/상태 provenance·episode split·실제 도달/반납; NoOp 정확도만 높은 후보 거부 |
| V11-05 | 독립 critic·PopArt·recurrent/event PPO | 초기 ratio=1, variable-duration return 수계산, critic-only update 후 actor 불변 |
| V11-06 | 완결 episode wave·상대 deficit scheduler·resume | 긴/짧은 경기 포함, 실제 가족별 step 노출, 중복·불완전 commit 차단 |
| V11-07 | reward/score/terminal 분리·paired map 평가 | 제공 runner 결과 동치, 불일치 보존, family/side/seed 통계 |
| V11-08 | 전체 ONNX·state export·provider 성능 | 긴 sequence와 실제 경기 비교·hash lock·opset/provider 기록 |
| V11-09 | CLI·백그라운드·자원·관전 | 다른 cwd·소유 자식 종료·원본 graphics·재개·상태 표시 |
| V11-10 | pilot·본 비교·ablation·confirmation·test | §5–7 결과; 실패 시 미달 기준과 다음 원인 실험 기록 |

## 2. 파일 배치와 재사용

[루트 README](../../../README.md)와 [배치 규칙](../../common/project_layout.md)을 따른다. 아래는 **예정 구조**이며 `analysis/audit_league.py`만 현재 존재한다.

```text
code/v11/
  analysis/audit_league.py
  blackout_v11/
    contracts.py          # 입력·objective·원본 hash
    adapter.py            # 공개 이름↔canonical team; stream별 state
    state.py              # 초기화·reset·respawn·schema
    semantic.py           # 좌표·11채널·흡수 phase·unknown feature
    tensor_planner.py     # 거리장·waypoint·FSM; tensor 연산
    assignment.py         # 목표 후보·용량·조건부 배정
    policy.py             # Attention/GRU/goal/gate/correction
    critic.py             # 독립 C0/C1·PopArt
    decision.py           # 결정 event·mask·log-prob·duration
    teachers.py           # hash 고정 과거 실행 policy
    demonstrations.py     # 압축 replay·teacher provenance
    trajectory.py        # episode·state·event·snapshot
    rewards.py            # objective별 outcome·potential·SMDP return
    recurrent_ppo.py      # burn-in·hidden 재계산·joint likelihood
    collector.py          # frozen wave·자연 종료·invalid 분리
    opponents.py          # 고정 리그·self-play snapshot
    scheduler.py          # family step deficit·episode quota·PFSP
    evaluation.py         # raw/score/official 정보 분리
    statistics.py         # paired cluster CI·seed·side·family
    export.py             # ONNX·state/init·I/O manifest
    runtime.py            # 자원·worker lifecycle·resume
    study.py              # 예산·stage·선택·test lock
    runner.py             # check/prepare/start/status/stop/resume
    viewer.py             # 동결 snapshot 관전
  configs/{default,ablations,opponents,splits}.json
  contracts/{original,development_io,state_schema,objective}.json
  scripts/{v11_experiments,export_v11,watch_v11_models}.py
  run_v11_fast.sh
  watch_v11_models.sh
  tests/v11/

docs/v11/{README.md,plans/,reports/}
docs/common/organizer_qa_2026-10-24.md
logs/v11/reports/                 # 자동 감사·export·benchmark
logs/v11/<study>/
  registration.json              # source/build/opset/provider/objective/splits
  evaluation_revision.json       # 승인된 평가 변경·source override; 원본 등록 보존
  task_plan.json
  status.json
  runtime_profile.json
  runs/<arm>_s<seed>/
    checkpoints/
    training.jsonl
    episodes.jsonl
    decisions.jsonl
    exposure.json
  evaluation/{dev,confirmation,frozen_test}/
  selection_lock.json
  summary.json
artifacts/data/v11/<dataset_hash>/
artifacts/submission/v11/<policy_hash>/
```

학습 checkpoint는 v7+처럼 run 내부에 둔다. 직접 작성한 분석·계획은 `docs/`, 자동 보고서는 `logs/`, 모델/데이터/제출 실물은 기존 제외 규칙을 따른다. 최종 packaging에서 추가 파일이 허용되지 않으면 state 초깃값·상수를 그래프나 지정 runtime 설정에 넣고, 로컬 검증용 manifest와 실제 제출 파일 집합을 구별한다.

| 기존 코드 | 활용 | 그대로 재사용하지 않을 부분 |
|---|---|---|
| shared `scripted_fsm.py`, `navigation.py`, `coordination.py` | 동결 교사·독립 reference, 의미를 tensor로 이식 | NumPy/heapq/Python 객체를 ONNX에서 호출 |
| v6 `mappo_v6.py` | 41-way 기준군·회전·확률 검증 | frozen latent, 단일 유닛 제한을 주 모델에 영구 적용 |
| v9/v10 `policy.py` | 작은 map/entity 표현·공통 장면 계산 | self 없는 query·절대 slot 의미·Python data-dependent 분기 |
| v10 `value.py`, `targets.py` | 독립 critic·동결 target·PopArt | stateless target을 recurrent 모델에 무검증 적용 |
| v10 trajectory/runtime/study | 압축·원자적 commit·등록·프로세스 관리 | v10 경로/상수, episode 비중을 step 비중으로 간주 |
| shared `agent_league/` | frozen opponents·trace·기존 objective 비교 | `infos` 폐기, proxy를 공식 outcome으로 대체 |

실제 helper는 고정 hash로 참조하거나 v11로 명시 복사한다. 과거 등록이나 checkpoint의 hash를 바꿔 resume를 통과시키지 않는다. 이번 평가 변경은 원본 등록과 완료 산출물을 보존하고 별도 revision을 검증하는 예외다. v11 실행기 구현 시 `code/shared/project_paths.py`, `code/run.py`의 버전 발견을 검사·확장한다. 현재 감사 스크립트는 `__file__`에서 루트를 찾는다.

## 3. 핵심 데이터 계약

`policy_identity = hash(weights, planner, decoder, state_schema, input_adapter, source_closure)`로 정의한다. 가중치만 같다고 동일 실행 정책으로 간주하지 않는다.

episode에는 `episode_id, stream_id, physical_side, opponent_hash, requested_seed, initial_map_hash, initial_observation_hash, policy_identity, objective_id`를 남긴다. decision에는 `decision_id, step, duration, state_before, state_after, head_selection_mask, goal, legal_mask, gate, correction, behavior_log_prob, sampling_config, executed_action, interruption_reason`을 저장한다. 큰 tensor는 압축 binary로 분리하고 JSON에는 경로·hash를 둔다.

목표 25-step·잔차 4-step·interrupt 시점의 **합집합을 하나의 decision event 시계**로 사용한다. 그 event에서 실제 재선택한 head의 조건부 log-prob만 합산하고, `k`는 다음 event까지의 길이다. 그 사이 reward를 한 번만 합산한다. 유지 중인 option의 확률을 매 frame 다시 곱하지 않는다. GRU는 event에서만 갱신하고 나머지 frame에는 hidden을 유지한다. planner/FSM의 관측·정체 감지는 매 frame 진행한다. critic의 baseline은 새 goal/action을 선택하기 **전** 상태로 계산한다.

결과 필드는 다음처럼 나눈다.

- `runner_reward_totals`, `runner_winner`: 제공 runner와 같은 누적 및 모델/진영 변환.
- `score_before_terminal`, `score_after_terminal`, `visible_score_winner`: 마지막 행동 전·후 실제 점수 시점.
- `official_winner`, `official_winner_available`, `terminal_info_raw`: 제공 정보가 있으면 출처와 저장, 없으면 null.
- `natural_terminal`, `truncated`, `watchdog`, `valid`: 불완전 경기는 무승부가 아니다.

게임/API reward·timer·score는 패치하지 않고 학습 reward만 별도 계산한다. terminal score reset 버전과 유지되는 버전을 구별한다. 자료 부재나 0을 임의의 확정 점수로 메우지 않는다.

마지막 득점으로 100점을 넘어 normalized score가 1보다 커질 수 있다(리그 사례 102점). 유한한 실제 관측을 잘못된 입력으로 거부하거나 평가 점수를 1로 잘라내지 않는다. potential의 bounded clipping과 원본 점수 보존은 별개다.

wave는 `staged → validated complete episode → frozen-critic targets → committed update`다. 모델·optimizer·RNG·scheduler를 일관되게 저장한다. Unity 중간 physics state는 복원하지 못하므로 재개 시 새 episode를 시작하고 버린 step을 예산에 남긴다. replay는 event 단위 연속 chunk, burn-in 32·학습 64를 초기값으로 사용한다. 저장 hidden+burn-in과 전체 prefix 재계산을 대조해 stale-state 오차도 측정한다. encoder가 학습되는 동안 오래된 latent만 재사용하지 않는다. BC가 추가되면 총 update 후 KL을 검사한다.

## 4. 구현 및 실환경 검사

테스트는 실제 실패 위험이 있는 계약을 겨냥한다.

| 예정 테스트 | 의미 있는 검증 |
|---|---|
| `test_adapter_state.py` | row shuffle·두 경기 interleave·같은 모델 양 팀 사용·reset/respawn·partial input·state 누출 |
| `test_tensor_planner.py` | 막힌 길·대각선 모서리·적 본진·창고 수용 실패·목표 경쟁; 독립 A* 대조 |
| `test_policy_decisions.py` | class 제약·event·41-way/최대 k 개입·선택 후 취소 없음·STOP·동률 |
| `test_recurrent_ppo.py` | ratio=1·old mask/log-prob 거부·hidden 재계산·encoder gradient·forcing 분리·가변 기간 return |
| `test_critic_rewards.py` | critic-only update 불변·PopArt 출력 보존·potential telescope·점수 루프·terminal reset |
| `test_scheduler_collector.py` | 긴 noop와 짧은 강자 패배 혼합의 실제 step 비중·episode quota·watchdog·중복 commit |
| `test_onnx.py` | 긴 sequence·all-masked 후보·state round-trip·routing·side swap·provider parity |
| `test_evaluation_runtime.py` | 제공 runner 동치·reward/score 불일치 유지·split 누수·source lock·소유 자식·다른 cwd |

synthetic 검사 뒤 20경기 lifecycle에서 graphics·자연 종료·timeout·state 초기화를 확인한다. 이어서 교사와 tensor planner를 같은 task/side에서 비교하고 ONNX를 실제 경기 루프에 넣는다. synthetic 성공을 실환경 검증 완료로 표시하지 않는다.

## 5. 실험과 예산

모든 숫자는 구현 시 config에 등록할 **초기 제안 예산**이다. step은 agent row가 아니라 environment step이다. teacher·실패 시도·evaluation은 PPO와 따로 집계한다. 완료 wave로 인한 초과량은 최대 동시 episode 수×22,000 step 상한 및 실제 값을 남긴다.

| Arm | 구성 | 검증 질문 |
|---|---|---|
| P | 동결 tensor planner, 3 worker+2 guard, 학습 없음 | 기본 능력·ONNX 이식 보존 |
| R | P + v6식 41-way residual + 독립 critic | 단기 보정의 한계 |
| G | learned 목표/역할·명시 state·GRU + 계획 이동, residual 없음 | 목표 결정의 기여 |
| F | G + 조건부 gate/correction, 초기 최대 1개입 | 전체 v11 제안 |
| D | 같은 공개 ID·Attention/GRU·critic, 직접 방향 head | 입력 개선과 계획 제어의 효과 구별 |

R/G/F는 초기 planner·역할을 공유하고 G/F는 encoder 초기값도 맞춘다. D는 입력·대략적인 파라미터 수·teacher 자료·예산을 맞추되 행동 공간 차이를 밝힌다. P는 학습 seed가 없는 기준이며 복제 checkpoint로 표본을 늘리지 않는다.

| 단계 | 실행 | 예산과 판정 |
|---|---|---|
| P0 오프라인 | 계약·수식·최소 ONNX·기존 trace | Unity 0 |
| P1 이식 기준 | lifecycle 20경기, 교사/P 양 진영 | 상대당 10 map pairs; 조기 결함 탐지용 |
| P2 teacher | 공개 입력/state와 일치하는 새 demonstration | 262,144 env step 공통 자료; episode split |
| P3 pilot | R/G/F/D × seed 11 | 131,072×4 = 524,288 env step |
| P4 본 비교 | R/G/F/D × seeds 11/22/33 | 각 2,097,152, 합계 25,165,824 env step; pilot에서 새로 초기화; 12 runs의 dev 총 1,800경기 |
| P5 확인 | 우승 arm과 비교 arm × seeds 44/55 | 각 2,097,152, 합계 8,388,608 env step; 두 arm의 전체 5 seeds에 대해 핵심 상대 1,000경기 진단 |
| P6 최종 | dev에서 선택한 checkpoint 하나·decoder·objective·ONNX hash 고정 후 test | 후보 500경기와 v4 기준 모델 440경기; G4·G5·G6 판정 |

기본 teacher+pilot+main+confirmation 합계는 **34,340,864 env step**이다. baseline·evaluation·실패 시도·초과량은 이 숫자 밖이며 실제 총량에 별도 포함한다. 계획의 예산이지 이번에 시작한 학습량이 아니다.

P5 비교 arm은 R/G/D 중 dev 강자 상대 macro가 가장 높은 arm이다. 우승 arm과 같으면 남은 가장 높은 arm을 택한다. F가 지면 반드시 F를 최종 모델로 선택하지 않는다. P4에서는 새 seed 초기화 후 동일 자료·schedule을 쓰고 pilot 가중치를 이어받지 않는다.

기술 gate는 모든 arm에 적용한다. 후속 사용자 지시에 따라 학습 전 G1과 반납·승점률 같은 G2 성능 기준은 진단 전용으로 바꾸었다. 미학습 고정 P와 학습된 v4의 차이, 짧은 pilot 성적만으로 본학습을 중단하지 않는다. F/R/G/D 모두 사전 고정된 동일 본학습 예산으로 진행하고 낮은 성적도 결과로 남긴다. 실행·데이터 오류, 본학습 G3 진행 조건과 최종 G4·G5·G6 기준은 유지한다. 축소한 P5 confirmation의 성능 미달은 진단으로 기록하고 P6를 계속한다. 기술 실패·rollback 한도로 예산을 끝내지 못한 run은 실패/중도 종료로 명시하며 동일 예산 비교를 완주했다고 주장하지 않는다.

curriculum은 기본 행동 보존 → 목표 학습 → 전략/residual → 상대 다양성 순서다. 보유물 귀환·창고 변경·위험 회피가 실패한 채 강자 혼합 비중만 늘려 강제 승급하지 않는다. 탐색률뿐 아니라 agent-action/team-step 개입 비율과 득점·반납·정체 변화를 기록한다. rollback은 모델/optimizer/scheduler/RNG를 함께 복원하고 이미 쓴 step을 지우지 않는다. 반복 rollback 3회면 run을 실패로 종료하고 원인을 분석한다.

### 초기 학습 설정

출발값은 actor LR `1e-4`, critic LR `3e-4`, PPO clip `0.2`, max-grad-norm `0.5`, update epoch `4`, 총 update KL 한도 `0.02`다. frozen behavior의 완결 wave를 모아 학습하고 event sequence를 섞되 순서를 깨지 않는다. minibatch는 16개 sequence부터 메모리 probe로 고정한다. 기본 `gamma=1`, primitive-step `lambda=0.9975`; event 간 k-step에서 discount/GAE는 설계의 기간 공식을 따른다. entropy는 goal/gate/correction 각각 로그로 남기고, 초기 coefficient `0.003`을 기준으로 pilot에서 지나친 방향 탐색과 완전 KEEP 고착을 검사한다. 이 값들은 검증할 초기값이며 최적 설정이라는 주장이 아니다.

### 가설별 추가 실험

기본 본 비교와 별도로 해결되지 않은 가설부터 실험한다. 한 추가 arm은 **3 seeds×524,288 = 1,572,864 step**이다. 같은 길이의 F 재실험 또는 고정된 schedule의 F prefix와 비교한다. 모든 조합의 전수 탐색은 기본 예산에 넣지 않는다.

1. 명시 목표 state와 GRU를 각각 제거하여 계획 기억·학습 기억을 구별한다.
2. C0 vector / C1 map / C1-blind critic을 같은 actor/init·상대 task에서 비교한다.
3. episode 균형 / step deficit을 비교하고 loss 가중치는 고정한다.
4. residual 최대 1/2/5유닛과 41-way/조건부 gate를 비교한다.
5. 목표 재검토 10/25/50 step과 greedy/sampled 실행을 비교한다.

시험할 arm 수·예산은 실험 전 등록한다. 유효한 요소를 합쳐 새 최종 조합을 만들면 조합 전체를 다시 확인한다. test를 보고 그에 맞춰 새 arm을 선택하지 않는다.

## 6. 노출·평가·선택

### 노출과 데이터 분리

주 학습의 step 목표는 planner 40 / 기타 scripted 30 / self-play 20 / 직접 정책 10%다. 부족 가족에 새 episode를 배정하고 매 wave마다 step·완료 경기·목표/잔차 결정 수·학습 가중치를 기록한다. 두 wave 연속 planner 비중이 30% 미만이면 우선 배정을 보정하고 비교의 노출 차이를 보고한다. 목표 비중만 기록하고 달성했다고 가정하지 않는다. 양 진영 step 목표는 50:50이다.

기본 PPO loss는 실제 수집된 결정의 분포에 대해 계산한다. 가족별 loss 재가중은 별도 arm으로 관리한다. PFSP는 가족 비중을 유지하며 가족 안에서만 사용하고 uniform 20% 하한을 둔다. snapshot이 없으면 해당 self-play 비중은 동결 초기 v11로 채운다.

train/dev/confirmation/test는 요청 seed뿐 아니라 **실제 초기 배터리/창고 map hash** 겹침도 검사한다. seed 적용이 불확실하거나 초기 배치가 반복되면 실제 독립 배치 수를 보고하고 새로운 맵에 대한 일반화를 주장하지 않는다. split 겹침은 학습/선택 전에 재등록으로 해결하며 원본 API에 숨은 reset/step을 넣지 않는다.

이번 리그는 설계에 사용했으므로 v11의 unseen test가 아니다. 새 task·map·action RNG seed를 등록한다. 고정 상대 checkpoint는 학습과 benchmark에 공통으로 등장할 수 있다. **test 경기·map·행동 seed·결과**를 학습/선택에 넣지 않는다. 미관측 상대 일반화는 별도의 held-out 상대 검증으로 구분한다. demonstration validation 역시 episode 단위로 분리한다.

### 평가량과 선택

- 완료된 baseline·pilot의 **15상대×10 map pairs×2진영=300경기/후보** 평가는 그대로 재사용한다. 기존 결과·task·source hash를 새 평가 조건으로 바꾸지 않는다.
- 본학습 dev: scripted 9개 + v4/v6/v7 두 개 + v9/v10 = **15상대×5 map pairs×2진영=150경기/run**. 4 arms×3 seeds의 총량은 **1,800경기**다.
- Confirmation: 선택한 두 arm 각각의 seeds 11/22/33/44/55를 **noop·random·collector·v4·v6의 5상대×10 map pairs×2진영=100경기/checkpoint**로 확인한다. 총 **1,000경기**이며 seed 간 재현성 진단에 사용한다. 이 5상대 결과로 전체 21상대 G5·G6 통과를 주장하지 않는다.
- Frozen test: dev에서 선택해 고정한 **checkpoint 하나**만 전체 21상대로 평가한다. 일반 19상대는 각각 10 map pairs, v4·v6는 각각 30 map pairs를 양 진영으로 실행해 **500경기**다. 동결 v4 기준 모델은 같은 상대·map pairs에서 자기 자신인 v4를 제외한 **440경기**다. confirmation과 test의 split은 서로 다르다.
- G3 등 진행 조건을 충족해 끝까지 실행할 때 남은 평가 합계는 **3,740경기**다. 완료된 baseline·pilot은 이 숫자에 포함하지 않는다. 모든 학습 예산과 추가 seeds 44/55는 그대로 유지한다.
- 동결 기준 모델과의 비퇴행 비교는 같은 공통 상대·map pair에서 다시 평가한다. 기준 모델 자신을 상대 목록에서 빼면 후보에서도 같은 상대를 빼 분모를 맞춘다. 과거 리그 90%를 새 test의 동일 조건 baseline으로 간주하지 않는다.
- 모델별 macro와 별개로 가족별 macro·최악 가족·각 상대 W/D/L·양 진영·seed별 결과를 낸다. planner checkpoint가 다섯 개라는 이유로 전략 가족 평균에 5배 가중하지 않는다.

dev arm 선택은 v4/v6 직접 평균 승점률, 가족 macro, worst-side, p95 지연, 작은 모델 순으로 한다. 제출 seed는 같은 dev 규칙을 사용하고 최종 동률은 seed 오름차순이다. 선택·decoder를 confirmation 전에 고정한다. G3 미달 시 진행을 중단하는 조건은 유지한다. G3 통과 이후에는 축소 confirmation의 성능 미달을 재현성 진단으로 남기고, 이미 선택한 checkpoint의 최종 test를 진행한다. confirmation이나 test 성적으로 seed를 다시 선택하지 않는다.

기본 제출 checkpoint는 P4 seeds 11/22/33 중에서 선택한다. P5의 44/55는 arm 재현성 확인용이며 자동으로 제출 후보가 되지 않는다. 추가 seed를 제출 후보로 삼는 후속 실험에는 그 dev 선택 이후 별도의 confirmation split이 필요하다. 경제 비퇴행 G6의 incumbent는 기존 리그에서 동결한 **v4-29794421**로 사전 고정한다. P는 이식/기반 행동 보존의 별도 기준이고 v6는 추가 성능 비교 대상이다. test 결과에 따라 G6 baseline을 유리한 모델로 바꾸지 않는다.

bootstrap은 training seed와 map pair를 cluster로 잡는다. 양 진영은 같은 map block이며 유닛/step을 독립 표본으로 늘리지 않는다. seed 3~5개의 한계를 밝히고 seed별 원수치를 보관한다. 모두 5:5여서 bootstrap 폭이 0이 되어도 불확실성이 0이라고 하지 않고 pair 단위 bounded interval과 유효 배치 수를 보고한다. v4/v6 두 1차 우월성 비교에는 Holm 보정을 적용한다. 결과를 본 뒤 검정법을 바꾸지 않는다.

## 7. 승격과 최종 성능 gate

다음은 **개발 목표값**이다. 현재 이미 달성한 결과가 아니다. 승률과 무승부 반점인 승점률을 둘 다 보고하며 아래 gate는 승점률을 쓴다.

| Gate | 기준 | 실패 시 |
|---|---|---|
| G0 계약 | 공개 입력·routing/state/reset·최소 ONNX 검사 통과 | 장기 학습 전 수정 |
| G1 기본 행동 진단 | P의 공통 dev macro가 reference보다 5%p 초과 하락 없음; noop/random ≥90%, collector ≥80%; score·반납 함께 확인 | 성능 미달을 기록하고 P2 교사 학습 계속 |
| G2 pilot 진단 | 양 진영 반납 존재; collector에서 P−5%p 이상; random 이상 | 성능 미달을 기록하고 P4 본학습 계속; invalid action/자연 종료 위반은 별도 기술 gate에서 중단 |
| G3 유망 후보 | v4/v6 직접 평균 ≥55%; 공통 scripted에서 P 대비 5%p 초과 회귀 없음 | 강제 승급 대신 원인 실험 |
| G4 최종 강자 | held-out v4와 v6 각각 ≥60%, Holm 보정 95% 하한 각각 50% 초과 | 최강 달성으로 표기하지 않음 |
| G5 전반 대응 | 21상대 macro ≥90%, 모든 scripted ≥80%, 강자 가족 양 진영 각각 ≥50% | 특정 상대/진영 약점 후보로 분류 |
| G6 경제 비퇴행 | 공통 상대·map에서 baseline 대비 관측 score 승점률 차이의 단측 95% 하한 ≥−5%p; 반납/정체 악화 원인 검토 | raw reward만 높인 후보를 기본 최종에서 제외 |
| G7 배포 | lock한 ONNX의 긴 sequence·실제 경기·state isolation·provider 성능 통과 | PyTorch 실력과 분리해 배포 미완료 |

선택한 두 arm의 5개 학습 seed 재현성은 핵심 5상대 confirmation으로 보고하고, G4–G6는 선택 제출 checkpoint의 전체 상대 frozen test에서 판정한다. 기준과 통계적 하한을 낮추지 않으며, 표본이 부족해 기준을 입증하지 못하면 미달로 보고한다. 특히 v4·v6는 각각 30 map pairs를 유지한다. test를 반복해서 들여다보며 유리한 시점에 멈추거나 다른 seed로 바꾸지 않는다.

G6는 주 objective 변경이 아니라 경제 능력 보존 조건이다. 공식 winner가 제공되면 새 objective와 해당 평가를 별도로 추가한다. 로컬 raw reward test를 공식 대회 인증으로 바꾸지 않는다.

## 8. 자원·시간·운영

planner+actor+state 전달+ONNX 전체를 측정한다. 리그에서 팀 추론 p95는 v6 약 10.88ms, v7 약 12.3ms, v10 약 2.70ms였지만 당시 worker 경합 조건의 관측값이다. 개발 목표는 같은 호스트/thread 조건에서 **팀 1회 warm p95≤10ms**, p99·최대·cold load 별도 보고다. 이는 공식 대회 제한이 아니다. 초과 시 거리장·캐시·width를 측정하고 성능을 만드는 planner를 검증 없이 제거하지 않는다.

worker 2/4/8 probe로 처리량·메모리를 보고 고르되 본 비교 설정은 고정한다. actor CPU와 learner CPU/MPS도 실제 forward/backward/전송으로 측정한다. 합성 추론 속도만으로 Unity throughput을 보장하지 않는다.

완료 wave는 압축 보관하고 demonstration replay에는 크기 상한을 둔다. one-hot map을 검증한 뒤 pixel당 semantic ID 1 byte로 저장해도 262,144개 장면의 map 원자료는 약 2.4GB다. nonbinary graphic은 float32로 보존한다. vector/state/target·checkpoint·임시 파일을 포함해 wave 최악 사용량을 계산하고, 초기 RAM soft limit 32GiB·disk reserve 25GiB에서 probe한다. 예상 wave 사용량까지 여유가 없으면 시작하지 않는다.

ETA는 `남은 env step/실측 유효 steps/s + 평가 소요 + update/export`로 갱신한다. 리그 실행 약 32.1시간을 v11 학습 시간 보장으로 옮기지 않는다. 긴 timeout과 조기 종료 시간을 따로 측정한다.

다음은 **앞으로 구현할 명령**이다. 현재 실행 가능한 v11 명령이 아니다.

```bash
bash code/v11/run_v11_fast.sh --check
bash code/v11/run_v11_fast.sh --prepare --name goal_memory_v1
bash code/v11/run_v11_fast.sh --name goal_memory_v1
bash code/v11/run_v11_fast.sh --name goal_memory_v1 --status
bash code/v11/run_v11_fast.sh --name goal_memory_v1 --stop
bash code/v11/run_v11_fast.sh --name goal_memory_v1 --resume
bash code/v11/watch_v11_models.sh --study goal_memory_v1 --opponent v6
```

`--check`는 Unity/학습 없이 계약·자산·합성 ONNX를 검사한다. `--resume`은 같은 source/config를 요구하며, 이번 승인된 평가 변경은 `evaluation_revision.json`에 기록된 범위만 검증하여 허용한다. 중단된 현재 실험은 사용자가 다음 한 줄로 재개한다. 완료된 학습과 baseline·pilot을 다시 실행하지 않는다.

```bash
bash /Users/safeailab_macmini/Desktop/2026-IST-tech-RL/code/v11/run_v11_fast.sh --name goal_memory_v3 --resume
```

관전은 동결 snapshot으로 실행하고 학습/test와 분리한다. 원본 graphics를 유지하고 소유 프로세스만 정리한다.

## 9. 다음 구현의 첫 작업

먼저 **공개 이름 routing + 명시 state의 ONNX 왕복**과 **한 worker의 배터리 탐색·반납 tensor 경로**를 구현한다. 이를 다섯 유닛 baseline으로 확장하고 실제 능력 보존을 확인한 뒤 Attention 목표 학습과 recurrent PPO를 연결한다.

이번 산출물은 [리그 분석](../reports/league_findings.md), [설계](v11_design.md), 이 구현 계획, [운영진 답변](../../common/organizer_qa_2026-10-24.md), 감사 도구/자동 결과다. 향후 구현 보고서는 실제 수행·미수행·학습량·실패 run·성능 gate를 기록한다. 장점 결합은 설계 의도이고 최고 성능은 동결 비교가 입증해야 할 결과다.
