# BlackOut RL v8 구현 계획서

**구현 명세 · 실험·검증 계획 · 참고 문서·논문 목록**

기준일: 2026-09-27 KST  
문서 버전: 1.1 — 기존 구현 계획에 참고자료·읽을 위치·PR별 연결을 보강한 판.
문서 상태: 구현 제안. 코드 수정·새 학습·Unity 실행·성능 검증을 완료한 문서가 아니다.
대상 저장소: [thislis/2026-IST-tech-RL](https://github.com/thislis/2026-IST-tech-RL)

## 문서 안내

이 파일은 구현 범위, 동작 계약, 테스트, PR 순서, 참고자료를 한곳에 모은 인계용 계획서다. 본문의 새 경로·API·CLI는 **구현 예정 명세**이며 현재 저장소에 존재하거나 실행 검증되었다고 가정하지 않는다. `INT`/`EXT`는 첨부 원문, `Sxx`/`SUPxx`/`DOCxx`는 §17의 외부 참고자료다. `D01–D08`은 두 조사자가 사용한 공동 의사결정 ID이며 참고문서의 `DOCxx`와 다르다.

| 먼저 찾는 내용 | 바로가기 |
|---|---|
| 전체 방향과 범위 | [§0 결정 요약](#sec-0), [§1 성공 기준](#sec-1) |
| 모듈·데이터 흐름 | [§2 구현 구조](#sec-2) |
| 필수 수정 | [§3 환경·보상](#sec-3), [§4 정책·planner](#sec-4), [§5 계측·재현성](#sec-5) |
| 본선 및 조건부 정책 | [§6 C1](#sec-6), [§7 PPO·노출](#sec-7), [§8 C2](#sec-8), [§9 C3](#sec-9) |
| 실험·출시 판단 | [§10 평가](#sec-10), [§11 실행·export](#sec-11) |
| 작업 순서와 회귀 | [§12 PR](#sec-12), [§13 테스트](#sec-13), [§14 F01–F36](#sec-14) |
| 프로젝트 문서·원자료 | [§16 내부 자료 목록](#sec-16) |
| 논문·공식 문서 | [§17 참고자료 29개](#sec-17), [§18 구현 저장소·라이선스](#sec-18) |
| 담당자별 읽을 자료 | [§19 PR별 독서·검증표](#sec-19) |
| 초안 값과 근거 파일 | [§20 잠정값·미확정 계약](#sec-20), [§21 원문 provenance](#sec-21) |

논문을 모두 읽을 때까지 개발을 미루는 순서는 아니다. PR00–04의 필수 계약·계측부터 구현하고, C2/C3의 조건이 충족될 때 해당 심화 문헌을 읽고 구현한다.

<a id="sec-0"></a>

## 0. 결정 요약

**v8 본선은 `정확한 경기 결과 계약 + 실패를 격리한 planner + 명시적인 단일 슬롯 개입 gate + 동일한 학습·실행 계약을 사용하는 PPO`로 구성한다.** 부분회로/전뇌 확대는 본선의 전제가 아니다. 우선 기존 최종 checkpoint의 선택 규칙을 재검증하고, 작은 비-connectome 정책을 대조 가능한 기준으로 구현한다. 목표/역할 지속 정책은 미세 조정 상쇄가 관측될 때만, 전뇌 readout 개선은 별도 연구 분기에서 진행한다.

여기서 “학습·실행 계약 일치”는 sampling으로 학습한 정책과 deterministic 평가 정책의 기대 성과가 같다는 뜻이 아니다. 실제 behavior의 log probability를 저장·재계산하고, 평가 decoder를 별도 정책 구성요소로 고정·측정한다는 뜻이다. 분포 인수분해만으로 argmax 문제를 해결했다고 주장하지 않는다.

### 자료의 지위

- **INT**: `internal_result.md`. 내부 담당자가 원본 23개 run·checkpoint·1,380경기 등을 재집계하고 오프라인 계산·합성 검사를 수행한 보고서다. 이번 계획 작성자가 원 checkpoint를 다시 계산한 결과는 아니다. 보고서가 연결하는 `internal_analysis/*.json`, 분석 script, 실제 checkpoint·Unity 빌드는 이번 첨부에 없다. PR00에서 인계받는다.
- **EXT**: `v8_external_research.md`, 구조화 동반 자료 `v8_external_research.json`. 같은 외부 조사의 두 표현이며 독립적인 두 연구로 세지 않는다. 문헌 결과는 BlackOut에서 재현된 성능 증거가 아니다.
- **REF**: §17의 논문·공식 문서 목록. 상세 독해·코드 pin·라이선스 확인 수준은 EXT의 조사 기록에서 인계했으며, 이번 문서화에서는 핵심 공식 페이지의 연결·서지와 일부 API 본문을 다시 확인했다. 전체 논문 재독해·코드 재실행·최신 대회 규정 확인을 새로 수행한 것은 아니다. 외부 원문·구현 링크와 확인 범위를 각 항목에 적었다.
- 아래 아키텍처, 파일명, 수치 임계값, 실험 예산, PR 순서 중 기존 보고서의 측정값이 아닌 것은 **이번 구현 제안**이다. 모든 신규 경로와 CLI는 아직 존재한다고 가정하지 않는다.

### 이번에 바뀐 진단

| 관찰 | 자료가 지지하는 판단 | v8 결정 |
|---|---|---|
| 서로 다른 A2/A3 graph·가중치, 최종 correction logit 상한 -2.079234~-0.934818, KEEP=0 | 보고서의 보수적 수치 bound 아래 최종 greedy 경로는 항상 KEEP. 전 부동소수점 형식 증명이나 역사적 physics trace 복구는 아님 | 그래프 크기보다 **선택 decoder와 개입 확률 구조** 먼저 수정 |
| A0/A1 sampled 개입률 약 71.7~85.3%, 성과 저하 | 개입이 많다고 유용한 것은 아님. 인과적인 해로움까지 이 로그로 확정하지 않음 | 강제 개입 증가·entropy 증대부터 하지 않음 |
| F1 seed 22/33: 역사 training rate 20,000개에 최종 head 적용 시 각각 SE/West만 선택 | offline readout 편중. 실제 최종 평가 전체 trajectory는 아님 | neural activity·logits·행동·성과를 따로 검사 |
| final score 1,211/1,380건이 0:0, wrapper winner도 reward 부호 유도 | score 사용 불가, wrapper 승률은 역사 재현용 제한 지표 | 새로운 engine outcome과 보상 파이프라인 필수 |
| PPO ratio/GAE 합성 검사 통과, 실제 graph edge도 변화 | “PPO가 전부 틀림”, “graph gradient가 영원히 죽음”은 근거 부족 | 알고리즘 전체 교체 대신 실제 transition 정합성·모듈별 학습 계측 |
| 실제 env-step A 비중 F1 33.062%, residual 45.704~49.540% | episode 비중과 데이터 노출 비중은 다름 | 모든 transition 노출을 기록; 분포 변경은 별도 대조 |
| 238 tests 중 236 pass/1 fail/1 skip | 실패는 resumed latest와 과거 hash의 부적절한 비교. backup 손상 근거 아님 | immutable backup·정상 lineage를 따로 검사 |

근거: INT L138–178, L213–228, L324–340, L82–113, L350–375, L407–431.

<a id="sec-1"></a>

## 1. 범위·성공 기준·진행을 막는 조건

### 1.1 세 가지 완료 상태를 구분한다

1. **engineering_ready**: 종료·보상·행동·상태·provenance 회귀가 통과하고 새 실험을 신뢰성 있게 실행할 수 있다.
2. **research_improved**: 검증된 로컬 build와 고정 상대·평가 분포에서 기존 기준선보다 개선된 근거가 있다. 공식 대회 성능과 동일하다고 주장하지 않는다.
3. **submission_ready**: 실제 운영 loader·입력·state/reset·패키지·장치·자원 계약 및 해당 배포 환경의 경기 검증을 통과했다.

공식 계약 답변이 늦어져도 offline 진단·합성 검사·로컬 수정 build의 연구는 진행한다. 다만 수정 build가 공식 build와 동등하다고 선언하거나 stateful 정책을 제출 인증하지 않는다. authoritative local outcome이 없으면 새 승률로 모델을 선택하는 단계는 보류한다.

### 1.2 목표와 기준선

주목표는 고정 target 정책 전체를 상대로 한 유효 경기의 `W/N` 개선이다. draw는 N에 포함하고 승리로 세지 않는다. 이는 이번 계획의 **로컬 연구 endpoint 선택**이다. EXT의 `win+0.5draw`는 보조 utility로만 저장하며 공식 무승부 승점으로 단정하지 않는다. 공식 평가가 다른 점수체계이면 별도 계약/실험 ID로 endpoint를 재등록한다. engine winner와 공식 runner 결과가 다르면 두 지표를 모두 보고하고 자동으로 같다고 간주하지 않는다.

최소 실용 개선폭 `δ=0.05`(5%p)는 잠정 프로젝트 판단값이지 논문 최적값·공식 요구조건이 아니다. 사전 등록한 본확인에서 `Δhat≥0.05` 및 적절한 차이 구간의 하한 `>0`을 만족하면 관측 범위의 개선 후보로 인정한다. “개선이 최소 5%p 이상임”이라는 강한 주장은 하한 자체가 `>0.05`일 때만 가능하다. 소수 run에서는 그 구간의 coverage 한계를 별도로 표시한다.

기존 target 85% 이상 목표는 별도 장기 목표로 유지한다. 과거 10경기 9승 gate와 새로운 다중 seed·맵 비교를 동일한 통계적 인증으로 취급하지 않는다. 성능 평가 분모·맵·빌드·상대가 바뀌면 과거 50%와 직접 증감하지 않는다.

### 1.3 기준 정책의 세 층

| ID | 내용 | 쓰임 |
|---|---|---|
| R_hist | 원본 v7 checkpoint와 원본 실행 소스·build·decoder | 역사 재현. 결과를 덮어쓰지 않음 |
| R_plan | 수정 결과 계약에서 동작하는 pure planner; failure-isolation revision 고정 | 본선 최소 우위 기준. 학습 seed가 없는 고정 정책 |
| R_flat | v8와 같은 encoder·planner·critic·보상·수집·예산을 사용하는 flat41 정책 | C1 gate의 구조/선택 효과를 분리하는 재학습 대조군 |

기존 v7 final도 새 build 아래 inference-only로 재평가하여 이동 효과를 확인한다. 다만 예전 보상으로 학습한 모델과 새 보상으로 학습한 모델의 차이를 gate 고유 효과로 해석하지 않는다. R_flat 재학습이 그 혼동을 줄인다. target은 neural core만이 아니라 **planner override·코드·설정까지 포함한 완전한 고정 상대**로 고정한다. learner planner 실패 격리 패치가 frozen opponent까지 조용히 바꾸면 안 된다.

<a id="sec-2"></a>

## 2. 권장 구현 구조

```text
Unity final-event + physical tick/decision metadata
  → V8EnvAdapter / OutcomeBroker
  → 공통 ObservationAdapter + ActorContext(합법 정보만)
  → PlannerAdapter(슬롯별 실패 격리)
  → CandidateBuilder + ValidityMask
  → V8Policy
       R_flat: flat41
       C1: intervention gate + conditional40
       C2: optional goal manager (조건부)
       C3: optional coordinate/context readout (연구 분기)
  → DecisionDecoder + deterministic Executor
  → executed actions / next observation / OutcomeEvent
  → RewardLedger + TransitionBuilder
  → PPO / diagnostics / immutable checkpoint
  → 동일 PolicyFactory의 evaluator와 export adapter
```

새 기본 경로는 `blackout_rl/v8/` 아래에 둔다. v7 기록 소스는 수정하지 않고 호환 adapter 또는 명시적인 migration manifest로 읽는다. 옛 checkpoint의 source hash를 새 코드 hash로 덮어쓰는 방식은 금지한다.

### 제안 파일 배치

```text
blackout_rl/v8/
  contracts.py          # OutcomeEvent, StepContext, Transition, capability schema
  environment.py        # OutcomeBroker, terminal/decision join, legacy diagnostic adapter
  reward.py             # score ledger, end bonus, idempotent event processing
  policy_factory.py     # train/eval/export 단일 factory
  planner_adapter.py    # prepare/execute 분리, slot-local 실패 처리
  candidates.py         # stable candidate ID, physical-rule/heuristic mask 구분
  distributions.py      # flat41, exact factorization, gated conditional40
  decoders.py           # joint_argmax / gate_then_argmax / sampled
  model.py              # frozen encoder + 작은 correction/gate head + critic
  collector.py          # episode 유지, behavior version, trace 저장
  returns.py            # k=1 bootstrap/GAE; 조건부 SMDP
  trainer.py            # 공동 ratio·loss·optimizer·체크포인트
  telemetry.py          # 상시 집계, ring buffer, 실패/무작위 control windows
  checkpoints.py        # immutable blobs, parent lineage, resume contract
  exposure.py           # every-step map/side/opponent accounting
  evaluation.py         # 공통 outcome, policy lock, attempt ledger
  statistics.py         # independent-run + shared-map resampling
  export.py             # capability 계약 검사, clean-room parity
  options.py            # C2 활성화 시에만 구현/로드
  readout.py            # C3 활성화 시에만 구현/로드
configs/v8/
  protocol.yaml
  experiments/*.yaml
contracts/v8/
  environment_contract.json
  submission_contract.json
  permissions.json
scripts/
  validate_v8_contracts.py
  audit_v8_behavior.py
  train_v8.py
  evaluate_v8.py
  analyze_v8.py
  verify_v8_export.py
tests/v8/
  test_outcome.py / test_timer_integration.py / test_distributions.py
  test_episode_state.py / test_planner_isolation.py / test_returns.py
  test_factory_parity.py / test_checkpoint_lineage.py / test_statistics.py
  test_export_contract.py
```

이는 구현 후의 책임 분할안이며, 현재 사용할 수 있는 CLI 목록이 아니다. 비-connectome 본선은 graph 원자료나 MaleCNS 전체를 의무적으로 로드하지 않도록 registry의 의존성을 mode별로 분리한다.

<a id="sec-3"></a>

## 3. WP-A: 환경·결과·보상 계약 복구 — PR01/02

**구현 참고:** [S01](#s01) · [S02](#s02) · [SUP01](#sup01) · [SUP02](#sup02) — 읽을 위치와 적용 한계는 §17 참조.

### 3.1 엔진에서 reset 전에 결과를 확정한다

새 `OutcomeEvent`의 최소 필드는 다음과 같다.

```text
schema_version, build_sha256, protocol_sha256
run_id, episode_id, event_id, decision_id, unity_tick, game_time
final_score_points[2] or null
winner_team or null, winner_source
termination_reason, terminated, truncated
final_observation_id or null, score_snapshot_revision
```

`preterminal_score_points`, `terminal_raw_info`, `legacy_wrapper_winner`, `unity_shaping_sum`은 별도 진단 필드다. null은 누락이지 0 또는 draw가 아니다. score의 단위는 원점수 정수로 표준화하고 actor의 normalized score와 필드명을 분리한다.

종료 상태는 `RUNNING → ENDING → ENDED → RESETTING`으로 정의한다. 첫 종료 요청이 episode를 ENDING으로 바꾸고, 엔진의 기존 게임 규칙이 정하는 사건 처리 순서를 유지한 채 final snapshot을 만든다. 이후의 늦은 callback이 snapshot을 바꾸지 못하게 한다. 동시 득점·전투 처리 순서를 임의로 재정의하지 말고 실제 game coordinator의 우선순위를 계약으로 문서화한다.

winner는 이 엔진 결과에서 전달한다. terminal reward 부호나 전체 shaping 합으로 재구성하지 않는다. score와 winner의 관계는 종료 reason/게임 규칙별 fixture로 검사한다. 모든 상황에서 단순 score argmax가 규칙이라고 가정하지 않는다.

`OutcomeBroker`는 `(run_id,episode_id,event_id)`로 수신·저장·중복 제거하고 TerminalSteps와 episode별로 결합한다. 메시지는 step/reset 교환 때 전달되므로 다음 episode에서 늦게 도착한 event를 현재 episode에 붙이지 않는다. 누락을 메우기 위해 암묵적인 gameplay step을 추가하지 않는다. custom side-channel은 가능한 전달 수단이지 대회 허용이 확인된 기능이 아니다. [EXT §2.1; S01, S02]

### 3.2 terminal과 truncation을 분리한다

| 사건 | 정책 |
|---|---|
| 게임 규칙상 목표 점수·게임 시간 소진 | true termination: bootstrap 0 |
| collector rollout 경계 | 환경 reset 아님. final observation에서 bootstrap, 해당 buffer의 trace는 경계에서 끝남 |
| 외부 중단, 정상 final observation 보유 | truncation: 마지막 관측으로 bootstrap, reset 넘어 GAE 연결 금지 |
| crash/관측 부재/잘못된 episode | invalid attempt. 결과나 value를 지어내지 않음. 오류 transition 학습 금지, 실패 기록 보존 |
| option 종료·안전 abort | 게임 종료가 아니므로 보통 continuation/bootstrap. 실제 done과 별도 flag |

게임 길이 21,003이라는 숫자만으로 timeout으로 분류하지 않는다. 일반 LLAPI에서 `env.step()`은 Unity FixedUpdate 1회와 같다는 보장이 없으므로 실제 `unity_tick`과 `game_time`을 계측한다. 일정 dt가 검증될 때만 step→초 환산 및 SMDP γ 지수를 고정한다. [INT L93–103, L234–240; S01, S02]

### 3.3 마지막 score delta와 bonus를 정확히 한 번 반영한다

초기 reward 계약은 기존 의도를 유지한 `score_delta_terminal_v8`로 둔다.

`r_team = ((Δscore_own − Δscore_other) / target_score) + terminal_bonus`

단, 마지막 delta는 authoritative final snapshot과 ledger의 마지막 반영 점수 사이에서 계산한다. score target과 bonus coefficient는 config에서 읽고 source hash에 포함한다. 예를 들어 직전 97:42, 종료 100:42, target=100, 승리 bonus=1이면 마지막 reward는 `0.03 + 1 = 1.03`이다. 동일 event 재수신은 0회 추가 반영; 다음 reset 100→0은 score 사건이 아니다.

Unity score handler가 새 점수>0을 보고 shaping을 부여하는 경로도 delta/event 종류로 분리한다. v8 학습에는 Unity shaping을 기본 합산하지 않는다. 엔진 버그 수정은 별도 build ID로 남기고 기존 opponent의 추론 의미를 검증한다. 위 score-delta surrogate는 γ<1에서 원 승률 목적의 정책 불변성을 보장하는 PBRS라고 부르지 않는다. 최초에는 새 pickup/거리/충돌 보상을 추가하지 않는다. terminal-only 또는 PBRS 비교는 보상 지연/critic 진단 후 별도 실험이다. [INT L105–113; EXT §2.3]

### 3.4 TimerManager 생명주기 수정

배열 index로 완료 타이머를 지우는 대신 object ID와 generation으로 수명주기를 관리한다. 프레임 시작의 타이머 snapshot을 순회하고 완료 타이머를 identity로 제거한 뒤 callback을 처리한다. clear/restart는 generation을 증가시켜 같은 프레임에 남은 old-generation 타이머/callback을 무효화한다. callback에서 생성한 새 타이머를 old snapshot의 후속 iteration이 tick하거나 삭제하면 안 된다.

테스트는 단순 Python mirror에 그치지 않는다. C# edit/play-mode fixture와 실제 build에서 자동 재시작 timeout 3회, 명시 reset timeout 3회를 수행하고 timer 생성·완료·삭제 ID를 확인한다. 외부 watchdog 종료를 성공 timeout으로 세지 않는다. [INT L130–136]

### WP-A 완료 기준

99→100, 97→100, 감점/약탈, draw+shaping, 동시 사건, 정상 reset, 중복·지연 event에서 engine→broker→reward→evaluation 값이 일치한다. 결과 누락·중복 부여·reset 음의 delta는 모두 0건이어야 한다. 검증된 build/source 대응과 fixture 산출물을 보존한다. 공식 운영 승인과는 별도의 local engineering gate다.

<a id="sec-4"></a>

## 4. WP-B: 공통 정책 호출·planner·관측·mask — PR03

**구현 참고:** [S02](#s02) · [S04](#s04) · [S11](#s11) — 읽을 위치와 적용 한계는 §17 참조.

### 4.1 하나의 factory와 상태 계약

`make_policy(config, checkpoint, capabilities)`를 train/eval/export가 공통 사용한다. mode, controlled slots, sensor interval, dynamics, intervention, decoder, teammate revision을 모두 전달하고 **resolved effective config**를 저장한다. 인자를 받았지만 무시하는 설정은 reject한다. F1의 기존 intervention 전달 누락을 이 구조로 막는다.

state key는 환경 인스턴스를 포함한 `(env_id,episode_id,agent_id 또는 team_id)`다. 같은 decision 재호출은 동일 action을 반환하고 FSM/brain/RNG를 다시 진행시키지 않는다. 같은 key인데 관측 hash가 다르면 오류로 기록한다. 누락·역행 step은 공식 계약에 맞춰 reject하거나 명시적 tick API로 처리한다. 관측 배열이 같다는 이유로 다른 decision을 중복으로 취급하지 않는다.

agent ID→canonical slot 변환은 loader가 제공하는 메타데이터에 근거해야 한다. 공식 self-ID/순서 계약이 없는데 batch index로 유닛 정체성을 가정하지 않는다. stateful이 불허되면 cache·GRU·option을 숨겨 유지하지 않고 stateless 재계산 정책으로 재설계한다. 그 경우에도 identity 계약은 별도로 필요하다. [INT L269–288, L338–340, L433–435]

### 4.2 planner 실패 격리

팀 계획 생성과 슬롯별 경로/행동 실행을 분리한다. 한 슬롯의 `PathNotFound`는 해당 슬롯의 `status=unreachable`와 재계획/fallback으로 처리하고 다른 네 슬롯의 action·goal·cache를 보존한다. fallback은 합법적인 정지 또는 별도 검증된 recovery로 한정하고 그 이유를 기록한다. teacher label은 `abstain/invalid`이지 NoOp 정답으로 바꾸지 않는다.

shared assignment에 변경이 필요한 진짜 팀 사건과 한 슬롯의 path 실패를 구분한다. C2의 single-slot goal edit가 다른 슬롯의 action을 자동 재배정하면 안 된다. 이것을 의도할 경우 팀 전체 manager라는 새 action contract로 다시 등록한다.

회귀 테스트에서 고장 없는 입력의 기존 planner action과 새 action이 일치해야 한다. 실패 주입에서는 정상 네 슬롯을 보존해야 한다. 다만 v6 실제 teacher_failures=0이므로 이 수정을 v6 성능 회귀 해결책이라고 과장하지 않는다. [INT L220, L248–255]

### 4.3 관측과 mask

Y top-down↔world bottom-up, 진영 x/y 교환을 공통 변환 모듈에 둔다. 진영 교환은 임의 180도/90도 회전이 아니다. wall/창고 후보/현재 활성 창고를 다른 cache로 두고 episode reset에서 활성 영역을 갱신한다.

mask는 `hard_rule`, `geometry_estimate`, `heuristic_preference`, `duplicate_candidate`로 원인을 구분한다. 물리 FP/FN은 실제 collider·class·buff·radius 기준이 있을 때 측정한다. planner KEEP는 fallback label이지 안전 인증이 아니다. 검증 전 “모든 유효 행동”을 의미하는 hard mask로 근사 물리를 설명하지 않는다.

먼저 raw semantic ID/decoded ID/renderer 설정의 golden fixture와 zero/stale 검사를 추가한다. 실제 경계 오염이 재현되면 categorical ID 렌더링의 nearest/no-interpolation·색공간·샘플링 경로를 수정하고 별도 build로 검증한다. 양자화 후 임의 ID 치환만으로 해결했다고 하지 않는다. batchmode는 유지하고 렌더링 기반 관측을 끄지 않는다.

<a id="sec-5"></a>

## 5. WP-C: 재현 가능한 계측·checkpoint — PR00/04

**구현 참고:** [S16](#s16) · [S17](#s17) · [DOC04](#doc04) — 읽을 위치와 적용 한계는 §17 참조.

### 5.1 상시 저장과 event window를 분리한다

모든 step에 `(run,env,episode,decision,unity_tick,game_time,update,behavior_version,map,side,opponent_hash)`를 연결한다. episode가 완료되지 않았거나 재개에서 폐기되어도 상대·진영·step 노출을 잃지 않는다.

상시 counter/histogram은 q, p_KEEP, max correction, greedy margin, mask 수/원인, sampled/decoded/executed 개입률, displacement, noop/stuck, reward 성분, loss/entropy/KL/clip, 모듈별 gradient·parameter change를 기록한다. 학습 aggregate는 sampled 개입률만으로 종료하지 않는다.

상세 trace는 사건 전후 ±250 **환경 transition**을 기본 시작값으로 사용한다. actual tick/time을 함께 보관하여 nominal 약 5초씩이라는 가정을 검증한다. outcome, path failure, 높은 q이지만 KEEP, nonzero action 무이동, 큰 TD error, decoder action 차이가 trigger다. 비교용으로 무작위 정상 window도 같은 포맷으로 저장하고 inclusion rule을 기록한다. trigger-only 데이터의 빈도를 전체 빈도로 환산하지 않는다.

필수 상세 필드: original legal obs 또는 lossless 재구성 참조, frozen feature, planner proposal/state digest, 후보 stable ID·mask, logits/q/maxp, sampled label/old_logp, decoded action, executor action/reason, position before/after, score event, V/next_V/raw advantage/return, sensor·heading timestamp. 실제 full-state counterfactual을 지원하지 않으면 그 flag를 false로 둔다.

관측은 semantic class ID uint8와 vector로 compact 저장한다. 같은 팀 공통 map은 확인된 경우 중복 제거한다. 모든 graph neuron을 상시 저장하지 않고 별도 연구 진단에서 population 통계를 샘플한다. writer buffer·용량·dropped-window count를 기록하고 필수 transition 누락이면 데이터셋을 invalid로 처리한다.

### 5.2 policy degeneracy 감시

초기·정기 checkpoint에 chronological probe set과 별도 합성 stress를 적용한다. 입력 의존성, per-side greedy 행동 분포, q 대비 개입률, bias-only 일치, neural feature block 변화, margin을 기록한다. graph는 interval-bound 결과도 저장한다. 모든 correction upper bound가 KEEP 아래이면 해당 decoder를 `no_effective_residual`로 표시한다. 자동으로 bias를 올리거나 실패를 좋은 성능으로 처리하지 않는다.

새 checkpoint마다 **행동이 달라졌는지**와 **그 변화가 좋았는지**는 별도 판정이다. 단일 방향 비중이 높다는 사실만으로 무조건 실패시키지 않고 미리 고정한 대칭/랜드마크 probe와 실제 closed-loop 결과를 함께 확인한다.

### 5.3 checkpoint와 provenance

step0, 매 65,536 env transitions을 넘긴 첫 rollout 경계, stage boundary, 종료/안전 중단에 immutable checkpoint를 저장한다. 이 cadence는 초기 제안값이며 저장 부하를 측정해 시작 전 고정한다. `latest`는 포인터이지 불변 artifact가 아니다.

checkpoint는 model/optimizer/RNG/scaler·normalizer/scheduler/step/실행 소스·config·decoder·executor·runtime fingerprint와 parent hash를 포함한다. 정상 학습으로 바뀐 latest를 과거 hash와 비교하는 테스트를 고치고, backup 불변성·parent lineage·현재 파일 hash를 각각 검사한다. original v7 파일을 새 fingerprint로 다시 저장하지 않는다.

Unity physics를 복원하지 않는 resume는 `episode_restart_resume`이다. 폐기한 partial의 side/opponent/step과 rollout 처리 정책을 기록한다. exact resume라고 이름 붙이지 않는다. 정책 state·RNG 복구와 physics 복구의 범위를 명시한다. 재개/중단이 곧 새로운 독립 training run은 아니다.

<a id="sec-6"></a>

## 6. WP-D: v8-C1 단일 슬롯 개입 gate — 본선 첫 후보, PR05

**구현 참고:** [S03](#s03) · [S04](#s04) · [S05](#s05) · [S17](#s17) — 읽을 위치와 적용 한계는 §17 참조.

### 6.1 아키텍처

초기에는 현재 slot별 `128 frozen latent + 49 planner context = 177`을 유지한다. actor 입력 권한을 확장하지 않는다. 작은 shared per-slot MLP `177→64→64`에서 correction score 8개씩을 출력한다. 다섯 slot embedding의 canonical concatenation `320→64→1`에서 team gate logit을 출력한다. 모든 hidden activation과 initialization은 config에 기록한다. 마지막 logit은 선형으로 둔다. 이 폭은 비교 가능한 시작 설계이지 최적값이 아니다.

critic은 최초 대조에서 기존 299차원 predecision 입력과 구조를 유지한다. 현재 선택한 correction을 V 입력에 넣지 않는다. 이전 action·기존 planner/option context 추가, critic normalization, actor/critic optimizer 분리는 계측이 지지할 때 한 축씩 대조한다.

R_flat은 같은 trunk·scalar head·40 correction head를 사용하되 scalar를 KEEP logit으로 쓰는 flat41 대조다. 따라서 head 수·용량 차이를 줄일 수 있다. 기존 v7 flat 경로도 역사 기준선으로 보존한다. 초기 분포를 맞춘 조건과 gate semantics를 바꾼 조건을 구분한다.

### 6.2 간단한 gate + flat conditional40을 먼저 쓴다

초기에는 slot gate와 direction gate를 추가로 나누지 않는다. `V(o)`는 유효 correction의 집합이고:

`q = sigmoid(b)` (V가 비어 있으면 q=0)

`c_j = exp(l_j) / Σ_(k∈V) exp(l_k)` (j∈V)

`P(KEEP)=1−q`, `P(j)=q c_j`.

mask는 conditional 안에 적용하며, V가 비어 있지 않으면 q를 유지한다. 유효 후보가 줄어들 때 전체 개입 질량도 줄어드는 기존 flat41과 **다른 분포 설계**다. 유효 후보가 적으면 한 후보에 위험하게 확률이 몰릴 수 있으므로 후보 수별 결과를 검사한다.

초기 q=.10, gate bias=log(.1/.9)≈−2.197225, correction logits 동일값 0을 제안한다. flat41 참조는 KEEP=0, correction bias≈−5.886104로 초기화한다. 모두 유효하면 두 초기 분포는 동일하지만 mask가 있을 때는 달라짐을 명시한다. q 하한·개입 bonus·hard intervention budget은 최초에 추가하지 않는다.

### 6.3 PPO 확률·entropy 계약

`logP(KEEP)=log(1−q)`

`logP(j)=log q+log c_j`

`H(P)=H_B(q)+q H(c)`

`ratio=exp(logP_new(a|o,m)−stored_logP_old(a|o,m))`.

하나의 joint ratio를 한 번 clip한다. KEEP 표본에 사용되지 않은 conditional logprob를 더하지 않고 conditional entropy를 무조건 가산하지 않는다. all-invalid correction이면 KEEP prob=1/logp=0/entropy=0이며 빈 softmax를 호출하지 않는다. numerical stability는 logsigmoid/logsumexp로 처리한다.

최초 behavior는 masked stochastic policy에서 sample한다. ε mixture를 추가하는 후속 실험은 old/new logprob·entropy 모두 실제 mixture에 대해 계산한다. training에서 threshold로 action을 강제한 뒤 stochastic logprob를 붙이지 않는다.

Executor가 proposal을 다른 action으로 바꾸는 경우를 숨기지 않는다. 기본 경로는 사전 mask로 유효 proposal만 뽑는다. 불가피한 고정 shield는 **latent proposal을 학습 action으로 하고 고정 executor를 전이 과정으로 정의**하거나 executed-action pushforward 확률을 합산하는 방식을 명시해야 한다. 원 proposal logprob를 변경된 actuator action의 likelihood라고 이름 붙이지 않는다. runtime 예외는 정책 action으로 위장하지 않는다. [EXT §2.2–2.3; S04, S05]

### 6.4 decoder는 모델의 일부다

| decoder ID | 동작 | 용도 |
|---|---|---|
| `joint_argmax_v1` | KEEP 확률과 가장 큰 개별 수정 확률 비교 | 역사/flat 기준 |
| `gate_then_conditional_argmax_v1` | q>τ이면 valid conditional40의 최대, 아니면 KEEP | C1 deterministic 후보 |
| `sampled_v1` | 실제 P에서 sampling | behavior 및 별도 진단 평가. 제출 가능 여부 별도 |

τ=.5 하나를 기본 파일럿으로 등록한다. 나중에 threshold를 더 탐색하면 HP trial 수에 포함하고 dev에서만 선택한다. tie는 KEEP 우선, correction tie는 stable candidate ID 순서다. `q>.5`는 여러 correction의 총질량과 KEEP을 비교하는 **새 decoder**이며 PPO가 그 decoder의 승률을 직접 최적화한다는 보장은 없다.

기존 frozen graph checkpoint를 정확히 gate 형태로 인수분해한 adapter를 먼저 구현한다. joint_argmax를 유지했을 때 기존 logp/entropy/action과 같아야 한다. 그 adapter에서 threshold decoder만 바꾼 비교는 학습 0-step의 값싼 진단이다. 이 진단이 성공해도 새 gate 학습 성과와 구분한다.

### C1 채택/기각

채택은 유효한 행동 변화와 R_plan 대비 유효 endpoint 개선이 함께 관측될 때다. q가 낮아 fragmentation이 아니거나, actuator 실행이 무효하거나, 개입 증가가 검증 경기 성과를 떨어뜨리면 gate 증량은 기각한다. 차이가 없으면 원인이 해결된 것이 아니라 이 후보가 개선을 못 만든 것이다. A0/A1·v6 회귀를 고려해 entropy·LR·reward·curriculum을 동시에 올리지 않는다.

<a id="sec-7"></a>

## 7. WP-E: collector·PPO·노출 — PR06

**구현 참고:** [S05](#s05) · [S07](#s07) · [S08](#s08) · [S15](#s15) — 읽을 위치와 적용 한계는 §17 참조.

### 7.1 최초 학습 설정은 가능한 보존한다

C1 vs R_flat에서 γ=.9995, λ=.99, lr=1e−4, rollout=2048, epochs=4, minibatch=128, clip=.2, value_clip=.2, value_loss_coef=.5, entropy_coef=.003, max_grad_norm=.5, target_kl=.02, action_repeat=1을 동일하게 사용하는 안을 기본으로 제안한다. 이는 v7 등록 출발점이며 v8 최적값이라는 뜻이 아니다. source/config의 실값을 PR00에서 대조하고 차이는 명시한다.

진행 중 episode는 rollout 경계에서 유지한다. 같은 update의 모든 표본은 같은 behavior version에서 수집하며 evaluator RNG는 training RNG를 소비하지 않는다. final remainder가 minibatch보다 작은 경우를 검사하고 actual update 수를 저장한다. frozen encoder 사용 시 cached features로 학습할 수 있지만 encoder를 풀면 관측을 다시 계산해야 하므로 저장 계약을 변경한다.

return은 raw reward 단위로 계산한다. critic normalization을 도입하면 bootstrap 전에 denormalize하고 정상화 통계의 업데이트 시점·old value 단위를 고정한다. 모듈별 gradient와 critic-only/actor-only 진단을 먼저 보고 value clipping/optimizer를 바꾼다. 첫 output zero-init backward의 edge gradient=0은 예상 fixture로, 다음 update 도달성도 검사한다.

### 7.2 exposure는 먼저 기록하고 변경은 분리한다

C1 vs R_flat 최초 비교에서는 기존 고정 상대 schedule·uniform train maps를 공유한다. until 200,000: scripted/weak/target=.65/.25/.10; until 600,000: .20/.30/.50; 이후 .10/0/.90을 출발 설정으로 보존한다. 실제 상대 선택 코드의 meaning과 episode 경계 적용을 PR00에서 확인한다. 어떤 한 후보만 target-only 또는 failure replay로 바꾸지 않는다.

모든 step의 side·opponent를 기록한다. A 비중33.062% 같은 노출이 다시 나타나면 원인을 episode 길이와 sampling cycle로 분리한다. step-quota 실험은 C1을 동결한 다음 별도 `exposure_balance` 실험 ID로 둔다. 구현 대안은 두 side의 persistent env에서 같은 behavior version으로 정해진 transition quota를 모아 update하는 방식이다. 한 경기를 조기 종료해서 quota를 맞추지 않는다. 총 transitions/update·optimizer epochs를 유지하고 env 수 변경도 별도 요인으로 기록한다.

PLR·역사 상대 pool·역할 배분·특수 아이템은 순차 ablation이다. PLR을 시도할 때 과거 표본을 PPO에 재투입하지 않고 최근 raw GAE·score age·uniform floor로 다음 train cell을 고른 뒤 현재 정책으로 새 경험을 수집한다. 허용되지 않은 opponent identity를 actor feature에 넣지 않는다.

<a id="sec-8"></a>

## 8. WP-F: C2 목표/역할 option — 조건부 PR07

**구현 참고:** [S06](#s06) · [S07](#s07) · [S09](#s09) · [SUP01](#sup01) · [SUP02](#sup02) · [SUP03](#sup03) — 읽을 위치와 적용 한계는 §17 참조.

**시작 조건:** 사건 trace에서 유효 미세 개입이 다음 planner tick에 상쇄되며, 목표 선택/지속이 실패의 중요한 기전이라는 증거가 있다. 그 빈도·성과가 아직 보고서에 없으므로 처음부터 기본값으로 켜지 않는다.

초기 구현은 단일 slot의 `KEEP_PLANNER`, `현재 관측에서 생성된 합법 goal 후보 선택`에 한정한다. role edit는 goal edit만으로 충분하지 않을 때 별도 옵션으로 추가한다. 후보는 관측 위치·활성 창고·도달성에서 결정하고 stable goal ID를 저장한다. 숨은 stack 수량이나 privileged enemy class를 사용하지 않는다.

manager가 목표를 유지하는 동안 low-level planner는 매 환경 transition마다 관측을 받아 방향을 갱신한다. 고정 방향 repeat와 다르다. initiation mask, target reached/disappeared, path failure, 위험 abort, max duration을 명시한다. duration 값은 phase-1 trace로 정하되 최대 두 값만 사전 등록한다. 고정 종료/abort 규칙을 먼저 쓰고 종료 확률·low-level policy를 동시에 학습하지 않는다.

실제 k transition 지속 시:

`R_i=Σ_(j=0..k−1) γ^j r_(t+j)`, `Γ_i=γ^k`

`δ_i=R_i+Γ_i(1−terminated_i)V(h_(i+1))−V(h_i)`

`A_i=δ_i+Γ_i Λ_i continuation_i A_(i+1)`.

λ 시간척도를 유지하려면 Λ_i=λ^k를 선택할 수 있지만 primitive GAE와 수치적으로 동일한 estimator가 아니다. k=1에서는 기존 식으로 회귀해야 한다. 실제 Unity tick 수가 environment transition과 다르면 verified elapsed 시간 단위로 지수를 수정하고 protocol ID를 바꾼다.

manager option의 old_logp는 initiation 때 한 번만 저장한다. tick마다 복제해 같은 선택을 여러 PPO sample처럼 세지 않는다. option 종료/abort는 환경 terminal이 아니며 실제 k를 기록한다. buffer 경계에서 option을 임의 종료하지 않는다. 첫 구현은 시작된 option이 종료되는 경계까지 behavior weight를 고정하여 완전한 manager transition을 수집하고 actual primitive budget의 경계 초과량을 기록한다. strict budget 종료의 censored option은 action-conditioned continuation value를 설계하지 않았다면 actor update에서 제외하고 계수를 지어내지 않는다. 이 때문에 C2의 env-step/decision/update 예산 차이를 반드시 비교표에 남긴다.

대조군은 같은 manager head의 항상 종료(k=1), 원 planner, 별도 등록한 고정 방향 repeat다. duration만 늘려도 collision·abort가 증가하거나 동일 head k=1과 차이가 없으면 시간추상화의 기여를 주장하지 않는다. state/reset이 제출에서 불허되면 연구 전용으로 남긴다.

<a id="sec-9"></a>

## 9. WP-G: C3 정보·좌표 readout과 connectome 연구 — 조건부 PR08

**구현 참고:** [S10](#s10) · [S11](#s11) · [S12](#s12) · [S13](#s13) · [S14](#s14) — 읽을 위치와 적용 한계는 §17 참조.

### 9.1 연구 분기에서 먼저 고칠 것

F1은 동적 six rates가 존재해도 greedy 방향이 고정될 수 있다. 따라서 다음 순서로 비용을 늘린다.

1. 학습된 F1과 저장 rates를 사용해 bias-only/constant/zero/shuffle/feature-dependent logits를 오프라인 재검사한다.
2. 공통 factory에서 `frozen_frame`, `sensory_block`, `neural_feature_block`, `pool_shuffle`, `actuator_block`을 구분해 전달한다. neural feature=0이어도 readout bias로 움직일 수 있으므로 actuator_block만 확실한 정지 조작이다.
3. 실제 같은 초기 조건·같은 teammate에서 제한된 closed-loop 평가. 행동 분기 후 상태 분포가 달라지는 사실과 OOD 개입의 한계를 기록한다.
4. 같은 retina·slot0·4 teammates·action cadence의 작은 encoder/readout 대조와 비교한다. 개선 신호가 없으면 전뇌 장기학습을 늘리지 않는다.

### 9.2 좌표·합법 정보의 최소 대조

첫 구현 우선안은 `[six rates, sinθ, cosθ]→작은 2-layer MLP→9 world actions`다. 같은 MLP 용량의 rates-only를 대조로 둔다. θ는 기존 정책이 사용한 **가상 sensor heading**이며 실제 body orientation 관측으로 이름 바꾸지 않는다. 정지/충돌에서도 heading 갱신 규칙을 명시한다. vector 정보 추가는 별도 요인이고 heading·용량·추가 정보 효과를 한 번에 결론내리지 않는다.

상대 action decoder도 변환 fixture와 제한 비교 후보로 구현한다. 다만 retina frame과 13-tick pooled signal은 여러 시점의 heading을 담을 수 있다. 단일 θ를 붙여 평균 신호의 좌표를 정확히 복구했다고 주장하지 않는다. frame heading/time, 현재 heading, history window가 다른 상황을 검사한 뒤 어떤 기준의 상대 행동인지 명세한다. 내부 discrete eight-direction heading에 대해서는 유효한 bijection mapping을 먼저 시험하고, 일반 연속 heading 양자화의 alias/tie는 별도로 처리한다.

held type/class/score/time은 actor가 제공받는 필드만 추가한다. 배터리 정확 수량·숨은 상대 클래스·추론 때 없는 reward를 actor에 넣지 않는다. raw legal input은 구분하지만 현재 feature가 구분하지 못하는지 episode/map-group probe로 확인한다. 이전 v3 replay에는 group ID가 없어 그대로 독립 probe split으로 쓰지 않는다.

작은 GRU64·부분 encoder fine-tuning은 C3의 후속 ablation이며 최초 본선 모델이 아니다. GRU라면 contiguous sequence, padding mask, rollout hidden detach와 episode reset의 구분, burn-in/recompute, update 후 hidden-state drift를 구현한다. actor inference에 없는 reward를 training 때만 넣고 inference에서 0으로 대체하는 것은 금지한다.

### 9.3 wiring 주장과 성능 주장을 분리한다

부분회로 비교는 real/multiple-degree-null/작은 MLP부터 시작한다. 기존 rewiring seed1729 하나를 training seed5개로 반복한 것은 5개 graph null이 아니다. 3~5 graph seed는 예산 제안이며 충분성 정리가 아니다. real이 먼저 성능상 가치가 있을 때 port/weight/sign 통제를 추가한다. 학습된 checkpoint를 즉시 rewire하는 shock test와 null을 재학습한 inductive-bias 비교는 다른 질문이다.

본선 복귀는 동일 합법 정보·제어범위·tuning·env budget에서 작은 모델보다 개선되고 전체 비용·제출 계약을 만족할 때만 검토한다. 활동/움직임/뉴런 수는 생물학적 우위나 경기 우위를 대신하지 않는다. [INT L308–342; EXT §2.4–2.5]

<a id="sec-10"></a>

## 10. WP-H: 평가·모델 선택·본확인 — PR09

**구현 참고:** [S16](#s16) · [S19](#s19) · [S20](#s20) · [S21](#s21) — 읽을 위치와 적용 한계는 §17 참조.

### 10.1 split과 선택 이력

기존 train/dev는 진단용으로 사용한다. v6 confirmation은 rollback 선택에 반복 사용되었으므로 새 독립 confirmation으로 재사용하지 않는다. 독립성이 확인된 v8 confirmation/test manifest를 새로 잠그고 접근 이력을 남긴다. 이번 계획에 최종 test seed나 내용을 열거하지 않는다. 기존 test의 미사용을 증명할 수 없다면 새 held-out manifest를 별도 관리한다.

`model_lock`은 checkpoint뿐 아니라 policy factory, decoder/threshold, executor, renderer, reward, build, full opponent, runtime, evaluation RNG, split, primary metric, invalid handling, 분석 코드까지 고정한다. 중단한 최종 test를 성공한 episode만 모아 다시 완료하는 방식을 막고 attempt ledger와 사전 중단/재시도 규칙을 둔다.

### 10.2 검증 단계와 제안 예산

| 단계 | 실행 내용 | 제안 예산/판정 |
|---|---|---|
| S0 offline | 기존 자료 이관, bound 재실행, 분포/GAE/상태 fixture | 새 학습 0, gameplay 0 |
| S1 contract | engine snapshot·마지막 delta·timer·reset·renderer | 사건별 fixture, timeout 자동3/명시3. 승률 최적화 아님 |
| S2 behavior | R_plan/옛 final/decoder-only 변화의 chronological·paired 진단 | 기존 dev 6 maps×양 side를 시작 단위로 제안. 탐색용 표본, 최종 성능 인증 아님 |
| S3 mechanics pilot | R_flat와 C1, 필요시 C2/C3 중 증거가 있는 것만 | arm당 3 training runs, 262,144까지 기술 검사; 문제 없으면 등록된1,048,576 env-step endpoint까지. dev30×양side=60경기/run/등록 endpoint |
| S4 fixed-budget study | 파일럿에서 구조·HP·decoder 고정 후 선택된 후보와 학습 대조 | 초기 제안 2,000,000 env steps/run. run 수 R은 파일럿 분산과 목표폭·비용을 보고 confirmation 전에 고정. 3–5개라면 파일럿/제한 근거라고 표시 |
| S5 confirmation | 고정 절차의 새 runs·새 maps 또는 artifact-only 확인 범위를 명시 | 새 map100×양side를 예산 출발값으로 제안하되 요구 정밀도에 따라 사전 조정. 결과 보며 유리할 때 중단 금지 |
| S6 final test | dev에서 선택·잠근 하나의 배포 artifact와 고정 reference | 전체 잠긴 test 1회. 결과 본 뒤 수정하면 새 test/protocol 필요 |

262,144-step 지점은 200k 전환 이후의 두 번째 상대 구간에 들어갔지만 600k 이후 target 중심 구간에는 도달하지 않았으므로 **여기서 target 무개선을 학습 불가능 증거로 기각하지 않는다.** 해당 endpoint는 메커니즘·무결성 검사다. 1,048,576은 기존600k 이후 target 중심 구간을 포함하기 위한 제안값이며 최적 학습량이 아니다. 기존 stage schedule을 압축해 같은 실험이라고 부르지 않는다.

최초 R_flat+C1, 각3run×1,048,576이면 새 학습량6,291,456 env steps다. 후보3개+학습 대조1개까지 모두 실행하면12,582,912다. S4는2,000,000×R×학습 arm 수이며 S3와 별도 fresh-run이면 두 예산을 더한다. 평가 비용은 `Σ arms(runs × maps × 2 sides × opponents × evaluation_replicates)`로 산정하고 고정 planner는 가상의 training-seed 복제 없이 추가한다. 이는 향후 작업 소요시간 예고가 아닌 실험 예산 산술이다.

파일럿 상태를 중간에 폐기/rollback하여 성능을 회복한 run은 고정 예산 순수학습과 구분한다. 기본 comparative training에서는 자동 성능 rollback을 끄고, 안전/무결성 오류는 실패 run으로 남긴다. 배포 best pointer는 학습 상태를 되돌리는 기능과 분리한다.

### 10.3 통계 구현

outcome cube는 `Y[algorithm, training_run, map, side, opponent, eval_rep]`다. run ID는 seed 문자열과 다르다. 같은 map의 양 side를 묶고 알고리즘 전체에서 동일 map 재표집 index를 공유한다. 알고리즘별 training-run 행은 기본적으로 독립 재표집하며 같은 seed 번호라고 자동 pairing하지 않는다. 실제 공통 난수 paired-block 설계가 검증된 경우만 별도 분석한다. 이 선택은 INT의 참고 paired 계산과 구분되는 EXT 기반 **새 분석 규약**이다.

고정 reference는 training 변동을 만들어내지 않는다. 하나의 policy가 평가한 모든 map에 같은 row index를 사용하고 map별로 다른 run을 새로 뽑지 않는다. 동일 결과의 퇴화 CI에 인위적 jitter를 넣지 않는다. crossed CI, fixed-map seed CI, fixed-policy map CI, run별 spread와 최저 진영 성과를 함께 보고한다.

기본 resample 20,000회는 Monte Carlo 제안값이지 독립 실험 수가 아니다. 3~5runs로 catastrophic tail을 인증하지 않는다. 후보 최대 3개를 동시에 confirm하면 comparison family·보정을 잠그고, 가능하면 dev에서 하나의 finalist를 먼저 고른다. seed를 버리거나 best seed만으로 algorithm 평균을 만들지 않는다. [EXT §2.7; S16, S19–S21]

### 10.4 승격·중단

무결성 gate는 outcome/정책 상태/로깅 오류가 0건이라는 fixture 통과를 요구한다. 학습 NaN·비정상 action·환경 crash·누락된 최종 결과는 조용히 제외하지 않는다. 정책 오류와 인프라 오류를 구분하고 사전 등록한 재시도·몰수 규칙을 적용한다. 정책 오류로 censor된 셀을 빼서 유효 경기 승률만 높이는 것을 금지한다.

연구 승격은 주 대비 Δ와 불확실성, 유효 이동/행동 기전, 진영별 회귀, 전체 자원 비용을 함께 본다. 진영 비열등 한도 −5%p는 출발 제안값이며 사용한다면 확인 전에 고정한다. 충분한 표본에서 side 차이 구간의 하한이 이 한도를 못 넘으면 안전한 진영 성능 유지 주장을 보류한다. 전체 승률 상승만으로 큰 한 진영 퇴행을 숨기지 않는다.

최종 artifact 선택은 dev에서 사전 정의된 규칙(평균 주지표, 동률이면 낮은 지연, 재동률이면 고정 hash순)을 적용하고 confirmation/test 전에 잠근다. 이 선택 artifact의 test 성과와 여러 독립 training runs의 알고리즘 평균은 다른 결론으로 보고한다.

<a id="sec-11"></a>

## 11. WP-I: 성능·가속·export — PR10

**구현 참고:** [S02](#s02) · [S18](#s18) · [DOC01](#doc01) · [DOC02](#doc02) · [DOC03](#doc03) · [DOC04](#doc04) · [DOC05](#doc05) — 읽을 위치와 적용 한계는 §17 참조.

현재 검증된 native protobuf/queue/thread 제한/선택적 CSR backend를 자동 업그레이드하지 않는다. 일반 source fingerprint와 82-file runtime overlay를 모두 확인한다. 비-connectome 본선은 CPU를 우선 reference로 하고, 특정 연산만 장치에 배치하는 선택은 실측으로 판단한다. 외부 문서의 다른 버전에서 MPS 지원 여부를 본 기록으로 현 runtime을 추정하지 않는다.

계측은 encoder/planner/mask/통신/copy/CSR/PPO/save/reset/eval의 exclusive span과 전체 request를 구분한다. 개별 warm request p50/p95/p99, 실패·deadline 초과, cold load, 최대 메모리, 전체 학습·평가 wall time을 기록한다. nested cumulative time이나 모듈별 p99를 합쳐 전체 p99로 만들지 않는다. accelerator isolated benchmark에는 완료 동기화가 필요하고 전체 policy E2E의 실제 반환 경계도 측정한다.

기존 512/256-step parity는 종료 없는 구간이므로 새 regression은 최소3경기의 terminal/reset, 중복 호출, 명시 resume boundary를 포함한다. 같은 source/backend의 exact test와 cross-backend 수치 tolerance test를 구분한다. logits 오차가 작아도 near-tie action이 바뀌면 별도 사건으로 보고 speed-only 동등성을 선언하지 않는다.

export에는 tensor 외에도 planner/FSM·canonical routing·decoder·feature 전처리·normalizer·state lifecycle을 포함한다. 공식 loader가 허용하는 format으로만 작성하고 graph/custom op/외부 파일/NOTICE를 검증한다. `research_export`와 `submission_certified_export`를 분리하고 후자는 운영 계약 미확정이면 실패시킨다. 임의의 milliseconds 또는 RAM 상한을 공식 기준처럼 만들지 않는다. [INT L407–437; EXT §2.8]

<a id="sec-12"></a>

## 12. PR 순서·의존성·책임 역할

| PR | 변경 단위 | 선행 | 책임 역할 | 종료 산출물 |
|---|---|---|---|---|
| PR00 | 원본 이관·source/runtime manifest·불변 checkpoint/lineage | 없음 | 내부 분석/재현 담당 | input manifest, missing-artifact 목록, 기존 회귀 실패의 정정 테스트 |
| PR01 | engine OutcomeEvent·broker·reward/evaluation 공통 계약 | PR00 | Unity/API 담당 + 평가 담당 | 사건 fixture·마지막delta/winner 반례를 통과하는 결과 |
| PR02 | TimerManager generation·관측 renderer 검증 | PR00 | Unity 담당 | 자동/명시3연속 timeout, raw-ID golden fixtures |
| PR03 | policy factory·state/ID·planner slot 실패 격리·mask reason | PR00 | 정책/환경 담당 | train/eval effective config 동등, 정상4slot 보존 |
| PR04 | every-step provenance·ring buffer·module stats·degeneracy 검사 | PR00/03 | 학습 기반/분석 담당 | 실제 transition 재연 가능한 trace |
| PR05 | C1 gate/distributions/decoders, same-capacity flat 대조 | PR03/04 | 정책·PPO 담당 | 확률·entropy·ratio·decoder 반례 테스트, offline adapter |
| PR06 | collector/PPO/return·exposure accounting | PR01/04/05 | 학습 담당 | live end-boundary ratio/value/reward 검사, exposure 합계 일치 |
| PR07 | C2 goal option + SMDP (조건부) | PR06 및 상쇄/지연 진단 | 계층 정책 담당 | k=1·실제k·abort·same-head 대조 |
| PR08 | C3 coordinate/context + trained-F1 interventions (조건부) | PR03/04 및 field audit | 표현/신경회로 담당 | rates-only/heading/capacity·actuator구분 fixture |
| PR09 | split·experiment ledger·통계·locks·승격 보고서 | PR00/01/04 | 평가/통계 담당 | cube·crossed resampler·선택/attempt 기록 |
| PR10 | end-to-end 성능·backend parity·격리 export | PR01/02/03/09 | 실행/배포 담당 | 실제3경기 parity, 자원 측정, capability 인증 |

PR00 뒤 PR01/02/03과 통계 명세는 병렬 개발할 수 있다. PR05도 합성 검사는 먼저 개발 가능하지만, 성능 선택은 PR01/02의 live gate를 통과한 뒤에 한다. PR07/08은 조건을 만족하지 않으면 구현되지 않은 optional 상태로 남겨도 v8-C1 본선 완료를 막지 않는다. 공식 승인 문의는 별도 외부 담당이 병행하되 답변을 만들거나 발송 완료로 기록하지 않는다.

<a id="sec-13"></a>

## 13. 필수 회귀·실패 테스트 목록

| 범주 | 실패시켜야 할 잘못된 구현 |
|---|---|
| 결과 | terminal reset score를 final로 저장; reward 부호 winner; 지연 event를 다음 episode에 결합 |
| 보상 | 마지막 +3 누락/중복, reset 음수delta, draw+shaping 승리, score 감소 양의 dense reward |
| 타이머 | callback 재시작으로 생성된 새 timer 삭제; old generation callback의 새 경기 침범 |
| 관측/ID | canonical 순서 변경에 다른 slot출력, zero/stale map 묵인, top-down/진영 변환 오류 |
| planner | 한 slot PathNotFound가 다른 네 slot state/action을 초기화 |
| 분포 | all-invalid NaN; KEEP에 branch logprob 추가; 잘못된 entropy 가중; 별도head ratio clipping |
| decoder | exact factorization이 joint_argmax를 바꿈; threshold decoder를 기존 greedy로 이름 붙임 |
| execution | overwritten action에 원 proposal likelihood로 라벨 변경; 중복 호출에 RNG/tick 진행 |
| PPO/GAE | reset 관측 bootstrap; truncation zero bootstrap; rollout 경계에서 environment reset |
| graph/F1 | 첫 backward0을 영구 사망으로 판정; neural feature block=actuator stop라고 단정 |
| options | initiation logp를 매tick 복제, early abort에서도 계획k 사용, option end를 game terminal로 처리 |
| provenance | legitimate resumed latest를 immutable backup과 비교; 원 fingerprint 재기록 |
| 통계 | map마다 새 training row 선택, seed번호 자동 pairing, reference 복제, missing을draw로 대체 |
| parity/export | 종료 없는 short trace만으로 연전 통과; 외부 package/file/state 미허용을 묵인 |

실제 target backend가 필요한 필수 검사가 skip이면 그 backend의 승격은 통과가 아니다. 비-connectome CPU 본선에 불필요한 optional Metal 연구 검사는 별도 suite로 분리할 수 있다.

<a id="sec-14"></a>

## 14. F01–F36 처리 대장

아래는 모든 항목을 즉시 새 기능으로 바꾸라는 목록이 아니다. 보고서상 수정된 항목은 유지·회귀 대상, 근거 부족 항목은 계측 후 조건부 변경이다.

| F | v8 처리 | 담당 WP/PR |
|---|---|---|
| F01 | 최종 점수 snapshot과 공통 evaluator | A / PR01 |
| F02 | engine winner와 공식 scorer의 정의·계약 분리 | A·I / PR01·10 |
| F03 | 마지막 점수 변화를 정확히 1회 반영 | A / PR01 |
| F04 | ID·generation 기반 timer와 실제 연속 timeout 검사 | A / PR02 |
| F05 | wall·후보 창고·활성 창고별 cache | B / PR03 |
| F06 | raw-ID/renderer golden fixture; 실제 재현 후 수정 | B / PR02·03 |
| F07 | 제공/허용 정보·ID 구분; 숨은 배터리 수량 추가 금지 | B·G / PR03·08 |
| F08 | mask 원인·collider 대조; 근사를 안전 인증으로 사용 금지 | B / PR03·04 |
| F09 | 정확한 인수분해 adapter와 gate/decoder 대조 | D / PR05 |
| F10 | 개입 상쇄 계측 후 goal option 검토 | C·F / PR04·07 |
| F11 | frozen feature probe 후 필요 시 encoder 일부 학습 | C·G / PR04·08 |
| F12 | slot-local 실패 처리와 teacher abstain | B / PR03 |
| F13 | 사건 지연·critic 진단; k=1 return 우선 검증 | C·E·F / PR04·06·07 |
| F14 | 구조 비교와 curriculum 변경의 실험 ID 분리 | E·H / PR06·09 |
| F15 | 실효 residual 부재 알림과 연속 행동 검사 | C·D / PR04·05 |
| F16 | topology prior로 명명; 본선 graph 의존성 제거 | G / PR08 |
| F17 | graph seed와 training seed 구분; 복수 null은 조건부 | G·H / PR08·09 |
| F18 | 가상 sensor heading·좌표·시점 fixture | G / PR08 |
| F19 | 제공된 vector의 최소 추가 대조 | G / PR08 |
| F20 | 작은 readout/encoder 대조; readout 63개와 전체 학습 파라미터 구분 | G / PR08 |
| F21 | tick·sensor·pool window 계측; window를 고정 dead time으로 해석 금지 | C·G / PR04·08 |
| F22 | 학습된 F1 인과성 검사를 F0 검사와 분리 | G / PR08 |
| F23 | 동일 retina·slot·teammate의 작은 모델 대조 | G / PR08 |
| F24 | proxy 포트·부호·동역학 명세와 조건부 민감도 연구 | G / PR08 |
| F25 | 새 holdout, shared-map × 독립 training-run 비교 | H / PR09 |
| F26 | 현재 가속 보존 및 82-file runtime 검증 | C·I / PR00·10 |
| F27 | 전체 policy request의 tail 지연과 총 실행 시간 측정 | I / PR10 |
| F28 | 연구 export와 제출 인증 export 분리 | I / PR10 |
| F29 | 명시적 ID·reset·retry·capability 계약 | B·I / PR03·10 |
| F30 | 이번 첨부에 없는 실물·분석 JSON 인계와 상대경로 manifest | C / PR00 |
| F31 | episode-restart resume 및 partial 노출 기록 | C·E / PR00·06 |
| F32 | 합성과 live suite 분리; 필수 검사 skip은 통과 아님 | 전 PR |
| F33 | 문서 상태와 실제 source/runtime 상태의 일치 관리 | C·H / PR00·09 |
| F34 | factory에서 intervention·sensor·dynamics 명시 전달 | B·G / PR03·08 |
| F35 | Unity shaping handler의 delta 처리와 task reward 분리 | A / PR01 |
| F36 | 고정 target 상대의 item·role 개별 ablation | E / PR06 후속 |

<a id="sec-15"></a>

## 15. 지금 착수할 작업과 착수하지 않을 작업

첫 작업은 PR00–04다. 특히 ① internal_analysis 근거 묶음과 원본 checkpoint의 필요한 부분을 이관하고 ② engine outcome/마지막 reward/timer를 복구하며 ③ 하나의 policy factory와 행동 trace를 만든다. 이어 ④ 기존 final graph의 exact-factorization adapter에서 decoder-only 비교를 수행하고 ⑤ 작은 C1/R_flat을 같은 조건으로 파일럿 학습한다.

지금 하지 않는 것은 전체 전뇌 재학습, graph 확대, 무조건 entropy/개입 확률 증가, 임의 장시간 repeat, PPO→COMA 전체 교체, 여러 reward·curriculum·역할 변경 동시 투입, v3 DAgger 단순 재도입, 기존 test 열람, 공식 계약 없는 export 해제다. 이후 유용한 증거가 생기면 각각 독립된 실험으로 재검토한다.

v8의 완료는 “PPO update가 돈다” 또는 “수정 행동이 많다”가 아니라, **정확한 결과를 가진 경기에서 유용한 행동 변화가 재현되고, 동일 기준선보다 낫고, 그 정책 전체가 실제 실행 환경에서 동작한다**는 연결로 판정한다.

<a id="sec-16"></a>

## 16. 구현에 참고할 프로젝트 문서·코드·원자료

### 16.1 먼저 읽을 인계 문서

| 자료 | 읽을 부분 | 구현에서 사용하는 이유 | 증거 경계 |
|---|---|---|---|
| `internal_result.md` — **INT** | I00–I09, C01–C10, F01–F36, `v8_decision_inputs` | 실제 실행물·평가 오염·A2/A3 bound·F1 readout·회귀·노출 진단의 출발점 | 이번 계획 작성자는 연결된 원시 JSON·checkpoint를 재계산하지 않음 |
| `v8_external_research.md` — **EXT** | §1 결정표, §2.1–2.8 근거 카드, §3 코드 provenance, §4 실험, §5 계약 | 수식·실험 가정·적용/기각 조건 및 논문 독서 위치 | 문헌 조사이며 BlackOut 새 실행의 성능 증거 아님 |
| `v8_external_research.json` — **EXT-J** | `evidence_sections`, `decisions`, `official_contract` | 정확한 저자·URL·판·commit·검증 수준과 미확인 목록 확인 | EXT와 같은 조사의 구조화 표현; 독립적인 두 근거가 아님 |
| 앞서 작성한 `v8_implementation_plan.md` 및 `implementation_backlog.json` | WP-A–I, PR00–10, 24개 task와 coverage | 이번 문서의 기존 설계 기준 및 작업 분해 | 구현 완료 결과가 아닌 계획 초안 |
| 기존 감사 `blackout_repository_audit_2026-09-27.md` | F01–F36 및 범위 | 보고서 이전의 문제 가설과 상태 변화 비교 | 오래된 가설을 INT의 새 검증보다 우선하지 않음 |

### 16.2 저장소 안에서 함께 읽을 문서

아래 링크는 **기존 감사 기준 snapshot** `e6e98a63e9f401f6bf03e463a720c03878d1beb7`에 대한 탐색 경로다. 현재 HEAD를 새로 조사했다는 뜻이 아니다. INT가 분석한 로컬 HEAD는 `82ce2a1014c28b02408ef733c578a7ef367053cc`이고 요청 snapshot object가 없어 정확한 diff를 확인하지 못했다고 보고했다. 실제 구현 시작점은 PR00에서 정한다.

| 문서·경로 | 확인할 내용 | 관련 PR |
|---|---|---|
| [history.md](https://github.com/thislis/2026-IST-tech-RL/blob/e6e98a63e9f401f6bf03e463a720c03878d1beb7/history.md) | v1–v7의 실패 원인, 이미 고쳐진 문제와 미해결 문제 | PR00·05·06 |
| [game_spec.md](https://github.com/thislis/2026-IST-tech-RL/blob/e6e98a63e9f401f6bf03e463a720c03878d1beb7/game_spec.md) | 게임 종료·득점·이동·관측·활성 창고 | PR01·02·03 |
| [versions.md](https://github.com/thislis/2026-IST-tech-RL/blob/e6e98a63e9f401f6bf03e463a720c03878d1beb7/versions.md) | Unity/API/build·Python 의존성의 역사적 고정값 | PR00·10 |
| [issues/1st_issues_v0-v7.md](https://github.com/thislis/2026-IST-tech-RL/blob/e6e98a63e9f401f6bf03e463a720c03878d1beb7/issues/1st_issues_v0-v7.md) | score/winner/timer/관측/제출 이슈의 기존 재현과 범위 | PR00–04 |
| [reports/prep13_evaluation_contract_checklist.md](https://github.com/thislis/2026-IST-tech-RL/blob/e6e98a63e9f401f6bf03e463a720c03878d1beb7/reports/prep13_evaluation_contract_checklist.md) | stateless·입력·loader·공식 미답변 항목 | PR03·10 |
| [reports/v7/main_study_preregistration.md](https://github.com/thislis/2026-IST-tech-RL/blob/e6e98a63e9f401f6bf03e463a720c03878d1beb7/reports/v7/main_study_preregistration.md) | v7 고정 예산·상대 schedule·latest 선택·제외 범위 | PR00·06·09 |
| [reports/v7/acceleration_and_resume.md](https://github.com/thislis/2026-IST-tech-RL/blob/e6e98a63e9f401f6bf03e463a720c03878d1beb7/reports/v7/acceleration_and_resume.md) | protobuf/queue/CSR 가속, parity 범위, resume 한계 | PR00·10 |

### 16.3 소스 읽기 순서

| 실행 구간 | 기존 코드의 탐색 시작점 | 주의점 |
|---|---|---|
| 결과·보상 | `scripts/evaluate_v7.py`, `eval/evaluator.py`, `blackout_rl/reward.py`, `training_reward.py` | 기존 평가기의 방어가 새 평가기로 전달되지 않은 이유부터 비교 |
| Unity/API | `TimerManager.cs`, `BlackOutAgent.cs`, API `_collect_obs()`, `competition/match.py`, `model/loader.py` | 소스 저장소의 위치·commit은 INT I00 참조. 프로젝트 본체에 모두 들어 있다고 가정하지 않음 |
| residual | `mappo_v6.py`, `mappo_v6_training.py`, `v7_1/model.py` | planner state, candidate 의미, mask, KEEP=0, sampled→greedy를 함께 확인 |
| 학습·등록 | `v7_training.py`, `v7_registry.py`, `rollout.py`, `ppo.py` | 현재 PPO 계약 중 정상 동작하는 부분을 불필요하게 교체하지 않음 |
| 전뇌 | `v7_2/policy.py`, `sensory.py`, `dynamics.py`, `decoder.py`, `teammates.py` | 가상 heading, six-pool history, F0/F1, neural block/actuator block 분리 |
| 회귀 | `tests/v7/test_contracts.py`, `test_teammates.py`, `test_accelerated_runtime.py` | mock/합성/저장 결과 읽기/live 실행을 구분 |

### 16.4 PR00에서 받아야 할 비-Git 산출물

INT의 `internal_analysis/artifact_inventory.json`, `behavior_audit.json`, `graph_comparison.json`, `event_timeline.json`, `evaluation_cube.json`, `exposure_summary.json`, `direct_diagnostics.json`, 최소 재현 script·결과, input/output hash 목록을 우선 인계받는다. 이름은 **INT가 보고한 산출물**이며 이번 문서에 내장된 파일은 아니다.

그 다음 필요한 final/초기 재구성 기준 checkpoint, immutable 평가 사본, build와 source→binary manifest, graph/port metadata, frozen opponent, resolved config, 원본 로그를 복원한다. 모든 경로를 상대경로 색인으로 연결하고 누락 자료는 `blocked`로 둔다. 공개 저장소 검색이나 문헌 자료로 사라진 실제 trajectory를 대체하지 않는다.


<a id="sec-17"></a>

## 17. 구현 참고 논문·공식 문서 목록

### 17.1 선택 기준과 읽기 우선순위

아래 **29개 항목**은 기존 외부 보고서의 출발 자료 S01–S18과 계획에서 실제 사용하는 추가 근거를 선별한 목록이다. 그중 21개는 논문·연구보고서, 8개는 공식 문서다. S12의 정정문과 각 항목의 코드·보충 링크는 관련 자료로 연결했으며 독립적인 성능 연구 수로 중복 계산하지 않는다.

**당장 필요한 자료:** S01·S02 → S04·S05·S07 → S16·S17·S19·S21 순으로 읽는다. 이 순서는 종료 계약, 행동 확률, 추정기, 성능 평가에 대응한다. S03·S08은 본선 설계와 구현 비교를 보완한다. S18·DOC01–05는 실행 담당자가 병행한다.

**조건이 생겼을 때 읽을 자료:** C2를 시작하면 S06·S09·SUP01–03, 표현/기억 문제를 검증하면 S10·S11, connectome 연구를 진행하면 S12–S14, curriculum을 바꾸면 S15를 사용한다. 읽었다는 이유만으로 해당 알고리즘을 본선에 추가하지 않는다.

서지와 주요 읽을 위치는 EXT의 상세 조사에서 인계했다. 이번 웹 재확인은 주로 공식 페이지·서지·초록 및 일부 API 본문이다. 해당 논문의 모든 실험·보충 자료를 새로 독해하거나 저장소 코드를 실행한 것은 아니다. 코드 pin/라이선스의 근거는 §18과 EXT-J를 사용한다. 논문 초판, 수정판, 현재 구현 default와 논문 실험 script는 서로 다를 수 있다.

### 17.2 빠른 색인

| ID | 자료 | 사용 범위 |
|---|---|---|
| [S01](#s01) | Handling Time Limits | PR01·06 |
| [S02](#s02) | Unity ML-Agents Python Low Level API | PR01·03·10 |
| [S03](#s03) | Residual Reinforcement Learning for Robot Control | PR05 |
| [S04](#s04) | A Closer Look at Invalid Action Masking in Policy Gradient Algorithms | PR03·05·06 |
| [S05](#s05) | Proximal Policy Optimization Algorithms | PR05·06 |
| [S06](#s06) | The Option-Critic Architecture | 조건부 PR07 |
| [S07](#s07) | High-Dimensional Continuous Control Using Generalized Advantage Estimation | PR06·07 |
| [S08](#s08) | The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games | PR06 |
| [S09](#s09) | Counterfactual Multi-Agent Policy Gradients | 조건부 PR06·07 |
| [S10](#s10) | Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs | 조건부 PR08 |
| [S11](#s11) | A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning | PR03·04, 조건부 PR08 |
| [S12](#s12) | Transforming a head direction signal into a goal-oriented steering command | 조건부 PR08 |
| [S13](#s13) | Connectome-constrained networks predict neural activity across the fly visual system | 조건부 PR08 |
| [S14](#s14) | A Drosophila computational brain model reveals sensorimotor processing | 조건부 PR08 |
| [S15](#s15) | Prioritized Level Replay | PR06 후속 |
| [S16](#s16) | Deep Reinforcement Learning at the Edge of the Statistical Precipice | PR00·04·09 |
| [S17](#s17) | Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO | PR05·06·09 |
| [S18](#s18) | Performance Tuning Guide | PR10 |
| [S19](#s19) | The pigeonhole bootstrap | PR09 |
| [S20](#s20) | Bootstrapping data arrays of arbitrary order | PR09의 심화 참고 |
| [S21](#s21) | Empirical Design in Reinforcement Learning | PR09 |
| [SUP01](#sup01) | Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping | 조건부 PR01·06·07 |
| [SUP02](#sup02) | Reward Shaping in Episodic Reinforcement Learning | 조건부 PR01·06·07 |
| [SUP03](#sup03) | On Centralized Critics in Multi-Agent Reinforcement Learning | 조건부 PR06·07 |
| [DOC01](#doc01) | Benchmark Utils — torch.utils.benchmark | PR10 |
| [DOC02](#doc02) | torch.profiler | PR04·10 |
| [DOC03](#doc03) | torch.mps.synchronize | PR10 |
| [DOC04](#doc04) | Reproducibility | PR00·04·09·10 |
| [DOC05](#doc05) | Changes announced May 6, 2022 — Python Updates | PR00·10 |

### 17.3 자료별 읽을 위치·적용·제한

<a id="s01"></a>

#### S01. Handling Time Limits

**서지:** Farama Foundation. Gymnasium 공식 문서; 첨부 조사 및 이번 페이지는 1.3.0 호환 표기를 확인.

**원문:** [Handling Time Limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)

**읽을 위치:** Termination, Truncation, Importance in learning code, Solution.

**구현 연결:** PR01·06: terminated/truncated, reset 이전 final observation, bootstrap fixture.

**적용 한계:** 일반 MDP/API 의미 설명이다. BlackOut의 게임 시간초과·외부 watchdog·공식 판정 매핑은 직접 검증한다.

**확인 범위:** 이번에 공식 본문 재확인. 상세 근거: EXT §2.1/E01.

<a id="s02"></a>

#### S02. Unity ML-Agents Python Low Level API

**서지:** Unity Technologies. 공식 4.0 계열 문서.

**원문:** [Unity ML-Agents Python Low Level API](https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/Python-LLAPI.html)

**읽을 위치:** BaseEnv, DecisionSteps/TerminalSteps, agent_id, Communicating additional information, Side Channels.

**구현 연결:** PR01·03·10: outcome 전달, 명시적 ID 라우팅, decision/physical tick 구분, reset·side-channel join.

**적용 한계:** 이번 반환 페이지는 4.0.2로 표시되었고 EXT는 4.0.3 표시를 기록했다. 페이지 표시는 실제 설치 patch의 증거가 아니므로 둘을 덮어쓰지 않는다. custom channel·state 허용도 공식 대회 계약과 별도다.

**확인 범위:** 이번에 공식 API 본문 재확인. 상세 근거: EXT §2.1/E01.

<a id="s03"></a>

#### S03. Residual Reinforcement Learning for Robot Control

**서지:** Tobias Johannink et al.. ICRA 2019; arXiv:1812.03201v2 (2018-12-18); DOI 10.1109/ICRA.2019.8794127.

**원문:** [Residual Reinforcement Learning for Robot Control](https://arxiv.org/abs/1812.03201v2) · [저자 프로젝트](https://residualrl.github.io/)

**읽을 위치:** §III-A 식 (4)(5), Algorithm 1, §III-B, §IV–VI의 controller-only/RL-only/residual 대조.

**구현 연결:** PR05: planner 보존형 정책의 설계 배경; 방향 치환 residual과 연속 actuator 덧셈의 차이 명세.

**적용 한계:** 연속 로봇 제어·TD3 결과다. 정규화된 방향, gate threshold, BlackOut의 개선·안전성을 보장하지 않는다. 논문 exact 실험 코드 commit은 미확인.

**확인 범위:** EXT의 상세 독해 기록을 인계; 이번에는 공식 서지·연결을 재확인. 상세 근거: EXT §2.2/E02.

<a id="s04"></a>

#### S04. A Closer Look at Invalid Action Masking in Policy Gradient Algorithms

**서지:** Shengyi Huang, Santiago Ontañón. FLAIRS 35, 2022; arXiv:2006.14171v3; DOI 10.32473/flairs.v35i.130584.

**원문:** [A Closer Look at Invalid Action Masking in Policy Gradient Algorithms](https://doi.org/10.32473/flairs.v35i.130584) · [arXiv v3](https://arxiv.org/abs/2006.14171v3) · [조사한 구현 pin](https://github.com/vwxyzjn/invalid-action-masking/tree/6daedd29e4b48271c274c4593dcbc9a76be30ce4)

**읽을 위치:** Invalid Action Masking, Proposition 1, Strategies to Handle Invalid Actions, Appendix Multi Discrete Action Generation; 코드 CategoricalMasked·rollout mask·ratio.

**구현 연결:** PR03·05·06: sampling과 log-probability에서 동일 mask 사용, all-invalid 처리, entropy·ratio regression.

**적용 한계:** 독립 categorical의 entropy 합을 conditional hierarchy에 그대로 쓰지 않는다. 물리적 invalidity와 전략 선호 mask를 구분한다. gate q 보존·threshold 설계는 이 계획의 선택이지 논문 보장 아님.

**확인 범위:** EXT의 상세 독해 기록을 인계; 이번에는 공식 서지·연결을 재확인. 상세 근거: EXT §2.2/E02.

<a id="s05"></a>

#### S05. Proximal Policy Optimization Algorithms

**서지:** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov. 2017 기술보고서; arXiv:1707.06347v2 (2017-08-28).

**원문:** [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347v2) · [공식 계열 PPO2 reference pin](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py)

**읽을 위치:** §3 식 (6)(7), 식 (9), Algorithm 1; old-policy rollout과 minibatch epochs.

**구현 연결:** PR05·06: joint ratio를 한 번 clip, 실제 behavior log-probability, old value/return 단위 계약.

**적용 한계:** stochastic PPO를 학습하고 별도 greedy/threshold decoder를 쓰는 정책의 성능은 별도 측정한다. clipped objective를 실제 return의 단조 개선 보장으로 해석하지 않는다.

**확인 범위:** EXT의 상세 독해 기록을 인계; 이번에는 공식 서지·연결을 재확인. 상세 근거: EXT §2.2/E02.

<a id="s06"></a>

#### S06. The Option-Critic Architecture

**서지:** Pierre-Luc Bacon, Jean Harb, Doina Precup. AAAI 2017; arXiv:1609.05140v2; DOI 10.1609/aaai.v31i1.10916.

**원문:** [The Option-Critic Architecture](https://ojs.aaai.org/index.php/AAAI/article/view/10916) · [arXiv v2](https://arxiv.org/abs/1609.05140v2)

**읽을 위치:** Preliminaries, Learning Options 식 (1)–(4), Algorithm 1, termination과 intra-option policy, Discussion.

**구현 연결:** 조건부 PR07: goal option 수명·환경 종료의 구분, always-terminate 동용량 대조.

**적용 한계:** 처음 v8-C2는 사전 정의 planner option과 고정 abort를 쓴다. 자동 option 발견이나 원논문의 전체 학습 방법을 재현한다고 부르지 않는다.

**확인 범위:** EXT의 상세 독해 기록을 인계; 이번에는 공식 서지·연결을 재확인. 상세 근거: EXT §2.3/E03.

<a id="s07"></a>

#### S07. High-Dimensional Continuous Control Using Generalized Advantage Estimation

**서지:** John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, Pieter Abbeel. arXiv:1506.02438; 2015 초고, v6 (2018-10-20).

**원문:** [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438v6)

**읽을 위치:** §2 Definition 1/Proposition 1, §3 식 (16), §4, §6.3.

**구현 연결:** PR06·07: GAE·value bias/variance, k=1 회귀와 시간 단위 변환 검증.

**적용 한계:** GAE 가중치 척도는 장기 학습의 절대 상한이 아니다. γ^k·λ^k만 사용했다고 SMDP estimator가 primitive GAE와 같아지지 않는다. GAE 원 실험 공식 code pin은 미확인.

**확인 범위:** EXT의 상세 독해 기록을 인계; 이번에는 공식 서지·연결을 재확인. 상세 근거: EXT §2.3/E03.

<a id="s08"></a>

#### S08. The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games

**서지:** Chao Yu et al.. NeurIPS 2022 Datasets and Benchmarks Track; arXiv:2103.01955v4.

**원문:** [The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games](https://arxiv.org/abs/2103.01955v4) · [공식 on-policy pin](https://github.com/marlbenchmark/on-policy/tree/de66d7a4b23fac2513f56f96f73b3f5cb96695ac)

**읽을 위치:** §3.3, §5.1–5.5, Appendix C; value normalization·critic inputs·epochs·batch size.

**구현 연결:** PR06: 중앙 critic과 PPO 구현 요인을 분리한 최소 ablation.

**적용 한계:** 원 actor 분해와 v8 단일 team decision은 다르다. 공식 repo의 per-map 설정·재현 경고를 확인하고 최신 default를 논문 설정이라고 취급하지 않는다.

**확인 범위:** EXT의 상세 독해 기록을 인계; 이번에는 공식 서지·연결을 재확인. 상세 근거: EXT §2.3/E03.

<a id="s09"></a>

#### S09. Counterfactual Multi-Agent Policy Gradients

**서지:** Jakob Foerster et al.. AAAI 2018; arXiv:1705.08926v3 (2024-12-11 정정 포함).

**원문:** [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926v3) · [정정 포함 v3 본문](https://arxiv.org/html/1705.08926v3)

**읽을 위치:** v3 §4 식 (4), Figure 1, Errata, Lemma 1/Appendix A.

**구현 연결:** 조건부 PR06·07: 행동 독립 baseline, history-aware critic, 작은 Q(h,c) 진단의 참고.

**적용 한계:** 2024 v3는 joint history 의존성이 빠진 증명을 수정했다. 과거 PyMARL pin이 그 정정을 구현했다고 가정하지 않는다. v8를 COMA 5-actor 구조로 일괄 교체하지 않는다. Q의 예측을 관측되지 않은 반사실 정답으로 쓰지 않는다.

**확인 범위:** 공식 arXiv 서지와 v3 HTML/Errata 확인 경로 재확인. 상세 근거: EXT §2.3/E03.

<a id="s10"></a>

#### S10. Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs

**서지:** Tianwei Ni, Benjamin Eysenbach, Ruslan Salakhutdinov. ICML 2022; PMLR 162:16691–16723.

**원문:** [Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs](https://proceedings.mlr.press/v162/ni22a.html) · [조사한 pomdp-baselines pin](https://github.com/twni2016/pomdp-baselines/tree/e7c19c32a20033d75414b29fbc466c77c211e968)

**읽을 위치:** §4, §5.1–5.2, Fig.6, Appendix A/B/D/E; 입력·context 길이와 계산 비용.

**구현 연결:** 조건부 PR08: 같은 합법 입력의 작은 GRU 대조, sequence/reset·입력 권한 검사.

**적용 한계:** 원 실험은 off-policy 계열이다. 입력에서 버린 정보와 진짜 기억 필요성을 분리한다. 추론 때 없는 reward를 actor RNN에 넣거나 train-only reward를 inference에서 0으로 대체하지 않는다.

**확인 범위:** EXT의 상세 독해 기록을 인계; 이번에는 공식 서지·연결을 재확인. 상세 근거: EXT §2.4/E04.

<a id="s11"></a>

#### S11. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning

**서지:** Stéphane Ross, Geoffrey Gordon, Drew Bagnell. AISTATS 2011; PMLR 15:627–635.

**원문:** [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html)

**읽을 위치:** §2 이론, §3 Algorithm 3.1, §5 게임 실험과 stuck-state·expert mixture 논의.

**구현 연결:** PR03·04, 조건부 PR08: teacher failure를 abstain으로 분리, learner 방문 분포·NoOp 편향·폐루프 성능 검사.

**적용 한계:** v3에 이미 DAgger가 있었다. 단순 재도입은 개선안이 아니다. teacher 품질·관측 식별성·모델 근사 오차의 가정을 확인한다. 공식 2011 실험 code archive는 미확인.

**확인 범위:** EXT의 상세 독해 기록을 인계; 이번에는 공식 서지·연결을 재확인. 상세 근거: EXT §2.4/E04.

<a id="s12"></a>

#### S12. Transforming a head direction signal into a goal-oriented steering command

**서지:** Elena A. Westeinde et al.. Nature 626:819–826, 2024; DOI 10.1038/s41586-024-07039-2.

**원문:** [Transforming a head direction signal into a goal-oriented steering command](https://www.nature.com/articles/s41586-024-07039-2) · [Author Correction (2025-03-11)](https://www.nature.com/articles/s41586-024-08245-8)

**읽을 위치:** Methods: Network model 식 (5)–(10), Comparing model predictions with behaviour; 2025 Author Correction.

**구현 연결:** 조건부 PR08: heading·goal·상대 조향의 좌표 정의, sensor timestamp와 decoder 해석.

**적용 한계:** 생물학적 방향 제어 연구이지 BlackOut 학습 연구가 아니다. 2025 정정은 유전형·그림 label에 관한 것이며 모델식 변경을 보고하지 않는다. 우리 가상 sensor heading을 생물학적 body heading 측정값으로 바꾸어 부르지 않는다.

**확인 범위:** 원문·정정 페이지 연결과 서지 재확인; 세부 Methods는 EXT에서 인계. 상세 근거: EXT §2.5/E05.

<a id="s13"></a>

#### S13. Connectome-constrained networks predict neural activity across the fly visual system

**서지:** Janne K. Lappalainen et al.. Nature 634:1132–1140, 2024; DOI 10.1038/s41586-024-07939-3.

**원문:** [Connectome-constrained networks predict neural activity across the fly visual system](https://www.nature.com/articles/s41586-024-07939-3) · [조사한 flyvis pin](https://github.com/TuragaLab/flyvis/tree/92b3845cc426dd309a1a0e1b3890156c42e14021)

**읽을 위치:** Methods: Neuronal dynamics, Optic flow task, Importance of task optimization and connectome constraints, Unconstrained CNN; Supplementary Notes 1–5.

**구현 연결:** 조건부 PR08: topology·weight·sign·task optimization을 나눠 검증하는 대조 설계.

**적용 한계:** 시각 신경반응·optic-flow 예측 연구다. 원 논문의 생리 파라미터 수와 전체 decoder 학습량을 혼동하지 않는다. 생물학적 예측을 게임 승률 근거로 바꾸지 않는다.

**확인 범위:** 공식 원문 페이지 연결·서지 재확인; Methods·보충 상세는 EXT에서 인계. 상세 근거: EXT §2.5/E05.

<a id="s14"></a>

#### S14. A Drosophila computational brain model reveals sensorimotor processing

**서지:** Philip K. Shiu et al.. Nature 634, 2024; DOI 10.1038/s41586-024-07763-9.

**원문:** [A Drosophila computational brain model reveals sensorimotor processing](https://www.nature.com/articles/s41586-024-07763-9) · [조사한 brain model pin](https://github.com/philshiu/Drosophila_brain_model/tree/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960)

**읽을 위치:** Methods: Computational model, Neurotransmitter predictions, Assessment of model robustness, Computational modelling limitations; 출력 제거 조작.

**구현 연결:** 조건부 PR08: sensory/neural-output/actuator 개입 구분, 가중치·부호·port 가정 명세.

**적용 한계:** 고정 전뇌의 신경·행동 예측이며 게임 RL 성과가 아니다. FlyWire630 원 연구를 MaleCNS 또는 783 실행과 같은 데이터셋으로 취급하지 않는다. README의 silence 설명과 실제 구현 범위 차이도 확인한다.

**확인 범위:** 공식 원문 페이지 연결·서지 재확인; 상세 분석은 EXT에서 인계. 상세 근거: EXT §2.5/E05.

<a id="s15"></a>

#### S15. Prioritized Level Replay

**서지:** Minqi Jiang, Edward Grefenstette, Tim Rocktäschel. ICML 2021; PMLR 139:4940–4950; arXiv:2010.03934v4.

**원문:** [Prioritized Level Replay](https://proceedings.mlr.press/v139/jiang21b.html) · [조사한 level-replay pin](https://github.com/facebookresearch/level-replay/tree/ccecf452ee3342217ece964aaf10c2831625f9b3)

**읽을 위치:** §3.1–3.2 식 (2)–(5), Algorithms 1–2, supplementary score/staleness ablation.

**구현 연결:** PR06 후속: 최근 on-policy 정보로 다음 train cell을 정하는 coverage sampler.

**적용 한계:** 오래된 transition을 PPO에 재투입하는 알고리즘이 아니다. 높은 value error를 실제 학습 가능성과 동일시하지 않고 uniform 대조를 유지한다. 코드 pin은 CC-BY-NC4.0으로 보고되어 채택 전 사용 조건 확인이 필요하다.

**확인 범위:** EXT의 상세 독해 기록을 인계; 이번에는 공식 서지·연결을 재확인. 상세 근거: EXT §2.6/E06 (JSON에서는 R1–R4).

<a id="s16"></a>

#### S16. Deep Reinforcement Learning at the Edge of the Statistical Precipice

**서지:** Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, Marc G. Bellemare. NeurIPS 2021; arXiv:2108.13264v4 (2022-01-05).

**원문:** [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264v4) · [공식 proceedings](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html) · [조사한 rliable pin](https://github.com/google-research/rliable/tree/3ccd9f4dea577a04d3d2b557f259aac08badbd81)

**읽을 위치:** §2–4, §4.1, Appendix A.2–A.5; few-run uncertainty와 평가 protocol.

**구현 연결:** PR00·04·09: training-run 변동, 선택 편향, 차이의 구간·seed spread.

**적용 한계:** 여러 task마다 따로 훈련한 구조와 한 정책이 공통 map 여러 개를 평가한 구조는 다르다. 기본 rliable의 task별 run 재표집을 그대로 넣지 않는다. 3–5개 run으로 tail 안정성·동등성을 인증하지 않는다.

**확인 범위:** EXT의 상세 독해 기록을 인계; 이번에는 공식 서지·연결을 재확인. 상세 근거: EXT §2.7/E07.

<a id="s17"></a>

#### S17. Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO

**서지:** Logan Engstrom et al.. ICLR 2020; arXiv:2005.12729v1.

**원문:** [Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO](https://arxiv.org/abs/2005.12729v1) · [조사한 구현 pin](https://github.com/MadryLab/implementation-matters/tree/5ee6ecb12545365d9178135e65576adfc0d82f52)

**읽을 위치:** §3–5, Appendix A.1–A.2: initialization·normalization·clipping·LR·비교 실험.

**구현 연결:** PR05·06·09: gate 외 구현 차이 통제, R_flat 대조, 튜닝·실패 비용 대장.

**적용 한계:** 논문명만 같거나 final env-step만 같다고 공정한 비교가 아니다. 초기 구현 bug의 정확한 수정 patch까지 확인된 것으로 말하지 않는다.

**확인 범위:** EXT의 상세 독해 기록을 인계; 이번에는 공식 서지·연결을 재확인. 상세 근거: EXT §2.7/E07.

<a id="s18"></a>

#### S18. Performance Tuning Guide

**서지:** Szymon Migacz, PyTorch contributors. PyTorch 공식 튜토리얼; 페이지의 수정 표기 2025-07-09.

**원문:** [Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide)

**읽을 위치:** Disable gradient calculation, thread/OpenMP 관련 항목, Avoid unnecessary synchronization, dtype·copy·compile 관련 주의.

**구현 연결:** PR10: 실제 critical path부터 최적화하고 의미 보존 변경과 모델/학습 변경을 분리.

**적용 한계:** 문서 표시 버전은 설치 환경 검증이 아니다. CUDA 권고를 MPS에 자동 적용하거나 AMP·graph pruning·action repeat를 무료 가속으로 취급하지 않는다.

**확인 범위:** 공식 튜토리얼 본문·표시 metadata 재확인. 상세 근거: EXT §2.8/E08-S01.

<a id="s19"></a>

#### S19. The pigeonhole bootstrap

**서지:** Art B. Owen. Annals of Applied Statistics 1(2):386–411, 2007; DOI 10.1214/07-AOAS122.

**원문:** [The pigeonhole bootstrap](https://arxiv.org/abs/0712.1111) · [출판 DOI](https://doi.org/10.1214/07-AOAS122)

**읽을 위치:** §2.1 crossed effects, §3.2 행·열 독립 재표집, §4.2–4.3의 조건과 한계.

**구현 연결:** PR09: 같은 policy row를 모든 map에서 보존하고 map column을 공유하는 resampler 설계.

**적용 한계:** BlackOut의 알고리즘별 독립 row·공통 map 비교는 프로젝트 적용안이다. 원 논문이 적은 training run의 nominal CI coverage를 보장하지 않는다.

**확인 범위:** EXT의 상세 독해 기록을 인계; 이번에는 공식 서지·연결을 재확인. 상세 근거: EXT §2.7/E07.

<a id="s20"></a>

#### S20. Bootstrapping data arrays of arbitrary order

**서지:** Art B. Owen, Dean Eckles. Annals of Applied Statistics 6(3):895–927, 2012; DOI 10.1214/12-AOAS547.

**원문:** [Bootstrapping data arrays of arbitrary order](https://arxiv.org/abs/1106.2125) · [출판 DOI](https://doi.org/10.1214/12-AOAS547)

**읽을 위치:** factor 수준별 가중·product reweighting의 초록 및 원리; 추가 차원 도입 시 본문 재검토.

**구현 연결:** PR09의 심화 참고: 실제 표집되는 추가 factor가 생길 때 resampling 설계 재검토.

**적용 한계:** EXT는 초록·버전 metadata 중심 확인이다. 고정 상대를 자동으로 random opponent factor로 재표집하지 않는다.

**확인 범위:** 이번도 서지·초록 확인 수준; 전체 증명 재검증 아님. 상세 근거: EXT §2.7/E07.

<a id="s21"></a>

#### S21. Empirical Design in Reinforcement Learning

**서지:** Andrew Patterson, Samuel Neumann, Martha White, Adam White. JMLR 25(318):1–63, 2024.

**원문:** [Empirical Design in Reinforcement Learning](https://jmlr.org/papers/v25/23-0183.html) · [이번 확인된 공식 JMLR 경로](https://www.jmlr.org/beta/papers/v25/23-0183.html)

**읽을 위치:** §2.5–2.7, §3.2, §4.4–4.5, §7: few-run tail·튜닝·공통 난수·다중비교·보고.

**구현 연결:** PR09: procedure/artifact 성능 구분, dev/confirmation/test, trial ledger와 실패 표본 관리.

**적용 한계:** 특정 seed 수·δ·중단 규칙을 논문이 BlackOut에 처방한 숫자로 부르지 않는다. 이번 δ=5%p는 프로젝트 제안이다.

**확인 범위:** 공식 JMLR 서지·초록 재확인. 상세 근거: EXT §2.7/E07.

<a id="sup01"></a>

#### SUP01. Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping

**서지:** Andrew Y. Ng, Daishi Harada, Stuart J. Russell. ICML 1999, pp.278–287.

**원문:** [Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping](https://dl.acm.org/doi/10.5555/645528.657613) · [저자 소속기관 제공 PDF](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)

**읽을 위치:** §2.2/§3 Theorem 1: F=γΦ(s′)−Φ(s).

**구현 연결:** 조건부 PR01·06·07: shaping 도입 전 telescoping·경계·reward-hacking fixture 설계.

**적용 한계:** 현재 score-delta surrogate를 자동으로 정책 불변 PBRS라고 부르지 않는다. 부분관측·팀 정책·변하는 상대에 원정리를 무조건 전이하지 않는다.

**확인 범위:** 첨부 EXT의 원문/정리 확인 기록에서 인계; 이번에 PDF를 새로 분석하지 않음. 상세 근거: EXT §2.3/E03 PBRS1999.

<a id="sup02"></a>

#### SUP02. Reward Shaping in Episodic Reinforcement Learning

**서지:** Marek Grześ. AAMAS 2017, pp.565–573.

**원문:** [Reward Shaping in Episodic Reinforcement Learning](https://www.ifaamas.org/Proceedings/aamas2017/pdfs/p565.pdf)

**읽을 위치:** §2–3: episode 마지막 potential과 horizon 경계.

**구현 연결:** 조건부 PR01·06·07: true terminal의 Φ=0, 외부 truncation continuation, 서로 다른 길이의 shaping 검사.

**적용 한계:** γ<1이고 종료 길이가 다르면 같은 비영(非零) terminal potential도 안전하지 않을 수 있다. 경계식 통과가 finite-budget PPO 성능 향상을 보장하지 않는다.

**확인 범위:** 첨부 EXT의 PDF 독해 기록에서 인계; 이번 새 PDF 분석 없음. 상세 근거: EXT §2.3/E03 PBRS2017.

<a id="sup03"></a>

#### SUP03. On Centralized Critics in Multi-Agent Reinforcement Learning

**서지:** Xueguang Lyu, Andrea Baisero, Yuchen Xiao, Brett Daley, Christopher Amato. JAIR 77:295–354, 2023; arXiv:2408.14597 (2024 등록).

**원문:** [On Centralized Critics in Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2408.14597)

**읽을 위치:** state-based/history-based critic의 조건; S09 Errata와 함께 읽기.

**구현 연결:** 조건부 PR06·07: V(h)와 Q(h,c), actor와 critic의 정보 권한·history 표현 구분.

**적용 한계:** 중앙 state가 있다는 사실만으로 POMDP critic의 정합성이나 성능 향상을 보장하지 않는다. EXT도 상세 전체 증명·실험 재현까지 완료한 것으로 표시하지 않았다.

**확인 범위:** 공식 arXiv 서지·초록 재확인; 이 문서에서 전체 증명 재독해 없음. 상세 근거: EXT §2.3/E03.

<a id="doc01"></a>

#### DOC01. Benchmark Utils — torch.utils.benchmark

**서지:** PyTorch contributors. PyTorch 2.8 버전 고정 공식 문서.

**원문:** [Benchmark Utils — torch.utils.benchmark](https://docs.pytorch.org/docs/2.8/benchmark_utils.html)

**읽을 위치:** Timer, num_threads, blocked_autorange, Measurement.

**구현 연결:** PR10: isolated kernel timing의 warm-up·반복·thread 조건 고정.

**적용 한계:** block 평균의 분포를 요청별 p99로 사용하지 않는다. EXT의 2.8 Timer 조사 결과를 실제 2.13 runtime의 동작으로 일반화하지 않는다.

**확인 범위:** 버전 고정 공식 문서 연결 재확인. 상세 근거: EXT §2.8/E08-S02.

<a id="doc02"></a>

#### DOC02. torch.profiler

**서지:** PyTorch contributors. PyTorch 2.8 버전 고정 공식 문서.

**원문:** [torch.profiler](https://docs.pytorch.org/docs/2.8/profiler.html)

**읽을 위치:** activities, schedule, record_shapes, profile_memory, with_stack.

**구현 연결:** PR04·10: 병목 귀속용 trace와 계측 없는 전체 처리량 측정을 분리.

**적용 한계:** 중첩 cumulative span을 더하거나 CPU profiler만으로 모든 MPS kernel 시간을 측정했다고 하지 않는다. profiler 자체 overhead를 기록한다.

**확인 범위:** 버전 고정 공식 문서 연결 재확인. 상세 근거: EXT §2.8/E08-S02.

<a id="doc03"></a>

#### DOC03. torch.mps.synchronize

**서지:** PyTorch contributors. PyTorch 2.8 버전 고정 공식 API 문서.

**원문:** [torch.mps.synchronize](https://docs.pytorch.org/docs/2.8/generated/torch.mps.synchronize.html)

**읽을 위치:** 호출의 완료 동기화 의미; 실제 설치 버전 signature와 비교.

**구현 연결:** PR10: 비동기 실행 enqueue 시간과 완료 시간을 구분하는 MPS 측정.

**적용 한계:** 모든 operation마다 sync하여 pipeline을 바꾼 시간과 실제 end-to-end 요청 시간을 구분한다. MPS 사용 가능·대회 허용을 이 문서로 확정하지 않는다.

**확인 범위:** 공식 API 연결·완료 동기화 설명 재확인. 상세 근거: EXT §2.8/H01.

<a id="doc04"></a>

#### DOC04. Reproducibility

**서지:** PyTorch contributors. PyTorch 2.8 버전 고정 공식 문서.

**원문:** [Reproducibility](https://docs.pytorch.org/docs/2.8/notes/randomness.html)

**읽을 위치:** Controlling sources of randomness, Avoiding nondeterministic algorithms.

**구현 연결:** PR00·04·09·10: RNG 분리, runtime pin, same-backend/cross-backend parity.

**적용 한계:** seed 설정만으로 다른 장치·버전·재개 물리 상태가 같아지는 것은 아니다. 실제 검증한 범위만 재현성으로 보고한다.

**확인 범위:** 버전 고정 공식 문서 연결 재확인. 상세 근거: EXT §2.8/H07.

<a id="doc05"></a>

#### DOC05. Changes announced May 6, 2022 — Python Updates

**서지:** Protocol Buffers team. Protobuf 공식 변경 안내, 2022-05-06.

**원문:** [Changes announced May 6, 2022 — Python Updates](https://protobuf.dev/news/2022-05-06/)

**읽을 위치:** Python upb 변경, generated-code 요구와 호환 주의.

**구현 연결:** PR00·10: protobuf runtime/implementation·generator·wire descriptor hash를 분리해 고정.

**적용 한계:** upb가 빠를 수 있다는 설명이 현재 ML-Agents dependency override의 호환성 인증은 아니다. 현재 검증된 overlay를 자동 업그레이드하지 않는다.

**확인 범위:** 공식 변경 안내 본문 재확인. 상세 근거: EXT §2.8/E08-S03.


<a id="sec-18"></a>

## 18. 참고 구현 저장소·판·라이선스 확인 목록

아래 commit·라이선스는 **첨부 외부 보고서가 정적으로 확인한 기록을 인계**한 것이다. 이번 문서 작업에서 모든 원격 코드와 LICENSE를 다시 열어 확인하거나 코드를 실행한 것은 아니다. `pin 확인`은 `논문 당시 실험과 완전히 같은 코드 확인`이 아니다. 라이선스 표시는 사용 허가에 관한 보증이나 법률 검토 결과가 아니며, 코드·데이터·환경 의존물 각각의 실제 적용 조건은 채택할 때 다시 확인한다.

| 연결 자료 | 저장소·검토 pin | 읽을 코드 | 보고된 라이선스·채택 주의 |
|---|---|---|---|
| S04 | [invalid-action-masking](https://github.com/vwxyzjn/invalid-action-masking/tree/6daedd29e4b48271c274c4593dcbc9a76be30ce4) — `6daedd29e4b48271c274c4593dcbc9a76be30ce4` | `ppo.py`: CategoricalMasked, rollout masks, ratio | MIT. recommended default는 논문 표와 다르므로 그대로 실험 재현이라고 하지 않음 |
| S05 | [OpenAI baselines](https://github.com/openai/baselines/tree/ea25b9e8b234e6ee1bca43083f8f3cf974143998) — `ea25b9e8b234e6ee1bca43083f8f3cf974143998` | `baselines/ppo2/model.py` | MIT. 2017 논문 exact experiment snapshot 미확인 |
| S06 | [option_critic](https://github.com/jeanharb/option_critic/tree/5d6c81a650a8f452bc8ad3250f1f211d317fde8c) — `5d6c81a650a8f452bc8ad3250f1f211d317fde8c` | `learning.py`, 종료 규칙 | 본체 라이선스 미확인. 공개되어 있다는 이유만으로 복사·배포하지 않음 |
| S08 | [on-policy](https://github.com/marlbenchmark/on-policy/tree/de66d7a4b23fac2513f56f96f73b3f5cb96695ac) — `de66d7a4b23fac2513f56f96f73b3f5cb96695ac` | `shared_buffer.py`, per-map training scripts | MIT. wrapper의 masks/bad_masks와 final observation을 함께 확인 |
| S09 | [PyMARL](https://github.com/oxwhirl/pymarl/tree/c971afdceb34635d31b778021b0ef90d7af51e86) — `c971afdceb34635d31b778021b0ef90d7af51e86` | `src/learners/coma_learner.py`, `src/modules/critics/coma.py` | Apache-2.0. 2024 COMA 정정 이전 pin; 수정 이론을 구현했다고 가정하지 않음 |
| S10 | [pomdp-baselines](https://github.com/twni2016/pomdp-baselines/tree/e7c19c32a20033d75414b29fbc466c77c211e968) — `e7c19c32a20033d75414b29fbc466c77c211e968` | `policy_rnn.py`, `recurrent_actor.py` | MIT. 논문 당시 commit 미확인; reward 입력·off-policy replay를 우리 PPO에 복사하지 않음 |
| S12 | [WesteindeWilson_AnalysisCode](https://github.com/wilson-lab/WesteindeWilson_AnalysisCode/tree/df92d854c863f59c2f9c96358985516ab9dc095d) — `df92d854c863f59c2f9c96358985516ab9dc095d` | 모델 notebook | 본체 LICENSE 미확인. 논문 라이선스와 코드 허가는 별개 |
| S13 | [flyvis](https://github.com/TuragaLab/flyvis/tree/92b3845cc426dd309a1a0e1b3890156c42e14021) — `92b3845cc426dd309a1a0e1b3890156c42e14021` | `flyvis/network/dynamics.py`, network config | MIT. paper checkpoint·release와의 정확한 매핑은 미확인 |
| S14 | [Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model/tree/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960) — `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960` | `model.py`, 특히 `silence()` | MIT. README와 실제 출력 제거 범위를 대조 |
| S15 | [level-replay](https://github.com/facebookresearch/level-replay/tree/ccecf452ee3342217ece964aaf10c2831625f9b3) — `ccecf452ee3342217ece964aaf10c2831625f9b3` | `level_replay/level_sampler.py` | CC-BY-NC4.0. 사용·배포 조건 확인; pin은 논문 최종판보다 오래됨 |
| S16 | [rliable](https://github.com/google-research/rliable/tree/3ccd9f4dea577a04d3d2b557f259aac08badbd81) — `3ccd9f4dea577a04d3d2b557f259aac08badbd81` | `library.py`, `metrics.py` | Apache-2.0. 본 프로젝트 crossed run×map에 기본 함수 그대로 적용하지 않음 |
| S17 | [implementation-matters](https://github.com/MadryLab/implementation-matters/tree/5ee6ecb12545365d9178135e65576adfc0d82f52) — `5ee6ecb12545365d9178135e65576adfc0d82f52` | `custom_env.py`, config와 normalizer | MIT. 초기 bug의 정확한 수정 patch·원 장비는 미확인 |

GAE 원 실험과 DAgger2011 공식 archive/commit은 외부 조사에서 고정하지 못했다. 제3자 재구현을 원저자의 검증된 code라고 대신 적지 않는다. 외부 소스가 우리의 기대 동작과 다르면 논문·버전·wrapper 의미를 먼저 대조하고, 현재 프로젝트에 맞는 최소 구현과 회귀 검사를 작성한다.

**채택 기록 형식:** `reference_id`, 원문 URL/판, code repo/commit/경로, LICENSE 확인일, 사용한 알고리즘 부분, 원 실험과 다른 점, 최소 fixture, 실행 환경, 결과, 담당자를 저장한다. 이 기록은 `docs/v8/reference_adoption.md`라는 제안 경로로 관리한다.


<a id="sec-19"></a>

## 19. PR별 필수 독서와 구현 완료 증거

| PR | 착수 전 읽을 자료 | 읽고 구현에 남겨야 할 결정·테스트 |
|---|---|---|
| PR00 | INT I00/I09, 프로젝트 versions·preregistration, S16·S17, DOC04 | 기준 commit 불일치·누락 자료·runtime pin·backups/lineage의 책임 범위 |
| PR01 | INT C01, EXT E01, S01·S02 | engine winner·final score·reason, duplicate/late event, 마지막 delta, terminal/truncation 표 |
| PR02 | INT C02/I05, S02 | old/new timer generation·callback 순서, 실제 3연속 timeout, raw categorical-ID fixture |
| PR03 | INT C06/C08, S02·S04·S11 | ID/reset/retry, factory effective config, 정상 네 slot 보존, teacher abstain, mask 원인 |
| PR04 | INT I02–I07의 자료 부재, S16·S17, DOC02 | chronological trace·정상 대조 window·모듈 gradient·분모·입력 재연 |
| PR05 | INT C03, EXT E02, S03·S04·S05 | exact factorization과 새 decoder 구분, all-invalid·한 후보·entropy·ratio fixture, q/margin/effective 개입 분모 |
| PR06 | INT C04/C09, S05·S07·S08; 표집 변경 시 S15 | 동일 behavior version, raw return 단위, continuous episode, 전체 step 노출·실패 기록 |
| PR07 | INT C05, EXT E03, S06·S07; Q/PBRS 검토 시 S09·SUP01–03 | goal option 시작/끝·실제 k·abort·rollout censor 계약, 같은 head k=1 대조 |
| PR08 | INT C07/C08, S10·S11·S12; wiring 분기는 S13·S14 | 가상 heading·timestamp·권한표, rates-only 대조, 학습된 F1 개입, memory와 정보 확장 분리 |
| PR09 | INT C09, EXT E07, S16·S17·S19·S21; 추가 factor면 S20 | independent run/shared map 알고리즘, 기존 confirmation 사용 이력, δ·분모·다중비교·최종 lock |
| PR10 | INT C10, S02·S18, DOC01–05 | runtime/프로토콜·정밀도·실제 요청 tail, 종료 포함 parity, 격리 loader 연전·capability 인증 |

각 PR 설명에는 `해결 F-ID / 변경 전 반례 / 변경 후 fixture / 읽은 reference_id / 의도적으로 바꾸지 않은 조건 / 남은 미확정 사항`을 넣는다. 합성·오프라인·실제 Unity·공식 loader 검증을 다른 칸으로 기록하고, 합성 통과를 실제 환경 통과로 승격하지 않는다.

원문이 제공하지 않는 BlackOut 수치는 참고문헌으로 정당화하지 않는다. 예를 들어 q=.1, τ=.5, MLP 폭64, ±250-step window, 3-run 파일럿, 5%p 개선폭은 이 계획의 출발 제안값이다. 논문이 같은 값을 사용했더라도 현재 과제의 최적값이라는 뜻은 아니다.

<a id="sec-20"></a>

## 20. 구현 전에 잠글 값과 아직 미확정인 계약

### 20.1 잠정 설정의 지위

| 항목 | 출발 제안 | 잠글 시점·변경 규칙 |
|---|---|---|
| C1 제어 범위 | 팀에서 최대 1 slot, conditional40 | PR05 구현 계약. 여러 slot 동시 제어는 별도 실험 |
| 입력 | slot당 기존 177차원, frozen encoder | R_flat/C1 기본 비교에서 공유; 정보 추가는 별도 요인 |
| C1 gate / decoder | q 초기 .10, τ=.50, tie KEEP | dev에서만 선택·수정; 이후 model lock 포함 |
| 학습 시작 설정 | §7.1의 v7 출발 설정 | PR00 실값 대조 후 공통 config 고정 |
| 학습 상대 | §7.2의 기존 고정 schedule | 구조 대조 중 유지; curriculum은 별도 ID |
| 중간 checkpoint | 65,536 transitions을 넘긴 첫 rollout 경계 | 저장 부하 측정 후 파일럿 시작 전 고정 |
| 상세 사건 window | 전후 ±250 transitions | 실제 tick/time·용량과 함께 고정; 무작위 정상 대조 포함 |
| 파일럿 | arm당 3 runs, 1,048,576 steps endpoint | 작은 표본의 실패기전 선별용. 강한 통계 인증 아님 |
| 고정 예산 본실험 | 출발값 2M steps/run | R과 budget은 파일럿 분석 뒤 confirmation 전에 사전 고정 |
| 연구 주지표 | engine 검증 W/N; draw 포함 | 공식 scoring과 다르면 별도 endpoint·실험 ID |
| 실용 개선폭 | 잠정 +5%p | 비용·기존 목표와 함께 confirmation 전에 고정 |
| 진영 회귀 한도 | 잠정 −5%p | 사용 여부와 판정 방식을 사전등록; 자동 공식 기준 아님 |
| 지연·메모리 한도 | **미정** | 공식 장비/loader/자원 계약을 받아야 제출 합격 기준 설정 가능 |

### 20.2 공식 자료가 필요한 항목

현 운영 build·runner·loader의 commit/hash, 점수·승패 정의, 동점·외부 timeout 처리, actor/teacher/planner 허용, 팀 입력/agent ID, state/reset/retry, 외부 graph·추가 package/custom op, CPU/GPU·thread·메모리·cold/warm 시간 제한이 필요하다. 문의 초안은 EXT-J `official_contract.inquiry_draft`에 있으며 **발송 완료로 기록되어 있지 않다**.

로컬 연구는 검증된 연구용 계약 아래 진행할 수 있지만, 공식 미답변을 허용으로 바꾸거나 수정 build의 승률을 운영 서버 승률과 동일하게 선언하지 않는다. 자료가 오면 영향 범위가 있는 PR·config·model lock을 갱신하고 변경 전후 실험을 분리한다.

### 20.3 완료 체크리스트

- [ ] 정확한 final score/winner/reason과 마지막 reward 단회 반영을 실제 Unity에서 확인했다.
- [ ] reset·중복·지연 event·타이머 재진입을 실제 여러 경기에서 검사했다.
- [ ] 기존 graph의 bound·exact adapter와 새 decoder 동작을 원본 hash에 연결했다.
- [ ] sampled·greedy·executed action·실제 이동·보상·update를 같은 ID로 재연할 수 있다.
- [ ] R_plan/R_flat와 후보가 동일 build·상대·정보·제어범위·예산에서 비교되었다.
- [ ] 회귀·실패·재시도·재개·누락과 전체 비용을 trial ledger에 보존했다.
- [ ] 소수 run의 한계와 map/run 독립 단위를 지킨 결과 보고서를 만들었다.
- [ ] 새 confirmation·최종 test와 artifact 선택 규칙을 서로 분리하고 잠갔다.
- [ ] 선택한 전체 policy가 허용된 loader·state·의존성·장치·자원 계약을 통과했다.
- [ ] 채택한 원문·판·code pin·허가 조건·읽을 위치·프로젝트 적용 차이를 기록했다.

앞의 engineering/research 조건이 통과해도 마지막 제출 계약이 미확정이면 상태는 `submission_ready`가 아니다. 반대로 C2·C3·connectome 연구를 수행하지 않았다는 이유만으로, 조건을 통과한 C1 본선의 완료를 막지는 않는다.


<a id="sec-21"></a>

## 21. 근거·버전·작성 범위 기록

### 21.1 입력 파일 고정값

다음 hash는 이번 문서 작성 환경에 제공된 **입력 파일 자체**의 SHA-256이다. 파일 안에 기록된 원격 run·checkpoint·build hash를 이번에 다시 계산했다는 뜻은 아니다. 아래 원문 파일명·절/행 인용은 해당 입력판을 기준으로 한다.

| alias | 파일 | SHA-256 |
|---|---|---|
| INT | `internal_result.md` | `a9c10c3d1f759e70a4ea8624576274bb3e711edeb0011721cf0eb0e15e986167` |
| EXT | `v8_external_research.md` | `46624c40379ee0d519d71712c8b9ddd425a816da4e1ac889e998c2bb93bf88eb` |
| EXT-J | `v8_external_research.json` | `a9a8a25c54526df6af30253529ec7277b7ea8873f5245db8176e225212d681ce` |
| PLAN-0 | `v8_implementation_plan.md` | `cddbacfea75e81372433eb5b4a3f162cbab6fced2133cc2deddf071444c18517` |
| TASK-0 | `implementation_backlog.json` | `fea8dcc5fad3fb4cf561da7f0d9f5fc56049fc2cd3a0d5e6670ec5820072955c` |

### 21.2 내부 근거의 위치

INT의 조사 범위 L5–17, 실제 checkout·산출물 L45–78, 결과/보상/타이머 L80–136, A2/A3 bound L138–178, PPO/rollback L180–228, 지속성 자료 부재 L230–240, planner/관측 L242–265, 입력/heading/DAgger L267–306, F1/graph L308–342, 노출/통계 L344–377, 성능/테스트/export L379–437, F-ID/결정표 L439–517을 사용했다.

EXT는 §1 후보 C1–C3/결정 D01–D08, §2.1–2.8의 근거 카드, §3 코드·라이선스, §4 실험, §5 미답변 계약을 사용했다. 외부 보고서 안의 과거 tool citation ID는 이 문서에서 새로운 검증 ID로 재사용하지 않고, 자료 ID·원문 링크·판·읽을 위치로 연결했다.

### 21.3 이번 웹 확인 범위

S01·S02 공식 API 설명, S18·DOC01–05 공식 문서, S03–S17 중 목록에 명시한 arXiv/PMLR/AAAI/Nature 서지·공식 연결, S19–S21·SUP03의 공식 서지·초록을 재확인했다. S09 v3 본문 경로와 S12 Author Correction도 연결했다. SUP01·SUP02의 PDF는 이번에 새로 분석하지 않고 EXT의 독해 기록을 인계했다. 상세 논문 수식·부록·code pin은 각 항목의 확인 범위대로 구분했다.

이 문서화 작업에서 원본 코드 수정, 새 Unity 경기, checkpoint 학습·재개, 패키지 설치, 제출물 업로드, 대회 운영자 문의 발송은 수행하지 않았다. v8 성능은 위 검증 절차가 실행되기 전까지 미확정이다.

