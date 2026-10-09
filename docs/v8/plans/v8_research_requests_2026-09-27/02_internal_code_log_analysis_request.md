# 내부 코드·로그 분석자에게 전달할 요청서

제목: BlackOut RL v8 준비를 위한 실패 기전·실행 산출물 감사
기준일: 2026-09-27 KST
기준 snapshot: main@e6e98a63e9f401f6bf03e463a720c03878d1beb7

## 요청 목적과 범위

`00_common_brief.md`와 `reference/previous_audit.md`를 바탕으로, v6/v7의 성능 정체를 만든 실제 원인을 코드·원본 로그·checkpoint·최소 재현으로 구분해주세요. 기존 감사는 전체 원본 로그·학습 가중치·Unity 실행물을 확보하지 못했으므로, 거기에 적힌 숫자를 재인용하는 것만으로 재검증을 대신하지 말아주세요.

코드 전체를 다시 요약하는 것보다, 한 건의 실패라도 ‘설정 → 실제 상태 → 선택한 행동 → 실행된 이동 → 보상 → update → 평가 결과’로 연결하는 자료가 필요합니다. 아래 위치는 감사 snapshot의 탐색 시작점입니다. 실제 경로/스키마가 다르면 확인한 위치를 기록하고, 없는 로그 필드는 신규 계측 대상으로 남겨주세요.

## 분석 전에 지킬 것

원본은 읽기 전용으로 보존하고 분석 결과는 별도 디렉터리에 저장해주세요. 독립 test는 열지 말고 train/dev 중심으로 조사해주세요. 새 장기학습, 대량 재평가, 기존 checkpoint의 resume, upstream 교체는 조사 중 자동 실행하지 않습니다. 필요한 경우 별도 진단 run을 제안하고 기록합니다.

증거 등급은 ‘소스 확인/기존 원시자료 재집계/합성 재현/실제 Unity 재현/가설/자료 부재’로 구분해주세요. ‘문서상 보고’도 원시자료 재확인과 구별합니다. 수정된 v1/v5 결함은 역사적 원인과 회귀 여부를 나눠주세요.

## I00. 실제 실행물·버전·산출물 목록 확보 — 시작 전에 필요

연결: F30–F31, F33 / 외부 E07

찾을 자료: `docs/common/history.md`, `docs/issues/1st_issues_v0-v7.md`, `docs/common/versions.md`, requirements lock, code/v7/configs/main_study, logs/v7/reports*registration*, logs, checkpoints, Unity build와 인접 upstream 소스.

특히 확보할 것은 v6의 run_summary/training/training_episodes/target_eval/confirmation/rollback 기록, v7의 `logs/v7/main_study/accelerated/summary.json`이 가리키는 23개 result와 각 run의 training/episodes/diagnostics/resolved_config/experiment_manifest/status/runtime/lineage, 초기·중간·최종 checkpoint, graph NPZ+JSON·source manifest, frozen opponent/encoder checkpoint입니다. 중간 checkpoint가 없으면 없다고 적어주세요.

기준 commit과 현재 branch/HEAD·미커밋 변경을 대조하고, 실제 실행의 source hash/overlay/빌드와 묶어주세요. user local·서버에만 있는 자료는 소유자에게 합법적인 읽기 접근/복사본을 요청하고, 접근 가능한 범위만 사용합니다. 원본 데이터의 전체 복사가 불필요하면 manifest·해시·필요한 부분 추출을 우선합니다.

산출물: 각 run의 experiment ID, 학습 seed, map split, code/build/config/opponent/checkpoint/graph hash, 시작·중단·재개 시점, runtime backend, 완료/실패/미확인 상태, 실제 artifact 경로의 목록. 절대 경로는 portable alias도 제공합니다. raw log가 없으면 해당 분석의 제한과 최소 추가 파일을 기록합니다.

## I01. 점수·승자·종료·보상 의미 검증 — 최우선

연결: F01–F04, F35 / 외부 E01

코드: `code/v7/scripts/evaluate_v7.py`, `code/shared/eval/evaluator.py`, `code/shared/blackout_rl/reward.py`, `training_reward.py`, `env.py`, 고정 upstream의 `_collect_obs`, `competition/match.py`, Unity 종료/타이머/score handler.

기존 자료로 할 일: manager가 지목한 result만 집계하여 progress·백업·재평가 중복을 제거합니다. 전체 경기수, 0:0, 0:0+winner, 음수/범위/단위 이상, 종료 길이·진영·모델별 분포를 재계산합니다. 이전 감사의 1,380/1,211/1,204는 검증 대상이지 기대값으로 맞출 숫자가 아닙니다. 0이 아닌 terminal score도 정확한 최종 점수라는 보장이 없습니다.

새 계측이 필요한 일: engine event의 final_score/final_winner/reason, terminal raw info, last-nonterminal snapshot, reward 성분을 같은 episode에 연결합니다. 정상 득점·감점/약탈·100점 도달·동점+shaping·동시 이벤트·연속 timeout을 비교합니다. 게임 timeout과 외부 watchdog은 다른 사건입니다. 종료가 정상이어도 reset 후 fake delta가 발생하는지, 마지막 delta가 누락/중복되는지 확인합니다.

타이머는 자동 재시작과 Python 명시 reset을 따로 검사하고 최소 3회 연속 timeout의 timer ID와 시점을 남깁니다. C# 소스 검토만 한 결과를 실제 Unity 재현으로 표기하지 않습니다.

산출물: 지표별 사용 가능/제한 사용/사용 불가 표, 원본 사례 5~10개, 각 사건의 전후 프레임, 최초 오염 코드 위치, 최소 failing test, 구현 수정 범위. 최종 점수를 증명할 자료가 없으면 임의 복원하지 않습니다.

## I02. A2/A3가 왜 같은 경기 기록을 냈는지 — 최우선

연결: F09, F15 / 외부 E02

코드: `code/v7/blackout_rl/v7_1/model.py`, `mappo_v6.py`, `v7_registry.py`, `code/v7/scripts/evaluate_v7.py`.

먼저 평가 파일/경로 재사용, 모델·graph hash, config resolution, checkpoint load, stale cache를 확인합니다. 같은 파일을 두 번 평가한 경우와 실제 다른 모델이 같은 행동을 낸 경우를 분리해야 합니다. A2/A3 graph src/dst, control seed/ports, trainable weight가 다른지도 확인합니다.

기존 최종 episodes를 episode별로 비교하고, 확보되는 초기/중간/최종 checkpoint를 pure planner·초기 residual·A0·A1·A2·A3와 대조합니다. 최종 checkpoint만 있으면 학습 전환 시점을 정확히 알 수 없다는 제한을 적습니다.

새 행동 계측: 동일한 chronological observation sequence를 각 정책에 같은 reset/이력 조건으로 넣어 p_KEEP, p_OVERRIDE=1-p_KEEP, 최고 수정확률, correction logit minus KEEP logit, legal correction 개수, mask 제외 이유, entropy, sampled 선택, greedy 선택, 적용 후 action을 기록합니다. slot·진영·시간대·정체/적재 상태로 나눕니다. 저장 관측을 섞어 stateful planner에 독립 입력하면 다른 문제가 되므로 순서를 보존합니다.

구분할 가설: (a) greedy KEEP 유지, (b) 같은 correction, (c) 다른 action이나 충돌로 이동 없음, (d) 행동은 다르지만 같은 경기 요약, (e) 잘못된 artifact/caching. 수정확률이 높아도 특정 correction 하나가 KEEP을 이기지 못하는 경우를 따로 집계합니다.

산출물: 동일성의 단계별 판정(파일/모델/분포/행동/이동/요약), 최초 분기 episode/step, 학습 sampled 개입률과 평가 greedy 개입률. ‘A2/A3 결과 동일=전부 KEEP’으로 결론 내리지 말아주세요.

## I03. 성능 회귀와 PPO update의 시점 연결 — 높은 우선순위

연결: F11, F13–F14 / 외부 E03·E06·E07

코드/자료: `mappo_v6_training.py`, `v7_training.py`, `v7_2/training.py`, v6 train script·curriculum, `rollout.py`, run별 training/episodes/target_eval/confirmation/rollback.

v6 최초 회귀·두 rollback, v7 A0/A1 성능 저하를 중심으로 학습 step 기준 타임라인을 작성합니다. policy/value loss, KL, clip fraction, entropy, explained variance, 실제 완료 epoch/minibatch, return/advantage 분산, opponent 전환, checkpoint save/load, resume, 진영 배치를 연결합니다. 외부 평가 기간과 train 기간을 wall time만으로 합치지 말아주세요.

없는 값은 신규 계측합니다: actor/critic/encoder/graph/출력 head별 gradient norm과 parameter delta, clipped gradient 비율, first-step zero-init 뒤의 gradient 도달성, critic scale, keep/correction별 advantage 분포. zero-init 출력층 때문에 첫 backward에서 graph gradient가 0인 것과 계속 0인 결함은 다릅니다.

v6에서 개선이 생겼다가 사라졌는지, 끝까지 없었는지, rollback이 회복만 시켰는지 구분합니다. 예산 종료를 알고리즘 수렴으로 해석하지 않습니다. sampling log-prob와 저장된 실행 action의 ratio, terminal/truncation bootstrap, rollout/episode 경계를 합성 fixture로 재검사합니다.

산출물: 주요 회귀 사건의 직전/직후 update 구간, 재현 가능한 그래프·명령, 가능한 기전과 반증 자료. 다른 모델의 성능 차이를 gradient 통계 하나로 인과 확정하지 않습니다.

## I04. 수정 행동은 무효한가, 해로운가, 장기 보상에 묻히는가 — 높은 우선순위

연결: F10, F13 / 외부 E02·E03

코드/자료: `apply_joint_action`, `PlannerFeatures`, `scripted_fsm.py`, `strategy.py`, reward tracker, chronological action/position/score events. 고빈도 행동 자료가 없으면 신규 진단 trajectory가 필요합니다.

개입의 requested direction과 실제 displacement, 다음 planner가 되돌리는 정도, 연속 개입 길이, 목표 변경 빈도, 도달 가능한 목표까지 진행, deposit/약탈/전투/아이템 사건까지의 시간을 집계합니다. 1·5·25·50·250 step 등은 분석용 예시 창일 뿐 최적 action duration을 미리 정한 것이 아닙니다. 시간 단위는 실제 dt 계약과 확인합니다.

planner 경로를 유용하게 줄인 경우, 충돌로 아무 일도 없었던 경우, 다음 프레임 원위치로 돌아온 경우, 위험 회피를 성공/실패한 경우, 여러 슬롯 협력이 필요했던 경우를 분류합니다. 단순 거리 감소를 반드시 좋은 전략으로 간주하지 않습니다. 유효 행동률과 경기 성능을 함께 봅니다.

관측 분석과 인과 실험을 분리해주세요. 관측된 override 후 좋은 reward만으로 override의 효과라고 할 수 없습니다. 전체 상태 snapshot/동등 재현이 가능하면 같은 상태에서 KEEP과 correction을 비교하고 이후 정책·상대·난수 조건을 고정합니다. 불가능하면 episode-level paired 재평가로 낮추고 ‘동일 상태 반사실’이라고 부르지 않습니다.

산출물: 실패 유형별 빈도·분모·대표 trajectory, reward 지연분포, 유효한 행동 차이의 지속성, 최소 개입 실험 계획. 행동 지속성·목표 residual·credit assignment 중 어느 변경이 먼저 필요한지 조건부 판단합니다.

## I05. planner·관측·mask·전략의 실행 실패 지도 — 높은 우선순위

연결: F05–F08, F12, F36 / 외부 E01·E02·E04·E06

코드/자료: `env.py`, `observation.py`, `semantic_map.py`, `navigation.py`, `scripted_fsm.py`, `strategy.py`, `team_state.py`, `PlannerFeatures.prepare`, `v7_2/teammates.py`, BASE-S15 보고와 가능하면 원 로그.

찾을 패턴: PathNotFound 위치/역할/목표/활성 창고, 한 슬롯 실패가 다섯 슬롯 stop/reset으로 확장되는 비율, 벽 근처 진동, stuck recovery 실패, 목표 ping-pong, 여러 유닛이 한 아이템에 몰림, 합법 경로가 mask로 차단, 금지/충돌 이동이 mask를 통과, KEEP은 항상 허용되어 위험을 유지하는 상황.

mask의 false positive/negative는 실제 물리나 검증된 기하 기준이 있을 때만 정의합니다. 현재 근사 mask를 ground truth로 삼아 자기 자신을 검증하지 않습니다. 하드코딩 좌표, 진영의 y=x 대칭, semantic top-down Y, 반경·속도·buff/class 조건을 대조합니다. 90도 회전 fixture는 기본적으로 인터페이스 검사이며 실제 맵/게임 규칙이 회전 대칭이라는 가정은 금지합니다.

맵은 wall·후보 창고·현재 활성 창고를 나눠 seed별 cache reset을 검증합니다. raw rendering ID와 decoded one-hot, item/unit/storage 겹침, background/headless 및 실제 배포 플랫폼의 관측 갱신을 확인합니다. unit ID·class·holding item type과 stack quantity의 제공 여부도 분리합니다.

특수 아이템·guard/worker 역할 분담은 random 상대 포화 결과가 아니라 target에서 별도 가설로 검토합니다. 산출물은 `seed × side × role × location × failure_type` 표, 유효한 경로 실패 사례와 fixture, planner 로직 수정과 관측 개선의 구분입니다.

## I06. 표현·좌표계·기억·시간 정렬 진단 — 높은 우선순위

연결: F07, F11, F18–F21 / 외부 E04

코드/자료: `ippo_model.py`, `mappo_v6.py`, `v7_1/model.py`, `v7_2/sensory.py`, `policy.py`, `decoder.py`, `dynamics.py`, v3 teacher replay·BC 기록(있을 때).

각 feature가 원 관측의 어디서 왔고 actor/critic 중 어디로 들어가는지 입력 정보 흐름표를 만듭니다. frozen encoder, 49차원 planner context, retina, six output pools, readout에서 heading/held type/class/score/time/목표가 보존·누락·간접 추론되는지 구분합니다. 원 관측에 없는 battery quantity는 ‘누락된 합법 feature’가 아니라 원래 unavailable로 표시합니다.

가능한 오프라인 probe: frozen/random/부분학습 encoder 또는 작은 raw-feature 모델이 목표 방향·held type·stuck·역할 등 합법적으로 정의 가능한 진단 label을 구분하는지 평가합니다. 훈련/검증은 trajectory 또는 map 단위로 나눠 인접 프레임 유출을 피합니다. probe 성공은 정책 개선 증명이 아니며 probe 실패도 정보 부재의 엄밀한 증명이 아닙니다.

egocentric-allocentric 검사는 같은 자기 시점 패턴·다른 heading의 일관된 fixture와 실제 이력 재생을 나눠 수행합니다. (a) 상대 행동→heading 변환, (b) heading sin/cos→readout 입력을 비교할 최소안을 제시합니다. 실제 상태가 heading을 간접 부호화할 수 있음을 확인합니다.

v7-2의 sensor refresh, brain tick, output history, action 선택 시점을 교차 상관/step response로 측정합니다. 13-tick history가 있다는 사실만으로 정확히 260ms의 행동 지연이 있다고 단정하지 않습니다. 감각에는 이미 있는 정보를 더하는 것과 새 memory/새 neural dynamics를 넣는 것을 별도 실험으로 구분합니다.

v3는 overall BC accuracy 대신 role별 confusion matrix, NoOp 비중, 이동 label 정확도, learner 분포에서의 label 품질, first divergence, teacher-forced vs closed-loop 결과를 찾습니다. 산출물은 표현 병목 가설별 증거와 최소 추가 입력/기억 대안입니다.

## I07. connectome이 실제 계산·행동·성능에 기여하는지 — 높은 우선순위

연결: F16–F17, F20, F22–F24, F34 / 외부 E05

코드/자료: connectome graph artifact/extraction/controls/sparse ops, `v7_1/model.py`, v7_2 전 경로, `logs/v7/reports/data*`, `unity_causality*`, `whole_brain_causality*`, `validate_v7_2_unity.py`, evaluate_v7.

부분회로: graph/rewired src,dst가 다른지, 실제 loaded buffer가 설정과 맞는지, input→output 경로와 실제 활성/gradient가 있는지, trainable weights가 변하는지, zero init 이후 변화가 downstream action까지 도달하는지 확인합니다. synapse count/transmitter가 metadata에만 있는지 실제 연산에 쓰이는지도 기록합니다. action으로 연결되지 않은 feature 차이를 성능 기여로 부르지 않습니다.

전뇌: population별 rate 평균/분산, 침묵·포화·진동 비율, six pools의 상관/유효 rank, readout weight/bias가 방향 하나/NoOp으로 붕괴하는지, 변화하는 감각과 관계가 있는지 확인합니다. degree/port/graph seed 통제를 증명할 원자료를 남깁니다.

학습된 F1을 대상으로 normal/frozen-frame/sensory-block/output-block/pool-shuffle을 별도 진단으로 설계합니다. 원래 F0 gate를 F1 성능 검증으로 대체하지 않습니다. 같은 정책·초기 상태·noise seed 조건을 쓰되, 행동이 달라진 후 궤적 분포가 달라졌다는 점을 표시합니다. 차단 조건이 evaluation entrypoint까지 전달되는지도 확인합니다. OOD 차단의 영향을 배선 고유 효과와 구분합니다.

한 슬롯 전뇌+네 슬롯 scripted의 성능은 동일한 네 teammate·동일한 slot·동일한 input을 받는 작은 대조군과 비교할 계획을 만듭니다. 더 많은 vector를 주는 대조군은 정보 확장 실험으로 별도 표시합니다. 대조군이 현재 없으면 실행된 결과처럼 적지 않습니다.

산출물: 구조→활성→출력→행동→성능의 각 단계별 증거, causal gate의 확인 범위, 최소 대조군과 비용, 전뇌 유지/분리/보류의 조건부 결론.

## I08. 평가 통계·진영/상대 편향·일반화 — 공통 필수

연결: F14, F25 / 외부 E06·E07

자료: split manifest, 23-run result, episode logs, counterpart opponent identity, v6/v7 evaluation protocol. 우선 I01의 지표 사용 가능 판정을 따른 뒤 집계합니다.

평가맵×진영×학습seed×모델의 결과 cube와 훈련맵×진영×상대의 노출/실패 cube를 만듭니다. A/B 할당의 episode 수 비율과 실제 수집 step 비율을 따로 계산합니다. 특정 상대/진영의 episode가 길면 update 노출도 달라질 수 있습니다. dev B 열세를 모든 세대·맵의 고정 성질로 가정하지 않습니다.

seed별 승/무/패와 평균뿐 아니라 어느 paired map/side에서 바뀌었는지를 제시합니다. 여러 학습 seed가 같은 60경기를 공유하는 종속성을 보존한 통계안을 외부와 합의합니다. v6의 10경기와 v7의 60경기를 바로 이어 붙이지 않습니다. A2/A3의 [0.5,0.5] 같은 구간은 입력 표본/계산법부터 확인하고 불확실성 0이라고 하지 않습니다.

test/confirmation 접근 이력, checkpoint 선택 시점, best vs latest 규칙, resume 뒤의 discarded episodes, 평가 중단/재실행/실패 누락 여부를 감사합니다. 맵 수준 일반화와 고정 상대에 대한 과적합은 분리합니다.

산출물: 통제 조건이 맞는 비교표, 독립 표본 단위, 신뢰구간/검정의 전제, 현재 확정 가능한 결론과 불가능한 결론. 비용 지표는 episode 길이에 영향을 받는다는 점도 설명합니다.

## I09. 실제 시간·재현성·회귀 테스트·제출 경로 — 최종 필수

연결: F26–F29, F31–F32 / 외부 E01·E07·E08

코드/자료: `logs/v7/reports/acceleration*`, `code/v7/scripts/accelerated_connectome.py`, `connectome_fast_runtime.py`, `connectome_metal.py`, export_v7/export_mappo_v6, artifacts/submission/policy, tests, actual runtime/lineage.

수집의 env/통신/parsing/planner/mask/encoder/CSR/readout, PPO update, checkpoint 저장, 평가를 각각 측정하고 end-to-end wall time을 제공합니다. short collector benchmark와 장기 실행, cold start와 steady state, 합산 처리량과 단일 정책 p95/p99, CPU/MPS 전송·동기화 비용을 분리합니다. profiling overhead와 장비/스레드/동시 프로세스 수를 기록합니다.

overlay/source fingerprint 범위에 실제 runtime 변경이 들어가는지, 원래/가속의 동일성 검증이 terminal·reset·여러 episode·resume까지 포함하는지 확인합니다. seed만 같고 물리 상태가 복구되지 않은 resume를 bitwise 연속 재개라고 하지 않습니다.

공식 loader를 확인한 후 동일 모델을 완전 격리 환경에서 로드하여 여러 경기 reset·입력 순서·중복 호출·시간 제한·의존성·외부 파일 접근을 시험할 계획을 만듭니다. 현재 export 차단을 제거하는 것으로 통과 처리하지 않습니다. 연구용 checkpoint가 허용된 입출력만으로 재현되는지를 판단합니다.

테스트는 이번에 실행한 명령·버전·개수·실패·skip을 저장하고 과거 62개 통과 기록과 구분합니다. 최소 regression 항목은 terminal snapshot, 정답 winner, 마지막 delta, 연속 timeout, graph/ckpt resolution, sampled log-prob ratio, action mask, 슬롯별 경로 실패 격리, train/eval intervention 전달, episode/step/reset, 가속 전후 parity입니다. actual Unity가 없는 테스트는 합성으로 표시합니다.

산출물: 시간 분해표, 비용 대 성능 비교 전제, fresh-clone 재현에 필요한 artifact/명령, export blocker와 regression matrix.

## 공통 신규 계측 제안 — 현재 존재한다고 가정하지 말 것

매 decision 전체를 영구 저장할 필요는 없습니다. 안전한 counter/histogram을 상시 남기고, 실패 전후 ring buffer와 고정 진단 trajectory에만 고해상도 기록을 남기면 계측 자체의 병목을 줄일 수 있습니다.

식별 필드: run_id, code/config/build/checkpoint/graph/opponent hash, train_seed, map_seed, episode_id, side, env_step_id, game_time, update_id, policy_state_version, runtime_backend.

정책 필드: p_KEEP, p_OVERRIDE, max_correction_probability, best_logit_margin, mask_count, mask_reason, sampled_joint_index, greedy_joint_index, behavior_exploration, old_log_prob, planner_action, requested_action, executed_displacement, slot, role, goal, path_status.

학습 필드: reward 성분, score 출처/단위, last_nonterminal_score, terminal_snapshot, winner_source, termination_reason, V/next_V, return/advantage, KL/clip/entropy, module별 gradient norm/parameter delta, actual minibatches, discarded_partial_episode.

전뇌 필드: heading, sensory_refresh_id/time, tick_id, input/pool summary, readout logits, intervention, state_reset_reason, rate/rank/saturation 요약.

## 최종 제출물

`artifact_inventory`, `root_cause_evidence`, `event_timeline`, `behavior_audit`, `minimal_repros`, `v8_decision_inputs`를 제출해주세요. 표는 Markdown/JSON, 큰 원자료는 필요한 형태로 보존하며 모든 집계에는 명령과 입력 hash를 붙입니다.

각 문제 카드에는 F/I/D ID, 현재 상태(미해결/수정됨/회귀/미검증), 증거 등급, 파일:함수:행/commit, run/checkpoint/map/episode/step, 기대값과 관측값, 가능한 원인과 반증, 영향 범위, 최소 재현/실패 테스트, 수정 제안, 외부에 필요한 자료, 우선순위와 완료 조건을 적어주세요.

각 주요 가설마다 대표 실패 trace와 가능하면 같은 조건의 성공/정상 trace를 함께 제시해주세요. 자료 부족은 ‘문제 없음’이나 ‘원인 확정’으로 바꾸지 말고 부족한 파일/필드/실험을 명시하세요. 최종 요약은 반드시 고칠 결함, 검증 후 바꿀 설계, 지금은 보류할 변경으로 나눠주세요.


---

## 부록: 공통 지침

# v8 조사 공통 지침 (사본)

작성일: 2026-09-27 KST
대상: thislis/2026-IST-tech-RL
출발 감사 snapshot: main@e6e98a63e9f401f6bf03e463a720c03878d1beb7

## 조사 목적

v8은 새 모델 이름을 정하는 작업이 아니라, 신뢰할 수 있는 평가 아래 기존 planner보다 유용한 행동을 학습하고, 그 결과를 제출 환경에서 재현하는 작업이다. 외부 조사자는 해결책의 근거와 적용 조건을, 내부 분석자는 실제 실패의 기전과 반증 자료를 마련한다. 두 결과를 같은 의사결정 ID로 연결한다.

이 요청서는 이전 감사에 기반한 조사 배정이다. 이번 작성에서 새 경기·재학습·원본 로그 재집계를 수행한 것은 아니다. reference/previous_audit.md의 수치도 원래의 확인 수준을 유지한다. 특히 36개 항목은 모두 확정 버그가 아니라 코드 결함, 보고된 현상, 관측 제약, 연구 가설, 검증 공백을 함께 포함한다. 최신 checkout이 다르면 변경분을 먼저 기록한다.

## 공통 원칙

- v8 승률 개선과 생물학적 배선의 고유 효과 입증을 별도의 목표로 둔다. connectome을 반드시 유지한다는 전제로 조사하지 않는다.
- 외부 논문의 성공은 BlackOut에서의 성공 증거가 아니다. 내부 로그의 상관관계는 원인 증거가 아니다.
- 과거에 수정된 결함은 수정 버전과 회귀 검사를 적고, 현재 결함과 분리한다.
- 오염된 terminal score를 최종 점수로 사용하지 않는다. last-nonterminal 점수는 preterminal이지 최종 득점 snapshot이 아니다. 기존 winner도 독립 검증 전에는 출처를 표시한다.
- 기존 결과·checkpoint·registration을 덮어쓰지 않는다. 원래 소스/환경의 진단과 수정된 환경에서의 재평가는 별도 ID로 둔다. 빌드 변경 시 모든 대조군을 새 빌드에서 다시 맞춘다.
- 기존 train/dev로 조사하고, 최종 test의 seed·경기 내용을 원인 분석이나 설계 선택에 사용하지 않는다. confirmation도 반복 선택에 쓰면 더 이상 독립 확인셋이 아님을 기록한다.
- 공식 actor 입력, 학습 전용 critic 입력, 진단 전용 엔진 정보의 권한을 분리한다. 배터리 수량처럼 현재 관측에 없는 정보를 정답 feature로 몰래 추가하지 않는다.
- 동일 map seed는 동일 중간 물리 상태나 동일 난수 소비를 보장하지 않는다. 진짜 반사실 분기에는 환경·정책·RNG 전체 snapshot 또는 동등성이 검증된 재현이 필요하다.
- 상대/역할/진영/맵/학습 seed/종료 종류별 분모를 명시한다. env step, agent action, game second, wall second를 구분한다.
- 새 장기 학습이나 대규모 sweep보다 기존 artifact 확인 → 오프라인 진단 → 합성 fixture → 제한된 독립 재평가를 먼저 한다. 조사 단계의 예산과 수정 권한은 프로젝트 책임자와 별도로 정한다.

## 서로 교환할 것

내부 분석자는 먼저 artifact 목록, 신뢰 가능한 지표, 최소 실패 사례, 부족한 계측을 외부 조사자에게 전달한다. 외부 조사자는 문제별 대안 두 개 이내, 필요한 관측/실행 계약, 예상 장단점, 반증 실험을 돌려준다. 최종적으로 03_joint_decision_handoff.md의 결정표를 같이 채운다.

논문을 많이 모으거나 로그를 많이 그리는 것이 완료 조건은 아니다. 각 조사 항목에서 ‘이 증거가 나오면 v8의 무엇을 바꿀지, 무엇은 바꾸지 않을지’가 적혀 있어야 한다.
