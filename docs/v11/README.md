# v11: 목표·기억을 사용하는 팀 정책

v11은 tensor planner, 작은 Attention, 목표·역할 선택, unit/team GRU, 제한된 방향 보정, 독립 지도 critic을 결합한다. 코드와 전체 실험 실행기를 구현했으며, 학습된 최종 후보의 성능은 P0–P6 실험 결과로 판단한다.

## 실행

중단된 `goal_memory_v3`은 다음 한 줄로 이어서 실행한다. 승인된 평가 축소는 `evaluation_revision.json`으로 검증하며, 완료된 학습 checkpoint와 baseline·pilot 평가는 재사용한다.

```bash
bash /Users/safeailab_macmini/Desktop/2026-IST-tech-RL/code/v11/run_v11_fast.sh --name goal_memory_v3 --resume
```

명령은 백그라운드 supervisor를 시작하고 셸로 돌아온다. 게임 창을 숨긴 상태에서 graphic 관측을 위한 렌더링은 유지한다. 전체 순서는 필수 검사 → CPU/MPS·worker 측정 → planner 기준 비교 → 교사 자료 수집 → pilot → 본 비교 → 추가 seed 확인 → 동결 평가·패키징이며, 재개 시 완료된 단계는 건너뛴다. 학습 전 G1과 pilot G2의 성능 미달은 `passed: false, blocking: false`로 기록하고 예정된 학습을 계속한다. 본학습 G3의 진행 조건은 유지한다. 이를 통과한 뒤 추가 seed의 confirmation 성능은 재현성 진단으로 기록하고, 성능 미달이어도 최종 평가를 진행한다. 실행·데이터 오류는 중단 조건이며, 최종 G4·G5·G6의 기준과 미달 판정은 유지한다.

새 실험은 사용하지 않은 이름으로 `bash code/v11/run_v11_fast.sh --name <새_이름>`을 실행한다. 이 문서 수정 과정에서는 실험을 재개하지 않는다.

같은 한 줄로 실험 종료 후 상황별 행동 보고서(`behavior_report.json`, `behavior_report.md`) 생성까지 진행한다. 정상 종료뿐 아니라 중단 요청이나 오류로 종료할 때도 저장된 경기 요약으로 부분 보고서를 생성한다. 보고서 생성 상태는 `behavior_report_status.json`에 남기며, 보고서 오류가 원래 실험 결과를 덮어쓰지 않는다.

```bash
bash code/v11/run_v11_fast.sh --name goal_memory_v3 --status
bash code/v11/run_v11_fast.sh --name goal_memory_v3 --stop
bash code/v11/run_v11_fast.sh --name goal_memory_v3 --resume

# Unity 없이 핵심 검사와 두 파일 제출물 로딩 확인
bash code/v11/run_v11_fast.sh --check

# 짧은 숨김 실행과 실제 학습 업데이트 검사
bash code/v11/run_v11_fast.sh --name v11_smoke --smoke

# 장치·worker 설정만 측정; 전체 학습은 시작하지 않음
bash code/v11/run_v11_fast.sh --name v11_profile --benchmark
```

일반적인 소스·설정·패키지 변경에는 새 `--name`을 사용한다. 이번 승인된 평가 변경은 예외로 원본 `registration.json`을 보존하고 `evaluation_revision.json`에 허용한 변경과 source override를 기록한다. 기존 checkpoint와 등록 hash를 바꾸어 재개 검사를 우회하지 않는다. 중단 시 완료된 optimizer commit과 평가 경기는 보존한다. 진행 중 Unity 물리 상태는 복원할 수 없으므로 미완료 episode를 폐기하고 사용한 step을 별도로 집계한다.

## CPU와 MPS

actor, planner, learner 장치를 독립적으로 선택한다. 실제 실행 환경에서 MPS를 사용할 수 있으면 CPU/MPS의 전송·동기화·forward/backward 비용을 포함해 측정하고, 처리량이 개선되는 작업에 MPS를 사용한다. CPU 추론과 MPS 학습을 함께 사용할 수 있다.

학습 배치는 64-event recurrent 구간의 backward까지 측정한다. Unity worker 2/4/8 중 메모리·디스크 조건을 만족하는 설정을 실제 경기 처리량으로 비교한다. 처리량 차이가 5% 이내면 적은 worker를 선택한다. 기본 RAM soft limit은 32GiB, 디스크 여유는 25GiB다. 선택 결과는 `runtime_profile.json`에 저장하고 본 비교에서 고정한다.

필요하면 `--device cpu|mps|auto`, `--actor-device cpu|mps|auto`, `--planner-device cpu|mps|auto`로 측정 후보를 제한할 수 있다. 기본값은 모두 `auto`다.

## 실험과 산출물

기본 teacher·pilot·본학습·추가 seed 확인은 **34,340,864 env step**이다. baseline·평가·실패 시도·완결 wave 초과량은 별도다. P는 고정 planner, R은 41-way 보정, G는 목표/기억, F는 전체 정책, D는 같은 표현과 기억을 쓰는 직접 이동 정책이다.

평가만 다음과 같이 줄인다. 완료된 baseline·pilot 결과는 그대로 재사용하고, R/G/F/D × seeds 11/22/33의 본학습 및 선택한 두 arm × seeds 44/55의 추가 학습은 유지한다. 각 선택 arm의 학습 seed는 총 5개다.

| 남은 평가 | 구성 | 경기 수 |
|---|---|---:|
| 본학습 dev | 12 runs × 15상대 × 5 map pairs × 2진영 | 1,800 |
| Confirmation 진단 | 2 arms × 5 seeds × 핵심 5상대 × 10 map pairs × 2진영 | 1,000 |
| 최종 선택 checkpoint | 전체 21상대 중 일반 19상대는 10 map pairs, v4·v6는 각각 30; 양 진영 | 500 |
| 최종 v4 기준 모델 | 같은 공통 상대·map pairs, 자기 자신인 v4 제외 | 440 |
| 합계 | G3 통과 등 진행 조건 충족 시; 완료된 baseline·pilot 제외 | **3,740** |

Confirmation의 핵심 상대는 noop·random·collector·v4·v6다. 최종 제출 checkpoint 하나를 dev에서 선택해 고정하며, confirmation이나 test 성적으로 seed를 다시 고르지 않는다. 다섯 seed의 재현성 진단과 선택 checkpoint의 최종 G4·G5·G6 판정은 구분한다. 평가량을 줄였다고 최종 기준을 낮추거나 미달을 통과로 처리하지 않는다.

| 경로 | 내용 |
|---|---|
| `logs/v11/<name>/status.json` | 단계·진행량·메모리·실행 상태 |
| `logs/v11/<name>/runtime_profile.json` | 작업별 CPU/MPS·thread·batch·worker 측정과 선택 |
| `logs/v11/<name>/evaluation_revision.json` | 원본 등록을 보존한 승인 평가 변경·source override의 추적 기록 |
| `logs/v11/<name>/runs/` | 학습 checkpoint·episode·loss·상대별 실제 step 노출 |
| `logs/v11/<name>/evaluation/` | 상대·진영·seed별 승패, 관측 점수, 실제 map hash |
| `logs/v11/<name>/gates/` | 다음 단계 진행 여부와 근거 |
| `logs/v11/<name>/behavior/episodes/` | 경기 상황별 행동 통계, 스텝별 압축 trace, 목표·역할 결정 |
| `logs/v11/<name>/behavior_report.json`, `.md` | 실험 종료 시 자동 생성; `--report`로 중간 집계·재생성 가능 |
| `logs/v11/<name>/summary.json` | 최종 결과 또는 gate 미달 이유 |
| `artifacts/data/v11/` | 압축 교사 replay |
| `artifacts/submission/v11/<hash>/` | 최종 `policy.py`, `checkpoint.pt` |

최종 정책은 `MyPolicy(vector_size=96, n_channels=11)`의 `forward(vector[B,96], graphic[B,11,96,96]) → action[B,2]`를 유지한다. 현재 제공된 runner의 팀당 canonical 다섯 행 순서로 자기 개체를 연결한다. 외부 ID·state 입력은 요구하지 않는다. 임의 순서나 일부 행만 전달하는 별도 실행기는 이 팀 정책의 지원 계약에 포함되지 않는다.

ONNX 개발 그래프는 동일 core에 명시적 상태 입출력을 사용하며 `development_onnx/`에 저장한다. 두 파일 제출 경로와 분리하고, ONNX 호출자가 상태 출력을 다음 입력으로 전달해야 함을 manifest에 명시한다. sampled 제출 후보를 선택한 경우 deterministic ONNX와 같은 decoder라고 보고하지 않는다.

추가 비교는 기본 실행에 포함하지 않는다. 지원하는 선택 실험은 `no_gru`, `vectorcritic`, `blindcritic`, `max2`, `max5`, `goal10`, `goal50`이다. 각각 F 기준군과 변경군을 동일한 3 seeds × 524,288 step으로 비교한다.

```bash
bash code/v11/run_v11_fast.sh --name v11_max2 --ablation max2
```

새 환경에서 export 의존성이 없다면 기존 Torch·NumPy·protobuf를 유지하며 추가한다.

```bash
uv pip install --python .venv/bin/python --no-deps -r code/v11/requirements-export.lock
```

## 근거 문서

- [설계](plans/v11_design.md), [구현·실험 계획](plans/v11_implementation_plan.md)
- [리그 분석](reports/league_findings.md), [계획 검토](reports/plan_review.md)
- [구현·실제 실행·MPS 측정 결과](reports/implementation.md)
- [제출 I/O](../common/blackout_last_4_pages.md), [운영진 답변](../common/organizer_qa_2026-10-24.md)

기존 리그 감사 도구는 `.venv/bin/python code/v11/analysis/audit_league.py`로 별도 실행한다. 기본 v11 실행에서 과거 2,100경기를 다시 감사하지 않는다.

## 경기 상황별 행동 기록

교사 수집·학습·평가에서 양 팀의 상황별 행동을 기록한다. 리드/동점/열세, 초반/중반/후반, 자원 운반 여부, 적·보관소 근접 여부와 개체 종류별로 정지·방향 전환·이동 및 수집·납품·재등장 추정치를 집계한다. 추정 이벤트는 환경의 공식 이벤트 판정과 구분한다. v11의 목표·역할·방향 보정 결정도 남기며, 교사 수집에서는 제안한 행동과 실제 실행한 교사 행동을 구분한다.

상세 trace는 풀리그의 압축 형식으로 관측 벡터·실제 행동·보상·다음 관측·종료 상태·누적 보상·추론 시간·지도 변화 통계를 저장한다. 지도 이미지는 시작·종료 및 1,000스텝마다 저장하므로 모든 프레임의 영상 재생 자료는 아니다. 경기당 상세 기록은 기본 64MiB, 실험 전체 상세 trace는 8GiB로 제한한다. 전체 한도를 넘으면 종료된 오래된 trace부터 정리하고 경기 요약과 상황별 집계는 유지한다. 정리 또는 기록 상한으로 조회할 수 없는 스텝은 명시한다.

```bash
# 실행 중에도 완료된 경기의 행동 통계를 집계
bash code/v11/run_v11_fast.sh --name goal_memory_v3 --report
# episode는 behavior/episodes/ 아래 디렉터리 이름
.venv/bin/python code/v11/scripts/v11_behavior.py --name goal_memory_v3 --episode <episode> --step 100
```

위 명령의 `goal_memory_v3`은 현재 실험명이다. 기존 `goal_memory_v1`에서 출처 명세와 함께 재사용한 완료 600경기에는 새 상세 행동 기록이 없으며, 보고서에서 기록이 없는 재사용 경기로 구분한다. 이번 평가 변경에서도 완료된 baseline·pilot과 기존 행동 요약을 보존하고, 재개 후 새로 실행하는 경기부터 기록을 추가한다.
