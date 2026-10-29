# BlackOut 강화학습 실험

코드, 문서, 실험 기록을 실제 디렉터리로 분리했습니다. 이전 위치에 호환 링크는 두지 않습니다.

| 경로 | 내용 | Git 공유 |
|---|---|---|
| `code/v2/` ~ `code/v10/` | 버전별 구현·스크립트·설정·테스트 | 예 |
| `code/v11/` | 목표·기억·planner 정책, recurrent PPO, CPU/MPS 자동 선택·백그라운드 실험 | 예 |
| `code/v12/` | 왕복 작업·공동 배정 DP·event PPO, CPU/CUDA 측정·병렬 실행 | 예 |
| `code/shared/` | 여러 버전이 사용하는 Python 모듈·도구·의존성 목록 | 예 |
| `code/pre_v1/` | 초기 구현·실험 코드 | 예 |
| `docs/` | 버전별 계획·분석·직접 작성한 보고서 | 예 |
| `docs/common/` | 실험 이력·게임 및 제출 규격·공통 문서 | 예 |
| `logs/<버전>/` | 실행 로그·자동 생성 보고서·실험 등록 기록 | 아니요 |
| `artifacts/` | 가중치·데이터·원본 게임 빌드·제출 산출물·과거 소스 보관본 | 아니요 |
| `.venv/`, `.unity/` | 로컬 실행 환경 | 아니요 |

v1 당시 구현은 후속 구현으로 발전했으며, 현재 별도 `code/v1/` 디렉터리는 없습니다. 공통 구현은 `code/shared/`에서 확인합니다. 이슈 문서는 `docs/issues/`, 초기 보고서는 `docs/pre_v1/reports/`에 있습니다.

[전체 Agent 풀리그](docs/common/agent_league.md) · [실험 이력](docs/common/history.md) · [v9 안내](docs/v9/README.md) · [v9 계획](docs/v9/plans/v9_plan.md) · [v10 안내](docs/v10/README.md) · [v10 구현 계획](docs/v10/plans/v10_plan.md) · [제출 규격](docs/common/blackout_last_4_pages.md) · [재배치 안내](docs/common/project_layout.md)

[v11 안내](docs/v11/README.md) · [v11 리그 분석](docs/v11/reports/league_findings.md) · [v11 설계](docs/v11/plans/v11_design.md) · [v11 구현 계획](docs/v11/plans/v11_implementation_plan.md) · [운영진 답변 반영](docs/common/organizer_qa_2026-10-24.md)

[v12 안내](docs/v12/README.md) · [v12 풀리그 재분석](docs/v12/reports/league_findings.md) · [v12 경량 공동 배정 설계](docs/v12/plans/v12_design.md) · [v12 구현·검증 계획](docs/v12/plans/v12_implementation_plan.md)

## 실행

**v12 포함 22개 풀리그 (Windows)**: `powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v12\run_league.ps1 --resume`. 기존 21개 + 선택한 v12가 총 2,310경기를 최대 16개 병렬로 평가한다. 2026-10-28 사용자 실행에서 75경기 완료 후 발생한 heartbeat 접근 오류를 수정했고, 기존 결과를 보존해 재개 준비만 했다. 직접 재실행하지 않았다. [참가 모델·오류 수정·재개 안내](docs/v12/reports/full_league_windows.md).

**현재 준비된 v12 수정 실행**: `powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v12\finetune_learned.ps1`. 기존 추가 학습은 3,057,458스텝에서 중단했고, 그 가중치를 보존해 수동 prior·강제 작업 유지·근접 회피 override를 제거한 새 학습 모드를 준비했다. 새 실험은 아직 실행하지 않았다. [변경 범위와 한 줄 실행 안내](docs/v12/reports/learned_control.md).

**v12 규칙 기반 9종 추가 학습**: `powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v12\finetune_scripted.ps1` 한 줄로 시작·재개한다. 학습 경기 1개 표시 + 15개 백그라운드 수집, CUDA PPO를 사용한다. 초기·중간 평가는 창 없이 수행한다. [설정·승률 목표·상태 조회 안내](docs/v12/reports/finetune_scripted.md).

프로젝트 루트에서 다음 명령을 사용합니다. 다른 작업 디렉터리에서는 스크립트의 절대 경로를 사용하세요.

```bash
# v12 전체 파이프라인: 현재 사용 가능한 상대, 짧은 자원 측정 후 백그라운드 실행
bash code/v12/run_v12.sh --name joint_trip_v12

# v12 준비 검사 / 실행 계획만 확인 (Unity·학습 시작 안 함)
bash code/v12/run_v12.sh --check
bash code/v12/run_v12.sh --dry-run

# v11 전체 실험: 작업별 CPU/MPS 측정 후 창 없는 백그라운드 실행
bash code/v11/run_v11_fast.sh --name goal_memory_v1

# v11 상태 확인 / 핵심 검사
bash code/v11/run_v11_fast.sh --name goal_memory_v1 --status
bash code/v11/run_v11_fast.sh --check

# 전체 Agent 풀리그: 구조·설정별 대표, 쌍당 양 진영 5세트
bash code/shared/run_agent_league.sh --name all_agents_v1

# 참가 목록만 확인: Unity 실행 없음
bash code/shared/run_agent_league.sh --list

# v10 전체 비교 실험: 창 없는 백그라운드 실행
bash code/v10/run_v10_fast.sh --name mapcritic_v1

# v10 준비 검사: 실제 Unity·학습을 시작하지 않음
bash code/v10/run_v10_fast.sh --check

# 선택 모델 생성 후 v10 대 v9 관전 (게임 창 표시)
bash code/v10/watch_v10_models.sh --study mapcritic_v1 --opponent v9

# v9 상태 확인: 학습을 시작하지 않음
bash code/v9/run_v9_fast.sh --status

# v9 준비 상태 검사: Unity와 학습을 시작하지 않음
bash code/v9/run_v9_fast.sh --check

# 새 소스 등록으로 별도 실험을 백그라운드에서 시작
bash code/v9/run_v9_fast.sh --name attention_relocated_v1

# v6와 v7-1 모델 관전 (게임 창 표시)
bash code/shared/watch_best_models.sh

# v9 회귀 테스트: 버전별 패키지 경로를 자동 설정
.venv/bin/python code/run.py -m unittest tests.v9.test_v9
```

Windows PowerShell에서는 `powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v12\run_v12.ps1 --name joint_trip_v12` 한 줄로 실행합니다. 상태 조회는 같은 명령 뒤에 `--status`, 중단 요청은 `--stop`을 붙입니다. v12의 실행 범위·자원 배분·미측정 항목은 [v12 안내](docs/v12/README.md)와 [구현 기록](docs/v12/reports/implementation.md)을 참고하세요.

`joint_trip_v12`를 중단한 뒤 이어가려면 다음 한 줄을 사용합니다. 저장된 runtime revision의 자원 설정으로 완료 결과와 체크포인트를 재사용해 백그라운드로 재개합니다. 2026-10-26 추가 요청에 따라 데이터 수집의 2-worker 제한을 해제하고, 실제 처리량을 8개부터 2개씩 늘려 측정하는 기능을 추가했습니다. 확정된 설정과 측정 결과는 아래 변경 기록을 참고하세요.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v12\run_v12.ps1 --name joint_trip_v12 --resume
```

`--resume` 대신 `--resume-check`를 붙이면 실행 없이 준비 상태만 확인합니다. [worker 변경과 재개 기록](docs/v12/reports/runtime_resume.md)에 보존 범위와 변경 이력을 정리했습니다.

전체 버전의 테스트는 프로젝트 루트에서 아래처럼 각 테스트 디렉터리를 순회합니다. `discover -s code/shared/tests`만 실행하면 공통 테스트만 수집됩니다.

```bash
for test_dir in code/*/tests; do
  .venv/bin/python code/run.py -m unittest discover -s "$test_dir" -v || break
done
```

실행기는 필요한 Python 검색 경로를 자동으로 설정합니다. `python -m ...`을 직접 실행할 때는 `code/run.py -m ...`을 사용합니다. 개별 `code/<버전>/scripts/*.py`도 직접 실행할 수 있습니다.

기존 등록 실험은 당시 소스·설정 해시를 유지합니다. 재배치로 소스 해시가 달라졌으므로 과거 등록을 덮어쓰거나 같은 이름으로 이어서 학습하지 마세요. 기존 완료 결과는 상태 조회와 기록 열람으로 확인할 수 있습니다.

새 clone에는 가중치·게임·데이터·Python 환경이 없습니다. 해당 로컬 자산을 준비해야 실제 실행이 가능합니다. 제공 게임 및 API/obs 코드는 변경하지 않습니다.
