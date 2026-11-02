# v12: 왕복 작업을 함께 배정하는 경량 팀 정책

**새 게임 풀리그 결과 (2026-11-02)**: [한눈에 보는 HTML 보고서](../../logs/shared/agent_league/v12_full_league_new_game_20261101/analysis.html). 실행 종료, **2,294/2,310경기 완료**, worker 오류로 16경기는 미완료다. 최종 점수 기준 v12는 **188승 0무 20패, 잠정 1위**다. 전체 순위·상대별 성적·경기 검색·미완료 목록을 제공한다. v12 참가자는 추가 학습 전 고정 모델 `v12-before-learned`다. 보고서 작성 중 재실행하지 않았다. [실행·상태 확인](reports/full_league_new_game.md) · [빌드 기록](reports/new_game_build.md).

**2026-11-01 새 게임·API 문서 반영**: [새 환경·ONNX 제출 계약](../common/environment_update_2026-11-01.md)을 확인했다. Hunter 2명 상한은 이제 엔진 규칙이고, 공식 판정은 최종 게임 점수다. 기존 v12 Python 정책·부분 ONNX·2,298경기 리그 결과는 새 버전 이식 또는 공식 제출 검증을 완료한 것이 아니다. 이번에는 문서만 갱신했다.

**2026-10-29 풀리그 결과**: [인터랙티브 HTML 보고서](../../logs/shared/agent_league/v12_full_league_20261028/analysis.html). 2,298/2,310경기 완료, 12경기는 시간 제한으로 미완료다. 누적 보상 기준 v12는 179승 31패, 잠정 6위다. 전체 순위·상성 지도·경기 검색·행동 통계와 공식 종료 판정 비교를 제공한다.

**2026-10-28 풀리그 오류 수정**: **22개·2,310경기, 최대 16개 병렬 평가**가 사용자 실행 후 75경기 완료 상태에서 heartbeat 파일 접근 오류로 중단됐다. 읽기·쓰기 재시도와 재개 호환 처리를 수정하고 기존 결과를 보존했다. 직접 재실행하지 않았다. `powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v12\run_league.ps1 --resume`으로 재개한다. [수정 내역·검증·실행 안내](reports/full_league_windows.md).

**2026-10-27 수정 실행**: 기존 추가 학습을 3,057,458스텝에서 중단하고 [학습 기반 작업·회피 선택 모드](reports/learned_control.md)를 준비했다. `powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v12\finetune_learned.ps1`로 직접 시작한다. 새 실험은 아직 실행하지 않았다.

규칙 기반 9종 대상 추가 학습은 [전용 실행 안내](reports/finetune_scripted.md)를 참고한다. `powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v12\finetune_scripted.ps1` 한 줄로 시작·재개하며, 기본값은 실제 학습 창 1개와 백그라운드 15개, CUDA learner다. 기존 완료 실험과 별도 경로를 사용한다.

작성일: 2026-10-26. Python 정책·학습·실행기를 구현했다. 이후 사용자 실행으로 기준 성능 평가를 완료했고, 공통 데이터 수집 단계에서 실측 처리량에 따른 병렬 실행 개선을 적용했다. 현재 진행 상태는 `--status`, 자원 변경과 보존 범위는 [재개 기록](reports/runtime_resume.md)을 참고한다.

v12는 v4·v6의 안정된 수집·귀환·경로 추종을 유지하면서, **편도 거리로 후보를 자르는 방식과 순차 배정의 기회비용**을 보완한다. 5개 유닛의 작업을 작은 공유 MLP로 평가하고, 부분집합 동적 계획법으로 팀 전체 배정을 선택한다. Attention·GRU·이동 방향 보정 신경망은 기본 모델에서 제외한다.

- [풀리그 재분석과 선택 근거](reports/league_findings.md): 상위권 50개 진영 교대 쌍의 결과 일치, 후보 생성·배정의 맹점, 해석 한계.
- [v12 설계](plans/v12_design.md): 입력 계약, 왕복 작업, 6,273개 파라미터 scorer, 정확한 공동 배정 확률, 경로 계산 예산.
- [구현·검증 계획](plans/v12_implementation_plan.md): 단계별 산출물, 대조군, 학습·평가 예산, 채택 조건.
- [구현 기록과 실행 방법](reports/implementation.md): 실제 구현 범위, CPU/CUDA 배분, 짧은 검사 결과와 남은 한계.

v11은 목표 유지와 기억을 갖춘 표현 학습이 중심이다. v12는 **같은 기본 제어 위에서 어떤 작업을 누구에게 배정해야 팀 전체 손해가 줄어드는지**를 직접 다룬다. v11도 왕복 거리·역할 변경을 설계에 포함하므로, 그 개념을 v12가 처음 도입했다고 주장하지 않는다.

기본 scorer는 실제 구현에서 6,273개 파라미터다. FP32 가중치의 산술 크기는 25,092 bytes이며 v11 전체 actor 194,327개 대비 약 96.8% 적다. 파일 전체 크기·경로 탐색·캐시·ONNX 그래프·전처리 비용은 별도 항목이다. 작은 신경망만으로 실제 게임 처리량이나 승률 개선을 주장하지 않는다.

구현은 `code/v12/`, 자동 기록은 `logs/v12/`, 데이터·제출물은 `artifacts/` 하위에 둔다. [프로젝트 배치 규칙](../common/project_layout.md)을 따른다. 기존 버전 정책·게임/API/관측 전처리·등록 실험은 수정하지 않았다. 공통 `project_paths.py`에는 v12 패키지 검색 경로만 추가했다.

## 실행

프로젝트 루트의 PowerShell에서 다음 한 줄을 실행하면 실제 파이프라인을 시작한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v12\run_v12.ps1 --name joint_trip_v12
```

Bash에서는 `bash code/v12/run_v12.sh --name joint_trip_v12`를 사용한다. 명령 뒤에 `--check`를 붙이면 준비 검사만, `--dry-run`은 계획만, `--benchmark`는 짧은 합성 자원 측정만 수행한다. `--status`는 상태 조회, `--stop`은 중단 요청이다. 기본 실행은 창 없는 백그라운드 프로세스다.

최초 학습 등록 당시 과거 v4/v6 등 checkpoint가 없어 `available` 범위로 scripted 상대와 v12 self-play를 사용했다. 그 등록과 결과는 그대로 보존한다. 2026-10-28에는 전달받은 패키지에서 고정 21개 참가자 자산을 복원했으며, 별도 평가 전용 `run_league.ps1`로 전체 풀리그를 준비했다. 과거 학습 결과를 새로 수행한 정식 21상대 평가로 해석하지 않는다.

설계의 전체 관측·상태 ONNX는 아직 완성되지 않았다. 개발 산출물은 scorer+공동 MAP 배정의 ONNX이며, Python 호스트가 명시 상태와 경로 처리를 소유하고 event에서만 호출한다. 제출용 두 파일 Python 정책과 개발 ONNX의 지원 범위를 구분한다.

## 중단된 실험 재개

`joint_trip_v12`는 사용자 요청으로 중단하고, 기존 완료 경기와 원래 등록을 보존한 상태로 실행 자원 변경을 준비했다. `powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v12\run_v12.ps1 --name joint_trip_v12 --resume`로 이어간다. Bash는 `bash code/v12/run_v12.sh --name joint_trip_v12 --resume`다.

최초 재개 때는 4·6·8 worker를 비교했다. 이후 사용자의 추가 승인으로 8개부터 2개씩 상한 없이 늘리는 실측 기능과 데이터 수집 병렬화를 추가했다. 이미 측정한 설정은 runtime revision으로 보존하므로 일반 `--resume`은 측정을 반복하지 않는다. 완료 경기 재사용·중단 경기 재시도·학습 묶음 보존은 [재개 기록](reports/runtime_resume.md)을 참고한다.
