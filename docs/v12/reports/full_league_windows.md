# v12 포함 대표 모델 풀리그 — Windows 실행 준비

**2026-10-29 결과**: [HTML 보고서 열기](../../../logs/shared/agent_league/v12_full_league_20261028/analysis.html). 09:22:35(KST)에 2,298/2,310경기 완료 상태로 평가가 종료됐고, 12경기는 30분 제한으로 재시도까지 실패했다. HTML은 이 시점의 정적 결과이며 새 경기 실행 시 자동으로 갱신되지 않는다. 전체 220경기의 누적 보상/공식 종료 판정 불일치를 함께 표시하고 두 기준을 전환해 비교할 수 있다.

2026-10-28. 외부 `blackout_league_portable_20261028/setting.md`의 평가 규약을 따라 자산을 가져오고 Windows 실행기를 준비했다. 준비 당시에는 실행하지 않았으며, 이후 사용자 실행으로 75경기를 완료한 뒤 13:49:53(KST)에 파일 접근 오류로 중단됐다. **오류 수정과 재개 준비를 마쳤고 직접 재실행하지 않았다.** 이 실행기는 평가 전용이며 가중치를 업데이트하지 않는다.

## 2026-10-28 heartbeat 접근 오류 수정과 재개

완료된 **75/2,310경기**와 원래 manifest를 바이트 단위로 보존했다. 남은 2,235경기는 다음 명령으로 직접 재개한다. 기존처럼 최대 16개 병렬이며 모델·평가 기준·대진·seed는 같다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v12\run_league.ps1 --resume
```

실행 없이 준비 상태만 확인하려면 `--resume` 대신 `--resume-check`를 사용한다. 현재 확인 결과는 `Resume ready: 75/2310 completed, 2235 pending. No game launched.`다. 과거 오류가 담긴 `status.json`은 보존했으므로 실제 재개 전에는 `--status`에 이전 `failed` 상태가 표시된다.

오류는 worker가 `heartbeat.json`을 원자적으로 교체하는 작업의 `WinError 5`와 supervisor가 같은 파일을 읽는 작업의 `PermissionError` 양쪽에서 발생했다. 읽기·교체 시 Windows 공유/접근 오류에 한해 최대 8회, 합계 0.71초의 대기 시간으로 재시도하도록 수정했다. 파일 권한 설정이나 충돌을 일으킨 외부 프로세스를 특정한 것은 아니다.

재시도 후에도 heartbeat를 읽지 못하면 마지막으로 읽은 시각을 사용한다. 실패한 읽기를 새 heartbeat로 취급하지 않으므로 지속적인 장애는 기존 120초 watchdog으로 해당 경기만 종료한다. worker도 일시적인 heartbeat 저장 실패는 로그를 남기고 다음 갱신 때 다시 시도한다. 경기 결과 저장 실패·디스크 부족·손상된 JSON 등 다른 오류를 정상 결과로 처리하지 않는다.

추가로 Unity가 실행 중 갱신하는 `Blackout_Data/ML-Agents/Timers/Prototype_timers.json`이 원래 빌드 해시에 포함되어 있었다. 이 **정확한 진단 파일 하나만** 빌드 입력 검사에서 제외했다. 실행 파일과 나머지 빌드 자산 검사는 유지한다.

원래 등록을 덮어쓰는 대신 [runtime_revision.json](../../../logs/shared/agent_league/v12_full_league_20261028/runtime_revision.json)에 원본 manifest 해시, 수정된 I/O·supervisor·worker 세 파일의 전후 해시, 진단 파일 제외를 기록했다. 재개 시 이 기록에 일치하는 변경만 허용하고 그 밖의 모델·코드·환경 변경은 거부한다. 새 실행 세션에도 이 기록을 복사한다. 이전 소스는 `artifacts/source_snapshots/v12_league_heartbeat_20261028/`에 보관했다.

검증은 일시적 읽기/교체 실패, 재시도 소진 시 기존 JSON 보존, 다른 오류 전파, heartbeat 장애 시 watchdog 유지, 수정 등록과 무관한 변경 거부 등 **7개 오프라인 검사** 및 실제 저장된 75경기의 재개 준비 검사만 수행했다. Unity·학습·평가 경기는 실행하지 않았다.

- [오류 주입 검사 결과](../../../logs/v12/league_heartbeat_fix_20261028/tests.json)
- [완료 경기·원래 manifest 보존 및 재개 검사](../../../logs/v12/league_heartbeat_fix_20261028/resume_validation.json)

## 한 줄 실행

프로젝트 루트 PowerShell에서 실행한다. 백그라운드 supervisor가 시작되고 PID·로그 경로를 출력한 뒤 터미널로 돌아온다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v12\run_league.ps1
```

기본 이름은 `v12_full_league_20261028`이다. 중단 후 같은 명령을 다시 실행하면 완료 경기를 건너뛰고 미완료 경기부터 재개한다. 이미 실행 중이면 중복 실행하지 않고 상태를 보여준다. 코드·모델·게임 빌드가 바뀌면 기존 등록 재개를 거부한다. 변경된 조건의 별도 리그는 `--name 새로운이름`을 사용한다.

```powershell
# 현재 상태만 조회
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v12\run_league.ps1 --status
# 새 경기 배정을 멈추고 진행 중인 경기 종료까지 기다림
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v12\run_league.ps1 --stop
# 명시적 재개 (같은 명령을 옵션 없이 실행해도 재개)
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v12\run_league.ps1 --resume
```

Git Bash에서도 `bash code/v12/run_league.sh`를 사용할 수 있다. 게임 실행 없이 목록은 `--list`, 등록은 `--prepare`, 로딩·합성 관측 추론 검사는 `--check`다. 실제 실행 중인 경기에는 최대 1,800초 watchdog이 적용되므로 `--stop`은 즉시 모든 프로세스를 종료하는 명령이 아니다. Windows Job과 소유한 PID 트리로 각 실행의 자식 프로세스를 정리한다.

## 참가자와 경기 수

패키지의 고정 **21개**에 `v12-before-learned` 한 개를 추가했다.

- 규칙 기반 9종: collector, default, defender, noop, raider, random, rush, special, weak.
- 학습 모델 12종: v1~v6 각 1개, v7 그래프 2개, v8 competition 2개, v9-B, v10-C1.
- v12: `artifacts/submission/v12/scripted9_before_learned_latest/`의 `policy.py`와 `checkpoint.pt`를 함께 사용한다. 기존 규칙 기반 추가훈련은 반영됐고, 이후 learned 모드 추가훈련 직전의 모델이다. [20경기 직접 비교](../../../logs/v12/comparisons/before_after_20261028/result.json)에서 이후 모델에 20승 0패여서 이번 대표로 선택했다. 이 비교만으로 다른 모든 모델보다 강하다고 판단하지 않는다.

모든 참가자가 서로 맞붙는다. **22 × 21 × 5 = 2,310경기**이며, 상대 쌍마다 Blue/Red를 교대한 5세트(10경기)다. v12 자체는 21상대와 총 210경기를 치른다. 기존 21개끼리의 2,100경기도 새로 실행하며 과거 승패를 가져와 채우지 않는다. 패키지에 없는 v11은 참가하지 않는다.

기본 seed는 `20261016`이고 기존 모델 ID를 유지했다. 원래 21개 간 2,100개 대진과 요청 seed가 패키지의 등록과 동일함을 오프라인으로 확인했다. 실제 초기 맵 해시도 저장하여 진영 교대 쌍의 맵 일치를 보고서에 집계한다. 플랫폼 차이 때문에 과거 macOS 결과와 비트 단위 재현을 주장하지 않는다.

## 승패와 자원 설정

| 항목 | 설정 |
|---|---|
| 순위 집계 | 패키지와 같은 양 팀 **누적 raw reward** 비교 |
| 보조 기록 | `terminal_info.winner`, 위 승패와의 불일치, 실제 초기 맵 해시 |
| 병렬 경기 | 최대 16개, 모두 창 없는 백그라운드 |
| 정책 추론 | CPU, 각 worker의 Torch/BLAS 1스레드 |
| GPU | Unity의 시각 관측 렌더러 유지; `-nographics` 사용 안 함 |
| 시간 배율 | `time_scale=50` |
| 경기 한도 | 22,000 step, wall time 1,800초, heartbeat 지연 120초 |
| 실패 처리 | 세션 내 1회 재시도; 미완료·오류 경기를 무승부로 세지 않음 |
| 메모리 | 시스템 가용 RAM 4GiB와 신규 실행 여유분을 남기며 배정 제한 |
| 디스크 | 최소 15GiB 및 진행 중 trace 여유분 확보 |

16개는 이전 v12 Q2 실측의 최상 처리량 **468.95 env steps/s**에서 가져왔다. 같은 측정에서 12개는 464.64, 14개는 454.97, 18개는 458.24였다. [이전 측정 원본](../../../logs/v12/resource_tuning_20261026_162623/scale_summary.json). 이번에는 사용자의 실행 전 금지 요청에 따라 실제 benchmark를 다시 돌리지 않았다. 모델 구성이 더 다양하고 경기 trace를 저장하므로 468.95 steps/s나 같은 총 소요 시간을 보장하지 않는다.

진행 중에는 완료 경기 수, 실행 PID, 가용 자원, 잔여 시간 추정치를 `status.json`에 기록한다. 추정은 완료 경기의 시간과 병렬 수를 사용하므로 상대 조합·자원 대기로 바뀔 수 있다. 주기적으로 잠정 순위를 갱신하고 종료 시 상세 보고서를 생성한다. 중단·오류의 부분 trace도 남긴다.

## 가져온 자산과 Windows 이식

원본 외부 폴더는 수정하지 않았다. README의 배치 규칙에 맞춰 checkpoint·그래프·과거 소스는 `artifacts/`, 기존 실행의 선택 checkpoint·설정은 해당 `logs/v7`~`logs/v10` 경로에 복사했다. 기존 내용이 다른 파일은 덮어쓰지 않았다.

- [고정 21개 목록](../../../artifacts/league/portable_20261028/roster.json)
- [가져오기·해시 기록](../../../artifacts/league/portable_20261028/import_manifest.json)
- [자산 검증](../../../artifacts/league/portable_20261028/asset_verification.json)
- [보관한 setting.md](../../../artifacts/league/portable_20261028/package_metadata/setting.md)

23개 고유 모델 자산의 해시를 원본과 대조했다. v6는 checkpoint가 요구하는 정확한 과거 planner 소스를 보관했다. v7의 원래 config 및 checkpoint digest는 유지하고 검사 시에만 Windows 실행 파일과 정확한 LF 줄바꿈 split 보관본 경로를 사용한다. 기록된 옛 Mac 프로젝트 절대 경로도 해당 프로젝트 접두사에 한해 로컬 경로로 해석한다.

게임은 이미 설치된 `artifacts/builds/windows-v1.1/Blackout.exe`, Python은 프로젝트 `.venv/Scripts/python.exe`를 사용한다. macOS 전용 설치 스크립트·잠금 파일·`.app` 실행기는 사용하지 않았다. Windows 파일 잠금, 프로세스 트리 소유, 메모리 측정, 백그라운드 실행을 별도로 구현했다. Mac CoreGraphics 창 검사는 적용하지 않고 Windows 숨김 프로세스와 Unity `-batchmode`를 사용한다.

## 결과 위치와 확인 범위

`logs/shared/agent_league/v12_full_league_20261028/`에 기록한다.

- `manifest.json`, `request.json`: 참가자·자산/코드/환경 해시·대진·자원 설정.
- `status.json`, `console.log`: 진행 상황 및 오류.
- `report.md`, `ranking.csv`, `summary.json`, `figures/`: 순위·승패 행렬·보조 감사.
- `models/`, `pairs/`, `matches/`: 참가자·상대 쌍·경기별 상세 보고서.
- `attempts/`, `cells/`, `failures/`: 시도 기록·완료 경기·실패. 각 시도 폴더에 `trace/`를 남긴다.

최초 준비 중 확인한 것은 [22개 모델의 오프라인 로딩·양 진영 행동·RNG 검사](../../../logs/v12/league_setup_20261028/model_checks.json), 대진·해시 및 [가짜 경기 결과를 사용한 실행기 검사](../../../logs/v12/league_setup_20261028/windows_runner_validation.json)다. 이후 사용자 실행에서 별도 worker 검사와 실제 경기가 진행됐으며, 중단과 수정 내역은 위에 기록했다. 이번 수정에서 설치된 모델·게임 실행 자산·환경·체크포인트를 바꾸거나 다시 학습하지 않았다.
