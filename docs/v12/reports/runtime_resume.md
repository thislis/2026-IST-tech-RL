# v12 실행 자원 변경과 재개 준비

## 후보 선택 저장 오류 수정 — 2026-10-27

`P4_dev` 작업을 마친 뒤 후보 선택 결과를 저장하다 `Circular reference detected`로 중단됐다. `selected`가 `options` 안의 원본 dict였는데 `selected.options = options`에 해당하는 정보를 추가해 자기 참조를 만들었다. 선택 결과를 `dict(max(...))`로 복사한 뒤 메타데이터를 추가하도록 수정했다. 후보 순위·동률 처리·학습량·정책·16 worker 설정은 바꾸지 않았다.

Unity 없이 모의 study로 학습 후보 승리와 Q2 동률 우선 선택, JSON 저장, 재개 시 동일한 선택 잠금의 재저장을 확인했다. `runtime_revisions/0004`에는 이번 소스 변경과 보존 hash를 기록했다. 등록 준비 스크립트는 수정된 한 줄을 되돌린 소스 hash가 직전 등록과 정확히 일치하는지도 확인했다. 원래 등록·실패 상태·체크포인트는 덮어쓰지 않았다.

정상 경기 1,259개, 기존 실패 경기 1개, checkpoint/job result 47개를 보존했다. 별도 실패는 `main_S_s22_map`의 `scripted-noop_r001_s0`에서 `map_splits.json` 교체 시 발생한 `PermissionError`다. 이 경기를 성공으로 바꾸거나 삭제·재실행하지 않았다. 해당 MAP 후보는 기존 규칙대로 불완전 평가로 처리된다.

**사용자 요청에 따라 이번 수정 작업에서는 실험을 실행하지 않았다.** 아래 한 줄로 완료된 학습·평가 작업을 재사용하고 후보 선택부터 이어간다. 이후 필요한 추가 seed/confirmation/test가 실행된다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v12\run_v12.ps1 --name joint_trip_v12 --resume
```

`--resume-check`는 실행 없이 준비 상태를 확인한다. `status.json`에는 마지막 실패가 이력으로 남아 있으며, 실제 재개하면 실행 상태가 갱신된다.

## 추가 승인 후 실제 확장과 재개 — 2026-10-26

사용자가 실제 중단·측정·재개를 승인하고, 미리 정한 개수 상한 없이 2개씩 늘려 시간 단축을 우선하도록 요청했다. 아래의 최초 준비 기록에 있던 "실행하지 않음"은 당시 상태이며, 이번 추가 작업에서는 실제 측정과 백그라운드 재개까지 수행했다.

`runtime_revisions/0003`은 **전체 Unity 예산 16개, 데이터 수집 묶음 최대 16개, PPO 작업 최대 8개 × 기존 2개 에피소드**를 적용한다. 정책 추론·플래너는 CPU 1 thread, PPO learner는 CUDA, 미니배치 128을 유지한다. CPU 코어를 고정 예약하지 않고, 측정 전 가용 RAM에서 4 GiB를 데스크톱 프로그램의 여유로 남긴다. 16개가 항상 실행된다는 의미는 아니다. 기존 부분 묶음, 단계별 독립 작업 수, 경기 종료 시점 때문에 실제 개수는 달라진다.

Q2 실제 데이터 수집 정책과 scripted 상대를 동일 seed로 실행했다. event 복사·critic feature·reward·trip ledger 비용을 포함하고, 초기화·warm-up 뒤 각 후보를 12초간 측정했다. 자연 종료 시 파일 저장과 PPO 비용은 포함하지 않으므로 전체 파이프라인의 확정 속도 향상으로 해석하지 않는다.

| Unity 수 | 합계 steps/s | 판단 |
|---:|---:|---|
| 8 | 404.0 | 기준 |
| 10 | 423.2 | 개선 |
| 12 | 464.6 | 개선 |
| 14 | 455.0 | 감소 |
| 16 | 469.0 | 정상 완료 후보 중 최고, 채택 |
| 18 | 458.2 | 감소 |
| 20 | 463.2 | 진단값만 보존; worker 종료 30초 timeout으로 채택 제외 |

최초 자동 sweep은 14·16에서 개선 폭이 연속 2% 미만이라 멈췄다. 사용자 요청대로 아직 최고값을 낸 16을 넘겨 18·20도 추가 확인했다. 18에서 처리량이 낮아졌고 20에서 종료 timeout이 발생했으므로 그 이상은 실행하지 않았다. 모든 측정 자식 프로세스가 정리된 것을 확인한 뒤 재개했다. 16개는 이 짧은 수집 작업 측정에서 8개보다 약 16% 빠르며, 모든 상대·단계에 대한 전역 최적값을 보장하지 않는다.

원시 측정: `logs/v12/resource_tuning_20261026_162623/`, 추가 18·20 측정: `logs/v12/resource_tuning_20261026_162938/`. 합친 선택 근거는 첫 폴더의 `scale_summary.json`과 revision의 `throughput_evidence.json`에 보존했다. 20개는 모든 worker가 측정 JSON을 남겼지만 종료 완료 조건을 통과하지 못했으므로 정상 후보 통계에 섞지 않았다.

중단 당시 정상 평가 180개, 데이터 14개 에피소드/15,427 steps와 체크포인트를 보존했다. 중단된 wave 7의 기존 2개 task는 그대로 마친 뒤 새 wave부터 묶음을 확대한다. 데이터 seed serial은 `wave × 변경된 묶음 크기` 대신 누적 완료 episode 수를 사용해 seed 순서의 도약·중복을 방지한다. 이전 등록·완료 경기·trajectory·behavior snapshot은 변경하지 않았고, 데이터 checkpoint 사본과 각 파일 hash를 revision에 남겼다.

**데이터 수집의 이후 상대·진영 배정 순서와 마지막 묶음의 budget 초과량은 달라질 수 있다.** 배정은 앞선 묶음의 실제 경기 길이를 반영하기 때문이다. Q2는 고정 정책이며 optimizer update가 없고, 해당 데이터는 PPO trajectory로 재사용하지 않는다. PPO 학습의 2-episode 묶음·seed·optimizer·평가 조건은 유지한다. 변경된 수집 묶음의 serial 범위와 plan hash는 `data_schedule.json`, 실행 배정은 `data_dispatches.jsonl`에 기록한다.

추가 모의 검사 10개(데이터 재개 2, 확장 측정 7, 등록 이전 1)와 기존 worker 재개 검사 8개를 통과했다. 실제 실험은 같은 `--resume` 명령으로 백그라운드 재개했으나, 이후 사용자가 직접 이어가도록 세팅까지만 남겨 달라고 요청해 다시 중단했다. `stopped` 상태와 Unity 프로세스 0개를 확인했다. 사용자가 위 재개 명령을 입력하기 전에는 다시 시작하지 않는다. 이미 등록된 실측 설정을 다시 측정하지 않고 사용한다. 재개 직후에는 기존 부분 묶음의 남은 경기부터 처리한 뒤 16개 수집 묶음으로 전환한다.

## 최초 준비 당시 기록

2026-10-26. 사용자가 진행 중이던 `joint_trip_v12`를 잠시 중단하고 자원 설정을 개선하되, 직접 재개할 수 있도록 준비만 요청했다. 본 실험을 중단했고, 이 작업에서는 다시 시작하지 않았다.

## 재개 명령

프로젝트 루트 PowerShell에서:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v12\run_v12.ps1 --name joint_trip_v12 --resume
```

Bash에서는 `bash code/v12/run_v12.sh --name joint_trip_v12 --resume`를 사용한다. `--resume-check`는 파일·등록·보존 결과를 확인하는 실행 없는 검사다.

첫 재개 명령은 먼저 4·6·8개 Unity worker를 각각 15초 동안 비교한다. 초기화·warm-up 시간은 별도로 보고하고, 동일 seed·동일 scripted 상대·Q0/Q1 혼합 정책에서 전체 simulation steps/s를 비교한다. 후보 중 최고 처리량의 95% 이상인 것 중 worker가 적은 설정을 선택한다. 이 짧은 측정은 전역 최적값의 증명이 아니며, 경기 승패 평가나 학습 데이터로 사용하지 않는다.

측정이 끝나면 선택한 실행 자원을 새 runtime revision에 기록하고 본 실험을 백그라운드로 시작한다. 첫 측정 동안은 명령을 실행한 터미널을 열어 둬야 한다. 오류나 Ctrl+C 중단이 있으면 본 실험을 시작하지 않고 probe가 소유한 프로세스를 정리한다. 측정 기록은 `logs/v12/resource_tuning_<id>/`에 둔다.

**이번 준비 작업에서는 실제 Unity 측정을 수행하지 않았다.** 실행 도구의 자동 승인 검토가 실제 probe를 사용자의 “세팅까지만 하고 실행하지 말라”는 요청과 충돌한다고 판단해 거절했다. 따라서 최종 동시 실행 수는 아직 미선택이며, 위 명령을 사용자가 실행할 때 결정한다. 그전까지 기록상의 4 workers는 기존 설정이다.

## 보존한 결과

중단 확인 시 정상 완료된 경기 110개, 중단으로 기록된 셀 10개, 본학습 checkpoint 0개였다. 10개 중 4개는 진행 중 중단, 6개는 환경 시작 전 중단이었다.

- 정상 완료 경기 파일은 byte hash로 보존 여부를 확인한다. 정책·checkpoint·opponent·seed·decoder·map 비교 조건을 바꾸지 않는다.
- 기존 코드가 실패 셀로 저장했던 중단 기록 10개와 해당 집계 파일은 원본 그대로 revision의 `archived/`에 옮긴다. 이 셀들은 재개 때 다시 실행한다.
- 실제 오류나 watchdog 실패는 중단과 구분한다. 단순히 성적이 나쁘거나 오류가 났다는 이유로 삭제·재시도하지 않는다.
- 원래 `registration.json`, 평가 `plan.json`, `job.json`, 초기 checkpoint는 덮어쓰지 않는다. 새 등록·자원 설정은 `runtime_revisions/`에 추가하고, `resume_preparation.json`이 현재 revision을 가리킨다.

정확한 보존 hash와 이동 목록은 `runtime_revisions/0001/migration_plan.json`, 준비 상태는 `resume_preparation.json`에 기록한다. 같은 이름으로 아무 코드 변경이나 허용하는 기능이 아니다. 검토한 실행기·중단 처리·수집 스케줄 변경만 받아들이고 정책·feature·보상·상대·학습 조건 변경은 거부한다. 이번 등록 이전은 학습 전 기준선 단계에 한정한다.

## 병렬 실행과 학습 조건

이전에는 물리 코어 6개 중 1개를 예비로 남긴 5 workers를 두 실험에 정수 나눗셈해 실제 4개만 실행했다. 지금은 논리 CPU와 RAM으로 상한을 잡고, 실행 시 처리량 측정으로 4·6·8개 중 선택한다. 균등 배분에서 남은 worker를 버리지 않는다.

평가에서는 전체 worker 예산을 두 작업에 배분하고, 한 작업이 끝나면 남은 작업의 bounded queue에 여유 worker를 추가한다. 현재 진행 중인 경기나 그 입력은 변경하지 않는다.

학습은 `rollout_episodes_per_wave=2`로 기존 묶음을 고정한다. seed 계산과 optimizer 갱신 시점은 worker 수가 아닌 wave 크기를 따른다. 전체 예산이 6개라면 독립 학습 작업 최대 3개, 8개라면 최대 4개를 각각 2 episode씩 병렬 수집한다. CUDA update는 기존 lease로 순서를 조절한다. 기존 CUDA 장치와 128-event PPO minibatch는 유지한다.

단일 작업만 있는 공통 데이터 수집처럼 조건 보존 때문에 worker를 모두 채우지 못하는 단계는 있을 수 있다. 점유율 100%를 위해 학습 묶음·seed·갱신 빈도를 조용히 바꾸지 않는다.

## 확인 범위

Unity를 실행하지 않는 모의 검사 11개가 통과했다. 중단과 진짜 실패의 구분, 완료 셀 재사용, bounded 제출, 동적 allocation, 1/4 execution workers에서 같은 wave·seed, legacy checkpoint의 자원 변경 호환, 학습 조건 변경 거부, 등록 원본·완료 결과 보존을 확인했다.

실제 4·6·8 처리량과 최종 선택 결과는 아직 없다. 본 실험 재개도 수행하지 않았으며, 다음 실행은 사용자가 위 명령을 입력할 때만 시작한다.
