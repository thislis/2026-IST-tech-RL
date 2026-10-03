# v8 최종 집계의 모듈 경로 오류 수정

**2026-09-29 후속 변경:** 사용자 지시에 따라 이 연구 환경의 실행을 철회하고 원본 게임/API를
복구했다. 아래 동일 명령 재실행 안내는 철회 전 기록이다. [현재 상태](environment_restoration.md).

`No module named 'blackout_rl'`은 학습·평가 종료 후 백그라운드 관리자의 `aggregate()`에서 발생했다.
6개 학습 run은 각각 1,048,576 step을 완료했고, 78개 평가 shard의 780경기도 완료되어 있었다.
완료 기록 84개와 checkpoint/result 해시를 검증한 뒤 저장된 결과만 사용해 최종 집계를 복구했다.
현재 상태는 `complete`다. 이번 수정 작업에서 Unity, 새 학습, 새 평가를 실행하지 않았다.

## 원인과 수정

`python /.../scripts/v8_experiments.py`로 파일을 실행하면 Python의 기본 모듈 검색 경로에는
`scripts/`가 들어간다. 작업 디렉터리를 프로젝트 루트로 바꾸는 것만으로는 충분하지 않다.
worker는 `_v8_cli.py`에서 루트를 등록했지만 별도 프로세스인 관리자에는 적용되지 않았다.
기존 검사는 worker의 `--check`와 이미 패키지를 import한 테스트 프로세스를 사용하여 이 차이를 놓쳤다.

v1 계열 `run_base_r12_bc.py`, v3/v5/v6의 학습 CLI, v7의 `launch_v7_background.py` 및
`pilot_evaluation.py`와 같이 파일 위치에서 구한 프로젝트 루트를 `sys.path`에 등록했다.
이제 실행 전 검사에서도 관리자 자체의 statistics import를 확인한다.
새 회귀 검사는 `PYTHONPATH` 없이 다른 작업 디렉터리의 새 Python 프로세스에서 같은 경로를 검사한다.

## v8 실험 계약과 기록 보존

실험 완료 후 관리자 파일을 고치면 전체 source fingerprint가 달라진다. 기존 등록이나 checkpoint
해시를 새 값으로 덮어쓰지 않고, **완료된 실험의 집계에만 적용되는 별도 수정 기록**을 추가했다.

- 변경 허용 파일은 `scripts/v8_experiments.py` 하나이며 수정 전후 SHA256을 고정했다.
- 원래 등록 파일, 모델·PPO·planner·환경·보상·통계 코드, 설정, 체크포인트와 경기 결과를 유지했다.
- 원래 입력 파일 해시와 전체 소스 목록을 확인한다. 다른 파일의 변경은 거부한다.
- 학습/평가 84개 작업이 모두 완료된 경우에만 이 수정 경로를 허용한다. 새 실험 실행에는 사용할 수 없다.
- 집계 시 평가 policy lock의 내용 해시, 등록 소스, 경기별 lock 및 map/side 전체 coverage도 검사한다.
- 요약에는 원래 실험 등록 해시와 집계 관리자·수정 기록 해시를 별도로 남긴다.
- 원래 관리자, 등록, 오류 상태, console log는 `reports/v8/acceleration/import_fix/`에 보관했다.

이는 계획서의 원본 보존, 불변 checkpoint, 완료된 평가 전체 집계, 학습 정책·실행 계약 유지에
따른 사후 처리 수정이다. 새로운 성능 실험이나 held-out test를 추가하지 않았다.
v7의 `frozen_pilot_runtime.py`처럼 과거 결과가 생성된 소스와 현재 관리 코드를 구분한다.

## 동일한 실행 명령

```bash
bash /Users/safeailab_macmini/Desktop/2026-IST-tech-RL/scripts/run_v8_fast.sh
```

백그라운드 실행·중복 방지·상태 확인 인터페이스는 유지한다. 이 파일럿은 이미 완료됐으므로
같은 명령을 다시 실행하면 학습·평가를 건너뛰고 저장된 결과를 집계한다. 실험 예산을 자동 연장하지 않는다.
`--status`, `--check`, `--stop`도 그대로 사용할 수 있다.

## 검증 근거

- 새 interpreter 재현: 수정 전 실제 오류 재현, 수정 후 import/통계 호출 성공.
- 프로젝트 외부 디렉터리에서 같은 셸 실행기의 `--check` 통과.
- 수정 관련 13개 검사 통과: import, 변경 범위 제한, 미완료 작업 거부, 완료 작업의 worker 재실행 방지 등.
- 전체 회귀 276개: **275개 통과, 1개 Metal 장치 접근 조건으로 skip**.
- `--aggregate-only`로 기존 780경기의 최종 집계 완료. 학습/평가 worker를 생성하지 않는다.

[최종 요약](../../logs/v8/accelerated_pilot_v1/summary.json),
[수정 등록](../../reports/v8/acceleration/import_fix/repair.json),
[수정 전 재현](../../reports/v8/acceleration/import_fix/reproduction_before.txt),
[회귀 검사](../../reports/v8/acceleration/import_fix/regression_tests.txt),
[복구 검증](../../reports/v8/acceleration/import_fix/verification.json).
