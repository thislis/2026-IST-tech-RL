# v13: 작업 배정과 이동 제어를 분리한 정책

**2026-11-04 평가 축소·재개 준비:** 기존 `hierarchical_v13`을 정상 중단하고, 학습 상태를 보존한 `hierarchical_v13_eval_light`를 준비했다. **현재 재실행하지 않은 상태다.** 기존 실험을 이어갈 때는 아래 명령을 사용한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v13\run_v13.ps1 --name hierarchical_v13_eval_light
```

5,242,880 step부터 재개한다. 중간 평가는 **2,097,152 step마다 88경기**, 각 본 학습에서 선택한 모델의 종료 평가는 **220경기**, 최종 확인은 **1,720경기**다. 학습량·학습 seed·선정 arm은 유지했다. 기존 teacher와 완료한 pilot을 다시 실행하지 않는다. [변경 내역·상태 조회·원본 보존](reports/evaluation_reduction_20261104.md).

2026-11-03에는 구현과 게임 없는 검사를 완료했고, 이후 사용자가 전체 실험을 시작했다. 아래 일반 실행 명령은 새 기본 실험용이다. 기존 실험의 재개에는 위의 **새 이름이 포함된 명령**을 사용한다.

배정망은 거리·귀환 비용·ETA·경합 특징으로 팀 5명의 작업을 공동 배정하고, 이동망은 목표와 국소 지도·거리장을 받아 32방향 또는 정지를 선택한다. 최근 64개 관측 요약을 외부 state에 저장해 `{0,4,8,16,32,64}` 시점을 읽는다. 이 버퍼와 학습된 GRU hidden은 별개다. GRU 유무·현재 관측만 사용·균등 간격 기억을 B1~B6로 비교한다.

## 한 줄 실행

프로젝트 루트의 PowerShell에서:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v13\run_v13.ps1
```

Git Bash에서도 같은 Windows 환경으로 실행할 수 있다. 게임의 창 검사기가 Windows 전용이므로 Linux 실행을 보장하지 않는다.

```bash
bash code/v13/run_v13.sh
```

일반 기본 run 이름은 `hierarchical_v13`이다. 이 이름의 기존 기록은 평가 축소 전 원본으로 보존했으므로, 현재 작업에서는 `--name hierarchical_v13_eval_light`를 사용한다. 숨김 백그라운드 프로세스가 실행되어 셸을 반환한다. 같은 이름의 명령은 저장된 설정과 체크포인트로 재개한다. 실행 중이면 중복 실행하지 않는다. v13 실행끼리는 하드웨어 잠금을 공유한다. 이전 버전의 별도 실행을 자동 종료하지는 않는다.

처음 실행하면 full ONNX 합성 검사 → learner CPU/CUDA·배치 크기 측정 → 병렬 Unity 수 측정 → 공통 teacher 데이터 → B1~B6 pilot → 상위 2개 arm의 3개 seed 본 학습 → 1,720경기 확인 평가 → 최종 ONNX 순으로 진행한다. 환경 수는 최대 24개 안에서 처리량·RAM·VRAM 여유로 선택한다. 숫자가 큰 설정을 무조건 선택하지 않는다.

## 조회·중단·작은 범위 실행

```powershell
# 게임·학습을 시작하지 않는 준비 검사 / 계획 확인 / 상태 조회
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v13\run_v13.ps1 --check
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v13\run_v13.ps1 --dry-run
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v13\run_v13.ps1 --status

# 정상 중단 요청; 같은 기본 명령으로 재개
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v13\run_v13.ps1 --stop

# 선택 사항: 별도 이름으로 B5 pilot까지만 실행
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v13\run_v13.ps1 --name v13_b5_pilot --scope pilot --arm B5
```

별도 이름을 사용했다면 조회·중단·재개에도 `--name`을 붙인다. `--resume-check`는 소스·설정·런타임 해시를 읽기 전용으로 확인한다. 변경된 코드로 기존 실험을 이어 실행하지 않으며, 변경 후에는 새 이름을 사용한다. `--foreground`는 터미널에 붙여 실행할 때 사용하는 옵션이다.

예산·worker 수 등을 바꾸려면 [기본 JSON](../../code/v13/configs/default.json)을 복사해 수정하고 새 이름과 `--config <파일>`을 지정한다. 기본 full 예산은 teacher 1,048,576 + pilot 6×262,144 + 본 학습 최대 2×3×8,388,608 = **52,953,088 팀 환경 step**이다. BC 계산·자원 측정·평가·chunk 반올림은 별도이며, 규칙 제어 B1은 Worker 학습량을 생략한다. pilot 전용도 teacher와 모방 학습·dev 평가를 포함한다.

## 기록과 산출물

| 경로 | 내용 |
|---|---|
| `code/v13/blackout_v13/` | 모델, 상태·기억, PPO, 수집·평가·export |
| `code/v13/configs/` | 기본 예산과 B1~B6 설정 |
| `logs/v13/<run>/console.log`, `status.json` | 콘솔 기록, 현재 단계·진행량·실패 이유 |
| `logs/v13/<run>/resources/` | 하드웨어 정보, 환경 수·CPU/CUDA 측정과 선택 |
| `logs/v13/<run>/runs/` | arm/seed별 update·dev·결과 |
| `logs/v13/<run>/confirmation/`, `result.json` | 고정 확인 평가와 통계 |
| `artifacts/data/v13/<run>/teacher/` | 압축 teacher chunk와 체크섬 |
| `artifacts/checkpoints/v13/<run>/` | latest/best, optimizer·RNG·버전·자기대결 snapshot |
| `artifacts/submission/v13/<run>/` | full `model.onnx`, `meta.json`, 합성 검증 기록 |

`.venv`의 기존 CUDA 환경을 사용한다. [확인한 compute 의존성](../../code/v13/requirements-compute.lock)을 기록했고 환경을 교체하지 않았다. 기존 실행에서 선택한 **환경8개·CUDA·sequence batch8**을 재사용한다. 공식 실행기의 100ms 충족 여부는 별도 검증 항목이며, 최종 export도 공식 컨테이너 검증 전에는 후보 상태다.

- [구현 범위·병목 대응·검증](reports/implementation.md)
- [모델 설계](plans/v13_design.md), [원 구현·검증 계획](plans/v13_implementation_plan.md)
- [v12 리그 근거와 발언 해석](reports/design_evidence.md)

파일 배치는 [루트 README](../../README.md)와 [공통 학습 실행 원칙](../common/training_runtime.md)을 따른다.
