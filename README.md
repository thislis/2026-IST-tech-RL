# BlackOut 강화학습 실험

코드, 문서, 실험 기록을 실제 디렉터리로 분리했습니다. 이전 위치에 호환 링크는 두지 않습니다.

| 경로 | 내용 | Git 공유 |
|---|---|---|
| `code/v2/` ~ `code/v9/` | 버전별 구현·스크립트·설정·테스트 | 예 |
| `code/shared/` | 여러 버전이 사용하는 Python 모듈·도구·의존성 목록 | 예 |
| `code/pre_v1/` | 초기 구현·실험 코드 | 예 |
| `docs/` | 버전별 계획·분석·직접 작성한 보고서 | 예 |
| `docs/common/` | 실험 이력·게임 및 제출 규격·공통 문서 | 예 |
| `logs/<버전>/` | 실행 로그·자동 생성 보고서·실험 등록 기록 | 아니요 |
| `artifacts/` | 가중치·데이터·원본 게임 빌드·제출 산출물·과거 소스 보관본 | 아니요 |
| `.venv/`, `.unity/` | 로컬 실행 환경 | 아니요 |

v1 당시 구현은 후속 구현으로 발전했으며, 현재 별도 `code/v1/` 디렉터리는 없습니다. 공통 구현은 `code/shared/`에서 확인합니다. 이슈 문서는 `docs/issues/`, 초기 보고서는 `docs/pre_v1/reports/`에 있습니다.

[실험 이력](docs/common/history.md) · [v9 안내](docs/v9/README.md) · [v9 계획](docs/v9/plans/v9_plan.md) · [제출 규격](docs/common/blackout_last_4_pages.md) · [재배치 안내](docs/common/project_layout.md)

## 실행

프로젝트 루트에서 다음 명령을 사용합니다. 다른 작업 디렉터리에서는 스크립트의 절대 경로를 사용하세요.

```bash
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

전체 버전의 테스트는 프로젝트 루트에서 아래처럼 각 테스트 디렉터리를 순회합니다. `discover -s code/shared/tests`만 실행하면 공통 테스트만 수집됩니다.

```bash
for test_dir in code/*/tests; do
  .venv/bin/python code/run.py -m unittest discover -s "$test_dir" -v || break
done
```

실행기는 필요한 Python 검색 경로를 자동으로 설정합니다. `python -m ...`을 직접 실행할 때는 `code/run.py -m ...`을 사용합니다. 개별 `code/<버전>/scripts/*.py`도 직접 실행할 수 있습니다.

기존 등록 실험은 당시 소스·설정 해시를 유지합니다. 재배치로 소스 해시가 달라졌으므로 과거 등록을 덮어쓰거나 같은 이름으로 이어서 학습하지 마세요. 기존 완료 결과는 상태 조회와 기록 열람으로 확인할 수 있습니다.

새 clone에는 가중치·게임·데이터·Python 환경이 없습니다. 해당 로컬 자산을 준비해야 실제 실행이 가능합니다. 제공 게임 및 API/obs 코드는 변경하지 않습니다.
