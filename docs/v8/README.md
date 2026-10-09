# v8 구현 및 실행

**최신:** 제출 안내에 맞춘 원본 환경용 `provided_competition_v1` 경로를 새로 구현했다.
[현재 학습·평가·제출 안내](competition.md)를 참조한다. 아래 실행 예시는 과거 수정 환경 경로다.

**2026-09-29: 아래 수정 Unity 연구 환경은 철회했다.** 제공 게임과 obs 생성 코드를 그대로
사용하라는 사용자 지시에 따라 원본 환경을 복구하고 새 모델 코드는 보존했다.
기존 학습 실행기는 현재 비활성화되어 있다. [최신 복구 상태](environment_restoration.md)를 참조한다.
아래 내용은 철회 전 구현과 검증의 역사 기록이다.

v8-C1과 같은 용량의 flat41 대조군을 `code/v8/blackout_rl/v8`에 구현했다. 단일 policy factory를
학습·평가·연구 export가 사용한다. 원본 v7 소스·체크포인트·등록 fingerprint를 수정하지
않고, 별도 Unity 연구용 빌드에서 engine outcome, reward ledger, timer 수명을 검증한다.

구현 기준: `docs/v8/plans/v8_implementation_plan_with_references.md`.
현재 검증 결과와 남은 실험은 [implementation_status.md](implementation_status.md),
웹 자료 확인 범위는 [reference_adoption.md](reference_adoption.md)를 참조한다.

2026-09-28 추가: 3 seeds × 2 arms의 백그라운드 가속 실행기를 준비했다.
실제 실행은 시작하지 않았다. [한 줄 실행 명령과 가속 검증](acceleration.md)을 참조한다.
아래 2026-09-27의 smoke·export 기록은 가속 전 소스를 사용한 과거 검증이다.

## 구성과 실행 계약

- 정책: frozen 128-d encoder + 슬롯별 177→64→64 trunk + correction 8개/슬롯,
  팀 gate 320→64→1. Critic은 299→256→128→1. flat은 동일 trainable parameter 수를 사용한다.
- 학습은 joint41 sampling과 실제 old log-probability를 사용한다. C1 평가는 `q > 0.5`일 때
  conditional argmax, 동률은 KEEP이다. flat은 joint argmax이다. decoder는 별도 정책 계약이다.
- actor에는 제공된 vector/graphic만 전달한다. engine score·opponent identity는 진단/학습
  장부용이며 actor feature에 넣지 않는다. 배터리 수량이나 숨은 상대 class를 추가하지 않는다.
- 한 슬롯의 PathNotFound는 해당 슬롯의 abstain으로 격리한다. 예기치 않은 팀 전체 fallback은
  유효 데이터로 학습하지 않고 attempt 오류로 남긴다. 고정 target의 기존 planner는 수정하지 않는다.
- 매 decision마다 map/side/opponent hash 노출을 기록한다. episode는 rollout 경계에서 유지된다.
  resume은 완료된 PPO 업데이트 경계부터이며 physics 복원을 주장하지 않는 episode-restart 방식이다.
- checkpoint는 SHA256 이름의 불변 파일이다. `latest.json`은 포인터이며 parent lineage를 확인한다.
  실패 suffix는 `recovery/`에 보존한 뒤 마지막 완료된 로그 경계로 돌아간다. 성능 rollback은 없다.
- ±250 transition event/control window는 bounded ring buffer로 저장한다. semantic ID는 lossless
  zlib/base64이며 `observation.expand()`로 복원한다. window 수가 상한을 넘으면 drop 수를 기록한다.
  every-step exposure는 window sampling과 무관하게 전부 남는다.

## 로컬 환경과 빌드

기존 `.venv`와 `.unity/6000.3.8f1/Unity.app`을 사용한다. 설치/외부 파일 다운로드는 수행하지 않았다.
`prepare_v8_unity.py`는 `../blackout` commit
`d2220a7d01be88d413f551efd529f4758833be8b`의 Assets/Packages/ProjectSettings와 로컬 Library를
별도 `build/v8_unity`에 복사한다. 기존 출력이 있으면 덮어쓰지 않는다.

```bash
.venv/bin/python code/v8/scripts/prepare_v8_unity.py
V8_BUILD_OUTPUT="$PWD/builds/BlackOut-v8.app" \
  .unity/6000.3.8f1/Unity.app/Contents/MacOS/Unity -batchmode -quit \
  -projectPath "$PWD/build/v8_unity" -executeMethod V8Build.Build \
  -logFile "$PWD/reports/v8/unity_build.log"
.venv/bin/python code/v8/scripts/register_v8_build.py --log logs/v8/reports/unity_build.log
.venv/bin/python code/v8/scripts/register_v8_provenance.py
```

현재 workspace에는 검증된 빌드와 준비된 프로젝트가 이미 있다. 위 prepare를 반복할 필요는 없다.
빌드 식별은 launcher만이 아니라 bundle 파일 목록과 SHA256으로 한다. Unity가 생성하는
`Contents/ML-Agents/Timers/`와 `.DS_Store`만 제외한다. 준비 소스 비교에서는
`Assets/ML-Agents/Timers/` 자동 진단 메타데이터만 제외한다. 해시 오류를 우회해 실행하지 않는다.

## 계약 검사와 짧은 통합 실행

```bash
.venv/bin/python code/v8/scripts/validate_v8_timer.py
.venv/bin/python code/v8/scripts/validate_v8_contracts.py
.venv/bin/python code/v8/scripts/validate_v8_contracts.py --live \
  --output logs/v8/reports/live_contract_validation.json
.venv/bin/python code/run.py -m unittest discover -s tests -v
.venv/bin/python code/v8/scripts/train_v8.py --config code/v8/configs/experiments/c1_smoke.yaml \
  --run-dir logs/v8/c1_smoke_NEW
.venv/bin/python code/v8/scripts/train_v8.py --config code/v8/configs/experiments/flat_smoke.yaml \
  --run-dir logs/v8/flat_smoke_NEW
```

`*_smoke.yaml`은 timeout을 2 game-seconds로 줄인 별도 진단 실험이다. 정규 파일럿의 schedule을
압축한 것이 아니며 승률·학습 가능성 증거로 해석하지 않는다. 같은 디렉터리의 재시작에는
`--resume`이 필요하고, 완료된 예산은 자동 연장하지 않는다.

`planner_native.yaml`은 기존 native protobuf/queue overlay 검사에 사용한다. CLI가 원본
71개 source와 82개 overlay 파일·패키지를 확인한 뒤, ML-Agents import 전에 설치한다.
기본 학습 설정은 CPU/original backend이며 MPS/CSR를 요구하지 않는다.

## 평가·연구 export

```bash
.venv/bin/python code/v8/scripts/evaluate_v8.py --config code/v8/configs/experiments/c1_smoke.yaml \
  --checkpoint-store logs/v8/c1_smoke_v2/checkpoints --max-maps 1 \
  --output logs/v8/c1_dev_smoke_NEW
.venv/bin/python code/v8/scripts/verify_v8_export.py \
  --checkpoint-store logs/v8/c1_smoke_v2/checkpoints \
  --trace logs/v8/c1_smoke_v2/windows.jsonl \
  --bundle build/v8_research_export_NEW --output logs/v8/reports/export_parity_NEW.json
```

평가는 모든 셀의 시작·성공·실패를 기록하고 실패한 셀을 빼고 승률을 계산하지 않는다.
현재 CLI는 등록된 dev만 허용한다. 독립 confirmation/test manifest 및 사전 재시도 규칙은
아직 미등록이므로 해당 split 실행은 거부한다. 원본 held-out test는 열지 않았다.

export 검증은 실제 수집된 관측으로 2개 episode를 재연하며, 중복 호출·dictionary 순서 변경도
검사한다. 별도 `python -I`, 빈 작업 디렉터리, bundle 내부 import를 사용하고 설치된 고정
의존성은 재사용한다. OS/공식 runner를 포함한 완전 독립 배포 시험은 아니다.
`submission_contract.json`이 unresolved이면 공식 제출 export는 실패한다.

`analyze_v8.py`는 complete `[run,map,side]` W/N 배열 두 개를 받는다. 호출자는 등록된 동일
map 순서·상대·빌드·metric의 cube를 제공해야 한다. 알고리즘별 run은 독립 재표집하고 map은
공유한다. 고정 planner는 한 row만 주고 `--fixed-reference`를 사용한다. 같은 seed 번호를
paired run으로 취급하지 않는다. 누락·NaN 셀은 거부하며 crossed/fixed-map/fixed-policy CI를
함께 출력한다. 분석 결과만으로 자동 모델 승격이나 test 접근을 실행하지 않는다.

## 정규 파일럿과 후속 범위

`code/v8/configs/experiments{c1,flat}.yaml`은 각 1,048,576 step, 기존 200k/600k 상대 schedule,
train/dev 분리를 유지한다. seed11이 기본이며 독립 seed22/33은 config의 seed와 experiment ID,
run 경로를 별도로 고정해야 한다. 가속 실행기는 이를 `accelerated_pilot/` 설정과 별도 등록으로 준비했다.
3 seeds×2 arms의 전체 파일럿은 과거 기술 검증과 별도다.
S2 chronological/paired 진단과 실험 잠금 후 실행하고, 관찰한 성과에 따라 임의로 예산을 바꾸지 않는다.
C2/C3는 근거가 생기기 전까지 지원하지 않으며 요청하면 factory가 거부한다.
