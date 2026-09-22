# v7 구현과 백그라운드 실행

2026-09-15. 계획서의 초기 구현 단계인 v7-1 그래프 보정과 v7-2 F0/F1 연구 경로를 구현했다. 장기 승률 개선이나 전뇌 내부 학습을 완료했다는 의미는 아니다.

## 구현 범위

- **v7-1**: 고정 v6 인코더 + 49차원 계획기 문맥 → 실제 FAFB 부분 그래프 → KEEP+5×8 팀 행동. 새 그래프는 호출 간 상태를 유지하지 않는다. 기존 `joint_distribution`과 PPO 업데이트를 공유한다. 실제 배선(A3), v6 MLP(A0), 차수 보존 재배선(A2), 파라미터 수 대응 MLP(A1) 설정을 제공한다.
- **v7-2 F0**: 전체 MaleCNS 유지 그래프, 20ms proxy 동역학, 100ms 의미 지도 시야 갱신, 고정 방향·정지 출력. 학습은 하지 않는다.
- **v7-2 F1**: 동일한 고정 전뇌의 출력 집단 특징을 저장하고, 9-way 출력층과 중앙 critic만 PPO로 갱신한다. 기본은 한 유닛 전뇌 제어 + 네 유닛 고정 scripted 정책이다. 제어 슬롯에 계획기 우회·fallback을 사용하지 않는다.
- 명시적인 `(episode_id, env_step_id, team_id)`와 슬롯별 상태, 동일 tick 재호출 캐시, 새 경기 초기화, 시드 분리, RNG/옵티마이저/신경 상태 저장을 제공한다.
- `--check`, 데이터 감사, PID/파일 잠금, 원자적 체크포인트, SIGTERM 후 rollout 경계 저장, 원본·그래프·설정·소스 해시, 순차 파일럿 큐를 제공한다.
- dev/confirmation 양 진영 평가와 모델 잠금 후 test 평가를 제공한다. test 결과를 모델 선택에 재사용하지 않는다.

## 2026-09-17 오류 수정

v7-2의 scripted 정책이 전뇌 담당 슬롯까지 계획하던 오류를 수정하고, `teammate_v2` 별도 실행에서 마지막 32,768스텝 체크포인트를 이어받았다. [원인·수정·검증·재개 안내](teammate_v2_recovery.md)를 참고한다. v7-1 평가 명령은 등록 당시 코드 아카이브를 사용한다.

## 실제 데이터와 계획 조정

**FAFB v783:** Codex 토큰 없이 접근 가능한 [공식 연결 아카이브](https://zenodo.org/records/10676866)와 [저자 주석 저장소](https://github.com/flyconnectome/flywire_annotations/tree/8587524c1748ce5ef2080822a2fc890fc03bf597)를 사용했다. 현재 Codex 다운로드와 같은 상품이라고 표시하지 않는다. 주석 커밋과 파일 SHA-256을 고정했다. 영역별 연결 행을 합산하되 양의 원본 연결을 추가 임계값 없이 유지했다.

실제 주석의 EPG/PEN/PEG/Delta7/PFL 계열 전체를 선택한 결과 **205 뉴런, 7,820 방향성 연결, 54,785 접촉**, 좌 102/우 103, 고립 노드 0, 강연결 성분 1이다. 계획의 256~2,048개 목표보다 작은 회로를 사용하며 임의 노드 추가로 크기를 맞추지 않았다. 모든 출력 풀이 K=4에서 도달 가능하다. 외부 입력 114,773접촉, 외부 출력 110,434접촉이 잘려나가는 부분회로라는 한계도 기록한다. 원래 16개 포트 제안 대신 실제 유형·좌우별 풀이 사용되며 확정 수는 YAML에 기록한다.

**MaleCNS v1.0:** [공식 다운로드](https://male-cns.janelia.org/download/)의 세 Feather 파일을 사용했다. `superclass`가 비어 있지 않은 모든 뉴런을 유지하고 status로 추가 필터링하지 않았다. **166,700 뉴런, 25,582,938 방향성 연결, 124,177,617 접촉**으로 참고 포함 규칙의 수치를 재현했다. 런타임 가지치기를 하지 않는다.

망막 입력은 현재 `R1`~`R8` 유형 이름 조건에 해당하는 6,091개 실제 뉴런이다. 참고 구현의 6,006개와 동일한 선택이라고 주장하지 않는다. 공식 optic-column 표에서 직접 대응 2,628개, 연결 기반 추정 3,253개, 명시적인 미검증 좌우 눈 내부 대체 대응 210개를 기록했다. 좌우는 `somaSide`, 없으면 공식 `rootSide`를 사용한다. 각 원본 ID와 대응 근거는 그래프 JSON에 있다. 각도·색·가림·거리 변환은 BlackOut용 인공 센서이며 생리학적 광학 측정값이 아니다.

전뇌 식은 `v'=exp(-.02/.1)v+1.5 W spikes+.180+noise+retina` 후 임계값 1에서 발화·리셋한다. 잡음은 1.2Hz Bernoulli, 진폭 .22이며 GABA/glutamate/histamine을 음수로 취급하는 모델 가정이다. 입력 필터·gain·13-tick 출력 창을 설정에 고정했다. [Fly64 기술 설명](https://github.com/ornata/fly/blob/f2f4114e53eaa326e54129f27a5383f93c6957af/docs/technical-notes.md)을 참고해 독립 구현했으며 원본 코드를 복제하지 않았다.

초기 고정 decoder는 모든 조작이 정지하는 검증 실패를 보였다. 오프라인 자극으로 `movement_threshold=.02`, `turn_gain=40`, `turn_limit=2`를 정하고 재검증했다. 이 보정은 게임 승률에 맞춘 학습이 아니다. 최초 실패 보고서 `whole_brain_causality.json`도 보존한다.

## 준비와 실행

프로젝트 루트에서 실행한다. `.venv`에 필요한 추가 패키지는 설치되어 있다. 다른 환경에서는 기존 의존성을 설치한 다음 다음 명령을 사용한다.

```bash
uv pip install --python .venv/bin/python -r requirements-v7.lock
.venv/bin/python scripts/prepare_v7_connectome.py --config configs/v7/v7_1_flywire.yaml --download
.venv/bin/python scripts/prepare_v7_connectome.py --config configs/v7/v7_1_mlp_control.yaml
.venv/bin/python scripts/prepare_v7_connectome.py --config configs/v7/v7_1_matched_mlp.yaml
.venv/bin/python scripts/prepare_v7_connectome.py --config configs/v7/v7_1_rewired.yaml
.venv/bin/python scripts/prepare_v7_connectome.py --config configs/v7/v7_2_fixed.yaml --download
.venv/bin/python scripts/prepare_v7_connectome.py --config configs/v7/v7_2_readout_ppo.yaml
```

`prepare`는 새 그래프만 생성하고 기존 그래프를 덮어쓰지 않는다. 확보한 원본 파일을 해시 검증하고 설정의 null 데이터 필드를 실제 값으로 고정한다. 다른 주석·파일로 기존 manifest를 재사용하면 실패한다. `data/connectomes/`는 Git에서 제외하고 출처/해시/감사 사본은 `reports/v7/data/`에 보관한다. 합성 그래프는 단위 시험 전용이며 실행 검사를 통과할 수 없다.

```bash
# Unity 실행 없이 검사
scripts/start_v7_1_background.sh --check
scripts/start_v7_2_background.sh --check

# 단일 seed, 각각 독립 백그라운드 프로세스
scripts/start_v7_1_background.sh --seed 11
scripts/start_v7_2_background.sh --seed 11

# 전뇌 출력층 PPO
scripts/start_v7_background.sh --config configs/v7/v7_2_readout_ppo.yaml --seed 11

# 3 seed × A3/A0/A2, run별 독립 디렉터리에서 순차 실행
.venv/bin/python scripts/launch_v7_queue_background.py --queue configs/v7/pilot_v7_1_queue.json

# F0와 3 seed F1 엔지니어링 파일럿 순차 실행
.venv/bin/python scripts/launch_v7_queue_background.py --queue configs/v7/pilot_v7_2_queue.json
```

단일 run이 실행 중인 경우 같은 run을 포함한 큐를 동시에 시작하지 않는다. 파일 잠금이 중복 학습을 차단한다. 큐는 실패를 기록하고 정지하므로 실패한 seed를 조용히 제외하지 않는다. v7-1 파일럿 예산은 각 run 200,000 환경 스텝이다. F0 기본 예산은 22,000, F1은 128,000이다. 4천만 스텝 본실험은 자동 실행하지 않는다.

로그 기본 위치는 `logs/v7/<experiment_id>/<seed>/`이다. `console.log`, `status.json`, `training.jsonl`, `episodes.jsonl`, `diagnostics.jsonl`, `checkpoints/latest.pt`를 확인한다.

```bash
# 경로의 experiment_id는 해당 YAML 또는 launcher 출력 참고
cat logs/v7/v7_1_fafb783_flywire/11/status.json
tail -f logs/v7/v7_1_fafb783_flywire/11/console.log
kill -TERM "$(cat logs/v7/v7_1_fafb783_flywire/11/training.pid)"
scripts/start_v7_1_background.sh --seed 11 --resume
```

SIGTERM 저장은 현재 rollout 완료 시점에 수행된다. Unity 물리 상태는 복원할 수 없으므로 재개 시 진행 중 경기를 폐기하고 새 경기로 시작한다. 로그에 폐기 스텝과 경기 ID를 남긴다. 신경 상태는 재생 연구용으로 저장하지만 훈련 재개에서 이전 Unity 경기와 이어 붙이지 않는다. 설정·소스·seed가 달라지거나 등록 예산을 이미 완료했으면 동일 run의 resume를 차단한다.

## 검증과 평가

```bash
.venv/bin/python -m unittest tests.v7.test_contracts tests.test_mappo_v6 -q
.venv/bin/python scripts/validate_v7_2_causality.py --config configs/v7/v7_2_fixed.yaml --ticks 512 --output reports/v7/whole_brain_causality_calibrated.json
.venv/bin/python scripts/validate_v7_2_unity.py --config configs/v7/v7_2_fixed.yaml
.venv/bin/python scripts/evaluate_v7.py --run RUN_DIR --split dev
.venv/bin/python scripts/evaluate_v7.py --run RUN_DIR --create-lock
.venv/bin/python scripts/evaluate_v7.py --run RUN_DIR --split test --lock RUN_DIR/final_model_lock.json
```

공통 단위 시험은 간선 방향/밀집 연산 대응/그래프 gradient/v6 MLP 업데이트 동치/마스크/PPO ratio/시드·경기 경계/단기 상태·체크포인트 복원/벽 가림을 검사한다. 기존 v6 preflight 단위 시험 하나는 작업 폴더의 기존 체크포인트에 영향받지 않도록 출력 경로를 임시 디렉터리로 격리했다. v6 구현 파일은 변경하지 않았다.

실제 Unity smoke에서 v7-1, F0, F1 각각 64스텝을 실행했다. 짧은 실행에서 v7-1 약 23~31 env-step/s, v7-2 약 12~14 env-step/s를 관측했으며 장기 평균이나 승률 추정으로 사용하지 않는다. 제어 슬롯 실제 변위와 감각·출력 개입 결과는 `unity_causality.json`에 기록한다.

새 dev/confirmation/test는 계획의 41000/42000/43000 대역을 사용한다. 기존 텍스트 파일 484개의 seed 필드에서 겹침이 없었지만, 로그에 남지 않은 외부 과거 사용까지 미사용을 입증한 것은 아니다. 최종 test는 잠금 후 한 번만 시작할 수 있다. 단일 run의 평가 신뢰구간은 맵 A/B 쌍을 묶는다. 구조 개선 결론은 추가로 독립 학습 seed 간 변동을 포함해야 한다.

## 후속 범위

현재 설정으로 학습을 시작할 수 있지만, **v7-2 B2 재배선 전뇌와 B3 CNN/GRU의 완전한 대조 실험, 5유닛 장기 검증, F2 국소 가소성, 4천만 스텝 본실험은 완료하지 않았다.** F0/F1 파일럿만으로 생물학적 배선의 우월성이나 학습 성공을 주장하지 않는다. `controlled_slots`에 여러 슬롯을 지정하는 구조는 지원하되 5유닛 실전 성과는 별도 검증 대상이다.

공식 제출 경로의 reset/step·추가 의존성·자원 계약과 두 파일 독립 로딩이 검증되지 않아 `export_v7.py`는 명시적으로 차단된다. 이를 제출 가능 모델이라고 표시하지 않는다. F2 설정을 켜도 구현된 것으로 가장하지 않고 실패한다.

FlyWire 데이터는 CC BY-NC 4.0, MaleCNS는 CC BY 조건을 별도로 기록한다. 원본과 그래프/체크포인트의 배포 시 해당 데이터 귀속 조건을 확인해야 한다.

실제 Unity 개입 검증 결과(256스텝): 정상 입력 74스텝 이동, 감각 차단 66스텝 이동 및 정상 대비 행동 22스텝 차이, 출력 차단 0스텝 이동. 고정 프레임에서는 신경 활동은 달랐지만 행동은 같았다. F1 시작 검사는 이 보고서의 SHA-256과 그래프·감각·동역학 대응을 요구한다. 평가는 실행 중인 `latest.pt`를 직접 추적하지 않고 내용 해시로 고정한 `evaluation_checkpoints/` 사본을 사용한다.

## 2026-09-15 백그라운드 시작 기록

2026-09-17 확인: v7-1은 전체 완료, v7-2는 F0 완료 후 F1 첫 seed에서 경로 탐색 오류로 중단됐다. [현재 상태와 한 줄 평가 명령](pilot_status_2026-09-17.md)을 참고한다. 아래는 최초 실행 시점의 안내다.

2026-09-15에 두 순차 큐를 시작했다. 상태 파일은 `logs/v7/pilot_v7_1_queue/status.json`, `logs/v7/pilot_v7_2_queue/status.json`이다. 각각 현재 trainer PID와 run 경로를 표시한다.

```bash
# 큐 전체 정지: 현재 trainer에 종료 신호를 전달하고 저장을 기다린 뒤 큐 종료
kill -TERM "$(cat logs/v7/pilot_v7_1_queue/queue.pid)"
kill -TERM "$(cat logs/v7/pilot_v7_2_queue/queue.pid)"

# 이후 재개: 완료 run은 건너뛰고 미완료 checkpoint는 resume
.venv/bin/python scripts/launch_v7_queue_background.py --queue configs/v7/pilot_v7_1_queue.json
.venv/bin/python scripts/launch_v7_queue_background.py --queue configs/v7/pilot_v7_2_queue.json
```

큐 사용 중에는 큐 PID로 정지한다. 개별 trainer만 정상 종료하면 큐는 다음 job을 실행할 수 있다. 즉시 강제 종료 대신 SIGTERM을 사용해야 마지막 rollout 저장을 기다릴 수 있다.
