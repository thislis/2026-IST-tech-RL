# v9 실행 및 구현 안내

v9는 원본 `artifacts/builds/BlackOut.app`, 제공 `blackout_env`, 관측 생성·전처리, 게임 reward를 수정하지 않는다. 새 코드는 `code/v9/blackout_v9/`에 있으며 v1~v8 모델·실행 코드는 보존한다. 구현 기준은 [v9 계획](plans/v9_plan.md)이다.

## 실행

2026-10-09 재배치 후에는 새 실험 이름으로 실행한다. 기존 완료 실험은 `--status`로 조회하며, 이전 소스 해시 등록에 현재 소스를 이어 붙이지 않는다.

어느 디렉터리에서든 다음 한 줄을 실행한다. `&`나 `nohup`을 붙일 필요가 없다.

```bash
bash /Users/safeailab_macmini/Desktop/2026-IST-tech-RL/code/v9/run_v9_fast.sh --name attention_relocated_v1
```

사전 검사 뒤 background supervisor의 PID·로그 경로를 출력하고 셸로 돌아온다. 터미널을 닫아도 실행되며, Unity는 `-batchmode`와 `no_graphics=False`로 렌더링을 유지하고 창은 숨긴다. macOS idle sleep은 `caffeinate`로 방지한다. 로그아웃·재부팅 후 자동 재시작은 설정하지 않는다.

같은 명령에 다음 옵션을 붙인다.

| 옵션 | 동작 |
|---|---|
| `--check` | 원본 해시·의존성·격리된 제출 파일 검증. Unity·학습 시작 없음 |
| `--status` | 현재 단계, worker 진행, run별 유효 step·checkpoint·메모리 확인 |
| `--stop` | 정상 중단 요청. 현재 PPO update는 완료 후 저장할 수 있음. `--status`에서 종료 확인 |
| `--resume` | 같은 source/config/package 등록으로 재개. Unity 물리 상태는 복원하지 않고 새 경기부터 시작 |
| `--benchmark` | 녹화 관측으로 CPU/MPS 추론 측정. Unity·gradient update 없음 |
| `--name NAME` | 별도 실험 이름. source/config를 바꾼 경우 새 이름 필요 |
| `--config /absolute/config.json` | 새 등록용 설정. 기존 실험의 설정을 덮어쓰지 않음 |

기본 로그 경로는 `logs/v9/attention_original_v1/`이다. 전체 v9 supervisor는 한 개만 실행 가능하며 최초 4개, 최대 8개의 Unity worker를 사용한다. v8 실행 명령은 변경하지 않는다.

## 자동 진행 순서

1. 원본과 제출 계약 검사, CPU/MPS 소배치 추론·합성 역전파 probe. 실행할 때 실제 장비 상태로 learner를 선택한다.
2. 최소 20개 원본 경기 lifecycle 검사. 이 단계에서는 optimizer를 갱신하지 않는다. 정상 종료, 11채널 graphic, 창 없음, 초기 관측 중복을 기록한다. 긴 timeout 경기 때문에 이 단계에도 시간이 걸릴 수 있다.
3. seed 11 기능 pilot: 목표 131,072 env step. 관측으로 확인한 pickup·delivery와 이동이 없으면 본 실험을 시작하지 않고 원인을 보고한다.
4. A(single head), B(+history/PFSP), C(+전략 mixture/보조학습)의 3 arms×3 seeds(11/22/33), 각 1,048,576 step. 목표 합계 9,437,184 step. 각 run은 별도로 새로 초기화한다.
5. 중간 평가와 run당 최종 30 repetitions×2 sides×4 opponent families=240경기. 수집·rush·약탈·기존 target을 구분한다. random Attention 및 v8 제출물도 같은 runner로 비교한다.
6. 개발 평가로 선택한 arm을 seeds 44/55에서 추가 확인한다(목표 2,097,152 step). 선택 checkpoint는 이 확인 결과·test로 다시 고르지 않는다.
7. 별도 action RNG test 반복, 동일 초기 baseline test, 두 파일 제출 검증·export. 외부 사이트로 업로드하지 않는다.

완료 episode 단위 학습이므로 마지막 wave는 예산을 초과할 수 있다(최대 실제 worker 수×22,000 step). pilot·lifecycle·평가·확인·폐기 시도는 본 실험 step과 구분한다. 강제 종료 때 정확히 읽지 못한 step은 heartbeat의 하한값이며 `unmeasured_crash_attempts`로 표시한다.

## 모델·보상·속도

- `policy.py`는 torch와 Python 표준 라이브러리만 사용한다. 96-vector 전체와 96×96×11 graphic 전체를 입력으로 쓰는 CNN 없는 소형 Attention이다. 4×4 pixel patch 576개, entity 10개, context 1개를 latent 32개로 읽고 latent Attention 2층을 적용한다. 자기 class는 독립 query에 반영해 공통 장면 연산을 재사용한다.
- 학습·기본 평가·제출은 모두 같은 categorical sampling decoder다. 클래스가 같은 아군의 입력이 동일해도 별도로 행동을 sample한다. batch row 번호를 자기 유닛 ID로 사용하지 않는다.
- 기본 학습 목적은 제공 `run_match()`와 같은 누적 raw reward의 **최종 상대적 승패 부호**다. 원본 raw reward 자체를 dense 보상으로 더하지 않는다. γ=1의 bounded potential shaping으로 완료 경기 return의 절댓값은 1.25 이하이다.
- 완전 episode를 임시 버퍼에 저장하고 정상 종료 검증을 통과한 데이터만 PPO에 반영한다. 22,000-step watchdog·외부 heartbeat/wall watchdog·비정상 종료는 무승부가 아니다. invalid 비율 1% 초과 또는 3회 연속 결함이면 중단한다.
- CPU 소배치 actor와 CPU/MPS learner를 비교해 선택한다. 2026-10-04 검증에서 B=5 CPU 추론 p50 약 0.90ms, 128 scene 역전파 CPU 약 80ms/MPS 약 21ms였으나 전체 Unity 처리량 배속으로 해석하면 안 된다.
- 공통 graphic을 한 번만 저장하고 **정확한 one-hot**인 경우 uint8 ID map으로 lossless 압축한다. nonbinary 값은 fp32 그대로 저장한다. PPO encoder는 매 update 다시 계산한다.
- worker 수 2/4/6/8을 각각 세 wave의 실제 수집 처리량으로 비교하는 제한된 자동 조정이 있다. 상대·episode 길이가 변하므로 통제된 benchmark의 배속 주장으로 사용하지 않는다. 메모리 압력이 높으면 병렬 수를 줄인다.
- CUDA 전용 FlashAttention, API 통신 overlay, hidden reset step, observation 채널 축소, 게임 시간 간격 변경은 사용하지 않는다. compile·mixed precision·world model·self-ID가 필요한 stateful policy는 기본 구현에 넣지 않았다.

## 결과·제출 파일

학습 중에는 `runs/<arm>_s<seed>/progress.json`, `training.jsonl`, `episodes.jsonl`을 확인한다. `console.log`에는 supervisor 진행 요약이, 각 attempt에는 원본 Unity/worker 로그가 남는다. 잘못된 시도는 삭제해 성공률을 부풀리지 않는다. 큰 staging 파일은 학습 반영 또는 폐기 후 정리하고 최근 250 step의 진단 자료만 보존한다.

최종 `summary.json`에 선택 모델과 비교 결과가 기록되고 다음 두 파일이 생성된다.

```text
artifacts/submission/v9<checkpoint_sha256>/
  policy.py
  checkpoint.pt
```

`MyPolicy(vector_size=96, n_channels=11).forward(vector, graphic)`는 float32 `(B,96)`, `(B,11,96,96)`을 받아 float32 `(B,2)`, 각 좌표 `[-1,1]`을 반환한다. `checkpoint.pt`는 `{"policy_state": state_dict}` 형식이다. 원본 `load_checkpoint()`와 격리된 `python -I`에서 검증하며 모델 코드에서 프로젝트 모듈이나 외부 데이터 파일을 요구하지 않는다.

현재 제공 입력에는 self-ID가 없고, 문서의 점수 기준과 로컬 runner의 누적 reward 승패 기준도 일치하지 않는다. 구현은 계획의 **V9-S / provided_runner_v1**에 해당한다. 이 한계를 관측·게임 수정으로 우회하지 않는다. 결과의 `format_ready`, `local_runner_improved`, `game_outcome_verified`, `official_server_certified`를 구분하며 최종 두 상태를 임의로 참으로 표시하지 않는다. 초기 맵 적용을 검증하지 못한 seed 반복은 새로운 맵 일반화 결과로 쓰지 않는다.

## 준비 단계 검증

합성 환경 회귀 검증에는 shape·채널 gradient, sample decoder 일치, bounded reward/GAE 경계, PopArt 보존, nonbinary graphic 복원, 손상 episode의 update 차단, 실제 MPS optimizer/CPU 저장·복원, 원본 loader/runner, detached supervisor·중단·watchdog가 포함된다. 실행 시 전체 lifecycle과 pilot은 새로 수행한다.

```bash
.venv/bin/python -W ignore::DeprecationWarning -m unittest tests.v9.test_v9 -v
```

원본 Unity 32-step 연결 진단은 별도 명령이며 학습하지 않는다.

```bash
.venv/bin/python -m blackout_v9.smoke
```

검증 자료는 `logs/v9/reports`에 저장하고 Git에는 올리지 않는다. 모델·설정·launcher·테스트·CLI 두 개만 소스로 추적한다.
