# v8 가속 파일럿 준비 — 2026-09-28

**2026-09-29 철회:** 수정 게임 및 추가 환경 채널 사용을 중단했다. 이 페이지의 실험 실행 명령은
현재 학습을 시작하지 않는다. [원본 환경 복구 및 최신 상태](environment_restoration.md)를 참조한다.

후속 실행 상태: 학습 6개와 dev 780경기가 완료됐고 최종 집계의 import 오류를 수정했다.
현재 결과와 동일 명령 재실행 동작은 [오류 수정·집계 복구 기록](import_fix.md)을 참조한다.
아래의 미실행 표시는 2026-09-28 준비 작업 당시 상태다.

실제 Unity 학습·평가를 시작하지 않고 실행 준비와 오프라인 검증을 마쳤다.
사용자가 다음 한 줄을 실행하면 분리된 백그라운드 관리자가 시작된다.

```bash
bash /Users/safeailab_macmini/Desktop/2026-IST-tech-RL/scripts/run_v8_fast.sh
```

터미널을 닫아도 계속 실행하며 `caffeinate`로 유휴 절전을 막는다. 별도의 `&`나
`nohup`은 필요 없다. 중복 실행은 관리자와 worker가 공유하는 파일 잠금으로 막는다.

## 실행 범위

| 항목 | 등록한 값 |
|---|---|
| 연구 arm / seed | C1, flat41 각각 11·22·33 |
| 학습 예산 | run당 1,048,576 step, 합계 6,291,456 step |
| 동시 작업 | 최대 6개, 포트 26000–26005 및 Unity 로그 분리 |
| 평가 checkpoint | 262,144 / 1,048,576 step |
| dev 평가 | 각 checkpoint의 30맵 × 양 진영, 고정 planner 포함 총 780경기 |
| 평가 작업 단위 | 5맵 × 양 진영 = 10경기; 빈 worker에 배정 |
| 결과 | 전체 run·map·side cube, C1/flat/planner 차이 및 bootstrap 구간 |

처음에는 6개 worker에서 타이머·결과·렌더링 계약 검사를 동시에 실행한다. 기존 연구 빌드의
종료 decision/tick, winner, score, 원시 semantic 이미지 해시와 일치해야 학습으로 진행한다.
이 검사는 사용자 실행 시 수행하며 **이번 준비 과정에서 통과했다고 주장하지 않는다**.
각 run은 중간 평가 checkpoint를 저장해도 에피소드를 끊지 않고 학습을 계속한다.
해당 run이 끝나면 보존된 중간/최종 checkpoint를 평가한다.

이는 계획서의 로컬 파일럿/dev 범위다. confirmation·held-out test, 성능에 따른 자동 예산 확대,
C2/C3, 공식 제출 인증은 포함하지 않는다. 학습·평가 예산은 임의로 줄이지 않았다.

## 적용한 가속

1. v7에서 검증한 native protobuf/queue 경로를 사용한다. 실행 전 원본 71개 소스와
   overlay 82개 파일·패키지 해시를 다시 검사한다. 원본 v7 파일을 수정하지 않는다.
2. 이 Mac의 12 CPU 코어(성능 코어 8개), 48GiB 메모리를 기준으로 독립 run 6개를 병렬 배정했다.
   각 worker의 Torch intra/inter-op 및 BLAS 스레드를 1개로 제한해 과도한 스레드 경쟁을 줄인다.
   6개가 실측 최적 병렬도라는 주장은 하지 않는다.
3. 후보 mask의 geometry 계산을 NumPy로 일괄 처리한다. 기존 부동소수점 계산 순서와
   후보별 제외 사유를 보존한다. v8이 폐기하던 구형 mask 계산은 learner feature 경로에서 제거했다.
   고정 상대의 planner는 그대로 사용한다.
4. 한 호출에서 공유하는 동일 graphic 배열의 검사·압축을 재사용한다. 호출 간 캐시는 두지 않아
   다음 frame에서 수정된 배열을 그대로 다시 검사한다. 관측 해시의 바이트 순서는 같다.
5. transition JSON을 한 번 직렬화하고, 겹치는 event/control window는 행 ID로 참조한다.
   `window_rows.jsonl`에 고유 행을 무손실 zlib로 저장하고 `windows.jsonl`에는 동일한 window
   구성·검열 여부·sampling metadata를 보존한다. `window_codec.decode_window()`로 복원하며
   export 재연도 새 형식을 지원한다. 기존 원본 JSON window도 읽을 수 있다.
6. 쓰기 큐는 8개로 제한하고 별도 thread에서 압축·버퍼 쓰기를 수행한다. checkpoint의 로그
   offset을 계산하기 전에 flush/fsync한다. 실패 suffix는 복구 디렉터리에 보존한다.
7. Unity의 표시 크기를 128×128, target frame rate를 -1로 설정한다. actor가 읽는 semantic
   관측은 여전히 96×96이다. 렌더링을 유지하는 batchmode를 사용한다.
8. 완료된 작업은 재실행하지 않고, 남은 dev shard를 빈 worker에 배정해 평가의 긴 대기열을 줄인다.

PPO, rollout/minibatch, 학습률, 보상, 상대 schedule, 맵 분할, action repeat, physics dt,
`time_scale=50`은 유지했다. 높은 time scale, `-nographics`, MPS 전환, 혼합정밀도/양자화,
패키지 업그레이드를 추가하지 않았다. 물리·graphic 관측을 바꾸거나 작은 CPU 정책에서
전환 비용과 수치 차이를 검증하지 못한 방법이기 때문이다.

## 오프라인 검증과 측정 한계

- 전체 회귀: **271개 중 270개 통과, 1개 조건부 skip**.
- mask 경계값·양 진영, 같은 배열의 mutation, window 완전 복원, 로그 I/O 오류,
  중간 crash/재개·정상 중단, 최종 저장 직후 복구를 검사했다.
- 합성 FixtureEnv의 PPO update에서 기존/가속 모델 가중치와 노출 장부가 정확히 일치했다.
  실제 게임 학습은 수행하지 않았다.
- 과거 Unity 로그의 두 에피소드, 48번 추론 재생에서 행동·mask·planner state가 정확히 일치하고
  log probability 최대 차이는 0이었다. 기존 모델은 추론용으로만 읽었다.

| 기존 기록으로 측정한 작업 | 결과 |
|---|---|
| mask 중앙 처리 시간 | 약 5.55배 빠름 |
| 관측 검사 / 압축 중앙 처리 시간 | 약 1.79배 / 4.40배 빠름 |
| 추론 요청 중앙 시간 | 기존 2 threads 9.16ms → 가속 1 thread 5.29ms |
| cold planner 초기화를 포함한 48요청 합계 | 1.199초 → 1.018초 |
| 기존 8개 window 로그의 무손실 변환 | 67,129,865 → 3,511,467 bytes (5.23%) |

짧은 과거 smoke 표본의 결과이므로 전체 학습의 배속·완료 시간·최종 디스크 사용량을
확정하는 수치는 아니다. 6-worker scaling, 장시간 발열, 실제 Unity 처리량은 아직 측정하지 않았다.
공간이 20GiB 아래로 내려가면 저장 후 중단을 요청한다. 30분간 step 진전이 없는 worker는
watchdog가 실패로 기록한다. 중단 요청 후 유예 시간은 5분이며 초과 시 마지막 불변 checkpoint를
사용하게 된다. 살아 있는 다른 실험의 Unity PID에는 종료 신호를 보내지 않는다.

## 상태·중단·재개

```bash
bash scripts/run_v8_fast.sh --status  # 상태와 run별 step
bash scripts/run_v8_fast.sh --stop    # 완료된 PPO update에서 저장 후 중단 요청
bash scripts/run_v8_fast.sh --check   # 파일/환경 검사만; Unity를 시작하지 않음
```

위 상대 경로 명령은 프로젝트 루트에서 실행한다. 중단 후에는 처음의 한 줄을 다시 실행한다.
학습은 마지막 저장된 PPO update부터 재개하고 물리 상태는 새 에피소드에서 시작한다.
자동 실패 재시도는 하지 않는다. 사용자가 재실행하면 미완료 dev shard 전체를 새 attempt로
실행하고 이전 실패 기록을 보존한다. 성공한 셀만 골라 승률을 계산하지 않는다.

- 관리 로그: `logs/v8/accelerated_pilot_v1/console.log`
- 전체 상태: `logs/v8/accelerated_pilot_v1/status.json`
- 학습 진행/저장: `logs/v8/accelerated_pilot_v1/runs/<run>/`
- worker·Unity 로그: `logs/v8/accelerated_pilot_v1/jobs/<job>/attempt-*/`
- 완료 집계: `logs/v8/accelerated_pilot_v1/summary.json`

## 근거와 등록

- [설정·소스 등록](../../reports/v8/acceleration/registration.json)
- [전체 회귀 검사](../../reports/v8/acceleration/regression_tests.txt)
- [오프라인 성능·동등성 결과](../../reports/v8/acceleration/offline_benchmark.json)
- [실행 전 검사](../../reports/v8/acceleration/preflight.txt)
- [준비 상태 기록](../../reports/v8/acceleration/preparation.json)
- [가속 전 소스 목록](../../reports/v8/acceleration/base_sources.json)과
  [가속 전 소스 보관본](../../reports/v8/acceleration/pre_acceleration_sources.zip)

과거 `engineering_validation.json`과 smoke checkpoint는 그 당시 소스의 검증 기록이다.
새 소스 해시로 덮어쓰거나 새 실행의 검증 결과로 가장하지 않는다. 이번 연구는 별도 등록과
새 run 경로를 사용한다. 실행을 시도한 뒤에는 등록을 덮어쓸 수 없으며 소스 변경 시 재개를 거부한다.

웹 자료는 공식 문서를 브라우저로 확인했으며 파일 다운로드나 의존성 설치를 하지 않았다.
[PyTorch 성능 안내](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide)는 CPU 스레드와
추론 오버헤드 검토에, [multiprocessing 안내](https://docs.pytorch.org/docs/2.14/notes/multiprocessing.html)는
프로세스별 CPU 스레드 제한에 참고했다. 설치된 Torch 버전은 유지했다.
[Unity LLAPI 문서](https://unity-technologies.github.io/ml-agents/Python-LLAPI/)의 worker ID,
engine configuration 및 time scale 주의사항과
[Unity FAQ](https://unity-technologies.github.io/ml-agents/FAQ/)의 visual observation 렌더링 요구사항을
반영했다. 문서의 일반적인 가속 방법을 이 프로젝트의 실측 성능 증거로 취급하지 않는다.
