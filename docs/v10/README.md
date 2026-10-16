# v10 — 단일 Attention actor + 독립 지도 critic

[구현 계획](plans/v10_plan.md)에 따라 C0(vector critic)와 C1(지도 Attention critic)의 MAPPO 비교를 구현했다. actor는 두 구조 모두 120,905개 파라미터, critic은 C0 29,057개 / C1 128,769개다. 게임·제공 환경·obs 생성 코드는 수정하지 않는다.

## 실험 시작

다음 한 줄로 준비 검사, CPU/MPS 측정, P0–P6 실험, 두 파일 제출물 생성까지 백그라운드에서 진행한다. 가상환경 활성화, 별도 `cd`, `&`, `nohup`이 필요 없다.

```bash
bash /Users/safeailab_macmini/Desktop/2026-IST-tech-RL/code/v10/run_v10_fast.sh --name mapcritic_v1
```

학습·평가·lifecycle은 원본 앱의 렌더링을 유지하면서 `-batchmode` wrapper로 창을 숨긴다. `-nographics`는 사용하지 않는다. foreground에서는 PID와 로그 경로를 출력하고 반환한다. 시간이 걸리는 준비 검사도 supervisor에서 실행한다. 실패하면 `--status`와 `console.log`에 이유를 남긴다.

```bash
# 상태·현재 run·진행량·측정 이후의 잠정 ETA
bash /Users/safeailab_macmini/Desktop/2026-IST-tech-RL/code/v10/run_v10_fast.sh --name mapcritic_v1 --status

# 소유 worker만 종료하고 마지막 완료 wave checkpoint 보존
bash /Users/safeailab_macmini/Desktop/2026-IST-tech-RL/code/v10/run_v10_fast.sh --name mapcritic_v1 --stop

# 동일 소스·설정·의존성으로 재개; Unity의 중간 물리 상태는 재개하지 않음
bash /Users/safeailab_macmini/Desktop/2026-IST-tech-RL/code/v10/run_v10_fast.sh --name mapcritic_v1 --resume

# Unity·본학습 없이 원본 해시·모델 수학·독립 제출 loader 검사
bash /Users/safeailab_macmini/Desktop/2026-IST-tech-RL/code/v10/run_v10_fast.sh --check

# 저장되지 않는 임시 모델과 합성 입력으로 CPU/MPS 측정
bash /Users/safeailab_macmini/Desktop/2026-IST-tech-RL/code/v10/run_v10_fast.sh --benchmark
```

`--prepare`는 등록만 만들고 종료한다. `--device auto|cpu|mps`는 learner, `--actor-device auto|cpu|mps`는 수집 actor 장치다. 명시적 장치 옵션을 사용한 실험은 재개할 때 같은 옵션을 사용한다. 기존 등록이 있는 이름에서 소스/설정이 바뀌면 새 이름을 사용한다. 이전 실행 결과를 덮어쓰지 않는다.

## 가속과 자원

- actor는 팀 내 공통 장면을 검증한 뒤 한 번 인코딩하고 5 query를 계산한다. 원본 11채널 96×96 입력을 보존한다.
- worker는 actor만 실행한다. 자연 종료된 경기의 old value는 동결 critic snapshot으로 learner에서 일괄 계산한 뒤 GAE를 생성한다.
- 실제 CPU/MPS 수치 검증과 actor+critic forward/backward+Adam+전송 시간을 비교한다. `runtime_profile.json`에 장치·스레드·value batch(128/256/512) 측정을 남긴다. C0/C1에 같은 선택을 적용한다.
- 기본 worker 4, cap 8이며 본 비교 중 처리량을 이유로 임의 변경하지 않는다. 전체 study는 한 run씩 실행한다. 관전과 실험은 `logs/runtime/compute.lock`을 공유해 자원 경합을 막는다.
- 2026-10-11 M4 Pro에서 합성 scene batch 128의 C0+C1 합산 측정은 CPU 약 0.312초, MPS 약 0.144초였다. batch 5 actor는 CPU가 빨랐다. 전체 Unity 실험 속도에 대한 수치가 아니며, 시작 시 다시 측정한다.
- 정확한 one-hot graphic만 lossless uint8 저장하고 나머지는 float32로 보존한다. 디스크 최소 시작 여유 25 GiB, 중단 reserve 15 GiB 외에도 wave의 최악 float 저장량을 확인한다. 메모리 soft limit 32 GiB 초과 시 마지막 완료 checkpoint를 보존하고 중단한다.
- auto MPS 학습 중 지원/메모리 오류가 나면 완료 checkpoint에서 CPU로 재개하고 `device_fallback.json`에 기록한다. 실행 도중 장치가 바뀐 비교는 그 한계를 함께 보고한다. 명시적 MPS 실패는 조용히 다른 장치로 바꾸지 않는다.

## 비교 실험

기본 실행은 lifecycle 20경기, P2 고정 actor 16경기(12 fit/4 holdout), C0/C1 pilot, seeds 11/22/33 본 비교, 두 arm 모두 seeds 44/55 확인, 선택을 동결한 C1/C0/초기 actor test를 포함한다. 학습 목표 합계는 10,747,904 env step이고 완료 wave 단위 초과분은 별도로 기록한다. 정규 평가 3,520경기와 pilot 평가 64경기, lifecycle·P2는 구분한다.

같은 seed의 actor 초기 해시와 counter 기반 상대 task 순서를 맞춘다. PPO permutation RNG와 상대 일정은 독립적이다. 주 비교에는 BC·auxiliary·history/PFSP·self-ID·기억을 넣지 않는다. C1-blind critic은 P2에서 진단하며, 본학습의 지도 정보 인과 주장은 계획의 별도 후속 대조가 필요하다. ledger·teacher·self-ID·목표/GRU/역할 확장은 기본 자동 실행에 포함하지 않는다.

평가·관전의 `match.py`는 제공 `run_match`와 동일한 팀 reward 누적/진영 교체/승패 계산을 사용하면서 partial terminal·truncation·시간 역행을 추가 검사한다. 제공 runner가 truncation을 무시하는 문제 때문에 v10 쪽 public API 경기 루프에서 검사한다. 원본 runner와 게임 파일은 그대로이며 합성 경기에서 양 진영 결과 동치를 검증한다. 로컬 누적 reward 판정과 공식 대회 점수의 차이는 계속 명시한다.

## 이전 모델과 관전

학습 후 selection lock이 생성되면 다음 한 줄로 v10 대 v9를 볼 수 있다. 기본 2경기·정상 배속이며 두 번째 경기에서 진영을 교체한다.

```bash
bash /Users/safeailab_macmini/Desktop/2026-IST-tech-RL/code/v10/watch_v10_models.sh --study mapcritic_v1 --opponent v9
```

`--opponent v8`, `v6`, `v7-1`, `all`도 지원한다. `--list`는 가용한 가중치·경로·해시를 표시한다. `--check`는 지정 모델들을 로드·검증하지만 Unity를 시작하지 않는다. `--games N`, `--speed 1`로 경기 수와 배속을 지정한다.

학습 중 snapshot을 보고 싶으면 `--checkpoint /절대/경로/checkpoint.pt`를 지정한다. 체크포인트를 자동 export·검증하므로 별도 export 명령이 필요 없다. 학습이 자원을 사용 중이면 관전은 충돌을 안내하고 종료한다. v10 snapshot이나 선택 모델이 없는 상태에서 임의 미학습 모델로 대체하지 않는다.

- 기본 첫 경기: **Blue=v10, Red=지정한 이전 버전**. 다음 경기에서 반대다. 매 경기 버전·run·step(기록된 경우)·가중치 해시를 출력한다. 원본 TeamA/TeamB 색상 asset과 API agent 인덱스를 기준으로 연결한다.
- v8·v9는 제공 두 tensor 입력의 제출 모델, v6·v7-1은 planner 상태가 포함된 `legacy_exhibition` 상대다. 과거 adapter의 ID/상태는 v10 actor에 전달되지 않는다.
- 관전만 원본 게임 창을 표시한다. Ctrl+C·watchdog·오류 때 소유 Unity 프로세스를 정리한다. 관전 로그는 학습·성능 평가에 합치지 않는다.
- 기존 `bash code/shared/watch_best_models.sh`의 v6 대 v7-1 관전도 유지한다.

## 산출물과 검증 상태

| 경로 | 내용 |
|---|---|
| `code/v10/` | 새 모델·학습·실행·관전·설정·테스트 |
| `logs/v10/mapcritic_v1/` | 등록·runtime profile·학습·평가·selection lock·summary |
| `logs/v10/watch/<session>/` | 관전 모델 목록·진영·결과 |
| `logs/v10/reports/` | 장치 측정 등 자동 보고서 |
| `artifacts/submission/v10/<hash>/` | `policy.py` + `checkpoint.pt` |

제출 actor는 float32 vector `[B,96]`, graphic `[B,11,96,96]` 두 입력만 받고 float32 `[B,2]`, `[-1,1]` 행동을 반환한다. `python -I`와 제공 loader로 두 파일만 로드하며 batch 0/1/3/5/10, permutation·split·분포 일치·STOP을 검사한다. critic·optimizer·teacher·관전 adapter는 제출물에서 제외한다. 외부 제출은 자동 실행하지 않는다.

구현 검증은 합성 경기와 임시 optimizer, 실제 로컬 과거 모델 가중치로 수행했다. v10 테스트 15개와 기존 v9·관전 회귀 테스트 22개가 통과했고, 최종 통합 변경 2개도 재검증했다. 기록은 `logs/v10/reports/implementation_verification.json` 및 같은 폴더의 테스트 로그에 있다. **실제 Unity·학습 실험은 구현 과정에서 시작하지 않았다.** 창 숨김·자연 종료의 실환경 검증은 첫 실행의 P1에서, 실제 관전은 관전 명령 실행 때 확인한다. 소스/형식 준비와 실제 경기 검증·성능 향상을 구분한다.

```bash
.venv/bin/python -W ignore::DeprecationWarning code/run.py -m unittest tests.v10.test_v10 -v
```
