# v11 구현 및 실행 확인

2026-10-24. v11 정책·학습기·평가기·제출물 생성·백그라운드 실행기를 구현했다. 기본 전체 실험은 아직 시작하지 않았으며, 이번 기록은 구현 검사와 짧은 실제 실행·장치 측정 결과다.

## 구현

- torch/stdlib만 필요한 standalone 정책: tensor planner, 3 worker+2 guard 초기 역할, Attention, unit/team GRU, 목표·역할 및 제한된 residual. P/R/G/F/D와 두 tensor 공개 인터페이스를 지원한다.
- 제공 runner의 canonical 팀 다섯 행 순서로 자기 entity를 연결한다. B 진영은 y=x 반사를 사용하고 경기·진영·respawn 상태를 분리한다. 임의 partial/shuffled batch는 지원 계약에 포함하지 않는다.
- event 기반 lossless replay, 완결 episode wave, recurrent PPO, 독립 map critic·PopArt, 가변 기간 return, 실제 상대별 env step deficit, 원자적 checkpoint 및 재개를 연결했다.
- 교사 데이터는 별도 수집하고 PPO에서 제외한다. G/F는 같은 seed·자료에서 encoder/GRU 초기화를 일치시키고 D의 직접 행동 head 지도 학습은 공유 encoder gradient에 들어가지 않는다.
- dev 선택, confirmation 전 선택 고정, 추가 seed, frozen test 및 두 파일 패키징을 P0–P6 실행기에 연결했다. raw reward·관측 score·실제 map hash·진영·seed 결과를 분리한다.
- 기본 전체 학습 예산은 34,340,864 env step이다. 추가 ablation은 별도 명령으로 실행한다.

## 실제 확인한 범위

| 확인 | 결과 |
|---|---|
| 핵심 검사 | 22개 통과: planner 경로·용량·창고 실패, event/likelihood/GRU, teacher 공통 초기화, PPO·PopArt·scheduler, 평가·runtime |
| 두 파일 독립 로딩 | 제공 `load_checkpoint`로 12회 sequence 및 시계/진영 reset 동치 통과 |
| ONNX | planner·GRU·명시 상태를 포함한 32회 synthetic sequence의 action/state parity 통과, action 최대 오차 0 |
| 짧은 숨김 실행 | 128 env step, graphic 관측 존재, 소유한 화면 창 0개 |
| 자연 종료 → 학습 → 재개 | v6 상대 1경기, 1,355 env step, 1회 PPO update, invalid 0, 재개 후 동일 가중치·추가 경기 없음 |
| 자원 측정의 실제 경기 | worker 2/4/8에서 14개 짧은 probe, 모두 유효·창 숨김·소유 자식 종료 확인 |
| 다른 cwd | 절대 경로 셸 명령의 `--status` 동작 확인 |

자연 종료 검사는 patrol 도착 event의 마지막 최적화 이전 정책으로 수행했다. 그 이후에는 해당 event 회귀 검사·ONNX parity와 최종 소스의 실제 worker probe를 수행했다. 단일 경기의 11회 반납 관측 proxy는 파이프라인 작동 증거이며 성능 개선의 통계적 근거가 아니다. 처리량 probe의 조기 종료는 자연 종료 학습 episode로 집계하지 않았다.

검사 산출물:

- [필수 검사](../../../logs/v11/reports/preflight.json)
- [자연 종료·학습·재개](../../../logs/v11/reports/integration_smoke/result.json)
- [ONNX 상태 전달](../../../logs/v11/reports/onnx_initial/manifest.json)
- [짧은 숨김 실행](../../../logs/v11/implementation_smoke_01/smoke_result.json)

## M4 Pro 측정과 선택

M4 Pro 12 CPU cores, 통합 메모리 48GB, 실제 host 프로세스의 MPS 가용성을 확인했다. actor/planner 조합, learner CPU threads 1/4와 CPU/MPS batch 4/8/16/32/64, 실제 Unity worker 2/4/8을 측정했다. 학습 측정은 warm-up을 제외하고 32 burn-in + 64 learning event의 입력 이동·forward/backward·optimizer update 및 prefetch 메모리를 포함한다.

| 작업 | 선택 | 측정 |
|---|---|---|
| 추론·planner | CPU, 2 threads | 평균 2.41ms, p95 6.55ms/팀 호출 |
| learner | MPS, batch 32 | 약 1,678 learning events/s |
| 가장 빠른 CPU learner | CPU, 4 threads, batch 32 | 약 703 learning events/s |
| Unity | 8 workers | startup/종료 포함 약 116 env steps/s |

MPS learner는 이 연산 벤치마크에서 가장 빠른 CPU 후보보다 약 **2.39배** 빨랐다. 전체 학습 wall time의 배율이나 최종 게임 성능을 뜻하지 않는다. 8 workers는 4 workers의 약 76 env steps/s보다 빨랐으며 모든 화면 감사에서 소유 창이 없었다. 선택한 MPS 학습 후보의 보수적 통합 메모리 측정은 약 10.85GiB였고, 실제 worker probe의 최대 RSS는 약 6.20GiB였다.

[최종 runtime profile](../../../logs/v11/hardware_m4pro_v11/runtime_profile.json)은 현재 소스·설정·패키지와 일치한다. 같은 host 환경에서 기본 전체 실험을 시작하면 이 측정 결과를 캐시에서 재사용한다. 소스·설정·장치 조건이 달라지면 재측정한다.

## 실행과 남은 실험

```bash
bash /Users/safeailab_macmini/Desktop/2026-IST-tech-RL/code/v11/run_v11_fast.sh --name goal_memory_v1
```

기본 전체 P0–P6 학습·평가, 성능 gate 통과, 학습된 최종 제출물 선정은 미수행이다. 실행기는 필요한 gate에 실패하면 이유를 남기고 정지한다. 공개 두 파일의 PyTorch 실행과 명시 상태 ONNX 개발 그래프를 구별하며, 최종 ONNX 서버의 상태 연결 규약을 인증했다고 주장하지 않는다. 상세한 상태 조회·중지·재개·선택 실험 명령은 [v11 안내](../README.md)에 있다.
