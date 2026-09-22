# v7 본실험 중단·가속 재개 준비 — 2026-09-21

## 현재 상태와 실행 명령

사용자 요청으로 원래 순차 실행 관리자에 SIGTERM을 보냈고, 하위 학습이 rollout 경계에서
저장한 뒤 종료했음을 확인했다. A0·A1 시드 11은 각 2,000,000 step 완료,
A2 시드 11은 **147,456 step**에서 정상 중단했다. 이 작업에서 본실험을 다시 시작하지 않았다.
별도 임시 환경의 짧은 수집 벤치마크와 연산 동등성 검사만 실행했다.

```bash
bash "/Users/safeailab_macmini/Desktop/2026-IST-tech-RL/scripts/start_connectome_fast.sh"
```

이 명령이 중단한 본실험을 재개한다. `--check`는 사전 검사, `--status`는 동시 실행 목록과
현재 학습 스텝 표시, `--stop`은 정상 종료 요청이다. 종료 요청 후 모든 진행 중 rollout의
체크포인트 저장과 Unity 종료를 기다린다. 강제 종료나 PID 추측 대신 이 옵션을 사용한다.

## 보존한 실험 조건

원본 등록 `main_study_registration.json`, 학습 코드, 모델·그래프·보상·상대·맵·학습 예산,
rollout/minibatch/epoch, `time_scale=50`, action repeat 1, 감각 갱신 간격, float32 정밀도를
변경하지 않았다. 한 모델의 rollout을 여러 게임에서 합치지 않고 **서로 독립적인
모델·시드 실행**을 병렬 배치한다. PPO 조건과 시드별 RNG 스트림은 유지한다.

기존 세 체크포인트는 `logs/v7/main_study/pre_acceleration_backup/`에 복사했고
`acceleration_resume_points.json`에 SHA-256을 기록했다. 준비 과정에서 원본 checkpoint를
다시 쓰거나 새 source hash로 바꾸지 않았다. A2는 원래 `--resume` 경로로 optimizer와
RNG를 복구한다. Unity의 진행 중 물리 상태는 기존 재개 계약대로 폐기하고 새 경기에서 시작한다.

가속은 별도 실행 오버레이다. 추가 코드·패키지·생성된 protocol 모듈을
`acceleration_registration.json`에 고정하고, 각 run의 `acceleration_runtime.json`과 이후
체크포인트 `lineage`의 `execution_acceleration` 항목에 실제 실행 backend를 남긴다.
원래 source 검사와 새 runtime 검사를 모두 수행한다. 기존 스크립트와 공유하는 manager,
queue, evaluation lock 및 개별 run lock으로 중복 실행을 막는다.

## 병목과 적용한 가속

프로파일에서 순수 Python Protobuf 파싱과 같은 프로세스 안의 `multiprocessing.Pipe`
pickle 복원이 큰 비용을 차지했다. 기존 `.venv`는 Protobuf 3.20.3을 그대로 유지한다.
`logs/v7/fast_runtime/packages/`에 격리 설치한 Protobuf **4.25.9 upb**만 가속 자식에서
사용한다. 설치된 ML-Agents의 정확한 serialized descriptors로 호환 Python 모듈을 만들고,
원본과 모든 wire descriptor 해시가 일치함을 검증했다. gRPC wire protocol은 바꾸지 않았다.
upb의 native 파서와 Apple Silicon 지원은 [Protobuf 공식 설명](https://protobuf.dev/news/2022-05-06/)을 참고했다.

gRPC 서비스 스레드와 학습 스레드 사이의 handoff를 순서가 보존되는 in-process queue로
바꿔 불필요한 직렬화·역직렬화를 제거했다. 이 변경은 **동일 프로세스 내부**에만 적용하며,
서로 다른 학습 프로세스끼리 메시지나 rollout을 공유하지 않는다.

Unity 표시 창은 128×128, `target_frame_rate=-1`로 설정한다. semantic observation의
96×96×11 크기와 렌더링은 유지한다. `no_graphics`로 관측을 없애거나 time_scale·물리 dt를
바꾸지 않는다. 설정 경로는 [Unity ML-Agents 저수준 API](https://github.com/Unity-Technologies/ml-agents/blob/develop/com.unity.ml-agents/Documentation~/Python-LLAPI.md)를 확인했다.

PyTorch 학습 설정의 2스레드는 유지하고 BLAS/Accelerate의 과도한 중첩 스레드를 제한한다.
`OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `OMP_NUM_THREADS=2`,
`MKL_NUM_THREADS=2`를 자식 실행에 적용한다. 병렬화와 스레드 수의 상호 영향은
[PyTorch 성능 안내](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide)를 참고하고 이 Mac에서 측정했다.

## MPS 선택

실제 Mac16,11, CPU 12코어(성능 8 + 효율 4), 메모리 48GB, PyTorch 2.13.0에서 측정했다.
MPS는 샌드박스 안에서는 사용 불가로 보였으므로 실제 Metal 장치 접근이 가능한 실행에서 검증했다.

| 연산 | CPU | MPS, 전송·동기화 포함 | 결정 |
| --- | ---: | ---: | --- |
| v7-1 인코더, batch 5 | 1.73 ms | 3.89 ms | CPU 유지 |
| v7-2 전체 CSR × spikes | 19.10 ms | 5.15 ms | MPS 사용 |
| v7-2 신경 상태 한 tick 전체 | 20.85 ms | 7.14 ms | CSR만 MPS, 나머지 CPU |

전뇌 CSR은 [PyTorch 공식 Metal shader API](https://docs.pytorch.org/docs/main/generated/torch.mps.compile_shader.html)로 작성했다.
노드·간선 전체를 유지하며 행마다 원래 CSR 순서로 float32 곱과 합을 계산한다.
밀집 166,700×166,700 행렬을 만들거나 희소 간선을 잘라내지 않는다. 고정 CSR 배열은 GPU에
계속 두고 spike 배열과 결과만 전송한다. 잡음 RNG·전압·감각 필터·출력 이력은 CPU의 원래
코드 그대로여서 checkpoint 형식도 유지한다. 작은 인코더를 MPS에 억지로 옮기지 않았다.

## 속도 측정

| 조건 | 수집 처리량 |
| --- | ---: |
| v7-1, 원본, 512 step 비교 | 28.45 step/s |
| v7-1, native/queue/창 설정, 512 step | 57.25 step/s |
| v7-2, 원본, 256 step 비교 | 13.24 step/s |
| v7-2, 통신 + MPS + 스레드 제한, 512 step | 51.52 step/s |
| 4개 동시 수집, 그중 전뇌 1개 | 합산 200.21 step/s |
| 6개 동시 수집, 그중 전뇌 1개 | 합산 234.83 step/s |
| 8개 동시 수집, 그중 전뇌 2개 | 합산 287.47 step/s |
| 10개 동시 수집, 그중 전뇌 2개 | 합산 304.96 step/s |
| 12개 동시 수집, 그중 전뇌 2개 | 합산 327.68 step/s |

동시 수집은 모든 프로세스의 준비를 barrier로 맞춘 약 20초 구간을 측정했다.
이는 PPO 업데이트를 제외한 짧은 수집 벤치마크다. 단일 실행도 short rollout 기준이며
장시간 발열·다른 앱 부하·실제 후반 상대 구성에 따른 성능이나 승률을 보장하지 않는다.
12개를 기본으로 선택하되 전뇌 MPS 작업은 최대 2개로 제한한다. 학습 우선으로 빈 슬롯을
채우고, 의존하는 학습이 완료된 dev 평가도 독립 run 단위로 병렬 배치한다.

## 동등성 및 제외한 변경

- 원본/가속 v7-1의 512-step 수집: features, central, action, log probability, value,
  reward, next value, terminal/truncation, valid mask, advantage, return의 바이트 해시가 동일했다.
- 원본/가속 v7-2의 256-step 수집에서도 해당 필드의 바이트 해시가 동일했다.
- 실제 전뇌 graph에 대한 128-tick CPU/MPS 비교에서 CSR 결과·발화·출력률·최종 전압이
  동일했다. 작은 CSR의 빈 행·다중 슬롯도 별도 검사한다.
- 이 검사는 지정된 입력과 환경 구간의 동등성 검증이며, 모든 향후 trajectory에 대한
  수학적 증명은 아니다. 비동기 실행 순서, 새 경기에서의 resume, 하드웨어/라이브러리 변경은
  재현성을 논할 때 별도로 고려한다.
- FP16/AMP, 간선 제거, 센서 해상도 축소, action repeat 증가, PPO 업데이트 축소,
  관측 렌더링 제거 등 실험 조건을 바꾸는 속도 절약은 적용하지 않았다.
- 인코더 MPS는 측정상 더 느려 제외했다. GPU로 전체 모델을 옮기거나 dense 전뇌를 만드는
  대신 실제로 이득이 있는 고정 CSR 연산만 옮겼다. 원래 환경 패키지를 덮어쓰지 않았다.

실행 중에는 `caffeinate -i -w <manager pid>`로 자동 유휴 잠자기를 방지하고,
실험 종료·실패·정상 중단 때 해당 프로세스도 정리한다. 수동 잠자기나 재부팅을 막지 않는다.

원자료: `acceleration_baseline_profile.json`, `acceleration_mps_comparison.json`,
`acceleration_equivalence.json`, `acceleration_parallel_*/summary.json`.
