# PREP-11 처리량 benchmark

측정 원본은 [`logs/prep11_benchmark.json`](../logs/prep11_benchmark.json), 재실행 코드는
[`scripts/benchmark_env.py`](../scripts/benchmark_env.py)이다. 측정한 executable SHA-256은
PREP-01의 `49172f88b1ba2429677e7ec4a7876c4589876090be4888aeaef39269e29f4a69`와 일치한다.

## 조건

- 호스트: macOS 26.5 arm64, Apple M4 Pro, logical CPU 12개
- Python 3.10.12, PyTorch 2.13.0, MPS built/available
- Unity `time_scale=50`, macOS player는 `no_graphics=False`
- warm-up: 환경별 100 step, 측정: 환경별 1,000 step
- action: seed를 고정한 `float32[-1,1]` uniform random
- 여기서 environment step은 10개 agent를 동시에 전진시키는 PettingZoo parallel step 1회다.
- 복수 환경은 한 Python 프로세스에서 각 환경을 순서대로 step하는 동기식 측정이다.

## 결과

| 환경 수 | 합산 env steps/s | 환경별 steps/s | agent decisions/s | Python+Unity CPU 합계 평균 | RSS 합계 평균 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 52.278 | 52.278 | 522.779 | 97.786% | 638.024 MiB |
| 2 | 49.736 | 24.868 | 497.360 | 97.233% | 969.428 MiB |

2개 환경의 합산 처리량은 1개보다 약 4.9% 낮았다. 현재 동기식 collector에서는 환경 수를
늘리는 것만으로 throughput 이득이 없으며, CPU 한 코어 수준이 포화된 패턴이다. 이후 rollout
collector에서 multiprocessing을 구현할 때 동일 측정을 다시 수행해야 한다.

## GPU 계측

Apple `AGXAccelerator`의 `PerformanceStatistics`를 21회씩 읽었다. 이는 시스템 전체 GPU
카운터이며 Unity/Python 프로세스만의 사용률이 아니다.

| 상태 | Device 평균/최대 | Renderer 평균/최대 | Tiler 평균/최대 | GPU in-use system memory 평균/최대 |
| --- | ---: | ---: | ---: | ---: |
| 실행 전 단일 표본 | 20% | 19% | 12% | 1457.109 MiB |
| 환경 1개 | 16.619% / 44% | 16.286% / 43% | 8.667% / 13% | 1657.020 / 1878.656 MiB |
| 환경 2개 | 19.619% / 36% | 19.095% / 35% | 10.571% / 19% | 1977.685 / 2696.969 MiB |

idle 단일 표본보다 실행 평균이 낮을 수 있으므로 사용률 차이를 BlackOut의 순수 GPU 비용으로
해석하지 않는다. 다만 측정 중 GPU가 포화되지 않았고, CPU/동기식 stepping이 현재 병목이라는
판단과 모순되지 않는다.

## 한 episode 실행 비용

PREP-08~10에서 같은 executable, `time_scale=50`으로 끝까지 실행한 2경기를 재사용했다.

- episode 길이: 각 21,003 step
- wall time: 408.303초, 413.815초
- 평균: 411.059초/episode (약 6분 51초)
- 장기 실행 처리량: 51.095 env steps/s
- 위 RSS 기준으로 1개 환경의 resident memory 예산은 약 0.64 GiB로 잡는다.

짧은 1,000-step 측정의 52.278 steps/s와 완전한 episode의 51.095 steps/s가 가까워,
warm-up 구간만 재서 처리량이 과대평가된 징후는 작다. GPU 카운터는 다른 시스템 작업의 영향을
받으므로 공식 Linux/NVIDIA 학습 호스트에서도 `nvidia-smi` 경로로 다시 측정해야 한다.
