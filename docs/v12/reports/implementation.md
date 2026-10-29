# v12 구현 및 실행 준비 기록

2026-10-26. 실제 학습과 Unity 경기는 실행하지 않았다. 아래의 합성 검사와 하드웨어 측정은 게임 성능 결과가 아니다.

## 실행 범위

프로젝트 루트의 PowerShell에서:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v12\run_v12.ps1 --name joint_trip_v12
```

Bash에서는 `bash code/v12/run_v12.sh --name joint_trip_v12`를 사용한다. 위 명령은 사용자가 실행할 때 실제 파이프라인을 시작한다. 이번 구현 작업에서는 실행하지 않았다.

- `--check`: 의존성·게임 파일·현재 사용 가능한 상대와 설정을 확인한다. 게임은 시작하지 않는다.
- `--dry-run`: 실행 단계와 예산을 출력한다. 학습·경기는 시작하지 않는다.
- `--benchmark`: 짧은 합성 CPU/CUDA·프로세스 측정만 수행한다. 학습 가중치를 저장하지 않는다.
- `--status`: 해당 이름의 진행 상태를 조회한다.
- `--stop`: 실행 중인 해당 실험에 중단을 요청한다. 미완결 episode는 PPO에서 제외한다.

기본 범위는 `available`이다. 현재 프로젝트에는 게임 빌드와 scripted 상대 9개가 있지만 과거 신경망·동결 checkpoint 12개는 없다. 따라서 설치된 scripted 상대와 v12 self-play로 실행하며, 누락된 상대와 변경된 상대 비중을 등록한다. 현재 학습 가족 비중은 원래 40/30/20/10 중 존재하는 scripted/self-play를 재정규화한 60/40이다. 문서의 v4/v6 비퇴행·21상대 정식 평가를 대신하는 결과로 해석하지 않는다. 과거 가중치를 갖춘 환경의 정식 비교는 `--scope full`을 사용한다.

## 구현한 동작

| 영역 | 구현 |
|---|---|
| 공개 입력 | canonical 팀 5행, `vector[5,96]`, `graphic[5,11,96,96]`; Team B의 `y=x` 반사 |
| 작업 | 왕복 collect/special/raid, deliver, intercept/defend, 변신, private hold; 자원별 고유 ID·capacity |
| 명시 상태 | 작업 단계·commitment·보유 변화·respawn·반납 실패·정체·시계/진영 reset |
| 경로 | 24×24 8방향 양의 비용 Dijkstra, 대각선 모서리 통과 금지, 제한된 LRU 거리장 캐시 |
| scorer | 공유 `64→64→32→1`, 6,273개 파라미터, 마지막 layer 0 초기화 |
| 배정 | 용량을 직접 처리하는 subset DP logZ/MAP/sample/marginal; 순차 대조군은 남은 locked 배정의 feasibility 보존 |
| 학습 | frozen 완결 episode wave, event별 고정 support, joint log-prob PPO, 별도 critic, duration GAE |
| 목적 | `provided_runner_v1`, 누적 raw reward 승패와 bounded 관측 score potential; score 진단 별도 |
| 비교 | Q0/Q1/Q2, S/J/F; 공통 초기화에서 pilot과 본학습 분리, dev 선택 후 confirmation/test |
| 기록 | 원자적 checkpoint·등록·진행 로그, terminal 포함 명확/모호 trip proxy, 실제 초기 map hash |
| 제출 | 외부 프로젝트 모듈을 요구하지 않는 `policy.py`·`checkpoint.pt` 두 파일 bundle |

경로의 Dijkstra는 같은 양의 비용 그래프에서 설계의 수렴한 min-plus와 같은 최단거리 문제를 푼다. 후보별 전체 reverse field를 매번 만들지 않으며, event가 없는 프레임에는 후보 생성·scorer·배정 호출을 하지 않는다. 제공 게임 및 API/관측 전처리는 수정하지 않았다.

현재 공개 관측으로 식별할 수 없는 활성 buff·숨은 stack 수량·적 class는 만들어 넣지 않는다. 이동 profile은 공개 class/보유물과 walkability 변경을 구분하며, 관측할 수 없는 효과는 정체 회복으로 대응한다. 이 근사 때문에 실제 충돌·완주 성능은 경기에서 확인해야 한다.

feature의 경로 여유는 현재 목표 인접칸 여유, 위험은 목표 주변 적 거리의 proxy다. 경로 전체의 병목·위험 적분은 아니다. service time은 초기 0.5초이며 완료 이력으로 보정하지 않았다. 상대 ETA는 공개 변위와 가정한 속도로 계산하는 휴리스틱 범위로, 검증된 물리 상·하한이 아니다. fallback·cost-stale·보호 창고 feature는 현재 기본값으로 남아 있다. 이 범위를 넘어 feature가 보정됐다고 주장하지 않는다.

## CPU와 GPU

현재 호스트는 Ryzen 5 7500F(6 physical cores, 12 threads), RAM 약 63.6 GiB, RTX 4060 Ti 16 GiB다. 기존 CPU 전용 PyTorch를 같은 버전의 `torch==2.13.0+cu126`으로 교체했고 CUDA 인식을 확인했다. 설치에는 [PyTorch 공식 설치 경로](https://pytorch.org/get-started/locally/)의 CUDA wheel index를 사용했다. NVIDIA 드라이버와 제공 게임은 변경하지 않았다.

CPU는 팀별 정책·경로·episode 수집을 담당한다. 별도 실험 프로세스들은 수집과 학습을 겹쳐 실행할 수 있으며, CUDA learner는 프로세스 간 lease로 동시에 과도하게 실행되지 않도록 한다. BLAS/OpenMP thread 수와 전체 수집 worker 예산을 제한해 중첩 병렬화의 과구독을 피한다. Unity semantic camera에는 렌더링이 필요하므로 `no_graphics=False`를 유지한다.

짧은 benchmark는 실제 DP forward/backward와 host transfer를 포함해 CPU thread 수·CUDA·minibatch를 비교하고, 별도 합성 작업으로 worker 수를 선택한다. Unity 렌더링·IPC·실제 상대 정책을 포함한 처리량 측정은 하지 않았으므로 최대 하드웨어 활용을 실증했다고 주장하지 않는다. 실제 실행 로그의 steps/s로 후속 판단할 수 있다.

측정 결과는 [자동 runtime profile](../../../logs/v12/preparation/runtime_profile.json)에 있다. warm-up을 제외한 추론 8회, 학습 연산 4회, 전체 정책 경로별 6회에 불과하므로 안정적인 성능 평가 대신 초기 배치 선택에 사용한다.

| 짧은 합성 측정 | 결과 |
|---|---:|
| 120 edge scorer + 전송 + CPU MAP, CPU 1 thread p50 | 2.458 ms |
| 같은 작업, CUDA scorer p50 | 3.300 ms |
| 정확 DP gradient + critic, 128 events, CPU 2 threads | 2,183 events/s |
| 같은 작업, CUDA | 6,846 events/s |
| 공개 정책 전체, no-event p50 | 1.435 ms |
| 공개 정책 전체, 캐시된 decision p50 | 10.052 ms |
| 공개 정책 전체, reset/cold p50 | 63.772 ms |

선택은 **CPU actor/planner 1 thread, CUDA learner 1 thread·128 event minibatch, 동시 실험 2개·각 2개 경기 worker(총 4개)**다. 합성 worker 측정 최고치는 5개였지만 균등한 두 실험 예산으로 나누어 실제 동시 경기 worker는 4개다. 새 실험을 시작하면 해당 시점의 짧은 측정으로 다시 선택하므로 위 숫자를 영구 하드코딩하지 않는다.

첫 측정에서 추론 모드로 만든 DP 인덱스 캐시가 autograd에서 재사용되지 못하는 문제가 발견됐다. 캐시를 일반 tensor로 생성하도록 수정하고 해당 측정만 재실행했다. 오류가 있던 CPU 측정은 선택에 사용하지 않았으며 원래 진단은 `logs/v12/preparation_attempt1/`에 남겼다.

## ONNX와 검증 범위

개발 ONNX는 `features/prior/legal/capacities → assignment/scores/map_value/feasible`인 deterministic joint event core다. opset 17, ONNX Runtime CPU provider를 사용하며, 명시 상태와 경로 처리는 Python 호스트에 남는다. 선택적 `OnnxEventBackend`는 event에서만 호출되는 경계로 사용한다. `Where`로 결과만 버리며 계산을 계속하는 캐시를 구현하지 않았다.

전체 `obs,state_in → action,state_out` 단일 ONNX 그래프, 10,000호출 parity, 같은 호스트의 v4/v11 전체 지연 비교, 실제 게임 처리량은 미완료·미측정이다. 현재 두 파일 Python 제출과 이 개발 ONNX를 혼동하지 않는다. sampled decoder의 RNG 계약을 deterministic MAP ONNX 검사로 대신하지 않는다.

사용자의 요청대로 작은 수학 fixture·FSM fixture·합성 PPO·제공 loader·짧은 export 검사만 수행했다. 장시간 회귀, Unity smoke, 본학습, 실제 평가 경기는 실행하지 않았다. scorer 파라미터 수나 합성 지연이 작다는 이유로 게임 성능 향상을 선언하지 않는다.

- DP 6개, 정책 8개, 학습/제출 4개, 실행기 3개(합계 21개)의 집중 검사 통과. 전체 구버전 테스트 순회는 하지 않았다.
- 29개 전수열거 fixture의 logZ·MAP·marginal·gradient, sampled replay ratio=1, locked capacity, 추론→학습 캐시 재사용을 확인했다.
- 제공 checkpoint loader로 두 파일 정책을 로딩한 3회 합성 sequence가 원본과 일치했다. 합성 checkpoint 파일은 27,797 bytes다.
- event ONNX는 217,754 bytes이며, 6개 배정 fixture와 12호출 host sequence를 비교했다. 0이 아닌 합성 residual에서 최대 score 차이 `1.19e-7`, action/task 일치; 실제 backend 호출은 event 7회이고 유지 프레임 호출은 0회였다. reset·진영 변경·pickup·명시 상태 복원을 포함한다. [자동 결과](../../../logs/v12/preparation/onnx_probe.json).
- PowerShell 준비 검사·계획·상태·중단 명령과 Bash 전달을 확인했다. 별도 합성 모드의 백그라운드 시작→즉시 중단으로 프로세스 lease 해제도 확인했다. 실제 실험을 시작하지 않았다.

## 파일 위치와 후속 판단

코드·설정·실행기는 `code/v12/`, 이 기록은 `docs/v12/`, 자동 준비 보고서는 `logs/v12/preparation/`에 둔다. 사용자가 시작한 실험은 `logs/v12/<name>/`, 공개 관측 event 데이터는 `artifacts/data/v12/`, 제출물과 개발 ONNX는 `artifacts/submission/v12/`에 둔다. 로컬 환경·가중치·로그는 Git 공유 대상이 아니다.

설계의 초기 prior·feature scale은 코드에 동결했으나 실제 no-learning pilot으로 최적화한 값이 아니다. optional ranking warm-start는 교사 선택을 선호 label로 다룰 뿐 인과적 우월 label로 취급하지 않으며, 기본은 zero-residual 공통 초기화다. 성능 채택 기준은 [사전 계획](../plans/v12_implementation_plan.md)을 유지하고, 현재 결과 판정은 **게임 성능 미확정**이다.
