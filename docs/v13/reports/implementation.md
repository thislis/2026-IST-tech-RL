# v13 구현과 검증 기록

2026-11-03. 사용자 요청에 따라 **실험을 시작하지 않고 구현만 수행했다.** Unity·실제 teacher 수집·학습·리그·하드웨어 처리량 probe·백그라운드 study는 실행하지 않았다. 합성 입력, 가짜 환경, 작은 optimizer 단위 검사, 임시 디렉터리 ONNX export와 자산 읽기 전용 검사를 수행했다.

## 구현 범위

| 계층 | 구현 |
|---|---|
| 배정 | 64차원 비용 특징 + 64차원 시간 요약 + 32차원 팀 요약, MLP 160→128→64→1, 5명 공동 capacity DP |
| 이동 | 13×9×9 crop CNN, 32차원 목표·역할 벡터, 시간 요약, 선택적 GRU64, 32방향+STOP |
| 기억 | raw token128 × 과거64, 현재와 4/8/16/32/64 step 전을 다시 인코딩; reset·누락 mask·실제 시간차 |
| 추론 | 공개 vector/graphic부터 후보·정확한 격자 거리·DP·상태 갱신·행동까지 하나의 ONNX |
| 학습 | 두 actor/critic/optimizer 분리, teacher BC, Worker 동결/Manager 동결, event/sequence PPO, burn-in64/unroll128 |
| 수집 | Windows spawn, 지속 환경, 완료 경기 즉시 reset, 완료 chunk부터 수집, 동일 정책 버전만 update |
| 연구 파이프라인 | 공통 teacher, B1~B6 pilot, 상위 2개×3 seed, dev 선택, 고정 확인 평가, 과거 v13 snapshot 상대 |
| 운영 | 숨김 백그라운드 실행·조회·중단·재개, atomic 기록, 소스/설정/API/빌드/모델 체크섬 |

`features.py`의 내용은 `tasks.py`에 합쳤다. 상대 목록은 별도 복사본 대신 새 게임 리그의 고정 roster를 읽는다. 예전 `opponents.json`의 이동 전 v7 체크섬을 재사용하지 않으며 frozen portable dependency를 그대로 사용한다. 기존 정책·등록 실험·게임 빌드·가중치는 수정하지 않았다. 공유 변경은 `project_paths.py`의 v13 검색 경로 등록이다.

## 후보·상태·학습 계약

32개 task 열은 자원 source10 + 저장소 component5 + 변신2 + 상대 유닛5 + 개인 hold5 + 개인 evade5다. 같은 저장소 열은 보유 중이면 deliver, Hunter면 defend가 된다. 자원 후보는 학습 점수와 독립적으로 고르고 기존 예약·각 유닛의 최선 후보를 우선한다. 자원 capacity1, intercept2, 저장소5, Hunter 총2·Carrier 총1 제한을 DP에 넣는다. 누락 유닛의 기존 예약도 차감한다. 최소 개인 hold가 있어 완전 불가능한 배정이 생기지 않는다.

이동망이 출력한 방향을 그대로 사용하며 학습된 arm에 planner 행동 override를 넣지 않는다. 픽업·보유 변화·변신·목표 소멸·지도 변화·25 step 주기에 다시 배정한다. 운반 후보의 합법성은 현재 보유 상태에 맞춘다. 속도 EMA는 실제 이동과 이동 시도 중 정체를 반영하고 ETA에 사용한다. 정체 누적 시간도 특징이다.

외부 state는 history `[5,64,128]`, Worker `[5,64]`, task `[5,24]`, navigation `[5,2,576]`의 float32다. 합계 **188,640 bytes/팀**이며 입력·출력 양쪽 전송 비용은 공식 실행기에서 추가 측정해야 한다. 공식 team 입력의 고정 slot 순서를 사용하고, B팀은 위치·지도·유닛 순서·행동을 대칭 정규화한다. 누락 row는 기존 state를 유지한다.

raw token은 관측96 + 이전 행동2 + 목표2 + task one-hot10 + phase4 + 나이/거리/변위/respawn/시간/segment/goal version/valid 8 + 예약6이다. MLP가 과거 token을 현재 가중치로 재인코딩하므로 오래된 embedding을 다른 가중치와 섞지 않는다. 실제 시간 gap이 큰 복귀와 respawn proxy는 기억을 지운다. 누락 동안에는 해당 row의 history가 멈추므로 age 입력으로 실제 시간차를 구분한다.

행동 log-prob에는 실제 선택한 방향 index와 유효 유닛 mask를 사용한다. Manager의 joint log-prob는 공동 분포의 partition으로 계산하고 결정 event에만 loss를 준다. Manager return은 event 사이 보상과 실제 step 길이를 사용한다. 최종 보상은 최종 점수 비교이며 엔진의 raw reward/winner와 혼동하지 않는다. shaping은 `gamma=1`에서 score potential 차이와 terminal 보정으로 정의한다.

## 병목 대응과 하드웨어 설정

읽기 전용으로 확인한 장비는 Ryzen 5 7500F(6코어/12스레드), RAM 약63.6 GiB, RTX4060Ti 16GB다. Torch 2.13.0+cu126과 CUDA를 사용할 수 있다. **이 장비에서 v13 처리량을 측정한 것은 아니다.**

| 예상 병목 | 구현한 대응 |
|---|---|
| 긴 경기 때문에 전체 대기 | episode 종료 즉시 같은 slot에 다음 경기; PPO는 128 step chunk 경계에서만 동기화 |
| Unity 반복 시작 | arm 내 환경 프로세스 재사용, 평가에도 같은 pool 사용 |
| CPU thread 과다 생성 | collector·ONNX·BLAS intra/inter-op 각1, learner는 별도 배치 |
| 환경 수 증가로 자원 경쟁 | 1/2/4/6/8/12/16/20/24 후보의 완료 step/s 측정; 2회 plateau 이후 중단 |
| 작은 모델의 CUDA 호출 비용 | CPU와 CUDA의 sequence batch 1/2/4/8 측정 후 선택 |
| 개별 유닛·sequence마다 CNN/역전파 | 시간·환경·유닛을 묶어 CNN/MLP와 recurrent tick 계산, minibatch당 한 번 backward |
| event마다 Manager DP 따로 계산 | 여러 sequence의 결정 event를 모아 점수와 joint probability 배치 계산 |
| 수천 번의 작은 CPU→GPU 전송 | history를 CPU에서 복원한 뒤 연속 tensor로 한 번씩 전송 |
| 전체 graphic/frame stack IPC | collector에서 FP16 crop·128 token·필요한 event 특징만 저장; 전체 이미지 전송 제외 |
| 거리 계산의 Python 루프 | TorchScript tensor 격자 완화, 수렴 시 종료, goal/walk 변화 없으면 거리장 재사용 |
| 경험 메모리 증가 | worker당 최대 한 chunk 응답, 고정 수집량, teacher chunk zlib1 압축 |
| 매 BC minibatch의 checkpoint I/O | 30초 간격·단계 끝·정상 stop 때 저장; RL은 update commit마다 저장 |
| 자기대결 모델 반복 compile | 프로세스별 immutable snapshot 모델 4개 캐시; episode state는 새로 생성 |
| RAM/VRAM/disk 소진 | 각각 8/3/15 GiB 여유, worker 추가 메모리 예상량 검사, 실제 사용량을 다음 pool 시작에 반영 |

측정 결과는 `resources/learner.json`, `workers.json`, `selected.json`에 고정한다. learner 측정은 합성 계산 처리량, worker 측정은 게임 수집 처리량이므로 두 값의 의미가 다르다. 짧은 probe의 최선 설정이지 모든 상대·학습 단계에서의 전역 최적값을 보장하지 않는다. 단계별 `collection_seconds`, `env_steps_per_second`, update 시간과 수집기의 policy/opponent/environment 시간을 기록한다. 속도 개선률은 실제 실행 후 비교해야 한다.

## 복구·평가

모델·두 optimizer·critic·policy version·부모/worker RNG·진행량·상대 snapshot을 체크포인트에 저장한다. Unity 중간 상태는 복원하지 않으므로 재개할 때 미완료 chunk를 버리고 새 episode를 연다. BC 강제 종료 시 마지막 30초 이내 작업은 다시 수행될 수 있다. 오류 시 watchdog과 소유 프로세스 정리를 거쳐 실패 이유를 남긴다. 무한 자동 재시작으로 실패를 숨기지 않는다.

train/dev/confirmation seed 영역과 실제 초기 map hash를 분리한다. 다른 split과 지도 중복 또는 같은 seed 진영 쌍의 map 불일치는 실패로 처리한다. dev는 22상대×5seed×2진영=220경기, confirmation은 기존21상대에 candidate840+v12baseline840 및 직접40경기다. baseline은 frozen v12-before-learned bundle이다. 점수 기반 macro와 상대별 성적, 진영 쌍을 보존한 seed-block bootstrap을 기록한다. 경계값에서 bootstrap 구간이 퇴화할 수 있어 통계 gate만으로 자동 채택하지 않는다.

관측 진단은 pickup, 저장소 근처에서 보유가 사라진 return proxy, 운반 손실 proxy, 운반 시간·ETA 오차, 정체, task 변경, teacher 지원 event 비율이다. 엔진에서 확인한 delivery 사건과 구분한다. 공식 컨테이너·시간 한도·회귀 검토 전에는 export 결과를 후보로 표시한다.

## 원 계획과의 구체적인 차이

- B1은 v13 거리장의 규칙 경로 제어다. 기존 v12 제어를 바이트 단위로 복제한 모델이 아니므로 B1↔B2를 이동망의 완전히 순수한 인과 비교로 해석하면 안 된다. 원 v12 자체는 고정 baseline으로 따로 평가한다.
- 32개 후보와 feature64의 의미는 위 구성으로 고정했다. v12 가중치를 새 Manager에 그대로 적재하지 않으며 teacher BC로 초기화한다. 실제 왕복 완료시간 회귀 모델 대신 관측 속도·정체 EMA를 비용 계산에 반영한다.
- pickup 후 운반 후보로 event 재배정하는 상태 전이를 사용한다. 원 계획의 모든 복잡한 task FSM·목표 강제 유지 규칙을 추가하지 않았다.
- 기본 B1~B6와 선택 actor의 별도 lag 설정은 구현했다. 후보 폭24/전체와 planner 방향 양자화 단독 평가는 기본 sweep에 포함하지 않았다. 추가 연구는 새 schema/설정/실험 이름으로 등록해야 한다.
- P4/P5 단계 사이의 실제 dev 성능 문턱은 실측 전 숫자를 정하지 않았다. 기본 실행은 정해진 예산으로 단계 전환하고 주기적 dev로 best를 고른다. P0~P7의 연구상 성능 조건을 통과했다는 의미는 아니다.

## 검사와 남은 실측

자동 검사 21개와 읽기 전용 준비 검사를 통과했다. `--status`는 `not_started`, `game_started: false`다. 테스트 재현 명령은 아래와 같다.

```powershell
$env:PYTHONPATH='code/v13;code/shared;code/v12;code/v11'
.venv/Scripts/python.exe -m unittest discover -s code/v13/tests -v
```

합성·가짜 환경 검사는 lag200step+, Dijkstra 거리 일치, capacity DP 전수 열거, Python/TorchScript/ONNX 다중 step parity, 진영 대칭, 누락 state·예약, frozen log-prob, 서로 다른 burn 길이 배치, event joint probability, actor별 파라미터 갱신, GAE, 버전 혼합 거부, 즉시 episode refill, teacher bundle, checkpoint identity, map split과 paired schedule을 포함한다. 자동 출력은 `logs/v13/validation/`에 둔다.

`--check`는 게임을 시작하지 않고 등록된 API66파일·전체 게임 빌드·상대22개의 자산을 검증한다. `--dry-run`과 `--status`도 실행을 시작하지 않는다. 초기 full ONNX gate는 합성 CPU 호출과 상태 전달을 확인하고 warm 호출이100ms를 넘으면 긴 실험 전에 중단한다. 이 검사는 공식 GPU 컨테이너의 p99·최대 지연, 실제 복잡한 지도, 장시간 안정성의 증거를 대신하지 않는다.

아직 수행하지 않은 항목은 실제 Unity 수집·학습·승률 비교, 실경기 처리량 선택, CUDA 학습 경로의 실전 안정성, 공식 제출 컨테이너 검사다. 사용자가 [한 줄 실행 명령](../README.md)을 실행하면 준비된 파이프라인이 앞의 단계부터 진행한다.
