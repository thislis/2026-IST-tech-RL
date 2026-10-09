# v8 제공 환경·대회 제출 경로 — 2026-09-29

기준은 [대회 제출 안내](../common/blackout_last_4_pages.md)와 설치된 제공 `blackout_env`의 loader/match 코드다.
게임·obs 생성·제공 API·전송 구현을 수정하지 않는 별도 실험 `provided_competition_v1`을 준비했다.
초기 구현 이후 사용자가 실행한 6개 run은 각각 2,048 step에서 중지·저장되었다.
후속 창 숨김 수정에서는 32 step 기술 검사만 수행했으며 학습·평가를 재개하지 않았다.

## 실행

이전과 같은 한 줄이다. 기존 수정 환경 실험을 재개하지 않고 새 등록의 실험을 실행한다.

```bash
bash /Users/safeailab_macmini/Desktop/2026-IST-tech-RL/code/v8/scripts/run_v8_fast.sh
```

관리자는 백그라운드로 분리되며 터미널 종료 후에도 유지된다. 중복 실행 잠금, `caffeinate`,
worker별 포트 27000–27005, CPU/BLAS 1 thread, 최대 6개 worker를 사용한다.
native protobuf overlay, UnityEnvironment 교체, `-nographics`, Unity renderer/타이머 패치는 사용하지 않는다.
원본 API의 공개 인자인 `no_graphics=False`, `time_scale=50`을 사용한다.
`env_path`에는 원본 실행 파일을 `exec ... -batchmode "$@"`로 호출하는
`launchers/v8/BlackOutRendered.app` 실행 스크립트를 전달한다. 이 디렉터리는 ML-Agents의
macOS 경로 탐색을 위한 실행 스크립트 컨테이너이며 새 게임 빌드가 아니다.
원본 앱·제공 API·obs 전처리기는 변경하지 않는다. `-nographics`는 사용하지 않으므로
graphic 렌더링은 유지되며 Unity 게임 창은 표시하지 않는다.

이전 구현에서 빠진 `-batchmode`를 보완했다. 2026-09-29 원본 게임 32 step 검사에서
10개 에이전트의 `96×96×11` float32 맵, 픽셀별 one-hot, 벽·양 팀 유닛 채널을 확인했고
시작부터 종료까지 수집한 화면 창 목록 28회에서 해당 Unity PID의 창은 발견되지 않았다.
아이템이 없는 채널은 0일 수 있으며 11개 채널 모두가 항상 비제로라는 의미는 아니다.
[실행 검사 결과](../../logs/v8/reports/background_window_fix/live_rendered_check.json).
이후 전체 회귀 검사 287개 중 286개 통과, Metal 접근 관련 1개 skip을 확인했다.
[창 숨김·재개 준비 결과](../../logs/v8/reports/background_window_fix/preparation.json).
실행 옵션 참고: [Unity Player 공식 문서](https://docs.unity3d.com/6000.0/Documentation/Manual/PlayerCommandLineArguments.html).

| 작업 | 예산 |
|---|---|
| C1-obs9 / flat9 | 각 seed 11·22·33, run당 1,048,576 step |
| 학습 합계 | 6,291,456 environment step |
| 평가 지점 | 262,144 / 1,048,576 step |
| 평가 | 각 모델당 30회 × 양 진영, scripted planner 기준선 포함 총 780경기 |
| shard | 5회 × 양 진영, 완료되지 않은 shard 전체만 재시도 |

학습 완료 후 중간/최종 checkpoint의 dev 평가, 전체 결과 집계, 모델 선택, 두 파일 export와
제공 loader 검증까지 자동 진행한다. 다음 명령들은 프로젝트 루트에서 실행한다.

```bash
bash code/v8/scripts/run_v8_fast.sh --check   # 원본 파일/등록만 검사, Unity 실행 없음
bash code/v8/scripts/run_v8_fast.sh --status
bash code/v8/scripts/run_v8_fast.sh --stop    # PPO update 완료 후 저장 요청
```

중단 후 같은 시작 명령으로 재개한다. 물리 상태를 복원하지 않는 새 에피소드 재개다.
실패 자동 재시도는 하지 않는다. 저장 후 재개 시 실패 구간 로그는 `recovery/`에 보존한다.
원본 환경의 종료 이상은 watchdog 오류로 남기며 승리·무승부나 정상 timeout으로 바꾸지 않는다.
30분 step 진전 없음, 중단 유예 5분, 디스크 20GiB reserve를 적용한다.

## 제출 인터페이스 때문에 변경한 모델 범위

`forward(vector, graphic)`에는 agent ID, canonical slot, run/episode/decision ID나 reset callback이 없다.
원본 전처리기는 Unity의 `unit_index`를 라우팅에만 쓰고 vector에서 제거한다. 같은 클래스의
아군들은 같은 관측을 받을 수 있다. 따라서 배치 행 번호를 자신의 위치로 가정하거나, 없는
정보를 보충해 기존 상태 유지형 joint41 planner 정책을 그대로 제출했다고 주장하지 않는다.

새 모델은 임의의 B에 대해 행별로 동작하는 **stateless C1-obs9 / flat9**다. 기존 C1/flat41과
동일한 정책이라고 부르지 않으며 기존 연구 checkpoint의 head를 재개하지 않는다.

| 기존 계획 요소 | 이번 경로 |
|---|---|
| 합법 obs만 actor에 전달 | `float32[B,96]`, `float32[B,11,96,96]`만 사용 |
| C1의 KEEP/개입 gate와 conditional correction | 각 입력 행에 KEEP+8방향, threshold `q>0.5`, tie KEEP |
| 같은 용량 flat 대조 | 같은 trunk/gate/correction 모듈의 flat9 joint argmax |
| frozen encoder + 작은 64×64 trunk | 기존 target encoder 가중치 동결, 128+49차원 feature 사용 |
| planner 기준 KEEP | 제출 정책은 동결 신경망의 관측 기반 baseline으로 변경; 기존 scripted planner는 별도 dev 기준선 유지 |
| slot embedding / local crop | self ID가 없으므로 평균 slot embedding·관측된 아군 중심점 crop 사용. 역사 target와 동일한 추론이라고 주장하지 않음 |
| 한 팀당 최대 1개 slot 수정 | 행별 개입으로 변경. 팀 행동은 5개 분포의 곱이며 PPO는 합산 joint log probability 사용 |
| 관측에 없는 값 없이 critic 학습 | 기존 299차원 centralized critic, 제공된 10개 관측만 사용 |
| PPO/GAE·3 seeds·상대 schedule | 유지. rollout 2048, minibatch 128, epochs 4, LR 1e-4, gamma .9995, lambda .99 |
| 명시적 엔진 점수·종료 채널 | 제거. 원본 reset/step이 반환하는 obs/reward/termination/truncation/info만 사용 |
| 엔진 score-delta 보상 | 원본 팀원 5명의 reward 평균을 학습에 사용. 환경 reward를 바꾸거나 terminal bonus를 추가하지 않음 |
| map-paired bootstrap | 실제 map seed를 검증할 수 없어 독립 run/episode-block 집계로 변경; 진영은 묶어 유지 |
| C2/C3, confirmation/test | 자동 활성화/접근하지 않음 |

49차원 추가 입력은 제공 vector의 앞 49개 값이다. 새 숨은 정보나 다른 관측 종류가 아니다.
학습 진영 순환도 기존 v8의 `[A,A,B,B,B]`를 유지하고 실제 step별 진영 노출을 기록한다.
자기 위치를 알 수 없으므로 geometry mask를 추정하지 않고 baseline과 같은 방향만 중복 후보에서 제외한다.
정책의 배치 순서 변경·분할 호출·재호출은 행동에 영향을 주지 않도록 검사한다.
이 인터페이스 적응이 성능을 보장하지는 않는다. 특히 동일 관측을 받는 같은 클래스 유닛의
개별 행동 분화는 이 입력 계약 아래 보장할 수 없다.

## 원본 환경과 경기 결과의 실제 계약

`make_original_env()`는 위 실행 스크립트 경로를 받는 제공 `BlackOutEnv` 인스턴스를 그대로 반환한다. 내부 메서드를
교체하지 않으며, 추가 side-channel, cache 삭제, score 주입, hidden step을 넣지 않는다.
worker가 종료 정리를 위해 읽는 Unity PID는 프로세스 관리용이며 모델 입력에는 전달되지 않는다.
매 에피소드 `close()` 후 새 원본 환경을 만든다. 원본 타이머의 재시작 문제를 코드 패치 없이
피하기 위한 프로세스 수명 선택이며, 경기 내 타이머·물리·보상은 그대로다. 초기화 비용은 발생한다.

학습은 공개 `reset(seed=...)`를 그대로 호출한다. 기존 seed 전달 시점 문제가 있으므로
요청 seed가 실제 적용된 map이라고 기록하지 않는다. 평가에는 제공 `run_match()`를 그대로 사용한다.
이 함수에는 seed 인자가 없으므로 평가 30회는 **검증된 서로 다른 30개 맵이 아니다**.
최초 관측 해시와 distinct count를 기록하며, 원본 seed 문제를 고치거나 map pairing을 가장하지 않는다.

제출 안내에는 보상이 대회 점수 집계와 무관하다고 적혀 있으나, 제공된 로컬
`competition/match.py::run_match()`는 경기 전체 팀 reward 합을 비교하여 winner를 결정한다.
현재 로컬 지표 이름은 `provided_run_match_reward_total_W_over_N`으로 이 사실을 명시한다.
`info['winner']`나 추정 점수로 다른 승패를 만들지 않는다. 제공 runner의 실제 결과를 그대로 사용하며,
공식 서버가 엔진 승자를 따로 사용하는지는 미확정이다. bootstrap 구간은 이 로컬 실행 자료의
기술 통계이며 실제 map 표집 독립성·서버 성능·5%p 개선 인증을 뜻하지 않는다.

## 제출 파일

완료 후 선택된 checkpoint별로 다음 두 파일만 생성된다.

```text
artifacts/submission/v8<checkpoint_sha256>/policy.py
artifacts/submission/v8<checkpoint_sha256>/checkpoint.pt
```

`MyPolicy(nn.Module)`의 생성자는 `vector_size=96, n_channels=11`, forward 출력은
`float32[B,2]`, 각 성분은 [-1,1]이다. checkpoint는 `{"policy_state": model.state_dict()}` 형식이며
encoder 가중치도 포함한다. `policy.py`는 torch와 Python 표준 라이브러리만 사용하고 프로젝트
패키지·환경·외부 파일을 import하거나 읽지 않는다. 제출할 때 위 두 파일을 함께 전달한다.

선택 규칙은 최종 endpoint의 arm 평균 W/N, 동률이면 C1, 해당 arm 내 가장 높은 seed-run W/N,
동률이면 seed 11→22→33 순이다. dev로 선택한 점수는 독립 검증 성능으로 취급하지 않는다.
학습 전 초기 가중치나 과거 수정 환경 checkpoint는 실제 제출 export에서 거부한다.

수동으로 새 실험의 특정 checkpoint를 내보낼 수도 있다.

```bash
.venv/bin/python code/v8/scripts/export_v8_submission.py \
  --checkpoint-store logs/v8/provided_competition_v1/runs/c1_s11/checkpoints \
  --output artifacts/submission/v8/manual_c1 \
  --report logs/v8/reports/competition/manual_c1_export.json
```

선택된 checkpoint와 export의 전체 tensor 가중치를 비교하고, 제공 `load_checkpoint()`로
로드한다. 별도 `python -I`, 빈 작업 디렉터리에서 shape/dtype/범위, B=0/1/3/5,
배치 permutation/분할, 반복 호출, HWC→CHW 제공 loader 경로, 학습 중 저장한 8개 관측 배치의
행동 parity를 검증한다. 현재 준비 검사는 합성 관측을 사용했고 실제 관측 재연은 사용자 학습 후 수행한다.
CPU를 검증했다. 공식 서버의 장치·시간/메모리 제한·업로드 주소는 문서에 없으므로 인증하거나
임의로 외부 제출하지 않는다.

## 산출물과 검증

- 새 등록: `logs/v8/reports/competition/registration.json`
- 관리 로그/상태: `logs/v8/provided_competition_v1/{console.log,status.json}`
- run checkpoint/로그: `logs/v8/provided_competition_v1/runs/<arm>_s<seed>/`
- 경기 attempt/loader 검사: `logs/v8/provided_competition_v1/jobs/<job>/attempt-*/`
- 최종 결과/선택 제출물: `logs/v8/provided_competition_v1/summary.json`
- 초기 준비 검증(변경 전 기록): [preparation.json](../../logs/v8/reports/competition/preparation.json)
- 테스트: [unit_tests.txt](../../logs/v8/reports/competition/unit_tests.txt), [regression_tests.txt](../../logs/v8/reports/competition/regression_tests.txt)
- loader 형식 검사 전용 초기 모델: `logs/v8/reports/competition/contract_fixture` (학습·제출용 모델이 아님)

소스·설정·제공 게임/API·원본 checkpoint·패키지 버전·실행 스크립트를 등록한다.
소스가 임의로 바뀌면 재개를 거부한다. 이번 실행 방식 변경은 별도 migration으로 기록했다.
각 2,048 step checkpoint의 원본 blob과 이전 등록을 보존하고, 새 등록을 참조하는 자식 checkpoint를
만들었다. 모델·optimizer·RNG·수집 상태·학습 로그 위치는 tensor/필드별 완전 동일성을 확인했다.
같은 실행 명령으로 2,048 step부터 새 에피소드로 재개하며 학습 step은 추가하지 않았다.
[변경 이력](../../logs/v8/reports/background_window_fix/migration.json).
