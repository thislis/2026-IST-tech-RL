# v8 구현·검증 범위 — 2026-09-27

**C1/flat41의 로컬 연구 실행 경로를 구현했고 실제 Unity 기술 검증을 통과했다.**
정규 파일럿·성능 개선·공식 제출 인증은 완료하지 않았다. 원본 v7 소스, 82-file 가속
runtime, 기존 checkpoint를 보존했다. 핵심 결과는
[engineering_validation.json](../../logs/v8/reports/engineering_validation.json)에 해시와 함께 기록했다.

## 계획서와 구현 대응

| 범위 | 구현·확인한 내용 | 후속 경계 |
| --- | --- | --- |
| PR00 | 원본 71 source/등록 파일과 82 overlay 검증, 인계 자료 해시, 패키지·상대·빌드 고정, 불변 checkpoint lineage | 외부 인계 보고서 4개 누락은 명시; 역사 trajectory를 새로 복구했다고 주장하지 않음 |
| PR01 | reset 전 engine score/winner 확정, run/episode/decision outcome join, idempotent 최종 delta/bonus | 수정한 로컬 연구 build 계약이며 공식 승인과 별개 |
| PR02 | 실제 C# generation timer, 자동/명시 reset 각각 3연속 timeout, raw semantic GPU/LLAPI 해시 | production 420초 전체 길이·Linux/Xvfb·Windows는 별도 검증 필요 |
| PR03 | 공통 factory, canonical IDs, 명시 reset/decision, 중복 호출 cache/RNG, 슬롯별 실패 격리, reason mask | geometry margin은 근사이며 공식 물리 legality로 부르지 않음 |
| PR04 | 매 transition 노출 장부, bounded event/control windows, lossless legal observation, q/log-prob/action/displacement/value 기록 | true state restore를 통한 인과적 counterfactual은 구현하지 않음 |
| PR05 | C1 gate+conditional40, 같은 용량 flat41, masked entropy, 세 decoder, legacy exact factorization | A2/A3의 재검사는 합성 feature 입력; chronological 경기 비교 S2는 미실행 |
| PR06 | 실제 behavior joint PPO ratio, raw GAE/return, terminal/truncation/rollout 분리, episode 유지, update 경계 save/resume | 384 step 기술 검증과 1,048,576 step 정규 파일럿을 구분 |
| PR07/08 | factory에서 C2/C3 비활성 모드를 명시적으로 거부 | 계획서대로 상쇄/정보/표현 증거가 생길 때만 구현 |
| PR09 | dev model lock, 모든 attempt 기록, engine W/N, independent-run/shared-map bootstrap와 민감도 CI | 독립 confirmation/test 및 다중 후보 family는 미등록; 해당 실행 CLI는 거부 |
| PR10 | CPU 정책 전체 request latency, original/native 종료·픽셀 parity, 두 연구 export의 격리 재연 | 공식 loader 미확정, peak RAM/cold-start 전용 benchmark·세부 profiler 완성은 후속 |

## 실제 수행한 검증

- 전체 회귀: **263개 중 262 pass, 1 skip**.
  [전체 로그](../../logs/v8/reports/full_regression.txt). skip은 기존 v7의 Metal 장치 접근 검사다.
  v8 필수 unit test는 skip 없이 통과했다.
- 실제 Unity: 자동 재시작 timeout 3회와 명시 reset timeout 3회.
  97→100 마지막 reward **1.03**, 99→100 **1.01**, 감점/이전, draw+shaping, duplicate/late
  event, reset score 검사. B→A 순서로 두 target score를 쓰는 fixture에서는 기존 엔진의
  첫 종료(B 승리, snapshot 0:100)가 보존된다. 물리적으로 모든 동시 충돌을 열거한 검사는 아니다.
  [결과](../../logs/v8/reports/live_contract_validation.json).
- 렌더: CPU numeric RGBA8→GPU texture→LLAPI→one-hot에서 복원한 RGBA8의 SHA256 일치.
  seed3001 초기 map의 ID counts는 `[5844,2688,320,320,1,1,42,0,0,0,0]`이며 양 가장자리
  row는 wall ID1이다. `[96,96,11]` one-hot만 검사해서는 all-empty map을 발견하지 못하므로
  양 storage 채널도 검사한다. 원시 바이트 비교는 reset golden fixture 범위다.
- 기존 native protobuf/queue에서도 같은 timeout 6건의 terminal/decision/tick/score/winner와
  semantic golden hash 일치. [parity 결과](../../logs/v8/reports/runtime_contract_parity.json).
  학습 정책의 장기간 cross-backend trajectory가 동일하다고 확대하지 않는다.
- 실제 C1과 flat 각각 **384 transitions, 6 PPO updates, 3 timeout episodes**.
  exposure는 각 384개로 partial episode까지 일치한다. encoder는 고정하고 trunk/gate/correction/
  critic parameter 변화를 기록했다. `logs/v8/{c1,flat}_smoke_v2`가 실행 산출물이다.
- 실제 평가 CLI도 공통 factory로 고정 target 상대, dev map41000 양 side 2경기를 완료했다.
  `logs/v8/c1_eval_smoke_v2`의 attempt/model lock/outcome으로 연결을 확인했다.
  2초 진단 경기의 1승1패는 정규 승률이나 개선 증거가 아니다.
- 두 연구 export: 실제 관측 2 episodes×8 decisions, dictionary 역순·동일 ID 재호출,
  별도 `python -I`에서 행동 및 planner state digest 일치.
  [C1](../../logs/v8/reports/export_parity.json), [flat](../../logs/v8/reports/export_parity_flat.json).
- 기존 final A2/A3 seed11 checkpoint의 합성 feature 96개 exact factorization에서 joint argmax
  변경은 각각 0건. threshold decoder는 다른 정책이며 바뀐 action 수를 별도 기록했다.
  [A2](../../logs/v8/reports/legacy_A2_seed11.json), [A3](../../logs/v8/reports/legacy_A3_seed11.json).

| 진단 arm | request p50 | p95 | p99 | 수집 wall time |
| --- | ---: | ---: | ---: | ---: |
| C1 | 9.95 ms | 19.28 ms | 100.71 ms | 15.45 s |
| flat | 9.59 ms | 15.12 ms | 52.58 ms | 14.83 s |

384개 개별 요청의 관측값이며 episode 첫 호출도 포함한다. 공식 deadline과 비교한 합격
판정이나 충분한 성능 benchmark가 아니다. planner/encoder/features는 하나의 합산 span으로
기록하며 개별 모듈 p99 합으로 전체 p99를 계산하지 않는다.

## 구현 중 발견해 수정한 통합 문제

1. 실제 frozen checkpoint의 loader 반환은 `SubmissionPolicy`이다. 내부 `actor_critic`을
   factory에서 명시적으로 사용하도록 수정했다. 작은 합성 actor만으로는 이 차이를 찾지 못했다.
2. 새 side-channel 등록은 대기 중인 reset을 동기적으로 전달한다. renderer 구독/agent setup보다
   먼저 등록했을 때 map이 모두 empty였다. 등록을 초기화 마지막으로 이동해 해결했다.
   이는 v8 통합 과정의 오류이며 기존 Xvfb Blit 보고와 같은 원인으로 확정하지 않았다.
3. numeric semantic 복사는 linear RGBA8 `CopyTexture`를 사용하고 실제 원시 바이트를 대조한다.
   관측 형식과 공개 정보 종류는 유지한다. 매 reset 검증 실패는 조용한 빈 지도 진행을 막는다.
4. Unity가 app bundle에 생성하는 profiler timer JSON 때문에 실행 후 bundle hash가 변했다.
   알려진 진단 디렉터리만 제외하고 나머지 게임 코드·자산은 계속 검증한다.
5. v8 파일 추가는 기존 v7의 전체 파일 집합 fingerprint를 바꾼다. v7 validator를 느슨하게
   고치지 않고 v8 adoption에서 등록된 원본 파일 각각을 검증한다. scheduler 회귀 fixture는
   해당 원본 검증 후 과거 snapshot fingerprint로 실행한다.
6. native overlay 전에 `blackout_rl`을 import하면 eager import가 이미 protobuf를 읽는다.
   CLI의 검증을 별도 프로세스에서 먼저 수행하고 parent가 overlay를 설치한 뒤 import한다.
7. 과거 resume 회귀는 현재 latest를 옛 정지 시점 해시와 비교했다. immutable backup의 해시,
   진행된 global step, parent lineage를 각각 검사하도록 정정했다.

초기 진단 빌드 r1/r2/r3와 실패 run은 결과를 성공으로 덮어쓰지 않고 별도 파일로 보존했다.
현재 검증된 bundle hash는
`70893bf06ea30239df6ff25666ac7cb595aa45a3c7b6e5859093174d39bfdbf5`이며,
source→binary 목록은 [unity_build_manifest.json](../../logs/v8/reports/unity_build_manifest.json)에 있다.

## 완료 상태와 다음 실험

- 로컬 engineering smoke: 통과. production-duration·새 OS 전체 인증을 포함하지 않는다.
- `research_improved`: **false**. S2 paired chronological 진단, 3 seeds×2 arms 정규 파일럿,
  독립 run/map confirmation은 미실행이다. 본선 설정 파일은 준비했지만 자동 장기 실행을 시작하지 않았다.
- `submission_ready`: **false**. 공식 loader, state/reset, dependencies/device/resource limits가
  unresolved다. 연구 export가 공식 제출물 인증을 대신하지 않는다.
- 원본 held-out test는 열지 않았다. 새 confirmation/test의 독립성 확인·manifest·retry 규칙을
  잠그기 전에는 평가 코드가 해당 split을 허용하지 않는다.
- C2/C3 및 target의 item/role 후속 ablation은 조건부 연구로 남긴다. 신경망 크기 증가나
  forced intervention으로 기술 검증을 성능 개선처럼 만들지 않는다.
