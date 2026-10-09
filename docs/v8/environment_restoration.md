# 제공 게임·환경 복구 — 2026-09-29

후속 상태: [제출 계약에 맞는 별도 학습·평가 경로](competition.md)를 구현했다.
아래 실행 불가 표시는 원복 직후의 과거 상태이며, Unity 패치 철회는 계속 유지된다.

사용자 지시에 따라 **제공된 게임 코드와 obs 생성 경로를 변경하지 않는다**는 조건을 최우선으로 적용했다.
기존 v8 계획서의 Unity 수정·추가 채널·타이머/렌더러 패치 항목은 철회한다.

## 복구한 실행 환경

- 게임 소스: `../blackout`, commit `d2220a7d01be88d413f551efd529f4758833be8b`.
- 제공 API 소스: `../blackout-env`, commit `6ba7d9993cf1bdefe1ed480c8efbcabcb923f539`.
- 사용 가능한 게임: 원래 `artifacts/builds/BlackOut.app`. 실행 파일 SHA-256은 `versions.md`에 기록된
  `49172f88b1ba2429677e7ec4a7876c4589876090be4888aeaef39269e29f4a69`와 일치한다.
- 두 제공 저장소는 모두 clean이며, 설치된 `blackout_env`의 소스 파일 13개는 제공 저장소와 일치한다.
- 제공 `BlackOutEnv` 자체를 반환하는 진입점만 남겼다. reset/step, obs·reward·info 반환값을
  가공하거나 cache를 지우거나 숨겨진 step을 삽입하지 않는다. UnityEnvironment 교체,
  별도 side-channel, 전송 구현 monkey patch도 사용하지 않는다.
- 수정된 `BlackOut-v8*.app` 4개, Unity 복사 프로젝트, C# 패치, 진단 실행물 및 수정 환경의
  연구 export는 `build/retired_v8_environment_2026-09-29/`로 옮겼다. 활성 `builds/`에는 원본 게임만 남는다.
- 빌드 패치·timer fixture·native overlay 설치·점수 주입·연구 worker 시작 경로를 비활성화했다.
  `prepare_v8_unity.py`를 실행해도 연구용 패치를 다시 생성하지 않는다.

원본 저장소와 원본 app 자체는 처음부터 변경되지 않았으므로 외부 저장소를 강제 reset하거나
제공 코드를 다시 작성하지 않았다. 현재 사용 경로에서 수정 복사본을 제거한 것이다.

## 모델과 과거 결과

새 모델, encoder 연결, C1/flat 분포·decoder, planner feature, PPO·GAE, checkpoint·통계 코드 등은
보존했다. 복구 전 모델 파일 해시와 비교하는 검사를 추가했다. 기존 v1–v7 소스도 유지했다.
오프라인 합성 모델 검사는 계속 실행할 수 있다.

이전 v8의 6개 학습과 780경기는 **수정 환경에서 얻은 과거 자료**다. 파일과 원래 등록·체크포인트
해시를 보존하되, 제공 환경에서의 성능이나 원본 환경용 학습 결과로 재분류하지 않는다.
기존 연구용 reward/outcome 자료형은 과거 로그 해석과 합성 검사에만 남아 있다.

## 현재 실행 상태

기존 셸 파일 이름은 유지했다.

```bash
bash /Users/safeailab_macmini/Desktop/2026-IST-tech-RL/code/v8/scripts/run_v8_fast.sh --check
```

이 명령은 원본 파일·설치 API·import 경로만 검사하고 Unity를 실행하지 않는다.
`--status`도 복구 상태를 표시한다. 옵션 없이 이전 실행 명령을 호출하면 수정 환경 경로가
철회됐다는 설명과 종료 코드 2를 반환하며 백그라운드 학습을 시작하지 않는다.

**v8 학습·평가 연결은 현재 실행 불가다.** 기존 collector는 추가 채널의 정확한 점수·종료 사건·
물리 tick을 요구했다. 이 값들을 원본 obs에서 임의로 추정하여 동일하다고 가장하거나,
다른 보상·종료 조건으로 몰래 대체하지 않았다. 원본 API가 실제 제공하는 obs/reward/info만으로
모델 학습 연결과 평가 지표를 정하고, 별도 실험으로 등록해야 다시 실행할 수 있다.
이 복구 작업에서는 그 새 실험을 구현·실행하지 않았다.

## 검증 자료

- [복구 manifest](../../logs/v8/reports/environment_restoration/restoration.json): 원본 저장소·API·app 파일 해시,
  모델 보존 해시 및 격리 경로.
- [복구 전 소스 보관본](../../logs/v8/reports/environment_restoration/before_rollback.zip).
- [원본 경로 검사](../../logs/v8/reports/environment_restoration/preflight.txt).
- [복구 전용 검사](../../logs/v8/reports/environment_restoration/restoration_tests.txt).
- [전체 회귀 검사](../../logs/v8/reports/environment_restoration/regression_tests.txt).

복구 검사는 파일 비교와 mock 환경 반환값 검증으로 수행한다. 실제 Unity나 학습·평가는 실행하지 않았다.
