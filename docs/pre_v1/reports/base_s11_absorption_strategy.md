# BASE-S11 20초 흡수 주기 전략

구현은 [`blackout_rl/strategy.py`](../blackout_rl/strategy.py)의 `absorption_state`,
`choose_strategy_mode`와 [`blackout_rl/scripted_fsm.py`](../blackout_rl/scripted_fsm.py)의
역할별 연결부에 있다.

## 시간 계약

Unity `GameScenario`는 에피소드 시작 시 `AbsorptionInterval=20`인 loop timer를 0초부터
시작한다. 관측의 `time_left`는 420초 에피소드 남은 비율이므로 다음과 같이 숨은 흡수
시계를 복원한다.

```text
elapsed = (1 - time_left) × 420
seconds_until_absorption = 20 - (elapsed mod 20)
```

부동소수점 경계는 정수 초 `1e-7` 이내에서 고정해 3·16·20초의 상태가 흔들리지 않게 했다.

| 구간 | phase / 전략 | 동작 |
| --- | --- | --- |
| 흡수 직후 0~3초 | `POST_ABSORPTION / FRESH_COLLECTION` | 새 Battery 수집 재개 |
| 3~16초 | `COLLECT / COLLECTION` | 평상시 역할 FSM 수행 |
| 마지막 4초 | `SECURE` | 새 수집 출발 중단, 보유 Battery 즉시 아군 창고 전달 |
| 선택 조건 | `RAID` | 점수 차 ≥ 0.10, 적 창고 Battery 존재, 흡수까지 ≥ 8초일 때만 적 창고 목표 개방 |

RAID는 Battery-only 기본 전략의 보수적인 선택 branch이며, BASE-S14 평가에서 실제 이득을
확인하기 전에는 다른 조건으로 확대하지 않는다.

## 실제 Unity 검증

seed `241112`에서 1,050 step / game time `21.0003초`를 실행했다.

- step 0, elapsed `0.0200`: `fresh_collection`
- step 150, elapsed `3.0200`: `collection`
- step 799, elapsed `16.0002`: `secure`
- step 999, elapsed `20.0003`: 다음 cycle의 `fresh_collection`
- 흡수 전 아군 창고 Battery cell 최대 4개, 흡수 0.5초 후 0개
- 흡수 전후 normalized 점수 `0.34 → 0.34`: 적재 시 획득 점수가 흡수 후 유지

전체 결과는 [`logs/base_s11_s12_strategy.json`](../logs/base_s11_s12_strategy.json)에 있다.
