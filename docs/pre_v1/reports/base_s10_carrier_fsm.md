# BASE-S10 전달자 FSM

구현은 [`blackout_rl/scripted_fsm.py`](../blackout_rl/scripted_fsm.py)의 `CarrierPhase`에 있다.

## 상태 전환

```text
SEEK_SHRINE → TRANSFORM → SEEK_FAR_BATTERY → PICKUP → DELIVER → RETARGET
                                ↕                  ↕
                                      EVADE
```

- Team A/B 본진 Carrier 성소 `(1,20)` / `(20,1)`로 이동하고 class id `2`를 관측해 변신 확인
- 다른 역할이 예약하지 않은 Battery 중 A* 경로 비용이 가장 큰 도달 가능 목표 선택
- Battery 회수 후 가장 가까운 수용 가능 아군 창고로 전달
- 적이 3 cell 이내면 주변 도달 가능 셀 중 최근접 적과의 거리를 최대화하는 `EVADE`로 전환,
  5 cell 밖이면 이전 수집/전달 상태로 복귀
- 사망 후 Collector class가 관측되면 본진 성소부터 다시 시작
- 변신 전 Hunter 성소를 경로에서 금지

실제 Unity seed `240810`에서 `unit_4`는 step 19에 Carrier가 됐고, 경로 비용 `25.9706`의
원거리 Battery `(18,4)`를 배정받아 step 54에 회수, step 83에 적재했다. 최종 class도
Carrier로 유지됐다. NoOp 상대가 접근하지 않는 실환경 검증과 별개로 회피 상태 및 안전
목표 선택은 적 인접 위치를 주입한 단위 테스트로 검증했다.

- 실행 결과: [`logs/base_s08_s10_role_fsm.json`](../logs/base_s08_s10_role_fsm.json)
- trajectory: [`logs/base_s08_s10_role_fsm_trajectory.jsonl`](../logs/base_s08_s10_role_fsm_trajectory.jsonl)
