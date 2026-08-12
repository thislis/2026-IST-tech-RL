# BASE-S09 경비원 FSM

구현은 [`blackout_rl/scripted_fsm.py`](../blackout_rl/scripted_fsm.py)의 `GuardPhase`에 있다.

## 상태 전환

```text
SEEK_SHRINE → TRANSFORM → PATROL ↔ CHASE
                     respawn → SEEK_SHRINE
```

- 고정 중앙 Hunter 성소 `(11~12, 11~12)` 중 가장 가까운 도달 가능 셀로 이동
- 관측의 self class가 `Hunter=1`로 바뀐 뒤에만 변신 성공 처리
- 아군 창고 connected component에서 중앙 쪽 경계 셀을 뽑아 결정적인 순환 순찰
- 적이 Chebyshev 5 cell 이내에 있으면 추격하고 범위를 벗어나면 순찰 복귀
- 사망 후 Collector class가 관측되면 중앙 성소부터 다시 시작
- 변신 전 Carrier 성소 `(1,20)`을 경로에서 금지

실제 Unity seed `240810`에서 `unit_3`은 step 211에 Hunter 변신이 확인됐고 step 331에 첫
창고 순찰 waypoint에 도달했다. 검증 상대는 NoOp이라 추격 branch는 가까운 적을 주입한
결정적 단위 테스트로 별도 확인했다. 실환경 결과는
[`logs/base_s08_s10_role_fsm.json`](../logs/base_s08_s10_role_fsm.json)에 있다.
