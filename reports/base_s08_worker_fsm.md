# BASE-S08 노동자 FSM

구현은 [`blackout_rl/scripted_fsm.py`](../blackout_rl/scripted_fsm.py)의
`ScriptedTeamController`와 `WorkerPhase`에 있다.

## 상태 전환

```text
SEEK_BATTERY → PICKUP → DELIVER → RETARGET → SEEK_BATTERY
```

- `SEEK_BATTERY`: 세 노동자에게 A* 경로 비용 기반으로 서로 다른 중립 Battery를 greedy 할당
- `PICKUP`: 다음 관측에서 held item이 생기면 회수 완료, 목표 Battery가 먼저 사라지면 재탐색
- `DELIVER`: 도달 가능한 가장 가까운 아군 창고로 이동하고 held item 해제로 적재 확인
- `RETARGET`: 목표를 해제한 뒤 남은 Battery를 다시 할당
- 중앙 Hunter 성소와 본진 Carrier 성소는 노동자 경로에서 금지해 의도치 않은 변신 방지
- 창고가 12 step 동안 적재를 받지 않으면 해당 창고 component를 제외하고 다른 창고로 재배정

## 실제 Unity 검증

seed `240810`에서 초기 목표는 `unit_0=(1,17)`, `unit_1=(4,17)`,
`unit_2=(8,19)`로 모두 달랐다.

| 유닛 | pickup | deposit | deposit 이후 retarget |
| --- | ---: | ---: | ---: |
| unit_0 | 66 | 114 | 115 |
| unit_1 | 110 | 178 | 179 |
| unit_2 | 95 | 182 | 183 |

세 노동자 모두 전환 순서를 실제 관측으로 확인했다. 332 step 동안 팀 전체로 Battery pickup
10회, deposit 9회가 발생했고 최종 Team A 점수는 normalized `0.51`이었다. 결과는
[`logs/base_s08_s10_role_fsm.json`](../logs/base_s08_s10_role_fsm.json)에 있다.
