# BASE-S05 team-local unit state

구현은 [`blackout_rl/team_state.py`](../blackout_rl/team_state.py)에 있다.

`TeamStateTracker`는 observation dictionary 삽입 순서에 의존하지 않고 팀의 canonical agent
순서로 매 step의 `TeamSnapshot`을 만든다. 각 `UnitState`에는 다음을 보존한다.

- team-local `slot_id`와 global `unit_index`
- normalized 위치와 `24×24` grid cell
- 보유 item id와 각 agent 자신의 class id
- 고정 역할, 현재 목표, step
- 팀 관점 own/opponent score와 남은 시간

team B에서 `unit_5~unit_9`가 local slot `0~4`, global index `5~9`로 분리되는지와 각 agent
관측의 self-class를 사용하는지 테스트했다. 누락된 팀원이 있으면 조용히 잘못된 batch를
만들지 않고 `KeyError`를 낸다.

실제 seed `240513`에서는 team A의 5개 slot을 연속 추적했으며, 전 유닛의 이동과 Battery
item id `1` 보유 전환을 확인했다. 증거는
[`logs/base_s04_s07_s13_coordination.json`](../logs/base_s04_s07_s13_coordination.json)에 있다.
