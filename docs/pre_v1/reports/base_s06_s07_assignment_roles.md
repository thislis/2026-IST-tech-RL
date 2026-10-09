# BASE-S06~S07 task assignment와 기본 역할

구현은 [`blackout_rl/coordination.py`](../blackout_rl/coordination.py)의
`greedy_path_assignment`와 [`blackout_rl/team_state.py`](../blackout_rl/team_state.py)의 기본
역할 구성에 있다.

## 할당 계약

- A* 실제 경로 비용으로 모든 도달 가능한 유닛-목표 후보 계산
- `(path cost, slot, target x, target y)` 순으로 결정적인 greedy 선택
- 유닛과 목표를 각각 한 번만 사용
- 역할 필터와 이미 예약된 목표 지원
- wall 또는 단절 영역으로 도달할 수 없는 목표 제외

## 초기 역할

team-local slot에 따라 `worker ×3`, `guard ×1`, `carrier ×1`을 고정한다.

| slot | 역할 |
| --- | --- |
| 0, 1, 2 | worker |
| 3 | guard |
| 4 | carrier |

실제 Unity seed `240513`에서 역할이 붙은 5유닛에 서로 다른 도달 가능 Battery를 배정했고,
전원이 129 step 안에 목표 또는 Chebyshev 거리 1 이내에서 Battery를 회수했다. 시작 A* 비용은
`4.4142~10.8284`, 실제 normalized 이동 거리는 `0.1547~0.4020`이었으며 semantic wall cell
진입은 없었다.

이 검증은 S04~S07의 **상태·할당·동시 이동 기반**을 분리 확인하기 위해 guard/carrier에도
Battery 목표를 준 coordination smoke다. 중앙/본진 성소 이동과 역할별 행동은 BASE-S08~S10의
FSM에서 연결한다. 전체 할당과 pickup step은
[`logs/base_s04_s07_s13_coordination.json`](../logs/base_s04_s07_s13_coordination.json)에 있다.
