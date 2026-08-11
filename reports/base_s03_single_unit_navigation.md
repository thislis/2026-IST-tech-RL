# BASE-S03 이동 가능 영역과 path planner

구현은 [`blackout_rl/navigation.py`](../blackout_rl/navigation.py), 실제 재현 runner는
[`scripts/verify_base_s01_s03.py`](../scripts/verify_base_s01_s03.py)에 있다.

## Planner 계약

- semantic wall이 아닌 cell을 기본 이동 가능 영역으로 사용
- bottom-left `(x,y)` grid에서 deterministic A*
- cardinal cost `1`, diagonal cost `√2`, octile heuristic
- 대각 이동 시 양쪽 cardinal cell이 모두 walkable이어야 하므로 wall corner cutting 금지
- cell path를 continuous normalized `(dx,dy)` waypoint action으로 변환
- 위치 진행이 `stall_limit` 동안 없으면 `replan_required` 신호 발생
- 충돌한 다음 cell을 임시 block했을 때 alternate route가 생성되는 단위 테스트 포함

동적 unit 회피와 장기 stuck 판정은 BASE-S04에서 확장한다. 또한 현재 semantic channel은
`BlockAll` wall은 표현하지만 적 본진의 `BlockEnemy` 전체 영역은 별도 channel로 제공하지
않는다. 이번 검증은 중립 Battery까지의 wall avoidance 범위로 제한했다.

## 독립 live-Unity 검증

고정 executable과 seed `210301`에서 다른 9개 unit에 `NoOpPolicy`를 적용하고 `unit_0`만
이동시켰다.

| 항목 | 결과 |
| --- | --- |
| 시작 | normalized `(0.0625,0.9375)`, cell `(1,22)` |
| 선택 Battery | cell `(9,17)` |
| 직선 경로의 wall | `(7,18)`, `(8,18)` |
| A* path | 9 cells, cost `10.0711`, wall cell 0개 |
| 실제 실행 | 119 environment steps, wall time 약 2.2초 |
| 결과 | Battery item id `1` 보유 확인 |
| 다른 9개 unit 최대 이동 | normalized `0.0` |
| replan | 0회; 최초 static route로 성공 |

검증 조건은 모두 통과했다.

- vector/semantic 좌표 정렬
- 직선 경로가 실제 wall과 교차
- A* route는 모든 wall을 회피
- target 외 unit은 정지
- target unit의 실제 Battery pickup

전체 trajectory 표본과 경로는
[`logs/base_s01_s03_live_navigation.json`](../logs/base_s01_s03_live_navigation.json)에 있다.
