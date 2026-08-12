# BASE-S04 stuck detector

구현은 [`blackout_rl/coordination.py`](../blackout_rl/coordination.py)의
`StuckDetector`와 [`scripts/verify_base_s04_s07_s13.py`](../scripts/verify_base_s04_s07_s13.py)의
재계획 처리에 있다.

## 판정과 복구 계약

- 목표가 있고 이동 action 크기가 기준 이상인 구간만 감시
- 기본 30 step 동안 시작 위치 대비 normalized displacement가 `0.01` 미만이면 stuck event 생성
- `NoOp`, 목표 변경, step 역행, Battery 회수 시 해당 유닛의 감시 window 초기화
- stuck이면 현재 waypoint를 임시 blocked cell로 등록하고 같은 목표까지 A* 재계획
- 대체 경로가 없으면 다른 유닛의 목표를 예약한 채 새 Battery를 재할당

고정 위치를 주입한 테스트에서 stuck event가 발생하고, 충돌 cell `(1,2)`를 제외한 대체
경로가 목표까지 생성됨을 확인했다. 정상 이동 유닛, `NoOp`, 목표 변경에는 오탐이 없다.

seed `240513`의 실제 5유닛 주행에서는 129 step 안에 전원이 정체 없이 목표를 회수해
stuck/replan이 각각 0회였다. 따라서 실제 주행은 정상 경로를 검증하고, 정체 복구 branch는
결정적인 주입 테스트로 검증했다. 실환경 결과는
[`logs/base_s04_s07_s13_coordination.json`](../logs/base_s04_s07_s13_coordination.json)에 있다.
