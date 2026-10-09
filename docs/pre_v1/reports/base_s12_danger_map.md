# BASE-S12 위험 지도

구현은 [`blackout_rl/strategy.py`](../blackout_rl/strategy.py)의 `DangerMap`과
`WeightedAStarPlanner`에 있다.

## 역할별 비용

semantic wall 기반 walkable grid 위에 매 step vector observation의 적 5기 위치로 traversal
multiplier를 만든다.

- 노동자: 적 주변 4 cell에 거리 감쇠 회피 비용
- 전달자: 적 주변 5 cell에 노동자보다 큰 회피 비용
- 경비원: 적 주변 5 cell 비용을 최소 `0.4`까지 낮춰 추격 방향 선호
- weighted A*: `이동 거리 × 진입 cell multiplier`의 총합 최소화
- admissible heuristic은 전체 walkable cell의 최소 multiplier를 곱한 octile distance
- 기존 corner-cut 방지, 금지 성소, temporary blocked replan 계약 유지

단위 테스트에서 같은 적 cell의 비용이 `carrier > worker > 1 > guard`임을 확인하고, 노동자는
더 긴 안전 경로, 경비원은 적 쪽 경로 비용이 더 낮아지는 것을 검증했다.

## 실제 관측 위치 기반 경로 audit

Unity seed `241112`에서 상대 `unit_5`를 spawn `(22,1)`에서 중앙 `(11,9)`로 실제 이동시킨
뒤 같은 시작 `(4,2)`와 목표 `(17,16)`를 비교했다.

| 경로 | cell 수 | 노동자 weighted cost | 적과 최소 Chebyshev 거리 |
| --- | ---: | ---: | ---: |
| 일반 A* | 15 | 58.9828 | 1 |
| 위험비용 A* | 23 | 24.0711 | 5 |

위험 경로는 기하학적으로 더 길지만 관측된 적에게서 충분히 떨어져 총 위험비용이 낮았다.
좌표와 전체 cell path는
[`logs/base_s11_s12_strategy.json`](../logs/base_s11_s12_strategy.json)에 기록했다.
