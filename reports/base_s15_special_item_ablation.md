# BASE-S15 특수 아이템 정책 ablation

선택형 정책은 [`blackout_rl/scripted_fsm.py`](../blackout_rl/scripted_fsm.py)의
`enable_special_items`에 구현했다. 기본값은 `False`다.

활성화하면 대기 중인 노동자 최대 1명만 현재 semantic map의 특수 아이템을 우선 배정받고,
회수 후 기존 `DELIVER` 경로로 아군 창고에 적재한다. 나머지 노동자 2명과 전달자는 Battery
수집을 계속한다. 특수 아이템의 held id 2~5와 map channel 7~10 대응을 사용한다.

## 같은 조건의 paired ablation

BASE-S14와 같은 seeds `5141~5145`, side swap, random action stream으로 10경기씩 비교했다.

| variant | 승/무/패 | 평균 점수 차 | 평균 episode step | 특수 pickup/deposit |
| --- | ---: | ---: | ---: | ---: |
| battery-only | 10/0/0 | +97.7 | 841.1 | 정책 배정 0회 |
| special enabled | 10/0/0 | +96.0 | 915.2 | 7 / 6 |
| special − battery | 0%p | **−1.7** | **+74.1** | — |

특수 variant는 실제로 특수 아이템을 배정 18회, 회수 7회, 적재 6회 수행했지만 승률은
같고 평균 점수 차가 낮아졌으며 종료도 느려졌다. seed/team별 점수 차는 6경기 악화,
3경기 동일, 1경기 개선이었다.

## 승격 결정

승격 규칙은 `승률 하락 없음 AND 평균 점수 차의 엄격한 개선`이다. 이번 결과는 두 번째
조건을 충족하지 못했으므로 특수 아이템 정책을 기본 agent에 유지하지 않는다.

- 기본 설정: [`configs/policies/scripted_default_v1.json`](../configs/policies/scripted_default_v1.json)
- battery 원본: [`logs/base_s14_scripted_vs_random.json`](../logs/base_s14_scripted_vs_random.json)
- special 원본: [`logs/base_s15_special_vs_random.json`](../logs/base_s15_special_vs_random.json)
- paired 판단: [`logs/base_s15_special_item_ablation.json`](../logs/base_s15_special_item_ablation.json)

구현과 실험 옵션은 향후 더 강한 opponent에서 재평가할 수 있도록 보존하지만 기본값은
`special_items=false`로 확정했다.
