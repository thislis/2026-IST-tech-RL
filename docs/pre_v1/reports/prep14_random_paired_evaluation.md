# PREP-14 random-vs-random paired-seed 평가

실제 평가 원본은
[`logs/prep14_random_paired_5seeds.json`](../logs/prep14_random_paired_5seeds.json)이다.
PREP-08 evaluator에 물리 A/B side 통계와 attribution audit를 추가했고,
[`eval/paired_series.py`](../eval/paired_series.py)는 seed pair를 독립 Unity process에서
병렬 실행할 수 있게 확장했다.

## 평가 조건

- policy: 양쪽 모두 `numpy.random.default_rng` uniform `float32[-1,1]`
- environment seeds: `1401, 1402, 1403, 1404, 1405`
- 각 seed마다 model을 A와 B에 한 번씩 배치: 5 pair, 총 10경기
- role별 policy RNG base: model `914001`, opponent `914002`
- Unity `time_scale=50`, 5 worker
- 모든 경기 21,003 step에서 정상 동시 termination
- truncation과 deadlock 없이 모든 경기가 완료됐고 terminal에서 10 agent가 동시 종료
- 승패 근거는 모두 `terminal_info.winner`; Unity shaping reward는 승패 판정에 미사용

## 경기 결과

| seed | game | model side | A:B score | 물리 winner | model result |
| ---: | ---: | --- | ---: | --- | --- |
| 1401 | 0 | A | 41:35 | A | win |
| 1401 | 1 | B | 39:34 | A | loss |
| 1402 | 0 | A | 18:41 | B | loss |
| 1402 | 1 | B | 15:36 | B | win |
| 1403 | 0 | A | 33:27 | A | win |
| 1403 | 1 | B | 36:25 | A | loss |
| 1404 | 0 | A | 24:39 | B | loss |
| 1404 | 1 | B | 40:52 | B | win |
| 1405 | 0 | A | 10:30 | B | loss |
| 1405 | 1 | B | 10:32 | B | win |

## Model 관점

| model side | 경기 | 승/무/패 | 승률 | 평균 model−opponent 점수 차 |
| --- | ---: | ---: | ---: | ---: |
| A | 5 | 2/0/3 | 40% | −9.2 |
| B | 5 | 3/0/2 | 60% | +7.8 |
| 합계 | 10 | 5/0/5 | 50% | −0.7 |

각 seed pair에서 model은 정확히 한 번 이기고 한 번 졌다. 물리 side를 한 번씩 공평하게
배정하므로 특정 side의 이득이 model 결과에 그대로 고정되지 않았고, 전체 model 결과도
5승 5패로 균형을 이뤘다.

## 물리 side 관점

| 물리 side | 승/무/패 | 승률 | 평균 자기 side−상대 점수 차 |
| --- | ---: | ---: | ---: |
| A | 4/0/6 | 40% | −8.5 |
| B | 6/0/4 | 60% | +8.5 |

이번 5개 seed에서는 seed 1401·1403의 두 경기 모두 A가 이겼고, seed
1402·1404·1405의 두 경기 모두 B가 이겼다. 즉 동일 seed에서 random stream의 side를
교환해도 물리 winner가 유지되는 강한 seed/side 효과가 관측됐다.

표본이 5 pair뿐이므로 이것을 전체 환경의 고정 B-side bias라고 확정할 수는 없다. 다만
향후 모든 model 평가에서 paired seed와 side swap을 유지해야 한다는 근거로 충분하며,
모델 승격 평가에서는 seed pair 수를 늘리고 pair 단위 confidence interval을 보고해야 한다.

## Evaluator 편향 검사

summary의 audit 결과는 다음과 같다.

- model A/B 노출 수: `5 / 5` — balanced
- 모든 episode의 `model_result == model_result(terminal winner, model_team)` — true
- 물리 A/B winner와 model winner를 별도 집계 — true
- `winner_derived_from_unity_shaping` — false
- `evaluator_side_attribution_passed` — true

따라서 evaluator가 A 또는 B를 무조건 model 승리로 세는 논리적 편향은 없으며, 관측된
물리 side 차이도 숨기지 않고 별도 지표로 반환한다. 이는 “환경에 side 효과가 없다”는 결론과는
구분한다.
