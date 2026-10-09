# BASE-S14 scripted-vs-random paired 평가

평가 runner는 [`eval/scripted_series.py`](../eval/scripted_series.py), 원본 결과는
[`logs/base_s14_scripted_vs_random.json`](../logs/base_s14_scripted_vs_random.json)에 있다.

## 평가 조건

- model: BASE-S01~S13 battery-only scripted team (`worker×3 / guard×1 / carrier×1`)
- opponent: 고정 seed의 `numpy.random.default_rng` uniform random policy
- environment held-out seeds: `5141, 5142, 5143, 5144, 5145`
- 각 seed마다 scripted model을 Team A/B에 한 번씩 배치: 5 pair, 총 10경기
- model policy seed base `514001`, opponent policy seed base `514002`
- Unity `time_scale=50`, seed pair별 독립 process 5개
- 승패 근거는 terminal `winner`만 사용하고 Unity shaping reward는 미사용

이 seed들은 PREP·BASE 구현 및 앞선 live smoke/evaluation에서 사용하지 않은 held-out 집합이다.

## 결과

| model side | 경기 | 승/무/패 | 승률 | 평균 model−random 점수 차 |
| --- | ---: | ---: | ---: | ---: |
| A | 5 | 5/0/0 | 100% | +97.8 |
| B | 5 | 5/0/0 | 100% | +97.6 |
| 전체 | 10 | 10/0/0 | 100% | +97.7 |

모든 경기는 truncation 없이 전체 agent가 동시에 terminal에 도달했다. Unity terminal frame이
점수를 초기화하므로 기록된 마지막 observable model 점수는 95~99지만, 실제 winner는 모든
경기에서 scripted side였다.

## side audit

- model side A/B 노출: `5 / 5`
- A/B model 승률 차: `0.0`
- 물리 A/B 승률: 각각 `50%`
- 물리 A−B 평균 점수 차: `+0.1`
- evaluator model-result attribution: 전 episode 일치

따라서 이 held-out 표본에서 scripted agent는 random보다 양쪽 side에서 일관되게 우세했고,
결과를 설명할 물리 side 편향은 관측되지 않았다.
