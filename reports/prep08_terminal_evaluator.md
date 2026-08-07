# PREP-08 Terminal-Winner Evaluator

## 구현

- episode evaluator: `eval/evaluator.py`
- side-swapped runner: `eval/paired_series.py`
- 실제 검증 로그: `logs/prep08_10_paired_seed_810.json`

evaluator는 Unity reward 합계나 Python score reward로 승자를 추론하지 않는다.
각 episode의 승자는 terminal `infos[*]["winner"]`만 사용하며 model의 side를 적용해
`win/draw/loss`로 변환한다. Unity shaping reward는 진단용 누적 필드로만 기록하고
`unity_shaping_used_for_winner=false`를 schema validator가 확인한다.

## Side attribution

같은 environment seed와 role별 policy RNG stream으로 두 경기를 실행한다.

1. game 0: model=Team A, opponent=Team B
2. game 1: model=Team B, opponent=Team A

summary는 Team A/B 승수가 아니라 model 관점 결과를 집계한다. synthetic regression
test에서는 A에서 이긴 model과 B에서 이긴 model을 모두 model win으로 계산함을 확인했다.

## 실제 smoke series

seed `810`, random-v1 대 random-v1 한 pair 결과다.

| game | model side | A:B score | terminal winner | model result | steps |
| ---: | --- | --- | --- | --- | ---: |
| 0 | A | 18:44 | Team B | loss | 21,003 |
| 1 | B | 28:44 | Team B | win | 21,003 |

series summary는 `1 win / 0 draw / 1 loss`, model 평균 점수 차 `-5.0`이다.
이는 evaluator 기능 smoke test이며 여러 held-out seed의 side bias를 평가하는 PREP-14를
대체하지 않는다.
