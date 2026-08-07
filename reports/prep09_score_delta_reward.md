# PREP-09 Python Score-Delta Team Reward

구현은 `blackout_rl/reward.py`의 `ScoreDeltaRewardTracker`다. 매 step normalized
score를 integer point로 복원하고 score advantage 변화량을 팀 공통 reward로 사용한다.

```text
delta_a = current_a - previous_a
delta_b = current_b - previous_b
reward_team_a = (delta_a - delta_b) / 100
reward_team_b = -reward_team_a
```

각 팀의 5개 agent에 같은 값을 전달한다. Unity shaping reward와 이 reward를 혼합할지는
학습 config가 결정하지만 evaluator winner에는 둘 다 사용하지 않는다.

## 부호 계약

| 사례 | score transition | Team A reward | Team B reward |
| --- | --- | ---: | ---: |
| A 5점 획득 | `(0,0)→(5,0)` | `+0.05` | `-0.05` |
| A 3점 약탈당함 | `(5,0)→(2,0)` | `-0.03` | `+0.03` |
| B가 훔친 3점 적재 | `(2,0)→(2,3)` | `-0.03` | `+0.03` |
| B 승리 terminal | score 유지 | `-1.0` | `+1.0` |

약탈 후 상대 적재까지 score differential이 총 6점 이동하므로 두 event의 경쟁 reward가
각각 반영된다. terminal frame에서 Unity가 score를 `(0,0)`으로 reset하더라도 tracker는
이를 score 감소로 처리하지 않고 마지막 관측 score를 유지한 채 terminal bonus만 더한다.

## 실제 실행 확인

| game | final A:B | score events | cumulative A/B score reward | terminal A/B bonus |
| ---: | --- | ---: | --- | --- |
| 0 | 18:44 | 11 | `-0.26 / +0.26` | `-1 / +1` |
| 1 | 28:44 | 13 | `-0.16 / +0.16` | `-1 / +1` |

두 episode 모두 누적 score reward가 `(final_a-final_b)/100`과 일치했다.
