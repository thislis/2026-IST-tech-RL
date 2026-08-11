# BASE-S01 policy primitives

구현은 [`blackout_rl/policy.py`](../blackout_rl/policy.py)에 있다.

## 공통 계약

`ActionPolicy`는 evaluator와 scripted controller가 공유하는 다음 인터페이스를 정의한다.

```python
act(observations, agents) -> dict[agent, np.ndarray(float32, shape=(2,))]
```

- 요청 agent가 observation에 없으면 즉시 실패한다.
- action은 agent마다 독립된 배열로 반환한다.
- evaluator의 model/opponent type도 `RandomPolicy` 전용에서 `ActionPolicy`로 확장했다.

## 구현

| policy | 동작 | 재현성/범위 |
| --- | --- | --- |
| `RandomPolicy(seed)` | agent마다 uniform random `(dx,dy)` | 격리된 NumPy RNG, float32 `[-1,1]` |
| `NoOpPolicy()` | 항상 `(0,0)` | 결정적, unit 정지 검증용 |
| `FixedDirectionPolicy(direction)` | 한 방향으로 계속 이동 | 입력을 L2 unit vector로 정규화 |

동일 seed random action 일치, dtype/range, NoOp, direction normalization, 누락 agent,
배열 alias 방지와 evaluator protocol 적합성을 단위 테스트했다. 실제 BASE-S03 검증에서는
`NoOpPolicy`로 target 외 9개 unit을 고정했고 최종 최대 위치 변화가 정확히 `0.0`이었다.
