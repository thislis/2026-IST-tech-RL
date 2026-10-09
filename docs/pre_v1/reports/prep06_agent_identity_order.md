# PREP-06 Agent Identity and Batch-Order Report

## 관측된 ML-Agents row order

세 번의 reset(seed `40407`, `40407`, `40408`)과 terminal까지 다음 raw order가
안정적으로 관측됐다.

```text
Team A behavior rows: unit_3, unit_2, unit_4, unit_1, unit_0
Team B behavior rows: unit_6, unit_7, unit_5, unit_8, unit_9
combined obs dict:     unit_3, unit_2, unit_4, unit_1, unit_0,
                       unit_6, unit_7, unit_5, unit_8, unit_9
```

이 순서는 현재 build에서 안정적이지만 unit index 순서가 아니므로 학습 코드가
dict insertion order나 ML-Agents row를 의미론적 slot으로 사용하면 안 된다.
Unity raw vector의 routing-only `unitIndex`를 Python wrapper가 읽어 agent name으로
매핑하는 방식은 reset과 terminal 전후에 안정적으로 동작했다.

## Canonical batching 계약

`blackout_rl.batching.stack_observations()`는 입력 dict 순서와 무관하게 다음 순서를
강제한다.

```text
global: unit_0, unit_1, unit_2, unit_3, unit_4,
        unit_5, unit_6, unit_7, unit_8, unit_9
Team A: unit_0..unit_4  (batch 5)
Team B: unit_5..unit_9  (batch 5)
team-local slot_id: 0,1,2,3,4
```

세 reset과 terminal observation에서 canonical order와 팀별 batch 크기 5가 모두
유지됐다. Team A/B 관점 교환도 team sign 반전과 storage/unit channel swap으로
확인했다.

## Self-ID 결론

초기 상태에서 같은 팀의 다섯 agent는 vector와 graphic이 각각 완전히 동일했다.
96-vector에는 전체 10개 unit의 위치가 있지만 “이 row를 제어하는 자기 unit”을
가리키는 index가 없고, self-class만 있다. 시작 시에는 모두 Collector이므로 이것도
구분 신호가 아니다.

따라서 제출 모델이 유닛을 구분하는 방법은 다음으로 확정한다.

- PettingZoo agent name을 canonical unit index로 변환한다.
- 팀별 모델 batch에는 `slot_id ∈ {0,1,2,3,4}`를 별도 입력으로 전달한다.
- actor는 slot embedding을 사용하거나 slot별 head를 사용한다.
- observation dict 순서에서 slot을 추론하지 않는다.

이 결론은 이후 PREP-12 actor 인터페이스의 필수 입력 계약이다.
