# PREP-04 Observation Contract

검증 기준은 executable SHA-256
`49172f88b1ba2429677e7ec4a7876c4589876090be4888aeaef39269e29f4a69`과
`logs/prep04_07_contract.json`이다. 구현은 `blackout_rl/observation.py`,
canonical batching은 `blackout_rl/batching.py`에 있다.

## Vector: `float32[96]`

| 범위 | shape | field | 범위와 관점 |
| --- | --- | --- | --- |
| `9*i : 9*i+2` | `[2]` | `unit_i` absolute position | map origin 기준 `[0,1]`; `i=0..9` 절대 unit 순서 |
| `9*i+2` | scalar | `unit_i` team sign | 관찰자 기준 아군 `+1`, 적군 `-1` |
| `9*i+3 : 9*i+9` | `[6]` | held-item one-hot | index 0 없음, 1~5 Battery/4종 특수 아이템 |
| `90:93` | `[3]` | self class one-hot | Collector/Hunter/Carrier |
| `93` | scalar | own score | 관찰자 팀 기준, 목표 100으로 normalize |
| `94` | scalar | opponent score | 상대 팀 기준, 목표 100으로 normalize |
| `95` | scalar | time left | `[0,1]` |

파서는 shape, `float32`, finite 값, position/scalar 범위, `±1` team sign,
item/class one-hot을 검사한다. Unity routing용 `unitIndex`는 upstream
preprocessor에서 제거되므로 policy vector에는 self-ID가 없다.

## Graphic: `float32[96,96,11]` HWC

| channel | 이름 |
| ---: | --- |
| 0 | empty |
| 1 | wall |
| 2 | ally_storage |
| 3 | enemy_storage |
| 4 | ally_unit |
| 5 | enemy_unit |
| 6 | battery |
| 7 | buff_speed |
| 8 | debuff_speed |
| 9 | buff_size |
| 10 | debuff_size |

각 pixel은 정확히 하나의 binary channel에 속한다. 환경 출력은 HWC이며
`hwc_to_chw()` 또는 `ObservationBatch.graphics_chw`가 모델 입력
`float32[11,96,96]` / batch `[B,11,96,96]`으로 변환한다.

Team B 관점은 좌표를 회전하지 않고 channel `2↔3`, `4↔5`만 교환한다.
실제 Unity 관측에서 position은 A/B 관찰자 간 동일하고 team sign은 반대이며,
위 네 channel swap이 정확히 일치함을 확인했다.

## 검증

- 실제 10개 agent 모두 vector `(96,)`, graphic `(96,96,11)`, `float32`
- Gymnasium observation space `contains()` 통과
- synthetic field parser, HWC→CHW, channel naming, canonical batch 단위 테스트 통과
- 명령: `.venv/bin/python -m unittest -v tests.test_contract`
