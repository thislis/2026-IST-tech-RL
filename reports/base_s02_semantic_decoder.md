# BASE-S02 semantic map decoder

구현은 [`blackout_rl/semantic_map.py`](../blackout_rl/semantic_map.py)에 있다.

## 좌표 계약

고정 Unity source에서 다음 변환을 확인했다.

- world/vector position: map bottom-left 기준 normalized `(x,y)`
- semantic map: `96×96`, tile당 `4×4` pixel, 전체 `24×24` cell
- Unity `DynamicRTSensor.Write`: texture bottom-up row를 tensor top-down row로 뒤집음
- decoder grid: Unity와 맞춰 `[y,x]`, `y`가 위로 증가

따라서 vector의 normalized Y와 NumPy row를 직접 같은 방향으로 사용하지 않는다.
`normalized_to_pixel`, `pixel_to_normalized`, `normalized_to_cell`이 이 변환을 한 곳에서
관리한다.

## 추출 결과

`SemanticMapDecoder.decode()`는 11개 channel 각각에 대해 다음을 제공한다.

- active semantic pixel과 normalized position
- 중복 제거한 `GridCell(x,y)`
- storage 등 연속 영역의 connected components
- wall grid와 그 역인 기본 walkable grid

벽, 아군/적군 창고, 아군/적군 유닛, Battery와 4종 특수 아이템을 동일 API로 조회할 수
있다. synthetic one-hot map에서 좌표 반전, channel별 위치, storage component를 테스트했다.

## 실제 map 검증

seed `210301`에서 다음을 decode했다.

- wall 168 cells
- ally/enemy storage 각 20 cells
- Battery 48 cells
- 초기 special item 0 cells

`unit_0` vector 위치 `(0.0625, 0.9375)`와 가장 가까운 ally-unit semantic pixel 중심의
오차는 `0.7071 pixel`로, 허용 기준 `1.5 pixel` 이내였다. 이 결과는
[`logs/base_s01_s03_live_navigation.json`](../logs/base_s01_s03_live_navigation.json)에 기록했다.
