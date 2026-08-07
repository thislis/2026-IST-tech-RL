# PREP-10 Evaluation Logging Schema

## Schema와 검증기

- episode JSON Schema: `schemas/episode_v1.schema.json`
- paired-series JSON Schema: `schemas/paired_series_v1.schema.json`
- runtime validator/hashing: `blackout_rl/logging_schema.py`
- example: `logs/prep08_10_paired_seed_810.json`

runtime schema version은 episode `blackout.episode.v1`, series
`blackout.paired_series.v1`이다. evaluator는 파일을 쓰기 전에 모든 episode와 summary를
검증하며 누락 field, 잘못된 side, SHA 형식, winner/result 값, episode count 불일치를
fail-closed로 거부한다.

## 필수 기록

| 범주 | 주요 field |
| --- | --- |
| identity | `episode_id`, `pair_id`, `pair_index`, schema version |
| randomness | environment `seed`, model/opponent `policy_seeds` |
| side | model/opponent team과 A/B side |
| policies | policy ID, kind, checkpoint path, checkpoint SHA-256 |
| environment | game/API commit, executable path/SHA-256, adapter |
| result | A/B 및 model/opponent score, terminal winner, model result |
| length | steps, wall seconds, start/end UTC, last time-left |
| rewards | score-delta 누적, terminal bonus, Unity shaping 진단 합계 |
| termination | simultaneous termination, truncation, terminal-frame score 한계 |

random-v1은 weight checkpoint가 없는 baseline이므로
`configs/policies/random_v1.json`을 policy artifact로 고정하고 그 SHA-256
`08326e77ea3ed37e18948e5098de228157ed430e490c2cd4717a11fa3260c6d1`을 model과
opponent checkpoint field에 기록했다. Unity executable SHA-256은
`49172f88b1ba2429677e7ec4a7876c4589876090be4888aeaef39269e29f4a69`이다.

`tests/test_contract.py`는 schema JSON round-trip, checkpoint hash round-trip,
필수 field와 paired model-side 집계를 회귀 검사한다.
