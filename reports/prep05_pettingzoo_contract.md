# PREP-05 PettingZoo Contract Report

## 적용 환경

- build SHA-256: `49172f88b1ba2429677e7ec4a7876c4589876090be4888aeaef39269e29f4a69`
- project adapter: `blackout_rl.env.ContractBlackOutEnv`
- live evidence: `logs/prep04_07_contract.json`
- test suite: `tests/test_contract.py`

고정된 upstream Python/Unity 조합에서는 seed SideChannel과 reset command가 같은
ML-Agents exchange에 들어가면 Unity가 맵을 생성한 뒤 seed를 처리한다. 또한
semantic-map cache가 reset을 넘어 남는다. `ContractBlackOutEnv`는 최초 handshake를
완료하고 seed를 한 exchange 먼저 전달하며 reset cache를 비워, 호출한
`reset(seed=N)`이 반환하는 episode에 N이 적용되도록 한다.

## 실제 계약 결과

| 항목 | 결과 |
| --- | --- |
| `possible_agents` / reset agents | `unit_0..unit_9`, 10개 |
| reset obs/info | agent별 obs 존재, info는 빈 dict |
| parallel step 입력 | 모든 live agent의 `float32[2]` action |
| parallel step 반환 | obs/reward/termination/truncation/info 모두 같은 10개 key |
| observation | vector `(96,)`, graphic `(96,96,11)`, `float32`, finite |
| action space | `(2,)`, 각 원소 `[-1,1]` |
| same seed | `40407`을 반복했을 때 wall/storage channel bitwise 동일 |
| different seed | `40408`은 `40407`과 다른 storage layout |
| episode end | 10개 agent가 같은 step에 모두 termination |
| truncation | 전체 episode에서 없음 |
| terminal | 21,003 steps, A 19 / B 24, `winner=1`, 점수 판정 일치 |

terminal frame에서는 Unity가 새 episode score를 먼저 노출해 `score_0=score_1=0`이므로
최종 점수는 마지막 non-terminal info를 보존하고 winner는 terminal info에서 읽었다.

## 회귀 결과

`tests/test_contract.py`의 11개 테스트가 모두 통과했다. 이 범위에는 parser,
HWC→CHW, canonical batch, reset/step/termination, seed, team perspective,
agent order, zero/non-zero action 및 terminal winner 검사가 포함된다.

```bash
.venv/bin/python -m unittest -v tests.test_contract
```
