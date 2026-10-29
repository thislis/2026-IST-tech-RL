# v12 설계를 위한 풀리그 재분석

작성일: 2026-10-26. 결론은 **작은 신경망으로 팀의 왕복 작업 배정을 개선하는 방향**이다. [설계](../plans/v12_design.md)와 [검증 계획](../plans/v12_implementation_plan.md)은 아래의 관측 사실과 새 가설을 구분해 사용한다.

## 1. 근거와 재검증 범위

| 항목 | 값 |
|---|---|
| 사용자 지정 원본 | `C:\Users\HeXA\Desktop\analysis.html` |
| 파일 크기 | 792,212 bytes |
| SHA256 | `45e6802cfdc16360c595a93c89a942104d6b2625d17274d55ce73772603faae5` |
| 내장 데이터 위치 | `<script id="analysis-data" type="application/json">` |
| schema / run | `league.analysis.v1` / `version_best_ties_league` |
| 생성 시각 | 2026-10-20 15:30:19.120578 +09:00 |
| 규모 | 21개 정책, 210개 대진, 2,100경기, 17,695,695 simulation step |

이번에는 HTML 내장 `models`, `pairs`, `audit` 및 현존 소스를 읽고 재집계했다. 원본 trace 디렉터리 `logs/shared/agent_league/version_best_ties_league`는 현재 작업 공간에 없어 과거 2,100경기의 raw trace 감사를 다시 실행하지 않았다. 기존 [v11 분석](../../v11/reports/league_findings.md)의 raw-log 조사와 이번 검증 범위를 혼동하지 않는다. 원본 HTML은 수정하거나 저장소로 복사하지 않았다.

## 2. 유지해야 할 강점

| 정책 | 승/무/패 | 승점률 | 팀 추론 p50 / p95 (ms) |
|---|---:|---:|---:|
| v4 | 180/0/20 | 90.0% | 5.2917 / 7.4532 |
| v6 | 180/0/20 | 90.0% | 8.2724 / 10.8832 |
| v7 Rewired | 180/0/20 | 90.0% | 9.3070 / 12.2732 |
| v5 | 179/0/21 | 89.5% | 5.9822 / 8.2922 |
| v7 FlyWire | 179/0/21 | 89.5% | 9.3516 / 12.3483 |
| v9 | — | 36.5% | 1.9683 / 2.7474 |
| v10 | — | 37.0% | 1.9322 / 2.7047 |

추론 시간은 CPU 8병렬 실행에서 경기별 일부 chunk를 표집한 값이다. 동일 상태·동일 호스트의 통제된 벤치마크가 아니므로 v4를 v6보다 정확히 몇 배 빠르다고 일반화하지 않는다. v11은 이 풀리그 참가자가 아니다.

상위권은 수집→귀환→반납, 역할 분담, waypoint·정체 회복을 갖춘 planner 결합 정책이다. 반납 proxy는 경기당 약 20.7~21.1회, 큰 방향 전환은 약 1.7~1.9%다. v9·v10은 반납 proxy 1.41/2.07회와 큰 방향 전환 약 45~47%를 보였다. 목표 수행을 버리고 가벼운 직접 이동망만 남길 근거는 없다.

v8 C1/Flat의 전원 같은 행동 비율 약 100%/97%는 개체 연결의 중요성을 보여준다. 다만 동일 클래스의 같은 입력을 받는 결정론적 정책이라는 조건 차이가 있어 신경망 크기 하나로 실패를 설명하지 않는다. v12도 현재 제공 runner의 canonical 팀 다섯 행과 자기 entity의 연결을 유지한다.

## 3. 새 발견: 5:5보다 강한 상위권 결과 일치

상위 다섯 모델의 10개 대진에서 `pairs[].matches`를 `repeat`로 묶으면 양 진영 교대 쌍이 50개다. **모델을 바꿔 배치한 두 경기의 길이와 Blue/Red 누적 raw reward가 50쌍 모두 정확히 같았다.** JSON 수치의 정확 비교이며 허용오차로 만든 일치가 아니다.

| 비교 | 일치한 쌍 |
|---|---:|
| 승리한 물리 진영 | 50/50 |
| 경기 길이 `steps` | 50/50 |
| 두 진영 누적 `rewards` | 50/50 |
| Blue가 두 경기 모두 승리 | 21/50 |
| Red가 두 경기 모두 승리 | 29/50 |

따라서 상위권의 동률은 여러 경기의 차이가 우연히 상쇄된 것만은 아니다. **공통 기본 제어가 결과를 지배하고 학습된 부분이 실행 결과를 거의 바꾸지 못했을 가능성**을 우선 검증할 가치가 있다. 전체 중간 action·hidden·trajectory 동일성, residual 개입률 0, 진영 자체의 인과적 우위까지 증명하는 자료는 아니다.

이 결과는 v7의 복잡한 연결 구조를 그대로 옮길 이유를 약하게 하고, 기본 제어를 보존하면서 실제 배정이 바뀌는 작은 학습기를 시험할 이유를 강화한다. 새 정책의 학습 기여는 `scorer 제거`, `동일 입력에서 배정 변경률`, `closed-loop 결과`로 확인해야 한다.

재집계의 핵심 절차는 다음과 같다. `data`는 위 script의 JSON이며 상위 집합은 모델 이름·순위를 먼저 확인한 것이다.

```python
top = {m["id"] for m in data["models"][:5]}
checked = 0
for pair in data["pairs"]:
    if not {pair["a"], pair["b"]} <= top:
        continue
    for repeat in range(5):
        left, right = sorted(
            (m for m in pair["matches"] if m["repeat"] == repeat),
            key=lambda m: m["side"],
        )
        assert left["steps"] == right["steps"]
        assert left["rewards"] == right["rewards"]
        physical = lambda m: m["side"] if m["winner"] == pair["a"] else 1 - m["side"]
        assert left["winner"] in (pair["a"], pair["b"])
        assert right["winner"] in (pair["a"], pair["b"])
        assert physical(left) == physical(right)
        checked += 1
assert checked == 50
```

## 4. 이번에 겨냥할 구조적 맹점

### 4.1 학습기가 보기 전에 좋은 작업이 후보에서 사라질 수 있다

공통 [coordination.py](../../../code/shared/blackout_rl/coordination.py)의 `greedy_path_assignment()`는 현재 유닛→목표의 편도 경로 비용으로 edge를 정렬하고 greedy 배정한다. v11의 [policy.py](../../../code/v11/blackout_v11/policy.py) `TensorPlanner.candidates()`도 intent별 유효 셀을 편도 거리로 `topk(5)` 한 뒤 learned scoring에 전달한다. 실제 candidate feature는 좌표 2 + intent 7 + 직선거리 1 + role 3의 13개다.

예를 들어 A 배터리는 가는 데 2초, 반납에 12초가 들고 B는 가는 데 5초, 반납에 2초가 들 수 있다. 편도 우선의 후보 절단은 B를 배제할 수 있다. 상대에게 먼저 빼앗기는 가까운 배터리도 같은 문제다. Attention을 키워도 **입력 후보에 없는 작업을 선택할 수는 없다.** 이는 가능한 반례이며 해당 손실이 리그에서 몇 번 발생했는지는 미측정이다.

v12는 편도·귀환·상대 도착시간 범위·이미 진행 중인 목표를 함께 보존하는 후보 집합을 만든다. 실제 stack 수량은 관측에 없으므로 배터리 타일 수나 1~8 범위를 정확한 수량으로 바꾸지 않는다.

### 4.2 앞 유닛의 좋은 선택이 팀에는 나쁜 선택일 수 있다

v11 `evaluate_decision()`은 `for slot in range(5)`에서 앞 유닛의 선택을 예약하고 뒤 유닛을 제한한다. 학습된 순차 정책도 좋은 공동 전략을 배울 수 있으나, **고정 슬롯 우선순위 없이 모든 배정의 합을 직접 비교하는 구조는 아니다.** 기존 global-edge greedy 역시 정확한 최소합 배정과 다르다.

설명용 비용표:

| 왕복 비용 | 작업 A | 작업 B |
|---|---:|---:|
| 유닛 0 | 1 | 2 |
| 유닛 1 | 1.1 | 100 |

유닛 0이 A를 먼저 선택하면 합계 101이다. 유닛 0→B, 유닛 1→A라면 3.1이다. v12는 5개 유닛이라는 작은 팀 크기를 이용해 **가산 효용과 작업 용량 아래의 정확한 공동 배정**을 계산한다. 이 장난감 예제는 실제 경기 손실이나 개선량의 추정이 아니다.

### 4.3 역할 수 자체가 자원을 소비한다

Hunter는 수집할 수 없고 살아 있는 동안 Collector로 되돌아갈 수 없다. 초기 3 worker+2 guard는 강한 기준선이지만 모든 배치·상대·남은 시간에서 최선이라는 증거는 없다. v11도 역할 변경을 학습하므로 새로운 역할 종류를 추가하는 것이 핵심은 아니다.

v12는 **지금 Hunter로 전환해 얻는 방어 가치와 이후 포기하는 수집 기회**를 같은 작업 선택에서 비교한다. 기존 guard의 무목적 추격도 수집선 보호·보유물 회수 가능성·시간 제한으로 평가한다. 상대 class와 kill 가능성은 관측에 없는 확정 사실로 입력하지 않는다. 무조건 worker를 늘리는 설계도 채택하지 않는다.

## 5. 경량화의 다른 맹점: 가중치만 줄이면 충분하지 않다

v11 전체 `MyPolicy`는 코드의 Linear·LayerNorm·GRU·학습 Parameter를 합하면 **194,327개**다. FP32 가중치는 777,308 bytes이며 critic·buffer·직렬화 overhead를 제외한다. 두 GRU만 62,208개다. v9/v10은 각각 123,765/120,905개([v10 계획](../../v10/plans/v10_plan.md)의 비교값).

v11 `candidates()`는 5유닛×3역할의 15개 source distance field를 만들고, 기본 4-step residual event에도 후보 생성을 호출한다. `navigate()`는 한 유닛의 캐시가 바뀌어도 새 거리장 다섯 개를 계산한 뒤 선택한다. `_relax`는 최대 575회와 수렴 조기 종료를 사용하며 항상 575회를 도는 것은 아니다. 개발 `ExportCore`의 `force_compute=True`는 이러한 경로를 매 호출 실행한다.

따라서 v12의 경량 목표는 다음 두 축이다.

- 공유 `64→64→32→1` MLP: **6,273 parameters**, FP32 25,092 bytes, v11 대비 약 96.8% 감소.
- residual 주기 제거, 작업 결정 시에만 후보·배정 갱신, 유효한 거리장 재사용, ONNX에서도 실제 무계산 분기 확인.

v11의 기존 M4 Pro probe 평균 2.41ms/p95 6.55ms는 [별도 구현 보고](../../v11/reports/implementation.md)의 수치다. 풀리그의 v4 지연과 직접 비교하지 않고 같은 호스트·관측 sequence에서 다시 측정한다.

## 6. 평가에서 반드시 보완할 부분

**판정 기준:** HTML `audit`에는 raw reward 승자와 최종 관측 score 승자가 반대인 경기 95개, score 동점인데 reward 승패가 있는 경기 119개가 있다. 마지막 전이에 점수 우열이 바뀐 경기도 5개다. 돌격형은 reward 승점률 63.25%인데 score 진단 승점률은 17.0%, 반납 proxy는 0이다. score는 공식 서버 인증 결과가 아니며 두 승률을 합치지 않는다. [운영진 답변 정리](../../common/organizer_qa_2026-10-24.md)와 동일하게 현재 runner 결과를 보존하면서 score 비퇴행도 확인한다.

**행동 계측:** [trace.py](../../../code/shared/agent_league/trace.py)의 `after is not None and not terminal` 조건 때문에 기존 pickup/delivery/movement proxy는 마지막 자연 종료 전이를 제외한다. delivery는 같은 step에 팀 점수 증가와 아군 Battery→빈손이 함께 있으면 1회다. 여러 유닛의 동시 반납 수량이 아니며 사망과 다른 유닛의 득점이 겹칠 수도 있다. 마지막 score를 복원한 감사와 이 proxy 한계는 별개다.

v12는 terminal을 포함한 새 trip ledger를 기록한다. 관측만으로 구분되지 않는 반납/사망·동시 사건은 `ambiguous`로 남기고, 유닛별 성공으로 임의 분배하지 않는다. 기존 delivery 약 21회를 새 ledger의 성공 횟수와 직접 같은 숫자로 비교하지 않는다.

**일반화:** 초기 관측은 1,050개 진영 교대 쌍에서 일치했지만, 배터리 배치 1,050종·창고 5종·벽 1종이었다. 새 미로 일반화가 검증된 것이 아니다. 여러 policy training seed도 아니다. v12는 map-pair를 통계 단위로 하고, 창고 조합 보류 평가와 별도 학습 seed를 둔다.

## 7. 선택과 기각

| 선택지 | 판단 |
|---|---|
| v11에 Attention/GRU를 더 추가 | 이번 맹점을 직접 해결하지 않고 경량 요구에도 맞지 않음 |
| v4 checkpoint만 압축 | 유용한 보존 기준선이지만 후보·배정의 제한을 그대로 유지 |
| v9/v10식 직접 이동망 재시도 | 목표 완주를 다시 학습해야 하므로 기본 선택에서 제외 |
| 모든 planner 행동 BC | v3의 높은 분류 정확도와 낮은 closed-loop 성능을 반복할 위험 |
| **작업 후보 개선 + 작은 scorer + 정확한 공동 배정** | **v12 기본 선택** |

가설을 분리하려면 편도/왕복 후보, 순차/공동 배정, 고정/학습 scorer, 고정 3+2/변신 기회비용을 각각 대조해야 한다. 개선이 확인되지 않으면 단순한 기준선을 유지한다. 신경망이 작다는 사실이나 DP가 정확하다는 사실만으로 게임 성능 개선을 선언하지 않는다.
