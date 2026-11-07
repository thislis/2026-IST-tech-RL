# v13 설계 근거와 지수 간격 기억의 해석

작성일: 2026-11-02. 저장된 리그 결과·v12 코드·최신 상태 포트 구현을 읽어 정리했다. 새 경기를 실행하거나 기존 결과를 수정하지 않았다.

## 1. 리그에서 실제로 확인한 것

기준은 [새 게임 ranking.csv](../../../logs/shared/agent_league/v12_full_league_new_game_20261101/ranking.csv), [summary.json](../../../logs/shared/agent_league/v12_full_league_new_game_20261101/summary.json), [v12 상대별 보고서](../../../logs/shared/agent_league/v12_full_league_new_game_20261101/models/v12-before-learned.md)다. 로컬 logs는 Git에 포함되지 않으므로 새 clone에서는 [문서 요약](../../v12/reports/full_league_new_game.md)을 참고한다.

| 항목 | 확인값 | 설계에 주는 의미 |
|---|---|---|
| 전체 경기 | 2,294 / 2,310 완료 | 잠정 순위이며 실패 16경기를 성적으로 채우지 않음 |
| 참가 v12 | `v12-before-learned` | learned 제어 변경 후 모델이 아님 |
| v12 결과 | 188승 0무 20패 / 208경기 | 완료 경기 승률 90.38% |
| v12 macro 승점률 | 90.48%, 잠정 1위 | 상대별 표본 수 차이를 반영한 동일 가중 평균 |
| 비교 순위 | v5 85.71%, v6 84.76% | 새 환경에서 v12가 강한 baseline임 |
| v12 누락 | defender 상대 2경기 | defender 100%는 완료 8경기 기준 |
| 판정 불일치 | 점수/누적 보상 승자가 다른 196경기 | 학습 목적·평가를 최종 점수에 맞춰야 함 |
| v12 취약 상대 | weak 50%, default 70%, special 80% | 전체 평균만으로 개선점을 정하지 않음 |
| 상위 모델 상대 | v4 80%, v5 70%, v6 80%, v7 한 모델 70% | 기존 강한 상대와의 차이도 추적 |

weak 상대는 [10경기 중 5승 5패](../../../logs/shared/agent_league/v12_full_league_new_game_20261101/pairs/scripted-weak__v12-before-learned.md)이며, 진영 교대 5쌍 모두 1승 1패였다. 이름이 weak라는 이유로 쉬운 상대라고 보지 않는다. 쌍당 표본은 작으므로 이동 실패·배정 실패 중 무엇이 원인인지 이 성적표만으로 단정하지 않는다.

기존 2026-10-28 리그의 v12는 누적 보상 기준 179승 31패, 잠정 6위로 기록돼 있다. 새 리그는 게임 빌드·API·승패 기준이 다르다. 6위에서 1위로 바뀐 것을 추가 학습이나 특정 네트워크의 효과로 해석할 수 없다. [v12 안내](../../v12/README.md).

## 2. v12가 입증한 범위

실제 [TaskScorer](../../../code/v12/blackout_v12/model.py)는 `64→64→32→1`의 6,273개 파라미터 MLP다. [정책](../../../code/v12/blackout_v12/policy.py)은 event마다 작업을 배정하고, [navigation](../../../code/v12/blackout_v12/navigation.py)은 Dijkstra 거리장과 waypoint, 규칙 기반 evade로 실제 방향을 만든다. [부분 ONNX](../../../code/v12/blackout_v12/onnx_core.py)는 이 Python 호스트 기능까지 포함하지 않는다.

따라서 성적이 뒷받침하는 결론은 **계산된 비용으로 작업을 배정하고 안정적인 제어로 수행하는 정책 전체가 강하다**는 것이다. 다음은 아직 분리 검증되지 않았다.

- 공동 배정 DP와 좋은 후보 생성 중 어느 쪽의 기여가 큰가.
- learned scorer와 고정 prior·규칙 회피 중 어느 쪽의 기여가 큰가.
- 이동 신경망이 기존 경로 제어보다 좋은가.
- 시간 기억이나 GRU를 추가하면 승률이 개선되는가.

사용자의 ‘배정망과 이동망을 분리하자’는 제안은 이 강한 구조를 확장하는 타당한 연구 가설이다. 다만 두 actor를 처음부터 함께 바꿔 학습하면 손실 원인을 알기 어려우므로, v13은 Worker 학습 → Manager 적응 → 교대 학습과 대조군으로 분리한다.

## 3. 엔지니어 발언의 가능한 의미

> 저희모델 간격을 2배씩 늘려서 4 8 16 32 64프레임 전껄 보는데 그냥 RNN용 state에 집어넣는다

문장만으로 구현을 확정할 수 없다. 특히 ‘state’는 신경망 hidden state와 실행 서버가 보관하는 tensor를 모두 뜻할 수 있다.

| 해석 | 구현 예 | 판단 |
|---|---|---|
| 과거 관측의 지수 간격 stacking | 상태 버퍼에서 `o[t−4], o[t−8], …, o[t−64]`를 꺼내 현재와 concat | **우선 해석.** ‘RNN용 state에 그냥 넣는다’와 최신 서버 계약에 잘 맞음 |
| 과거 feature/embedding 저장 | `z=Encoder(o)`를 보관하고 동일 lag를 읽음 | 가능. 원본 이미지보다 작지만 encoder 변경 시 stale 표현에 주의 |
| 과거 hidden state를 입력으로 추가 | 현재 관측과 `h[t−4], …`를 RNN에 제공 | 가능하나 hidden 구조·갱신법을 알아야 함 |
| DilatedRNN | 층별로 `h_l(t)=cell_l(x_l(t),h_l(t−d_l))`의 skip recurrence | 관련 개념이지만 이 발언만으로 단정 불가 |
| 지수 간격으로 행동을 실행 | 4,8,16… step마다 행동/관측 갱신 | 발언의 ‘전껄 본다’와 다름. v13에는 채택하지 않음 |

핵심은 **가까운 과거를 상대적으로 촘촘히, 먼 과거를 성기게 읽어 여러 시간 범위의 변화를 함께 보는 것**이다. 예를 들어 현재 위치가 같아도 4 step 전과 64 step 전 위치를 함께 보면 ‘막 도착’, ‘장시간 정체’, ‘왔다 갔다 함’을 구별할 단서가 생긴다. 다만 위치 snapshot 몇 개만으로 원인이나 숨은 상태를 완전히 복원할 수는 없다.

현재 서버는 [onnx_policy.py](../../../../blackout-env/blackout_env/competition/onnx_policy.py)에서 `state_in<suffix>`와 같은 shape의 `state_out<suffix>`를 다음 호출에 전달한다. GRU 연산을 포함했는지 검사하는 계약이 아니다. 관측 버퍼를 이 포트로 보존하는 해석은 구현상 가능하다. [프로젝트에 정리한 규격](../../common/environment_update_2026-11-01.md)도 참고한다.

반면 내장 frame_stack은 K≤16인 연속 프레임이며 지수 lag를 지정하지 못한다. 다섯 과거 시점을 매 호출 정확히 읽으려면 그 사이 값도 보관하거나 별도 샘플링 의미를 정해야 한다. v13은 해석이 명확한 **64칸 raw-summary history + 고정 lag read**를 선택한다. 세부 shape·업데이트 순서·timeout 의미는 [설계 §5–6](../plans/v13_design.md)에 둔다.

## 4. 논문과 v13 제안을 구분한다

- [FeUdal Networks, ICML 2017](https://proceedings.mlr.press/v70/vezhnevets17a.html): 느린 Manager와 빠른 Worker를 분리한 계층 강화학습의 참고다. BlackOut 작업 후보·배정 DP·32방향 제어를 그대로 제공하는 논문은 아니다.
- [Dilated Recurrent Neural Networks, NeurIPS 2017](https://arxiv.org/abs/1710.02224): 여러 간격의 recurrent skip connection을 사용한다. 과거 관측을 concat하는 것과 구조적으로 다르므로 엔지니어 발언의 명칭으로 단정하지 않는다.
- [Recurrent Experience Replay in Distributed Reinforcement Learning, ICLR 2019](https://willdabney.com/publication/r2d2/): recurrent state staleness와 parameter lag를 다룬다. v13의 sequence 재생·burn-in 검토에 참고하며, 논문의 off-policy 알고리즘을 PPO로 이름만 바꿔 쓰지 않는다.

위 문헌은 일반적인 설계 원리를 뒷받침한다. lag `{4,8,16,32,64}`, token 128차원, GRU 64차원, 25 step 배정 주기, 32방향은 **이 프로젝트의 검증 전 제안값**이다. 논문에서 BlackOut 최적값으로 입증된 수치가 아니다.
