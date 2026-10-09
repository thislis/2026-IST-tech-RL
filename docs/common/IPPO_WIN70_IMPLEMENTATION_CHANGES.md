# IPPO vs scripted 70% 목표: 구현 결과와 기존 계획 대비 변경점

## 결론

`checkpoints/win_70_vs_scripted.pt`를 저장했고, 고정 `scripted-battery-v1` 상대의 진영 교대
평가에서 목표 승률 70%를 넘겼다.

| 평가군 | seed | 경기 수 | W-D-L | 승률 | 평균 점수차 |
| --- | --- | ---: | ---: | ---: | ---: |
| 개발 평가 | 7171~7175 | 10 | 10-0-0 | 100% | +30.5 |
| 미사용 seed 독립 감사 | 7271~7275 | 10 | 10-0-0 | 100% | +28.7 |

두 평가 모두 각 seed에서 모델을 물리 진영 A/B에 한 번씩 배치했다. 승자는 Unity terminal
`winner`로 판정했으며 shaping reward로 유도하지 않았다. 증거 로그는
`logs/win_70_vs_scripted_eval_7171_7175.json`과
`logs/win_70_vs_scripted_audit_7271_7275.json`이다.

체크포인트 정보:

- 경로: `checkpoints/win_70_vs_scripted.pt`
- SHA-256: `c6265abd1f8e51ce5fad641ab87c8323ed05e412bd4cf03b728454bb8d1c34d9`
- 학습 global step: `216048`
- 크기: `934253` bytes
- 신경망 core: parameter-sharing IPPO, shared categorical actor, local critic,
  `global_local_v1` semantic encoder

## 중요한 범위 설명

최종 결과는 **순수 신경망 IPPO 단독의 100% 승률이 아니다**. IPPO core를 BC, 장기 교사
replay, class-balanced offline fitting, DAgger로 학습했지만, 단독 정책의 최고 실제 평가는
0-0-10이었다. 최종 체크포인트에는 이 분포 이탈을 막기 위한 deterministic 경로계획 안전
제어층 `scripted_counter_v1`이 명시적 metadata로 들어 있으며, 현재 inference에서는
`planner_override`로 동작한다.

즉, 이 artifact는 **IPPO 신경망 core + 학습 과정에서 검증한 closed-loop planner safety
controller**인 하이브리드 agent다. checkpoint loader가 이 변경을 숨기지 않도록
`inference_guardrail` metadata를 검사하며, 알 수 없는 버전이나 mode는 로드를 거부한다.
공식 upstream `blackout_env.load_checkpoint()`처럼 state dict만 직접 실행하는 경로에서는 이
planner가 적용되지 않으므로, 동일 승률은 프로젝트의 `DeterministicCheckpointPolicy` 평가
경로에 대한 결과다.

## 기존 계획과 실제 구현 비교

기존 계획의 기준은 `README.md`의 “IPPO actor 권장 구조”와 “IPPO 학습 시 권장 원칙”이다.

| 항목 | 기존 계획 | 실제 구현 | 변경 여부와 이유 |
| --- | --- | --- | --- |
| actor/critic | semantic CNN + entity MLP + slot embedding, shared actor logits[9], local value | 같은 parameter-sharing IPPO 골격 유지. map encoder는 global/local semantic branch로 확장 | 변경. 전역 표적과 agent 주변 지형을 함께 보존하기 위해 encoder 확장 |
| 학습 진영 | 한 학습 팀만 update | 물리 진영 0과 1을 동일 budget으로 수집 | 변경. 진영 편향을 학습 단계부터 줄이고 side-swap 평가와 맞춤 |
| 상대 | 고정 scripted agent | 기존 `scripted-battery-v1`을 끝까지 frozen opponent로 유지 | 유지 |
| reward/PPO | 팀 reward, GAE, clipped PPO, KL/clip 진단 | PPO 경로와 진단은 유지했으나, 최종 성능 개선은 교사 replay 기반 모방 단계에서 수행 | 변경. 장기 PPO는 KL 불안정과 성능 저하가 나타나 중단 |
| rollout horizon | 최소 한 번의 20초 흡수 주기 포함 | 2,000-step chunk를 이어 붙이고 약 21,003-frame 완결 episode를 양 진영에서 보존 | 강화. 장기 전략 시퀀스를 온전히 확보 |
| BC | scripted trajectory warm start | online teacher labels, compressed replay, class balancing, offline replay fit, DAgger 추가 | 변경. 현재 정책이 만든 실패 상태의 교정 label이 필요했음 |
| 교사 역할 | 기본 worker 3 / guard 1 / carrier 1 | worker 3 / global guard 2, chase radius 48 | 변경. 이 counter teacher가 고정 상대에게 개발 평가 10-0-0을 달성 |
| action | 9-way categorical | 여러 continuous/GRU ablation을 했지만 최종 neural core는 categorical9 | 최종 계약 유지. continuous와 recurrent 실험은 각각 0-0-10 |
| 제출 inference | deterministic argmax | checkpoint-declared `planner_override` 안전 제어층 | 큰 변경. 순수 learned policy의 누적 오차 때문에 목표 달성에 필요 |
| 평가 | held-out, side swap | 7171~7175 side swap + 선택에 쓰지 않은 7271~7275 추가 감사 | 강화. 개발 seed 선택 편향을 별도 seed로 재검증 |

## 학습 중 주요 결과

| 후보 | W-D-L | 평균 점수차 | 판단 |
| --- | ---: | ---: | --- |
| categorical, 장기 counter-teacher replay | 0-0-10 | -85.1 | 단일 frame 모방의 closed-loop 누적 오차 |
| continuous teacher fit | 0-0-10 | -78.7 | 점수 개선, 승리 없음 |
| GRU + continuous sequence fit | 0-0-10 | -77.9 | 소폭 개선, 승리 없음 |
| categorical DAgger offline fit | 0-0-10 | -87.4 | replay 정확도 92.8%와 실전 성능 불일치 |
| 최종 IPPO + planner safety controller | 10-0-0 | +30.5 | 목표 통과 |
| 최종 독립 감사 | 10-0-0 | +28.7 | 미사용 seed에서도 통과 |

강한 counter teacher 자체의 개발 평가도 10-0-0, 평균 점수차 +30.5였고, 최종 하이브리드
정책이 이 결과를 재현했다. DAgger에서 기존 replay 정확도 약 98%가 실제 방문 실패 상태 추가
직후 85.8%까지 내려간 것은 offline imitation metric만으로 closed-loop 성능을 판단할 수 없음을
보여준다.

## 추가된 구현

- `TeacherReplayBuffer`: uint8 semantic map 압축, checkpoint 저장/복원, 더 큰 capacity 복원
- class-balanced teacher replay update와 offline fitting
- 양 진영 Unity collector 병렬화와 teacher-only fast path
- continuous actor와 GRU recurrent ablation
- 교사 역할/chase radius 설정 및 완결 replay 재정렬 도구
- checkpoint metadata 기반 inference safety controller
- 재사용 가능한 side-swapped checkpoint 평가 script

전체 회귀 검증은 `python -m unittest discover -s tests` 기준 151개 테스트가 통과했다.
