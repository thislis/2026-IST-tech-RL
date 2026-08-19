# AGENT-06/08/11 Representation Ablation

## 결론

고정된 동일 학습 budget과 Phase 3 dev seed에서 unit attention, global/local map encoder,
auxiliary loss 네 종류를 비교했다. 실험은 완료했지만 어느 후보도 실제 승률이나 평균 점수차를
개선하지 못했으므로 기본 learned policy로 승격하지 않는다.

- AGENT-06: `self_attention_v1` 구현과 held-out 비교 완료, 승격하지 않음
- AGENT-08: `global_local_v1` 구현과 held-out 비교 완료, 승격하지 않음
- AGENT-11: 네 auxiliary target을 각각 독립 평가, 선택 결과는 빈 집합

전체 결과는 [`phase3_representation_ablation.json`](../logs/phase3_representation_ablation.json),
arm별 학습 및 episode 증거는 `logs/phase3_repr_*.json`과
`logs/phase3_repr_*_dev.json`에 저장했다.

## 공통 실험 계약

| 항목 | 값 |
| --- | --- |
| teacher replay | `checkpoints/ippo_counter_teacher_ordered.pt` |
| replay row | 230,000 |
| model initialization seed | 36001, arm마다 동일하게 재설정 |
| gradient step | arm당 1,500 |
| minibatch | 256 |
| learning rate | 1e-4 |
| action class balancing | inverse frequency power 0.5 |
| auxiliary coefficient | 0.1 |
| dev seed | 3101~3105 |
| 평가 | seed마다 물리 진영 A/B 교대, arm당 10경기 |
| 상대 | frozen `scripted-battery-v1` |

test split `3201~3210`은 모델 선택에 사용하지 않았다. 승자는 shaping reward가 아니라 Unity
terminal `winner`로 판정했다. 신뢰구간은 개별 경기가 아니라 완전한 side-swapped seed pair를
단위로 bootstrap했다.

## AGENT-06: unit entity attention

기존 ten-entity flatten latent에 acting unit을 query로 하는 4-head attention latent를 추가했다.
attention 설정은 checkpoint `model_config.entity_encoder_version`으로 명시하며, 기존 checkpoint는
기본 `flatten_v1`로 그대로 로드된다.

| arm | replay action accuracy | W-D-L | 승률 | 평균 점수차 | score 95% paired CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| flatten + legacy | 89.84% | 0-0-10 | 0% | -90.3 | [-93.6, -85.9] |
| attention + legacy | 90.25% | 0-0-10 | 0% | -93.4 | [-95.3, -91.8] |

attention은 replay 정확도를 0.41%p 높였지만 실제 평균 점수차는 3.1점 악화됐다. offline action
accuracy 개선이 closed-loop 개선으로 이어지지 않았으므로 유지하지 않는다.

## AGENT-08: global map + local crop

`global_local_v1`은 저해상도 전체 map branch와 acting unit 중심의 32×32 고해상도 crop branch를
결합한다. unit 위치는 absolute entity block과 canonical team-local slot에서 계산하며,
`affine_grid`/`grid_sample`로 batch crop을 생성한다.

| arm | replay action accuracy | W-D-L | 승률 | 평균 점수차 | score 95% paired CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy | 89.84% | 0-0-10 | 0% | -90.3 | [-93.6, -85.9] |
| global/local | 91.04% | 0-0-10 | 0% | -95.4 | [-97.6, -92.4] |

global/local도 replay 정확도는 1.20%p 높았지만 실제 평균 점수차는 5.1점 악화됐다. 따라서 이
동일-budget 실험에서는 legacy encoder를 선택한다. 기존 연구 checkpoint에 포함된
global/local encoder를 삭제하거나 과거 artifact를 변경하지는 않는다.

## AGENT-11: auxiliary loss 선택

AGENT-10에서 정의한 네 target을 global/local baseline에 하나씩 추가했다. 분류 target은 cross
entropy, 연속 target은 Smooth L1을 사용했다. 승률이 증가하고 평균 점수차가 회귀하지 않는
target만 선택하도록 사전에 정한 gate를 그대로 적용했다.

| target | W-D-L | 평균 점수차 | baseline 대비 승률 | baseline 대비 점수차 | 선택 |
| --- | ---: | ---: | ---: | ---: | --- |
| 없음 | 0-0-10 | -95.4 | — | — | 기준 |
| role | 0-0-10 | -95.5 | 0%p | -0.1 | 제외 |
| holding item | 0-0-10 | -96.3 | 0%p | -0.9 | 제외 |
| seconds to absorption | 0-0-10 | -95.6 | 0%p | -0.2 | 제외 |
| score delta | 0-0-10 | -96.4 | 0%p | -1.0 | 제외 |

`agent11_selected_targets=[]`이며 기본 actor 학습에는 auxiliary loss를 추가하지 않는다.

## 해석과 제한

모든 learned arm이 0승이므로 표현 간 차이는 승률이 아닌 점수차에서만 나타났다. 이 결과는
후보가 절대적으로 쓸모없다는 결론이 아니라, 현재 강한 교사 replay와 1,500-step 동일 budget에서
승격 근거가 없다는 결론이다. 특히 replay accuracy와 실제 점수차가 반대로 움직였으므로 이후
표현 후보도 반드시 closed-loop dev 평가를 거쳐야 한다.

현재 `win_70_vs_scripted.pt`의 100% 결과는 별도 문서에 기록된 planner override 하이브리드이며,
이번 실험은 해당 guardrail을 사용하지 않은 순수 learned actor 비교다.
