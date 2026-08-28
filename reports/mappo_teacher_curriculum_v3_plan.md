# MAPPO teacher curriculum v3 구현 및 계획 변경

## 목적

`checkpoints/win_70_vs_scripted.pt`가 사용하는 planner 전략을 neural actor에
증류한 뒤 MAPPO로 미세 조정하고, 최종적으로 해당 frozen 상대를 dev seed
양 진영 10경기에서 9승 이상(승률 85% 조건을 만족하는 최소 정수 승수) 이기는
것이 목표다.

v2 장기 실행은 2,000,896 environment step과 1,643개 terminal episode에서
전패했다. 핵심 원인은 상대 체크포인트가 추론 시 `planner_override`를 사용하지만
MAPPO learner는 체크포인트의 neural actor만 초기화해 사용했다는 점이다. 따라서
v3는 실패한 v2 latest에서 재개하지 않고 원본 `win_70_vs_scripted.pt`의 neural
core에서 새로 시작한다.

## 적용한 해결책

1. **On-policy DAgger label 수집**: learner가 실제로 방문한 모든 상태에서
   `3 worker + 2 guard`, chase radius 48 planner의 5개 action label을 수집한다.
2. **Teacher-forcing warm-up**: 첫 100,000 step은 PPO를 끄고 teacher action으로
   환경을 진행한다. forcing 확률은 1.0에서 0.0으로 줄이고 BC만 수행한다.
3. **지속적인 BC 보조 학습**: 이후 learner action으로 rollout하면서도 teacher
   label은 계속 모은다. MAPPO update 뒤 압축 replay에서 BC update를 수행하고
   stage별 coefficient를 점차 0으로 낮춘다.
4. **상대 curriculum**: base scripted부터 시작해 weak planner, full win70 비중
   25%, 50%, 100% 순으로 난도를 높인다.
5. **Historical mixture**: 마지막 단계에서는 full win70 80%와 직전 learner의
   frozen snapshot 20%를 episode 단위로 섞는다. 한 episode 도중 상대는 바뀌지
   않는다.
6. **양 진영 평가와 fail-closed 산출물**: 매 평가에서 동일 dev seed를 A/B로
   side swap한다. 최종 target 파일은 전체 committed dev 평가에서만 저장한다.

## Curriculum 계약

| 단계 | 최소/최대 step | 학습 상대 | 승격 gate | BC 계수 |
| --- | ---: | --- | --- | ---: |
| `dagger_warmup` | 100k / 100k | base scripted | warm-up 종료 | 1.00→0.70 |
| `base_scripted` | 50k / 250k | base scripted 100% | 60%, 양 진영 승리 | 0.70→0.30 |
| `weak_win70` | 100k / 300k | weak win70 100% | 60%, 양 진영 승리 | 0.50→0.20 |
| `win70_mix25` | 100k / 300k | weak 75% + full 25% | full 상대 30%, 양 진영 승리 | 0.30→0.15 |
| `win70_mix50` | 150k / 400k | weak 50% + full 50% | full 상대 50%, 양 진영 승리 | 0.20→0.08 |
| `full_win70` | 200k / 1M | full 100% | full 상대 70%, 양 진영 승리 | 0.10→0.02 |
| `robust_historical_mix` | 0 / 2M | full 80% + history 20% | 최종 목표 85% | 0.05→0.00 |

중간 단계가 gate를 통과하지 못해도 최대 stage budget에 도달하면 다음 난도를
경험하도록 전환한다. 이 경우 `training.jsonl`에
`reason=maximum_stage_budget`을 남겨 정상 승격과 구분한다. 마지막 단계에서는
강제 승격이 없으며 최종 85% 조건만 사용한다.

## 구현과 산출물

- 학습기: `scripts/train_mappo_teacher_curriculum_v3.py`
- 실행 wrapper: `scripts/train_mappo_teacher_curriculum_v3.sh`
- curriculum/episode mixture: `blackout_rl/mappo_curriculum.py`
- weak 상대 설정: `configs/policies/weak_win70_v1.json`
- 최신 재개 checkpoint: `checkpoints/mappo_teacher_curriculum_v3_latest.pt`
- dev 최고 checkpoint: `checkpoints/mappo_teacher_curriculum_v3_best.pt`
- 최종 목표 checkpoint: `checkpoints/mappo_win_85_vs_win70.pt`
- historical snapshot: `checkpoints/mappo_v3_snapshots/`
- update log/평가/요약: `logs/mappo_teacher_curriculum_v3/`

v3 checkpoint에는 decentralized actor 외에도 centralized critic, optimizer,
압축 teacher replay, curriculum 단계, 양 진영 seed cursor, opponent-mixture RNG와
누적 episode 통계를 저장한다. resume 시 상대 및 초기 체크포인트 SHA와 학습을
정의하는 주요 설정이 일치하지 않으면 중단한다.

## 실행

```bash
./scripts/train_mappo_teacher_curriculum_v3.sh
```

중단 후 같은 설정으로 재개:

```bash
./scripts/train_mappo_teacher_curriculum_v3.sh --resume-latest
```

의도적으로 다른 hyperparameter를 쓰는 실험은 기존 v3 경로에 이어 쓰지 말고
`--latest-checkpoint`, `--best-checkpoint`, `--target-checkpoint`, `--snapshot-dir`,
`--log-dir`를 모두 별도 경로로 지정한다.

## 검증 결과

256-step Unity 축소 실험에서 다음 경로를 확인했다.

- DAgger warm-up: 128 step, teacher label 640개, teacher forcing 128회,
  replay BC accuracy 0.531
- 단계 전환: `dagger_warmup → base_scripted`
- MAPPO+BC: 두 번째 128 step에서 PPO 4 epoch/8 minibatch 수행,
  approximate KL 0.00525, clip fraction 0.0718, BC accuracy 0.234
- 양 진영 target/stage 평가, latest checkpoint와 run summary 저장 완료

축소 실험의 승률은 성능 판단용이 아니며 학습·전환·평가·저장 경로 검증만을
목적으로 한다.
