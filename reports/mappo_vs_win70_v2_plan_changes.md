# MAPPO vs win70 v2: 실패 원인 수정 및 계획 변경

## 결론

기존 v1 장기 학습은 2,000,384 step 동안 3,907개의 rollout을 수행했지만
완료된 학습 에피소드가 0개였다. 512-step마다 Unity 환경과 collector를
폐기했기 때문에 실제로 770~1,133 step 이상 걸리는 경기가 끝나기 전에
상태가 초기화됐고, terminal 승패 보상이 한 번도 학습에 포함되지 않았다.

v1 체크포인트와 로그는 증거 보존을 위해 그대로 유지한다. 수정된 학습은
서로 섞이지 않는 v2 경로와 체크포인트 schema를 사용한다.

## 기존 계획 대비 변경점

| 항목 | 기존 v1 | 수정된 v2 | 이유 |
| --- | --- | --- | --- |
| 환경 수명 | PPO update마다 생성·종료 | 양 진영별 환경과 collector를 update 사이에 유지 | rollout 경계에서 에피소드 상태 보존 |
| 진영 균형 | update마다 진영 교대, 매번 새 환경 | update마다 진영 교대, 진영별 persistent collector | 양쪽 노출 균형과 에피소드 연속성 동시 확보 |
| rollout | 512 step | 2,048 step | 실제 경기 종료와 terminal reward를 update마다 충분히 관측 |
| 초기 actor | `phase3_selfplay_gen_08.pt` | `win_70_vs_scripted.pt`의 neural core | 기존 초기 actor가 상대에게 0/10, 평균 점수차 -94.9로 지나치게 약했음 |
| planner 처리 | 해당 없음 | 초기 checkpoint의 planner override는 actor에 복사하지 않음 | 제출 가능한 decentralized neural actor만 학습해야 함 |
| 학습률 | `3e-4` | `5e-5` | pretrained neural core를 급격히 훼손하지 않는 fine-tuning 설정 |
| entropy 계수 | `0.01` | `0.001` | 초기 imitation 정책을 과도한 탐색으로 빠르게 무너뜨리지 않음 |
| Unity shaping | `0.05` | `0.25` | win70 neural core 생성에 사용된 검증된 reward scale과 정렬 |
| seed | rollout마다 seed를 주고 즉시 폐기 | 에피소드 종료마다 3001~3008 cyclic seed 적용 | 실제 완료 에피소드 단위 다양성과 재현성 확보 |
| 재개 상태 | 모델·critic·optimizer만 저장 | seed cursor와 진영별 episode/result counter도 저장 | 재개 시 학습 분포와 진단 누적값 보존 |
| 안전장치 | 없음 | 8,192 step까지 terminal episode가 0이면 exit 3 | 잘못된 수집으로 장시간 자원을 낭비하지 않음 |
| 로그/체크포인트 | `mappo_vs_win70` | `mappo_vs_win70_v2` | 실패한 v1 산출물과 새 실행 혼합 방지 |

목표와 평가 계약은 바뀌지 않았다. 고정된 dev seed 3101~3105를 양 진영으로
교환한 10경기에서 최소 9승해야 승률 85% 이상 조건을 충족한 것으로 보고
`checkpoints/mappo_win_85_vs_win70.pt`를 저장한다.

## 새 산출물

| 경로 | 의미 |
| --- | --- |
| `checkpoints/mappo_vs_win70_v2_latest.pt` | v2 재개용 최신 체크포인트 |
| `checkpoints/mappo_vs_win70_v2_best.pt` | dev 승률, 이후 점수차 기준 v2 최상 모델 |
| `checkpoints/mappo_win_85_vs_win70.pt` | 9/10 이상 통과 시에만 생성되는 목표 모델 |
| `logs/mappo_vs_win70_v2/training.jsonl` | rollout별 완료 에피소드·승패·seed·PPO 지표 |
| `logs/mappo_vs_win70_v2/run_summary.json` | 평가 결과, collector 상태, 중단 원인 |
| `logs/mappo_vs_win70_v2/eval_step_*.json` | side-swapped 평가 원자료 |

v1의 `checkpoints/mappo_vs_win70_latest.pt`는 terminal episode를 한 번도
학습하지 못한 상태이므로 v2에서 재개할 수 없도록 schema를 분리했다.

## 검증 결과

- 관련 단위 테스트 11개 통과.
- 실제 Unity 4,096-step smoke에서 양 진영 합계 완료/terminal 에피소드 3개 기록.
- 같은 체크포인트를 재개해 6,144 step, 누적 terminal 에피소드 4개로 증가.
- smoke의 PPO KL은 `0.0067~0.0089`로 target KL `0.03` 이내였고 NaN은 없었다.
- 한 개 dev seed만 사용한 smoke 평가는 모델 승격 자격이 없으며 성능 주장에
  사용하지 않는다.

## 실행

새 학습은 다음 명령으로 시작한다.

```bash
cd /Users/safeailab_macmini/Desktop/2026-IST-tech-RL
./scripts/train_mappo_vs_win70.sh
```

중단 후에는 다음 명령으로 v2 최신 체크포인트를 재개한다.

```bash
./scripts/train_mappo_vs_win70.sh --resume-latest
```

새 실행에서 이미 v2 산출물이 발견되면 스크립트는 덮어쓰거나 로그를
이어 붙이지 않고 실패한다. 이 경우 `--resume-latest`를 사용하거나 명시적으로
새 `--log-dir`, `--latest-checkpoint`, `--best-checkpoint` 경로를 지정한다.
