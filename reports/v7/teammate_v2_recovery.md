# v7-2 경로 탐색 오류 수정 및 재개

2026-09-17 사용자 요청에 따라 중단 원인을 수정했다. 수정 실행은 `*_teammate_v2` 실험 ID로 구분하며 원래 실패 실행을 덮어쓰지 않는다.

## 원인과 재현

기존 `Collector`는 한 유닛을 전뇌로 직접 제어하면서도 `FrozenScriptedOpponent`로 아군 5명 모두의 행동을 먼저 계산하고 마지막에 전뇌 슬롯의 행동을 덮어썼다. 따라서 전뇌 유닛에도 실제 동작과 맞지 않는 worker FSM과 경로 탐색이 적용됐다. 아이템을 소지한 상태로 `(1,20)`에 진입하면 이 위치가 자기 진영 carrier 변신 지점이어서 worker용 경로 지도에서 출발점 자체가 막혔다. `nearest_reachable_cell`이 `PathNotFound`를 올리고 전체 학습을 중단했다.

동일한 위치·소지 상태와 DELIVER 단계를 구성해 기존 코드에서 `no reachable target from GridCell(x=1, y=20)`를 재현했다. 기존 실패는 34,123 env-step, 마지막 원자적 체크포인트는 32,768 env-step이다.

## 변경

- `ScriptedTeamController`에 opt-in `active_agents`를 추가했다. 기존 full-team 호출은 원래 동작을 유지한다.
- 새 `ScriptedTeammates`는 전뇌 담당 슬롯을 전이·아이템 할당·전달/경비 계획·이동 계산에서 제외한다. 전뇌 유닛에 scripted fallback을 넣지 않는다.
- scripted 담당 유닛이 이미 역할상 금지된 지점에 서 있으면 그 지점에서 빠져나가는 경로를 허용한다. 실제 벽의 walkability는 바꾸지 않는다.
- 실제로 갈 수 없는 목적지는 해당 scripted 유닛만 정지시키고 다음 관측에서 재시도한다. `assignment_unreachable`를 진단 로그로 남긴다. 다른 종류의 예외는 숨기지 않는다.
- 학습, 평가, 실제 Unity 개입 검증에 동일한 teammate factory를 사용한다.

새 팀 정책 ID는 `fixed_scripted_active_slots_v2`다. 전뇌 동역학·그래프·감각 모드·행동 공간·보상·상대·PPO 설정·총 예산은 그대로다. controlled slot의 예약 목표를 제거하면 다른 scripted 유닛의 할당도 달라질 수 있으므로, 단순히 원본과 동일한 실험이라고 표시하지 않는다. F0도 새 teammate로 다시 실행하는 큐를 준비했다.

## 재개 기록

원본:

```text
logs/v7/v7_2_malecns_readout_ppo_S0_one_unit_fly64_proxy_v1_repeat1_plasticity_off/11/
```

수정본:

```text
logs/v7/v7_2_malecns_readout_ppo_S0_one_unit_fly64_proxy_v1_repeat1_plasticity_off_teammate_v2/11/
```

`recover_connectome_pilot.py`는 등록된 원본 소스 해시, checkpoint schema/config/seed와 허용한 설정 차이만 검사한 뒤 파생 체크포인트를 생성한다. 원본 파일은 변경하지 않는다. 모델·옵티마이저·정책/맵/상대 RNG·저장된 신경 상태가 동일함을 확인했다. 변경 전후 소스/설정/체크포인트 SHA-256과 이유는 새 run의 `recovery.json`, `experiment_manifest.json`, checkpoint의 `lineage`에 남긴다. 이후 저장되는 체크포인트에도 lineage를 유지한다.

새 run에는 체크포인트 시점까지 커밋된 로그만 복사했다. 이후 약 1,355스텝의 저장되지 않은 실행과 원래 실패 로그는 원본 폴더에 그대로 남아 있다. Unity 물리 상태는 저장되지 않으므로 이전 진행 경기는 폐기하고 새 경기로 시작한다. 32,768→128,000 스텝의 나머지 예산을 실행하며 이를 완전히 독립적인 신규 학습 seed로 해석하지 않는다.

복구 큐 순서: F1 seed 11 재개 → 수정 F0 seed 11 → F1 seed 22 → F1 seed 33. 큐는 `configs/v7/pilot_v7_2_recovery_queue.json`에 고정했다. 기존 큐의 실패 기록은 보존한다.

## 검증

- 53개 관련 단위/회귀 검사 통과: `teammate_v2_test_results.txt`.
- 원래 예외 재현 및 제어 슬롯 제외, scripted 유닛의 금지 출발점 탈출, 도달 불가 유닛만 정지, A/B와 복수·전체 전뇌 슬롯, 벽 불변성 검사.
- 복구 설정 allowlist 및 원본 로그 보존 검사.
- 실제 Unity 4조건×256스텝 개입 검증 통과: `unity_causality_teammate_v2.json`. 정상 입력에서 74스텝 실제 이동, 감각 차단과 22스텝 행동 차이, 출력 차단 시 실제 이동 0.
- 기존 v7-1 완료 모델 9개는 등록된 학습 당시 코드 아카이브를 검증·복원하여 `start_pilot_evaluation.sh --check` 통과. 기존 평가 한 줄 명령을 계속 사용할 수 있다. 현재 소스 해시 검사를 무시하거나 원본 체크포인트의 해시를 고쳐서 통과시키지 않는다.

## 실행·확인

수정본 백그라운드 큐를 이 작업에서 시작했다. 이후 중단한 큐를 다시 시작할 때는 한 줄을 사용한다.

```bash
bash "/Users/safeailab_macmini/Desktop/2026-IST-tech-RL/scripts/resume_connectome_pilot.sh"
```

동일 명령에 `--check`를 붙이면 복구·실행 전 검사만 한다. 현재 실행 중이면 중복 시작은 잠금으로 차단된다.

```text
큐 상태: logs/v7/pilot_v7_2_recovery_queue/status.json
큐 로그: logs/v7/pilot_v7_2_recovery_queue/console.log
학습 로그: 위 수정본 run의 console.log, training.jsonl, diagnostics.jsonl
```

정지 시에는 `kill -TERM "$(cat logs/v7/pilot_v7_2_recovery_queue/queue.pid)"`를 사용한다. 현재 rollout 종료 후 체크포인트를 저장한다. 장기 파일럿 완료나 승률 개선을 이 오류 수정 검증으로 주장하지 않는다.
