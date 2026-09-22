# 2026-09-17 파일럿 상태와 다음 명령

**후속 업데이트:** v7-2 오류 수정과 재개를 진행했다. [teammate-v2 복구 안내](teammate_v2_recovery.md)를 참고한다. 아래 표와 오류 설명은 수정 전 확인 기록이다.

두 학습 큐의 lifetime lock은 해제되어 있다. 현재 파일럿 학습이 계속 실행 중인 상태가 아니다.

| 경로 | 결과 | 체크포인트 확인 |
|---|---|---|
| v7-1 A3/A0/A2 × 3 seed | 9/9 완료, 각 200,000 env-step, 합계 1,800,000 | 9개 모두 200,000스텝 및 현재 코드/설정 해시 일치 |
| v7-2 F0 seed 11 | 22,000 env-step 완료 | 최종 체크포인트 존재 |
| v7-2 F1 seed 11 | 34,123 env-step에서 실패 | 마지막 저장 32,768스텝 |
| v7-2 F1 seed 22/33 | 시작하지 않음 | 없음 |

v7-2 중단 원인은 전뇌 출력층 자체가 아니라 나머지 아군을 움직이는 scripted teammate의 경로 탐색 예외다.

```text
Collector.collect → self.teammate.act → ScriptedTeamController._assign_delivery
blackout_rl.navigation.PathNotFound: no reachable target from GridCell(x=1, y=20)
```

이 확인 작업에서는 학습 코드와 체크포인트를 변경하거나 실패한 큐를 재시작하지 않았다. 오류를 수정하고 변경된 teammate 동작·재개 provenance를 명시한 뒤 v7-2를 재개해야 한다. 단순 재실행으로 해결됐다고 간주할 수 없다.

## 지금 실행할 한 줄

완료된 v7-1에 대한 다음 단계는 dev 평가다. 다음 한 줄로 9개 모델을 고정 target 상대에 대해 dev 30맵 × 양 진영, 총 540경기 평가한다.

```bash
bash "/Users/safeailab_macmini/Desktop/2026-IST-tech-RL/scripts/start_pilot_evaluation.sh"
```

- 어느 디렉터리에서 실행해도 프로젝트 루트와 `.venv`를 자동으로 사용한다.
- 먼저 9개 완료 checkpoint, 등록 예산, 코드·설정·데이터 해시, 모델 로딩을 검증한다.
- 검사 후 백그라운드 프로세스를 시작하고 PID를 출력하므로 터미널을 계속 열어둘 필요가 없다.
- Unity는 한 번에 하나씩 실행하며 중복 평가 프로세스는 파일 잠금으로 차단한다.
- 기존 평가 코드의 checkpoint provenance 검사를 그대로 사용한다. 새 도구는 실행 순서만 관리한다.
- 같은 명령을 다시 실행하면 동일 checkpoint에 대한 완성된 60경기 결과는 건너뛴다. 중단된 모델의 부분 평가는 성공으로 처리하지 않고 해당 모델부터 다시 평가한다.
- 실패하면 큐를 멈추고 `status.json`과 `console.log`에 기록한다.
- 이 명령은 v7-2 재개, 대규모 본학습, confirmation/final test, 제출 export를 실행하지 않는다.

상태와 결과:

```text
logs/v7/pilot_v7_1_dev_evaluation/status.json
logs/v7/pilot_v7_1_dev_evaluation/console.log
logs/v7/pilot_v7_1_dev_evaluation/summary.json
```

상태만 보고 싶으면 같은 명령 끝에 `--status`, Unity 실행 없이 준비 상태만 검사하려면 `--check`를 붙인다. 중단할 때는 `kill -TERM "$(cat logs/v7/pilot_v7_1_dev_evaluation/evaluation.pid)"`를 사용한다. 실행 관리기가 evaluator에 SIGINT를 전달해 Unity 정리를 수행한다.

검증 완료: 셸 구문 검사, 실제 9개 checkpoint의 read-only `--check`, 완료 결과 재사용/중복 map 거절/실패 시 큐 정지에 관한 단위 검사 3개 통과. 사용자에게 실행 명령을 안내하는 작업이므로 새 장기 평가는 아직 시작하지 않았다.
