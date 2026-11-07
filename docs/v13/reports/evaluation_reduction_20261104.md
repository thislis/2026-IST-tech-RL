# v13 평가 축소와 기존 학습 이어가기

2026-11-04 사용자 요청에 따라 중간 평가만 축소했다. 기존 `hierarchical_v13`은 정상 중단했으며, **새 실행을 시작하지 않았다.** 사용자가 아래 명령을 실행하면 재개한다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v13\run_v13.ps1 --name hierarchical_v13_eval_light
```

## 적용한 설정

| 항목 | 기존 | 변경 |
|---|---:|---:|
| 중간 평가 간격 | 1,048,576 step | 2,097,152 step |
| 중간 평가 | 22상대×5seed×2진영 = 220경기 | 22상대×2seed×2진영 = 88경기 |
| 본 학습별 선택 모델 종료 평가 | 220경기 | 220경기 유지 |
| 최종 확인 평가 | 1,720경기 | 유지 |
| 본 학습 | B1·B6, 각 seed11/22/33 | 유지 |
| 학습 예산·optimizer·보상·모델·PPO | 등록 설정 | 유지 |
| 수집 환경·learner | 환경8개·CUDA·sequence batch8 | 실측 설정 재사용 |

중단 지점은 `main_B1_s11`, policy version2304, 누적 **5,242,880 step**이다. 현재 실행 중이던 평가는 중단하고 완료한 경기 결과를 보존했다. 다음 중간 평가 시점은 **6,291,456 step**이다. 첫 본 학습 B1은 이 지점에서 학습 예산도 끝나므로, 88경기 선별 후 선택 모델의 종료 평가로 이어진다.

## 서로 다른 평가 규모의 점수를 섞지 않는 방법

본 학습 중 checkpoint 선택은 항상 동일한 상대22개·첫2개 seed·양 진영의 **공통88경기**로 비교한다. 과거220경기 결과에서도 정확히 이 경기 ID를 추출해 점수를 다시 계산했다. 중단 시점 version2304에도 공통88경기가 모두 완료돼 있어 선별 후보로 포함했다. 220경기 전체 승점률과 88경기 승점률을 직접 비교하지 않는다.

각 본 학습이 끝날 때 마지막 checkpoint도 공통88경기로 평가하고, 그 실행에서 공통88경기 점수가 가장 높은 checkpoint 하나를 총220경기로 평가한다. **해당 모델의 기존88경기를 재사용해 나머지132경기만 실행**한다. 같은 모델의 기존220경기 기록이 있다면 그것도 재사용한다. 이렇게 얻은 **220경기 점수만** 본 학습6개 사이의 최종 선택에 사용한다. 최종 확인1,720경기의 상대·seed·진영 구성은 바꾸지 않았다.

평가 횟수가 줄어 학습 중 최고점 checkpoint를 놓칠 가능성과 선별 점수의 표본 오차는 커진다. 원래 모델을 그대로 학습하는 양은 줄이지 않았다. 실제 종료 시간은 재개 시점·처리량·경기 길이에 따라 달라진다.

## 원본 보존과 새 실행의 출처

기존 `logs/v13/hierarchical_v13/`의 request·manifest·결과와 `artifacts/checkpoints/v13/hierarchical_v13/`의 원본 체크포인트는 변경하지 않았다. 새 설정을 이전 hash에 끼워 넣지 않고 **별도 이름의 continuation**으로 등록했다.

- 새 로그: `logs/v13/hierarchical_v13_eval_light/`
- 새 체크포인트: `artifacts/checkpoints/v13/hierarchical_v13_eval_light/`
- 이관 내역: 새 로그의 `continuation.json`
- 실제 보존 검사: 새 로그의 `continuation_validation.json`
- teacher: 원본 `artifacts/data/v13/hierarchical_v13/teacher/`를 체크섬 검증 후 읽기 전용 재사용
- 기존 자기대결 snapshot도 원본 경로와 해시를 유지하므로 원본 폴더를 삭제하지 않는다.

완료한 pilot6개의 결과·학습 상태, 진행 중인 본 학습의 가중치·critic·optimizer·RNG·단계·누적 step을 복사했다. 본 학습의 `best_score`는 공통88경기 기준으로 다시 계산하고 `next_eval`만 새 간격에 맞췄다. 완료한 pilot의 220경기 점수와 B1/B6 선정은 유지한다. train/dev/confirmation map 분리 DB도 복사했다. Unity 중간 상태는 이어붙이지 않고 새 episode부터 수집한다.

## 조회·중단·검사

```powershell
# 상태 조회
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v13\run_v13.ps1 --name hierarchical_v13_eval_light --status

# 게임을 시작하지 않는 재개 준비 검사
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v13\run_v13.ps1 --name hierarchical_v13_eval_light --resume-check

# 사용자가 재개한 뒤 정상 중단하려면
powershell -NoProfile -ExecutionPolicy Bypass -File .\code\v13\run_v13.ps1 --name hierarchical_v13_eval_light --stop
```

재개 시 완료한 teacher·pilot·하드웨어 측정을 다시 실행하지 않는다. 기존 이름만 지정하거나 이름을 생략하면 원본 실험을 가리키므로 **새 이름을 반드시 포함한다.** 준비 스크립트 `code/v13/scripts/prepare_reduced_evaluation.py`는 이미 실행했으며 다시 실행할 필요가 없다. 대상 폴더가 존재하면 덮어쓰기를 거부한다.

## 검증

자동 검사 **29개 통과**. 추가 검사는 평가 설정만 변경되는지, 가중치/optimizer/RNG 보존, 평가 외 소스 변경 거부, 잘못된 진영 캐시 거부, 88→220 확장 시132경기만 추가하는지, 선별 우승 checkpoint의 종료 평가, 원본을 수정하거나 게임을 시작하지 않는 이관을 포함한다.

실제 체크포인트7개의 학습 tensor·optimizer·RNG·진행량이 원본과 동일하고 새 identity로 로드됨을 확인했다. 원본 manifest·request·선택 checkpoint·teacher metadata 등 기록한 체크섬도 유지됐다. 새 실행의 `--resume-check`에서 `resume_identity_matches: true`, `game_started: false`를 확인했고 상태는 `prepared`다. 재개·Unity·실제 학습은 실행하지 않았다.
