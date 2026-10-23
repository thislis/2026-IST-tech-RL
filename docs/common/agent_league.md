# 전체 Agent 풀리그

초기 Agent부터 v10까지 구조·학습 설정별 대표 정책을 원본 게임에서 비교한다. 학습이나 가중치 변경은 하지 않는다. 기본은 각 모델 쌍에 대해 **진영 교대 2경기 × 5회**이며, N개 모델이면 `5 × N × (N−1)`경기다.

## 한 줄 실행

```bash
bash /Users/safeailab_macmini/Desktop/2026-IST-tech-RL/code/shared/run_agent_league.sh --name all_agents_v1
```

다른 작업 디렉터리에서도 실행할 수 있다. 가상환경 활성화나 `&`, `nohup`은 필요 없다. PID와 로그 경로를 출력하고 반환한다. 준비 검사를 통과한 뒤 짧은 성능 측정을 하고 전체 리그를 시작한다. 최초 준비에 수분이 걸릴 수 있다.

```bash
# 참가 목록·예정 경기 수만 확인: Unity 실행 없음
bash code/shared/run_agent_league.sh --list

# 모델 로딩, 양 진영 동작, 독립 RNG 검사: Unity 실행 없음
bash code/shared/run_agent_league.sh --name league_check_01 --check

# 자원 측정만: 짧은 Unity probe만 실행, 리그 시작 안 함
bash code/shared/run_agent_league.sh --name league_bench_01 --benchmark

# 자원 측정과 v6-v10 양 진영 최대 2경기: 실제 Unity 구간 총 20분 제한
bash code/shared/run_agent_league.sh --name league_smoke_01 --smoke

# 진행 상태 / 중단 / 완료 경기에서 재개
bash code/shared/run_agent_league.sh --name all_agents_v1 --status
bash code/shared/run_agent_league.sh --name all_agents_v1 --stop
bash code/shared/run_agent_league.sh --name all_agents_v1 --resume

# Unity 없이 저장 결과로 상세 보고서 재생성
bash code/shared/run_agent_league.sh --name all_agents_v1 --report
```

`--prepare`는 참가 목록과 소스만 고정하고 실행하지 않는다. 이후 `--resume`으로 실행한다. 같은 이름의 기존 디렉터리는 덮어쓰지 않는다. `--models ID1,ID2,...`로 목록 일부를 명시하거나 `--rounds 1`로 별도 소규모 리그를 만들 수 있다. 기본 실행은 전체 모델과 5세트다. `--workers 1..8`은 자동 병렬 수 탐색을 해당 값으로 제한한다.

## 모델 선택과 실행 계약

`registry.py`가 `artifacts/checkpoints/`와 v7~v10의 완료 run을 조사한다. 초기 모델의 이름별 학습 방식·모델 설정·추론 guardrail을 구분하고 initial/fit/latest 등의 저장 역할과 self-play 중간 세대를 묶는다. v7은 구조·controller·sensory 설정, v8은 legacy planner와 두 tensor 제출 구조, v9/v10은 arm별로 묶는다. 서로 다른 seed는 별도 참가자로 세지 않는다.

대표는 기존 선택 모델을 우선하고, 그 외에는 완료된 본학습의 가장 작은 seed에서 최종 모델을 사용한다. 본학습이 없으면 가장 큰 완료 학습량의 후보를 사용한다. 선택된 동일 추론 가중치·설정의 복사본은 하나로 합친다. 모든 후보 선택·제외 사유는 request/manifest에 남는다. 등록된 모델이 로딩에 실패하면 전체 시작을 막으며 무작위 정책으로 대체하지 않는다.

과거 모델의 planner·메모리·greedy decoder와 제출형 모델의 sampling을 보존한다. 양 팀은 독립 RNG를 가지며 worker 실행 순서가 행동 RNG를 바꾸지 않는다. 매 경기 새 모델 상태와 Unity 환경을 사용한다. legacy 모델은 공개 agent ID와 상태를 사용할 수 있으므로, 두 tensor만 받는 모델과 동일한 제출 계약을 가졌다고 해석하지 않는다.

승패는 제공 runner와 같은 **팀 누적 raw reward 비교**다. 관측에 나타나는 점수와 공식 대회 점수는 별개다. 환경 seed를 공개 reset API로 요청하고 초기 관측 해시를 기록하지만, 서로 다른 맵이 생성된다는 보장은 하지 않는다. 각 반복에서 동일 모델 쌍의 양 진영은 같은 모델별 action seed와 요청 환경 seed를 사용한다.

## 하드웨어와 재개

실제 로그 기록을 켠 짧은 probe에서 worker 1/2/4/6/8을 비교한다. 가장 빠른 안정 설정을 선택하며 5% 이내의 차이라면 worker가 적은 쪽을 선택한다. CPU 1·2스레드와 MPS의 대표 신경망 추론도 측정한다. 이질적인 planner·전뇌 모델이 섞인 현재 실행기는 CPU를 사용하며, GPU가 더 빠르다고 측정되어도 검증되지 않은 전체 모델 장치 전환을 하지 않는다. CPU 스레드 수는 실제 측정으로 고른다.

렌더링·graphic 입력을 유지한 창 숨김 launcher와 `time_scale=50`을 사용한다. 원본 게임·API·obs를 변경하거나 `-nographics`를 사용하지 않는다. 학습·관전과 공통 compute lock으로 자원 충돌을 막는다.

기본 제한은 전체 실행 자식 프로세스 RSS 36GiB, 디스크 중단 여유 15GiB, 경기 22,000 step/1,800초, heartbeat 120초다. 종료 시간이 긴 모델은 benchmark와 상태 기록에서 확인할 수 있다. manifest에 trace 필요 공간 추정치와 최대 step 가정의 예상 시간을 기록하며, 시작 시 공간이 부족하면 기록을 보존하고 중단한다.

자연 종료되지 않은 경기·truncation·부분 종료는 승패에 넣지 않는다. 실패 경기에는 자동 재시도 한 번을 허용하고 모든 시도의 로그를 보존한다. 다시 실패하면 실패 셀로 남겨 전체 완료로 표시하지 않는다. `--resume`은 완료 경기부터 이어가고 중단 경기의 Unity 내부 상태는 재사용하지 않는다. 소스·모델 자산 해시가 바뀌면 같은 이름으로 재개할 수 없다.

## 산출물과 상세 행동 조회

모든 실행 산출물은 `logs/shared/agent_league/<실험명>/`에 저장한다.

| 파일/디렉터리 | 내용 |
|---|---|
| `request.json`, `manifest.json` | 참가자·제외 후보·설정·소스/모델 해시·고정 대진 |
| `checks/`, `benchmark.json`, `probes/` | 로딩 검사와 성능 측정; 리그 성적에 미포함 |
| `status.json`, `console.log` | 진행량·현재 worker·오류·잠정 ETA |
| `attempts/` | 성공/실패/중단을 포함한 모든 경기 시도 |
| `cells/`, `failures/` | 확정된 경기 결과와 미완료 경기 |
| `report.md`, `summary.json`, `ranking.csv` | 전체 순위·집계·해석 범위 |
| `models/`, `pairs/`, `matches/` | 모델 특징·상대 관계·각 경기 흐름 Markdown |
| `figures/score_matrix.svg`, `observed_cycles.json` | 승점률 행렬 및 관측된 순환 우위 |

각 시도의 `trace/`는 512 step 단위 압축 NPZ, 인덱스, 압축 이벤트 JSONL, 주기적 지도 스냅샷으로 구성한다. 매 step에 10개 유닛의 원본 vector 관측, 실제 행동 벡터, 결과 보상, 다음 관측, 누적 reward, 게임 시간, 정책 추론 시간, 종료/truncation 여부를 기록한다. `before[t] → actions[t] → rewards[t], after[t]` 정렬이다. 마지막 관측이 없다면 `after_present=false`, `after=NaN`으로 표시하고 마지막 행동과 보상은 남긴다.

```bash
bash code/shared/run_agent_league.sh --name all_agents_v1 --trace 'MATCH_ID' --step 100
```

`MATCH_ID`는 경기 보고서 또는 `cells/` 파일명에서 찾는다. NPZ는 `numpy.load(..., allow_pickle=False)`로도 읽을 수 있다. 인접 step의 `after[t]`와 `before[t+1]`가 정확히 같으면 하나의 관측 체인으로 무손실 저장한다. float32 비트 패턴의 시간 방향 XOR 차분을 압축하므로 반올림·양자화를 하지 않는다. `agent_league.trace.load_chunk(path)`와 상세 조회 CLI는 이를 원래 before/after 필드로 복원하며, 값이 다르면 관측 쌍을 그대로 저장한다. 매 step 전체 graphic은 저장하지 않는다. 팀 공통 지도 채널 개수·변화량을 기록하고 시작·종료 및 1,000 step마다 양 팀의 지도 스냅샷을 남긴다. 따라서 전체 관측 입력을 그대로 재생하는 데이터셋은 아니다.

행동 ID와 관측 entity slot의 대응을 임의로 추정하지 않는다. 이동·팀 분산·창고 근접·되돌아감은 팀/관측 슬롯 통계이며, pickup/delivery/teleport는 관측 변화의 **proxy**다. 정책 의도나 실제 엔진 사건으로 단정하지 않는다. 임계값은 manifest config에 저장된다.

보고서는 상대별 승점률 평균으로 순위를 만들고 동률은 공동 순위로 둔다. 승률과 승점률 `(승+0.5×무)/경기 수`를 구분한다. 모델별 지표와 대표 경기·사건 step을 연결하며, 5개 진영 교대 세트의 결과를 별도로 제시한다. 미완료 순위는 잠정이다. 전략과 우위 관계는 소표본의 탐색적 관측이며 공식 경기 성능이나 새 맵 일반화의 증명이 아니다.

## 검증

```bash
.venv/bin/python -W ignore::DeprecationWarning code/run.py -m unittest discover -s code/shared/tests -p test_agent_league.py -v
```

대진·진영·RNG, 제공 runner 판정 동치, 종료 transition 보존, 실패 처리, 재개 중복 방지, trace 정렬과 보고서 계산을 검사한다. 구현 검증 로그는 `logs/shared/agent_league/implementation_checks/`, 제한된 실제 경기 자료는 `implementation_live/`에 보존한다. 이 디렉터리들의 결과는 전체 풀리그 결과가 아니다.

## 21개 대표 모델 리그의 HTML 분석

`version_best_ties_league`의 완료된 2,100경기에 대한 [HTML 분석 보고서](../../logs/shared/agent_league/version_best_ties_league/analysis.html)는 순위·상대별 전적·행동 특성·근거 경기의 저장 시점별 맵과 실제 행동·해석 한계·후속 실험 제안을 포함한다. 차트와 사례 데이터는 HTML 안에 들어 있어 파일 하나로 열람할 수 있다. 원본 링크는 이 프로젝트의 로그가 있을 때 사용할 수 있다.

이번 참가 목록과 완료 결과에 한정한 분석을 재생성하려면 프로젝트 루트에서 실행한다.

```bash
.venv/bin/python code/run.py code/shared/tools/build_league_analysis.py
```

Unity·학습·추론은 실행하지 않는다. 완료 cell, 초기 snapshot, 종료 관측, 추론 시간 표본을 읽어 같은 로그 폴더의 `analysis_data.json`과 `analysis.html`을 생성한다. 기존 경기·순위 파일은 수정하지 않는다. HTML만 다시 렌더링하려면 `--render-only`를 추가한다.

원래 리그 판정은 누적 raw reward 기준이다. HTML의 관측 점수 진단은 종료 직후 관측을 별도로 복원한 값이며 공식 서버 판정으로 간주하지 않는다. 자세한 검증 결과는 같은 폴더의 `analysis_validation.json`에 기록한다.
