# 2026-11-01 게임·환경 업데이트와 프로젝트 적용 상태

사용자가 지정한 로컬 두 저장소의 HEAD와 소스를 확인했다. 원격 저장소를 새로 pull하거나 게임을 빌드·실행하지 않았다. 이번 작업은 문서 갱신이며 `.venv`, 게임 빌드, 정책 코드와 기존 실험 결과는 변경하지 않았다.

## 확인한 버전

| 대상 | 위치 | 확인한 revision / 상태 |
|---|---|---|
| 새 Unity 소스 | `C:\Users\HeXA\Desktop\blackout` | `af4c5faaf2ef9eae24f536860737099ad56acba3` · working tree clean |
| 새 Python API·대회 실행기 | `C:\Users\HeXA\Desktop\blackout-env` | `bcd8aadad2f293bf45f3999286b1f174bae5586b` · working tree clean |
| 프로젝트에 설치된 API | `.venv/Lib/site-packages/blackout_env/` | 기존 리그 manifest에 기록된 Python 소스 해시와 모두 일치. 새 API가 설치된 상태는 아님 |
| 기존 API 설치 출처 | `artifacts/setup/blackout-env` | 배포 메타데이터 `direct_url.json`에 기록된 로컬 설치 원본 |
| 프로젝트의 기존 Windows 게임 | `artifacts/builds/windows-v1.1/Blackout.exe` | 기존 리그 사용 경로. 새 Unity HEAD로 다시 빌드했다는 증거 없음 |

새 API의 `pyproject.toml`과 설치된 패키지는 모두 버전 문자열이 `0.1.0`이다. **패키지 버전 문자열만으로 업데이트 여부를 판정할 수 없다.** 새 commit, 설치 소스 해시, 빌드 전체 자산 해시를 따로 기록해야 한다. [versions.md](versions.md)의 2026-08-05 macOS 고정 환경은 역사적 재현 기록이며 현재 Windows 설치 정보와 다르다.

## 게임 규칙·렌더링 변화

기존 문서의 게임 기준 `d2220a7d01be88d413f551efd529f4758833be8b`에서 새 HEAD까지 비교했다. 상세 규격은 [game_spec.md](game_spec.md)에 반영했다.

| 항목 | 변경과 영향 |
|---|---|
| Hunter | 성소의 변신 제한이 **팀당 동시 2명**으로 적용된다. Carrier는 팀당 동시 1명 제한을 유지한다. 변신 예약·역할 수를 정할 때 엔진의 실제 상한을 따라야 한다. |
| 맵 | `(1,16)`을 열고 `(5,18)`, `(6,18)`에 벽을 추가했다. 반대 팀에는 y=x 대칭 위치가 적용된다. 이전 맵의 고정 통로·경로 캐시를 새 맵에 그대로 적용하지 않는다. |
| semantic map | RenderTexture에 Point filtering과 Clamp wrap을 명시했다. semantic ID 경계의 보간을 방지하는 변경이며 관측 채널·해상도 자체 변경은 아니다. |
| timer | timer 완료 callback에서 timer를 해제하거나 새 episode를 시작하는 경우의 컬렉션 변경과 중복 처리를 보완했다. |

소스: [Sanctuary.cs](../../../blackout/Assets/Project/Runtime/Scripts/Region/Sanctuary.cs), [RegionData.cs](../../../blackout/Assets/Project/Runtime/Scripts/Map/RegionData.cs), [Prototype.unity](../../../blackout/Assets/Project/Runtime/Scenes/Prototype.unity), [SemanticMapRenderer.cs](../../../blackout/Assets/Project/Runtime/Scripts/ML/SemanticMapRenderer.cs), [TimerManager.cs](../../../blackout/Assets/Project/Runtime/Scripts/TimerManager.cs).

문서 검토 중 발견한 기존 설명 오류도 구분해 수정했다. 특수 아이템은 초기 생성기에서 4종을 각각 만들며, 이후 `LevelDirector`가 별도로 추적하는 동적 생성 경로도 있다. 따라서 “전역에 특수 아이템은 항상 1개뿐”이라는 예전 문서 설명은 맞지 않는다. 이것은 이번 commit에서 새로 추가된 규칙이 아니라 기존 코드 설명의 정정이다.

## API와 승패 판정 변화

비교 기준 API는 `6ba7d9993cf1bdefe1ed480c8efbcabcb923f539`이다.

- 새 `BlackOutEnv`는 관측 dict를 `unit_0..unit_9` 순서로 반환하고, `CheckpointModel.act()`도 유닛 번호로 정렬한다. 팀의 배치 행 순서를 명시적으로 보장한다.
- `BlackOutEnv(..., additional_args=[...])`로 Unity 실행 인수를 전달할 수 있다.
- 종료 step에서 다음 episode의 decision 관측이 함께 도착해도 **terminal 관측의 최종 점수를 우선** 보존한다. 예전 “종료 점수는 항상 0이므로 마지막 non-terminal 값 사용” 가정을 새 환경에 그대로 적용하지 않는다.
- 새 `run_match()`는 **최종 `infos.score_0`, `score_1` 비교**로 승자를 정한다. 누적 raw reward는 학습용 기록이며 승패 결정에 사용하지 않는다. 반환 winner는 진영 교대 후에도 `0=model_a`, `1=model_b`, `None=draw` 기준이다.
- `run_match(..., seed=..., on_step=...)`, `run_series(..., seeds=[...])`와 `BaseModel.reset()`이 추가됐다. 점수, seed, 진영 교대 여부도 결과에 저장한다.
- `infos.winner`는 새 wrapper에서도 첫 terminal agent reward로 추론하는 경로가 남아 있다. **새 `run_match`의 최종 점수 판정과 동일한 구현이라고 간주하지 않는다.**

소스: [env/blackout_env.py](../../../blackout-env/blackout_env/env/blackout_env.py), [model/loader.py](../../../blackout-env/blackout_env/model/loader.py), [competition/match.py](../../../blackout-env/blackout_env/competition/match.py), [model/base.py](../../../blackout-env/blackout_env/model/base.py).

seed를 reset 전에 확실히 전달하는 프로젝트 `ContractBlackOutEnv`는 기존 버전에서 확인한 동작이다. 새 upstream의 reset 구현 자체에는 이번 비교에서 seed 순서 수정이 없었다. 최신 README의 재현 설명만으로 프로젝트 adapter가 불필요해졌다고 결론내리지 않으며, 새 빌드/API 조합에서는 별도로 확인해야 한다.

## 공식 ONNX 제출 규격

새 기준 문서는 외부 저장소의 [competition_ko.md](../../../blackout-env/docs/competition_ko.md), 실행 계약은 [onnx_policy.py](../../../blackout-env/blackout_env/competition/onnx_policy.py)다. 예전 PPT의 `policy.py + checkpoint.pt`는 로컬 PyTorch 실행 형식으로 남고, 공식 컨테이너는 참가자의 Python 코드를 실행하지 않는다.

| 항목 | 새 규격 |
|---|---|
| 제출 폴더 | `meta.json` + `model.onnx` |
| team 모드 | `{"mode":"team"}` · 한 step당 B=5로 1회 호출, 행동 `(5,2)` |
| unit 모드 | `{"mode":"unit"}` · B=1로 유닛당 호출. 공유 `model.onnx` 또는 `unit_0.onnx`~`unit_4.onnx` |
| vector | float32 `(B,96)` |
| graphic | float32 `(B,11,96,96)` · CHW |
| slot | 선택 입력 int64 `(B,)` · 팀 안 순번 0~4 |
| 입력 사용 범위 | `vector`, `graphic`, `slot` 중 최소 하나와 선택적 `state_in*`. 사용하지 않는 입력은 생략 가능 |
| 행동 출력 | `state_out*`가 아닌 첫 출력 `(B,2)` · 규격 float32, 범위 `[-1,1]` |
| 문서상 파일 제한 | 모델별 2GB 이하 단일 `.onnx`, 표준 연산 사용 |

행은 유닛 번호순이다. Team A의 slot 0~4는 unit_0~4, Team B는 unit_5~9다. 자기 관측 블록은 `slot + (vector[:,2] > 0 이면 0, 아니면 5)`로 찾을 수 있다. unit 모드의 서로 다른 파일은 **클래스가 아닌 slot별** 선택이며, 변신 때 모델 파일이 자동으로 교체되는 규약은 아니다.

### 프레임과 상태

- `meta.json`의 `frame_stack=K`는 1~16이다. K>1일 때 vector는 `(B,K,96)`, graphic은 `(B,K,11,96,96)`이다. 오래된 프레임부터 놓고 마지막 프레임이 현재다. 처음에는 첫 관측을 K번 복제한다. slot은 쌓지 않는다.
- 상태 입력 `state_in<suffix>`에는 대응하는 `state_out<suffix>`가 필요하다. float32 `(B,...)`이며 B 뒤 크기는 고정한다. 서버가 slot별로 보관하고 episode 시작에 0으로 초기화한다.
- team 모드에서 누락된 slot의 입력을 0으로 채우며 해당 slot의 기존 상태는 유지한다. 답하지 못한 step에서는 상태를 갱신하지 않는다. 프레임 스택은 응답이 없는 step의 관측도 쌓는 규약이다.
- 모델 서버가 재시작되면 프레임·상태는 사라진다. Python 객체 멤버나 임의 외부 파일을 지속 상태로 사용할 수 있다는 뜻이 아니다.

v11의 GRU와 v12의 목표·경로 상태를 이 포트에 맞게 옮길 수 있는 계약이 생겼다는 의미다. **현재 개발용 scorer/부분 ONNX가 공식 제출 가능한 완성 모델로 바뀐 것은 아니다.** v12의 Python 후보 생성·계획·공동 배정·행동 생성 전체를 ONNX 내부와 허용된 상태 포트로 구현하는 작업은 별도로 필요하다.

### 실행 한도와 복구

공식 문서 기준 팀당 RTX 3090 SM 50%, GPU 메모리 11GB, CPU 4코어, RAM 16GB, 네트워크 없음이다. 정책 컨테이너는 CUDA 12.4 계열과 `onnxruntime-gpu==1.22.0`을 사용한다. 팀당 매 step 100ms에는 unit 모드의 전체 호출과 전송 시간이 포함된다. 첫 step 허용 시간은 30초다.

시간 초과·추론 오류는 그 step의 팀 행동을 0으로 대체한다. 프로세스가 종료되면 문서상 최대 60초 대기, 시리즈당 5회까지 복구 대기를 지원하고 컨테이너 로그를 남긴다. 이는 현재 로컬 v12 풀리그의 경기당 1,800초 watchdog과 별개 제한이다.

모델 export·형식 검사는 ML-Agents의 protobuf 의존성과 충돌하지 않도록 별도 export 환경을 사용한다. 도구는 `blackout-env/blackout_env/competition/check_submission.py`이며 이번에는 실행하지 않았다. 이 도구의 `OK`는 해당 컴퓨터에서의 형식·합성 입력 검사이지 실제 공식 GPU 제한 내 경기 성능 인증이 아니다.

## 대진·녹화와 기존 결과의 경계

공식 문서는 비공개 무작위 seed를 경기가 끝난 뒤 공개하고, 같은 seed로 진영 교대 2경기, 동률이면 새 seed 2경기를 반복한다고 설명한다. 그러나 함께 제공된 로컬 `run_competition.py` CLI는 사용자가 준 `--seeds`를 각각 두 번 소진하고 종료한다. **자동 seed 추첨·동률 추가 경기까지 그 CLI가 구현했다고 해석하지 않는다.**

`--record`는 action/seed 리플레이 `.npz`를 저장한 뒤 1배속으로 재생해 `.mp4`를 만든다. 로컬 공식 실행 경로는 Linux·Docker·NVIDIA/MPS·Xvfb 기준이며, Windows의 기존 `run_league.ps1`과 같은 실행기가 아니다.

2026-10-28~29 v12 풀리그의 2,298경기와 [HTML](../../logs/shared/agent_league/v12_full_league_20261028/analysis.html)은 **기존 게임/API·누적 보상 기준 등록 실험**이다. HTML의 “공식 종료 판정”도 당시 기록한 `infos.winner`를 뜻하며 새 API·새 맵에서 재평가한 결과가 아니다. 220경기의 두 판정 불일치는 보존한다. 기존 원본·순위를 최신 규칙의 결과로 덮어쓰지 않는다.

새 버전 실험을 준비할 때는 새 API 환경과 새 소스에 대응하는 빌드를 별도 경로에 두고, score 기반 평가·slot/state 계약·Hunter 제한·맵/seed/종료 관측을 확인한 뒤 **새 실험 이름**으로 등록해야 한다. 기존 12경기 재개와 새 게임 버전 평가는 분리한다. 이번 문서 갱신에서는 설치·빌드 교체·정책 수정·학습·평가를 수행하지 않았다.
