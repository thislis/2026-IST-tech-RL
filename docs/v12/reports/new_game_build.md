# 최신 게임 Windows 빌드 기록

**2026-11-02 상태: 빌드·등록 완료.** Unity 6000.3.8f1 설치 후 첫 시도는 라이선스가 없어 exit 198로 종료됐지만, 사용자의 로그인·활성화 후 재시도에서 성공했다. 빌드 요약은 오류 0건, 경고 486건, 135,282,360 bytes, 빌드 단계 약 163초다. 최초 패키지 준비·에셋 import 시간은 별도다. 게임 Player·리그는 실행하지 않았다.

등록 실행 파일은 `artifacts/builds/windows-af4c5faaf2ef/build-c8231087c08416cc/Blackout.exe`다. `20261102_001227/build_summary.json`과 `Editor.log`에 결과를 보존했다. 경고는 주로 패키지 `ConvGeneric` 셰이더의 일부 variant 제약이며, 기존 `MapObsAgent.Awake()`의 상속 메서드 가림 경고도 있다. 경고 제거를 위해 게임 코드를 바꾸지는 않았다. 실제 Player 연결·렌더링은 실행 금지 요청에 따라 확인하지 않았다.

빌드 복사본의 게임 소스·씬 59개가 원본 archive와 동일함을 확인했다. 원본 Git 저장소도 수정하지 않았다. `build_evidence.json`에 소스 archive, 빌드 helper, 요약 기록의 해시를 남겼고 리그 입력 등록에 포함했다. 일반 빌드 등록 형식의 `source_proof=user_attestation`은 유지하되, 이번에는 직접 수행한 빌드 기록을 추가로 연결했다.

사용자의 요청으로 Unity 설치와 Windows 빌드를 진행한다. 이 작업은 게임·리그 실행을 포함하지 않는다. 빌드 성공과 리그 등록 상태는 [새 게임 풀리그 안내](full_league_new_game.md)를 함께 확인한다.

## 빌드 입력과 위치

리그의 22개 모델 모두 새 API와 등록한 빌드 경로를 사용한 오프라인 합성 입력 검사에 통과했다. 검사 결과는 `logs/shared/agent_league/v12_full_league_new_game_20261101/checks/result.json`에 있다. 상태는 `checked`, 완료 경기 수는 `0/2310`이다. Unity Player를 통한 실제 환경 연결 검사는 수행하지 않았다.

- 원본 저장소: `C:\Users\HeXA\Desktop\blackout`
- 소스 커밋: `af4c5faaf2ef9eae24f536860737099ad56acba3`
- Unity Editor: `6000.3.8f1`, changeset `1c7db571dde0`
- 원본 보관: `artifacts/source_snapshots/unity-af4c5faaf2ef.zip`
- 작업 복사본: `.unity/build-source-af4c5faaf2ef/`
- 자동 빌드 코드: `code/shared/agent_league/build_new_game.py`, `code/shared/unity/BuildLeaguePlayer.cs`
- 기록: `logs/v12/new_game_build_20261101/`
- 출력: `artifacts/builds/windows-source-af4c5faaf2ef-<빌드 시작 시각>/Blackout.exe`

원본 저장소가 고정 커밋의 깨끗한 상태인지 확인한 뒤 `git archive`로 작업 복사본을 만든다. 빌드용 C# 스크립트만 복사본의 `Assets/Editor/`에 추가한다. 원본 저장소에는 스크립트나 Unity의 `Library/`를 생성하지 않는다.

Windows x86_64 Player, 기존 Mono backend, Development Build 비활성화, `Assets/Project/Runtime/Scenes/Prototype.unity` 한 씬을 사용한다. 자동 실행 옵션은 설정하지 않는다. Editor를 실행하는 명령의 `-nographics`는 빌드 과정에만 적용되며, 리그에서 실행할 Player는 의미 지도 렌더링을 유지한다.

## 다시 빌드할 때

동일한 Editor가 설치되어 있으면 프로젝트 루트에서 다음 명령으로 빌드한다. 이 명령도 게임이나 리그를 시작하지 않는다.

```powershell
.\.venv\Scripts\python.exe -B .\code\run.py -m agent_league.build_new_game
```

각 빌드는 별도 출력 폴더에 저장한다. 성공한 빌드 경로는 `logs/v12/new_game_build_20261101/latest_build.json`에 기록한다. `Editor.log`, `build_summary.json`, `request.json`으로 실제 실행한 빌드 명령과 결과를 확인할 수 있다. 빌드 후 새 리그에 등록하는 절차는 별도이며 이미 등록된 빌드를 이 명령이 덮어쓰지 않는다.

참고: [Unity 6000.3.8f1](https://unity.com/releases/editor/whats-new/6000.3.8f1), [Hub 설치 CLI](https://docs.unity.com/en-us/hub/hub-cli-reference), [사용자 지정 빌드 스크립트](https://docs.unity.com/en-us/engine/6000.3/manual/building-and-publishing/build-customize-build-pipeline/build-script-build).
