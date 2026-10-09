# v8 참고자료 적용 기록

2026-09-27. 구현 기준은 로컬 `docs/v8/plans/v8_implementation_plan_with_references.md`와
`docs/v8/plans/v8_research_requests_2026-09-27/internal_result.md`, 분석 코드 `code/v8/research/v8_research_requests_2026-09-27/internal_analysis/`, 생성 데이터 `logs/v8/research/v8_research_requests_2026-09-27/internal_analysis/`이다.
구현 시작 HEAD는 `ee5d0a9e33a7c7ab7e860eaab48bf80b231ae927`이다.

`v8_external_research.md`, `v8_external_research.json`, 구판 `v8_implementation_plan.md`,
`implementation_backlog.json`은 로컬에서 확인하지 못했다. 공개 검색에서도 해당 인계 문서
자체를 확보하지 못했다. 이를 읽었거나 복원했다고 간주하지 않았다. 필요한 일반 원리는
아래 원문을 웹에서 열어 확인했으며, 논문/PDF/외부 코드를 파일로 다운로드하지 않았다.
현재 구현의 수치와 예산은 계획서의 잠정값이며 문헌 최적값이라는 주장을 하지 않는다.

| 원문 | 실제 확인 범위 | 적용 |
| --- | --- | --- |
| [Gymnasium Handling Time Limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/) | 웹 본문 | termination의 bootstrap 0, truncation의 final-observation bootstrap |
| [Unity ML-Agents Python LLAPI](https://docs.unity3d.com/Packages/com.unity.ml-agents@4.0/manual/Python-LLAPI.html) | 웹 API 본문 | DecisionSteps/TerminalSteps, canonical IDs, side-channel 계약. 웹 문서 patch와 설치 버전을 동일시하지 않음 |
| [Invalid Action Masking](https://arxiv.org/html/2006.14171v3) | HTML 본문 | 정규화된 masked joint distribution, 실제 behavior likelihood 저장 |
| [PPO](https://arxiv.org/abs/1707.06347) | 초록·공식 원문 연결, HTML 시도 | 세부 식은 구현 계획서와 기존 로컬 PPO에 대조. 논문 전체 재독해로 표시하지 않음 |
| [GAE](https://arxiv.org/abs/1506.02438) | 원문 초록·서지 | raw TD residual/GAE 및 별도 advantage 정규화; BlackOut 종료 구분은 로컬 fixture로 검증 |
| [The pigeonhole bootstrap](https://arxiv.org/abs/0712.1111) | 원문 초록·서지 | run/map 재표집 배경. independent algorithm rows/shared map 규약은 계획서의 실험 설계 |
| [Empirical Design in RL](https://www.jmlr.org/papers/v25/23-0183.html) | 저널 페이지·초록 | run 변동과 artifact 성능 구분, 독립 confirmation/test 미등록 상태 보존 |
| [Statistical Precipice](https://arxiv.org/abs/2108.13264) | 원문 초록·서지 | 작은 run 수의 한계 표시. bootstrap 반복 수를 실험 수로 세지 않음 |
| [Implementation Matters](https://arxiv.org/abs/2005.12729) | 원문 초록·서지 | PPO optimizer·normalization·clipping·KL 설정을 명시적으로 잠금 |
| [Unity 6000.3 Graphics.CopyTexture](https://docs.unity.com/en-us/engine/6000.3/script-reference/unityengine/graphics/copytexture) | 공식 API 검색 결과의 형식 호환 설명 | 같은 크기의 linear RGBA8 데이터 복사. 실제 CPU→GPU→Python 픽셀 SHA256 비교를 별도로 통과 |

v8 Python 구현은 이 저장소의 기존 모델/planner와 새 구현을 사용한다. 외부 논문 저장소를
복제하거나 외부 구현 파일을 가져오지 않았다. Unity 오버라이드는 로컬 `../blackout`의
고정 commit 소스를 바탕으로 별도 프로젝트에 적용한다. 배포 라이선스 심사·공식 loader
승인은 완료된 것으로 표시하지 않는다. C2/C3 문헌은 조건부 후속 범위이며, 해당 모델을
구현하거나 검증했다고 표시하지 않는다.

설치 버전·원본 소스 검증·입력 자료 해시는
[adoption_manifest.json](../../logs/v8/reports/adoption_manifest.json)에 기록한다.
