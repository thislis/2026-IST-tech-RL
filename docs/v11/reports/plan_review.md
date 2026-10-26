# v11 계획의 검토 기록

검토일: 2026-10-24. 이 기록은 확인한 범위를 구별하며, 에이전트 구현·학습·성능 달성 보고가 아니다.

## 독립 수치 검토

[리그 분석](league_findings.md)의 주요 수치를 별도 검토자가 원본과 대조했다. 이 단계는 **수치 검토 완료**이며 전체 설계·구현 계획의 완료 판정은 아니다.

- [analysis_data.json](../../../logs/shared/agent_league/version_best_ties_league/analysis_data.json)의 v1–v10 대표 12개 모델에 대해 W/D/L, 승점률, 양 진영 승점률, 경기당 반납 proxy, 방향 전환 비율이 문서와 일치했다.
- 상위 다섯 모델의 상호 10개 대진은 모두 5승 0무 5패였다.
- [v10 summary](../../../logs/v10/mapcritic_v1/summary.json)의 frozen test는 C1 60/240승, C0 47/240승, initial 58/240승이었다. 본학습 6개 run에서 target step 비율은 0.892~1.656%였으며 C1 seed 11은 1,096,887 step 중 9,789 step이었다.
- [리그 감사 결과](../../../logs/v11/reports/league_audit.json)에 기록된 `manifest.json`, `analysis_data.json`, `summary.json`, `ranking.csv`, `status.json`의 SHA-256을 현재 원본 파일에서 다시 계산해 모두 일치함을 확인했다.
- 감사 도구와 문서가 명시한 범위를 대조했다. 감사 도구는 2,100개 cell·attempt result·초기 snapshot·마지막 transition chunk를 확인한다. 중간 trace 전체에서 행동 지표를 재계산한 검사는 아니다.

주요 수치 오류는 발견하지 못했다. v9의 실제 혼합 학습 구성을 history/PFSP로 명시했다. 상태 허용·ONNX 실행에 관한 운영진 답변과 로컬 개발 가정을 구별한 표현도 확인했다.

## 독립 설계 검토와 반영

[설계](../plans/v11_design.md)와 [구현 계획](../plans/v11_implementation_plan.md)을 별도 검토자가 교차 검토했다. 다음 사항을 문서에 반영했다.

- 목표·보정·interrupt의 합집합을 decision event로 정의해 reward와 log-prob가 중복 집계되지 않도록 했다.
- GRU는 event에서 갱신하고 critic은 선택 전 상태를 사용한다. 목표 완료와 역할 reset, 경로 비용 변경 시 cache 무효화를 구별했다.
- benchmark 상대 checkpoint 공유와 test 경기·맵·seed·결과 격리를 구별했다.
- 약한 대조군을 성능 gate로 제거하지 않도록 기술 gate와 주 후보의 성능 gate를 분리했다.
- 경제 비퇴행 baseline을 v4로 사전 고정하고, 기본 제출 seed 선택과 추가 seed 재현성 확인을 구별했다.
- 기본 34,340,864 env step 예산과 후보별 300/1,260경기, 5개 seed의 6,300경기 계산을 확인했다.

이는 설계의 내부 일관성 검토다. ONNX export 성공, 실제 학습 수렴이나 성능 gate 통과를 실험한 것은 아니다.

## 요청 범위에 대한 최종 감사

| 요청 | 현재 확인한 증거 | 판정 |
|---|---|---|
| 대표 리그를 충분히 분석 | 2,100개 cell/attempt·초기 map·마지막 trace, 21모델/210대진 재집계, 사례·행동·계약 분석 | 완료 |
| v1–v10의 강점과 약점 반영 | 리그 분석의 10개 버전 대응 표, 설계의 구성별 채택·제외 이유 | 완료 |
| 운영진 Q&A 반영 | 공통 Q&A 문서, 기존 PPT에 후속 답변 안내, ONNX·명시 state 설계 | 완료 |
| 미확정 때문에 계획을 중단하지 않음 | 기존 공개 이름·관측·runner를 개발 기준으로 명시, 구현 경계와 가정 구별 | 완료 |
| v11을 어떻게 설계할지 | 관측·역할/목표·planner·잔차·기억·critic·학습·ONNX·ablation | 완료 |
| 어떻게 구현할지 | V11-00~10, 모듈·데이터·검사·예산·선택·성능 gate·실행 순서 | 완료 |
| README에 맞는 문서 배치 | docs/v11 plans/reports, docs/common Q&A, code/v11 분석 도구, logs/v11 자동 결과, root 링크 | 완료 |
| 자료 탐색의 근거 | 설계 본문에 ONNX/PyTorch·DAgger·Residual RL·VIN·MAPPO·league·평가 1차 자료와 적용 범위 | 완료 |

문서 로컬 링크·코드블록·공백·분석 도구 문법을 확인했고, 원본 리그 파일의 기록된 hash도 현재 파일과 일치했다. **요청된 분석·설계·구현 계획은 완료**다. v11 모델 구현·실험 실행·최고 성능 달성은 이 요청의 계획을 실행하는 후속 작업으로 남는다.
