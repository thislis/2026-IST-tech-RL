# v7 초기 파일럿 등록

실행 구성과 예산은 `pilot_registration.json`, 각 `configs/v7/*.yaml`, `splits.json`에 고정했다. 소스와 설정의 실행 시점 사본은 `logs/v7/pilot_source_snapshot.zip`에 있다. 원본/그래프 SHA-256 및 가공 정책은 `data/` 보고서 사본과 각 run manifest에서 확인한다.

- v7-1: A3/A0/A2 × seed 11/22/33 × 200,000 env-step. 합계 1,800,000. A1 matched MLP는 실행 가능하게 준비했으나 기본 초기 큐에는 포함하지 않는다.
- v7-2: F0 seed 11, 22,000 env-step 후 F1 seed 11/22/33, 각 128,000 env-step. 이는 폐루프·학습 실행 안정성 파일럿이며 B2/B3까지 포함한 인과적 배선 비교가 아니다.
- 고정 학습 상대 일정, uniform train map 표집, A/A/B/B/B 경기 순서. 경기 길이가 달라질 수 있으므로 환경 스텝 기준 A/B 노출률은 실측 로그로 별도 계산한다.
- 모델 선택은 등록 예산 종료 시 latest. 실패 seed를 조용히 제거하거나 좋은 map만 보고 회로/포트를 바꾸지 않는다. 큐는 실패 시 정지한다.
- 목적 지표는 terminal 승률 W/(W+D+L), 무승부율·점수 차는 별도. 학습 seed 간 결론은 양 진영 map 쌍과 독립 학습 seed를 함께 고려해야 한다.
- 코드 검증·오프라인 sensory 검증·짧은 실제 Unity 개입 시험은 완료했다. 장기 파일럿 종료, dev/confirmation 평가, 최종 model lock 및 test는 아직 완료하지 않았다.

계획 조정은 실행 안내에 명시했다: 공식 FAFB 공개 아카이브 사용, 205개 실제 CX 뉴런, 실제 유형별 포트, 6,091개 광수용체와 210개 명시적 기하 대체 대응, 오프라인 고정 decoder 보정. 전체 그래프와 입력/출력 데이터 선택은 게임 승률에 맞춰 변경하지 않았다.
