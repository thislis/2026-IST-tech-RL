# 외부 자료 조사자에게 전달할 요청서

제목: BlackOut RL v8 설계 결정을 위한 외부 근거 조사
기준일: 2026-09-27 KST

## 요청 목적과 전달 자료

첨부한 `00_common_brief.md`와 `reference/previous_audit.md`를 읽고, 알려진 병목을 해결할 수 있는 논문·공식 구현·실험 설계·공식 환경/제출 계약을 조사해주세요. 일반적인 강화학습 동향 정리가 아니라, 우리 문제에 적용 가능한 최소 변경을 고를 수 있는 자료가 필요합니다.

우리의 출발점은 planner 보존형 residual이 실제로 개선 행동을 내는지 불명확하고, 전뇌 직접 제어는 감각/좌표계/출력 용량의 제약이 의심되며, 점수·종료·제출 계약이 완결되지 않았다는 것입니다. 이는 기존 감사의 관찰과 가설이며 실제 원인 기여도는 내부 분석자와 교차 검증해야 합니다. 승률 개선용 본선 모델과 connectome의 고유 효과 연구를 분리해주세요.

아래 문헌은 이번에 서지와 원문 소개/공식 문서의 주제를 확인한 출발 자료입니다. 전부 읽고 검증을 마친 재현 목록이나 최신 문헌의 완전한 목록은 아닙니다. 원문 Methods·보충자료·저자 구현까지 확인하고, 2026-09-27까지의 후속 연구·수정판·정정·재현 실패도 검색해주세요. 링크와 읽을 지점은 `04_verified_source_starters.md`에 있습니다.

## E01. 경기 종료·점수·승자·제출 계약 — 최우선

연결: F01–F07, F28–F29, F34–F35 / 내부 I01·I05·I09

찾아주세요: 실제 대회에서 사용하는 최신 공지, 빌드 release/changelog, runner/loader 소스, 담당자 답변, 고정 버전 Unity ML-Agents와 wrapper 문서. 일반 ML-Agents 기능이 곧 대회 허용 계약이라고 추론하지 말아주세요.

확인할 질문은 명시적 final score와 winner가 있는지, 언제 reset되는지, reward-sum 판정인지 게임 결과 판정인지, 게임 규칙의 timeout과 외부 watchdog을 어떻게 구분하는지입니다. 모델 내부 planner/길찾기, 학습 teacher, state 유지, episode/step/reset hook, agent 순서/ID, 호출 재시도, 허용 입력·패키지·외부 파일·메모리·스레드·지연 한도도 확인해주세요. 점수 tie와 terminal 보상을 누적 reward로 대체하지 않는 계약이 필요합니다.

출발 자료: S01 Gymnasium Handling Time Limits, S02 Unity ML-Agents LLAPI. 두 자료는 종료와 agent routing을 점검할 근거일 뿐 최종 점수 snapshot이나 대회 규정이 자동 보장된다는 증거가 아닙니다. 게임에 내재한 시간 제한은 termination일 수 있으므로 모든 timeout을 truncation으로 통일하지 말아주세요.

검색어 예: `Unity ML-Agents TerminalSteps interrupted terminal observation reset`, `BlackOut RL competition model loader reset agent order inference limit`, `finite horizon termination external time limit truncation`.

받고 싶은 결과: 항목별 ‘공식 확인/고정 소스에서 확인/미답변/상충’, 문서·답변 날짜, 적용 빌드/commit, 근거 문장 위치, v8 설계 영향의 계약표. 답변이 없으면 원인을 추측하지 말고 미확정으로 남기고, 공식 문의 초안을 작성해주세요. 실제 발송은 별도입니다.

결정: stateful 정책과 planner를 제출할 수 있는가? 허용되지 않거나 hook이 없을 때 쓸 입력만으로 동작하는 대안은 무엇인가?

## E02. KEEP 편향·residual 개입·action mask — 최우선

연결: F08–F10, F12, F15 / 내부 I02·I04·I05

찾아주세요: residual policy, 기본 제어기와 학습 정책의 결합, 개입 여부/슬롯/방향의 계층적 선택, invalid-action masking, sampling과 deterministic 평가를 함께 다루는 연구/구현. 출발점은 S03 Residual Reinforcement Learning for Robot Control, S04 A Closer Look at Invalid Action Masking in Policy Gradient Algorithms, S05 PPO입니다.

비교해야 할 대안은 현재 41-way KEEP+5×8, 개입 gate+conditional correction, 목표/역할 수준 residual입니다. 로봇의 연속 제어 신호를 더하는 residual과 BlackOut의 정규화된 방향 벡터 선택은 같지 않으므로 논문 구조를 그대로 복사하지 말아주세요.

분포, 행동 선택, log-probability, mask 적용 위치, entropy, 개입 비용/제약, 학습/평가 선택 규칙을 수식 또는 간단한 의사코드로 제출해주세요. 확률을 계층적으로 인수분해해도 joint distribution과 joint argmax가 같다면 행동은 달라지지 않습니다. gate threshold나 단계별 argmax를 쓰면 최종 선택 정책이 바뀐다는 점을 분리해야 합니다. sampling→argmax 자체가 PPO 버그인 것도 아닙니다.

검색어 예: `residual policy learning intervention gating default controller`, `hierarchical categorical policy gate action masking PPO log probability`, `stochastic training deterministic evaluation policy action collapse`.

받고 싶은 결과: 두 대안의 적용 전제, 실패 위험, 최대 한 슬롯 제한 여부, mask 이후 총 개입 확률, 학습 분포와 실행 행동 정합성, 정책 보존과 개선의 trade-off. 내부의 p_KEEP/개별 최대 수정 확률/greedy override 측정값에 따라 채택 또는 기각하는 기준을 적어주세요. 개입률 증가 자체를 성공 지표로 삼지 말아주세요.

결정: 개입이 실행되지 않는 문제인지, 실행되지만 해로운 문제인지에 따라 행동 분포를 바꿀 것인가?

## E03. 시간적으로 지속되는 행동·팀 보상의 기여도 — 높은 우선순위

연결: F10, F13 / 내부 I03·I04

찾아주세요: options/시간적 추상화, 목표·역할 선택, 반사실적 기여도 추정, GAE의 시간 단위 조정, shaping의 목적 보존 조건. 출발점은 S06 Option-Critic, S07 GAE, S08 MAPPO, S09 COMA입니다. 잠재함수 기반 shaping의 원 논문과 후속 연구도 검색하되, 단일-agent MDP의 보장을 팀 대전/POMDP에 그대로 옮기지 말아주세요.

우리에게 필요한 것은 20ms 방향 수정이 수초 이후 팀 득점으로 이어지는 상황에서 어떤 학습 단위가 적합한지입니다. 단순 action repeat와 고수준 옵션은 구분하고, 행동을 k step 유지하면 누적 보상과 할인 gamma^k, 중도 termination, 옵션 종료 및 안전 override를 어떻게 처리하는지 조사해주세요. PPO update 횟수를 줄이는 가속과 전략 표현 변경을 구분해야 합니다.

COMA는 다른 actor/critic 구조를 갖는 방법이므로 ‘COMA로 교체’가 아닌 반사실 baseline의 개념·필요 데이터·계산량을 우리 team-action 구조에 어떻게 맞출지 검토해주세요. 보상 항을 추가할 때 점수 누적, 왕복 이동, 아이템 반복 접촉 등 보상 악용 사례도 포함해주세요.

검색어 예: `temporal abstraction options semi Markov policy gradient gamma action duration`, `counterfactual multi agent credit assignment difference rewards`, `potential based reward shaping terminal state policy invariance`.

받고 싶은 결과: 현재 PPO의 최소 변경안과 고수준 행동 변경안 각 하나, 필요한 critic 입력·보상 사건·측정 지표, 내부 보상 지연 분석을 보고 정할 시간척도 후보. 1/(1-gamma*lambda)는 GAE 가중치의 한 척도이지 장기학습 가능 시간의 절대 상한이 아니라는 점을 명시해주세요.

결정: 보상/critic을 먼저 고칠지, 행동 지속성/전략 표현을 먼저 바꿀지?

## E04. 관측 정보·기억·좌표계·planner 모방 — 높은 우선순위

연결: F07, F11, F18–F21 / 내부 I05·I06

찾아주세요: POMDP의 재귀 baseline, history input과 state reset, egocentric/allocentric action 변환, frozen encoder의 표현 진단, closed-loop imitation과 DAgger. 출발점은 S10 Recurrent Model-Free RL, S11 DAgger, S12 Transforming a Head Direction Signal…입니다.

비교 대상은 current frozen features, 관측에서 얻을 수 있는 최소 vector context 추가, 부분 encoder fine-tuning, 작은 GRU, 상대 행동 decoder입니다. actor가 쓸 수 없는 critic/엔진 정보는 feature 후보에서 제외해주세요. 재귀 입력으로 보상을 권하는 연구가 있어도 공식 inference 때 보상이 들어오지 않으면 다른 인터페이스가 필요합니다.

자기 시점 retina와 절대 방향 readout의 정합성을 어떻게 시험할지, heading·보유 item·class·score/time의 정보가 실제로 필요한지 분리한 ablation을 제안해주세요. 작은 recurrent 모델과 전뇌를 비교할 때 같은 감각만 주는 구조 비교와 더 많은 합법 정보를 주는 성능 비교를 따로 설계해주세요. DAgger는 이미 v3에서 사용했으므로 재도입 권고가 아니라 당시와 달라져야 할 데이터/손실/closed-loop 기준을 제시해주세요.

검색어 예: `recurrent model free RL POMDP observation action history`, `egocentric observation allocentric action heading equivariance navigation`, `DAgger compounding error no op class imbalance imitation closed loop`.

받고 싶은 결과: 입력 권한표, 작은 baseline 설계, heading 보정 두 방법 비교, encoder probe 과제, teacher 실패/NoOp 편향에 강한 평가 계획. 현재 heading이 상태에서 간접 추론될 가능성도 남겨두고 구조적 불가능이라고 단정하지 말아주세요.

결정: 정보 소실을 고칠 것인가, 기억을 추가할 것인가, encoder를 학습시킬 것인가?

## E05. connectome을 유지할 근거와 대조 실험 — 조건부 높은 우선순위

연결: F16–F17, F20, F22–F24 / 내부 I07

찾아주세요: S12 heading-to-steering, S13 Connectome-constrained networks…, S14 A Drosophila computational brain model…와 그 저자 코드/데이터/정정/후속 재현. 입력·출력 포트, 연결 방향, synapse count와 실제 연산 가중치의 관계, 흥분/억제 부호, 동역학·학습 대상, 외부 회로 절단 가정을 표로 만들어주세요.

실제 topology를 쓰는 것, synapse 강도/부호를 쓰는 것, 생물학적 동역학을 모사하는 것, 게임에서 학습하는 것은 서로 다른 주장입니다. ‘모든 뉴런이 있다’거나 ‘움직였다’는 사실만으로 기능 이전 또는 전략 성능을 인정하지 말아주세요.

부분회로 대조군은 여러 차수보존 재배선 seed, 포트/가중치/부호 대조, 파라미터 대응 MLP 등을 검토해주세요. 어떤 null이 무엇을 보존/파괴하는지 명시해야 합니다. 전뇌 F1은 학습 후 normal·frozen frame·sensory block·output block·pool shuffle 비교를 요구하며, 입력 차단은 분포 밖 입력이 될 수 있으므로 단독으로 배선의 고유 효과를 입증하지 못합니다.

회로 제거/간선 축소·상태 reset·pool 추가는 가속이 아니라 모델 변경일 수 있으므로 구분해주세요. 전뇌 재배선이 비싸면 비용 대비 우선순위를 적고, 소규모 null의 결과를 전뇌 증거라고 쓰지 말아주세요.

게임 플레이 데모를 찾아도 실제 데이터셋·버전·제어 유닛·planner fallback·학습 여부·완주/승률·공개 코드 commit을 확인해주세요. 영상/보도만 있는 사례는 ‘시연’으로 분리합니다. 생물학 연구의 검증 결과를 BlackOut 승률 근거로 옮기지 않습니다.

검색어 예: `connectome constrained task optimization ablation random wiring`, `Drosophila PFL3 head direction goal steering model`, `whole brain simulation sensorimotor causal intervention trained readout`.

받고 싶은 결과: ‘v8 본선에 유지/실험 분기로 분리/현재 보류’의 조건부 근거, 부분회로·전뇌 각각 최소 대조군, 과제 성능·구조 효과·생물학적 타당성의 서로 다른 통과 기준.

결정: 동일 정보·제어 범위·튜닝 예산의 작은 대조군보다 비용 대비 이득이 있는가?

## E06. 상대 구성·실패 맵 재표집·전략 검증 — 중간 우선순위

연결: F14, F25, F36 / 내부 I03·I05·I08

찾아주세요: 학습 가능한 난도 선택, train map 재표집, 고정 target/상대 집단/역사 정책 혼합, held-out 평가와 curriculum의 분리. 출발점은 S15 Prioritized Level Replay와 S08 MAPPO입니다.

우리 v7의 고정 일정은 구조 비교용이므로 성능 개선용 curriculum을 별도 experiment로 제안해주세요. 실패만 반복하는 방식과 학습 가능성 추정의 차이, 전체 train 분포를 잃지 않는 방법, episode 길이로 인한 진영/상대 step 노출 편향을 다뤄주세요. PLR식 환경 재방문을 오래된 transition을 PPO에 재사용하는 것과 혼동하지 말아주세요.

검색어 예: `prioritized level replay learning potential on policy`, `opponent curriculum frozen opponent league exploitability`, `curriculum distribution shift evaluation generalization`.

받고 싶은 결과: 고정 target 성능을 위한 안과 상대 일반화를 위한 안을 나누고, 내부의 seed×side×opponent 실패행렬을 근거로 샘플링을 정할 규칙. 쉬운 random 상대에서 포화된 특수 아이템 ablation을 target에 일반화하지 않는 검증안.

결정: 상대 강도, train map 표집, 역할/특수 아이템 정책 중 무엇을 바꿀 것인가?

## E07. 통계·공정한 비교·회귀 검증 — 최우선의 공통 기반

연결: F25, F30–F33 / 내부 I00·I08·I09

찾아주세요: S16 Statistical Precipice 및 저자 rliable, S17 Implementation Matters. 독립 학습 seed와 공유 평가 map/양 진영의 종속성을 함께 다루는 paired/hierarchical 또는 two-way resampling 방식을 검토해주세요.

매 프레임이나 같은 정책의 60경기를 독립 학습 반복으로 취급하지 말아주세요. 훈련 seed 번호가 같다고 서로 다른 모델의 궤적이 완벽히 paired인 것은 아니지만, 공통 평가 map/side의 pairing은 보존할 수 있습니다. 학습 seed 3~5개에서 가능한 결론과 불확실한 결론을 구분하고, bootstrap 하나로 충분하다고 단정하지 않습니다.

검색어 예: `reinforcement learning evaluation independent training seeds paired maps confidence interval`, `multiway bootstrap crossed random effects algorithms seeds tasks`, `reinforcement learning reproducibility code level optimizations`.

받고 싶은 결과: 1차 성능 지표와 최소 의미 있는 개선폭, 무승부 처리, 모델 선택/confirmation/test 잠금, 훈련 seed별 결과와 신뢰구간, 튜닝 비용·계산 비용 정규화, 다중 비교/조기중단 원칙. 승률은 표본의 정책·상대·분할 범위 밖으로 일반화하지 않습니다.

결정: ‘v8이 좋아졌다’를 무엇으로 인정할 것인가?

## E08. 실행 비용·이식성·제출 지연 — 중간 우선순위

연결: F26–F29 / 내부 I09

찾아주세요: S18 PyTorch Performance Tuning Guide 및 장치별 공식 profiler/benchmark 문서, 실제 설치된 ML-Agents/Protobuf 버전의 호환 자료. 기본 대상은 현재 장비와 공식 제출 환경이며, 최신 버전 업그레이드 자체를 해결책으로 삼지 않습니다.

검색어 예: `PyTorch small batch CPU GPU overhead profiling`, `sparse CSR matrix vector multiplication MPS benchmark`, `Unity ML Agents protobuf serialization environment throughput`.

받고 싶은 결과: encoder/planner/CSR/통신/복사/PPO/저장/평가별 비용 측정법, warm-up·동기화·p50/p95/p99·메모리의 정의, 독립 실행 병렬화와 한 정책의 vectorized rollout의 구분. 속도만 바꾸는 변경과 action repeat/정밀도/그래프/관측을 바꾸는 알고리즘 변경을 분리해주세요.

결정: 현재 병목에서 실행 비용이 정당화되는 backend·병렬화·export 방식은 무엇인가?

## 제출 형식과 완료 조건

최종 제출물은 ‘의사결정 요약’, ‘문제별 근거 카드’, ‘문헌/공식문서 목록’, ‘최소 실험 제안’, ‘미답변 공식 계약’의 다섯 부분으로 구성해주세요. 형식은 Markdown과 JSON 또는 문헌 관리 형식이면 됩니다.

각 근거 카드에는 E/F/D ID, 원문 제목/저자/연도/버전/원문 주소/정정 여부, 정확한 절·표·그림, 저자 코드 commit과 라이선스 확인 여부, 원래 과제·관측·행동·보상·학습량·장비·대조군, 우리와의 차이, 제안 변경, 필요한 내부 증거, 비용(실측/추정 구분), 실패 조건, 채택/보류 판단을 넣어주세요.

우선 문제별 핵심 자료 1~3개를 깊게 확인하고 단순 제목 나열을 피해주세요. 최종 구현 후보는 최대 3개로 좁히되, 계약/측정 결함 수정은 모델 후보와 별도로 필수 기반으로 표시해주세요. 각 후보는 ‘어떤 내부 결과가 나오면 기각할 것인가’를 반드시 포함해야 합니다.


---

## 부록: 공통 지침

# v8 조사 공통 지침 (사본)

작성일: 2026-09-27 KST
대상: thislis/2026-IST-tech-RL
출발 감사 snapshot: main@e6e98a63e9f401f6bf03e463a720c03878d1beb7

## 조사 목적

v8은 새 모델 이름을 정하는 작업이 아니라, 신뢰할 수 있는 평가 아래 기존 planner보다 유용한 행동을 학습하고, 그 결과를 제출 환경에서 재현하는 작업이다. 외부 조사자는 해결책의 근거와 적용 조건을, 내부 분석자는 실제 실패의 기전과 반증 자료를 마련한다. 두 결과를 같은 의사결정 ID로 연결한다.

이 요청서는 이전 감사에 기반한 조사 배정이다. 이번 작성에서 새 경기·재학습·원본 로그 재집계를 수행한 것은 아니다. reference/previous_audit.md의 수치도 원래의 확인 수준을 유지한다. 특히 36개 항목은 모두 확정 버그가 아니라 코드 결함, 보고된 현상, 관측 제약, 연구 가설, 검증 공백을 함께 포함한다. 최신 checkout이 다르면 변경분을 먼저 기록한다.

## 공통 원칙

- v8 승률 개선과 생물학적 배선의 고유 효과 입증을 별도의 목표로 둔다. connectome을 반드시 유지한다는 전제로 조사하지 않는다.
- 외부 논문의 성공은 BlackOut에서의 성공 증거가 아니다. 내부 로그의 상관관계는 원인 증거가 아니다.
- 과거에 수정된 결함은 수정 버전과 회귀 검사를 적고, 현재 결함과 분리한다.
- 오염된 terminal score를 최종 점수로 사용하지 않는다. last-nonterminal 점수는 preterminal이지 최종 득점 snapshot이 아니다. 기존 winner도 독립 검증 전에는 출처를 표시한다.
- 기존 결과·checkpoint·registration을 덮어쓰지 않는다. 원래 소스/환경의 진단과 수정된 환경에서의 재평가는 별도 ID로 둔다. 빌드 변경 시 모든 대조군을 새 빌드에서 다시 맞춘다.
- 기존 train/dev로 조사하고, 최종 test의 seed·경기 내용을 원인 분석이나 설계 선택에 사용하지 않는다. confirmation도 반복 선택에 쓰면 더 이상 독립 확인셋이 아님을 기록한다.
- 공식 actor 입력, 학습 전용 critic 입력, 진단 전용 엔진 정보의 권한을 분리한다. 배터리 수량처럼 현재 관측에 없는 정보를 정답 feature로 몰래 추가하지 않는다.
- 동일 map seed는 동일 중간 물리 상태나 동일 난수 소비를 보장하지 않는다. 진짜 반사실 분기에는 환경·정책·RNG 전체 snapshot 또는 동등성이 검증된 재현이 필요하다.
- 상대/역할/진영/맵/학습 seed/종료 종류별 분모를 명시한다. env step, agent action, game second, wall second를 구분한다.
- 새 장기 학습이나 대규모 sweep보다 기존 artifact 확인 → 오프라인 진단 → 합성 fixture → 제한된 독립 재평가를 먼저 한다. 조사 단계의 예산과 수정 권한은 프로젝트 책임자와 별도로 정한다.

## 서로 교환할 것

내부 분석자는 먼저 artifact 목록, 신뢰 가능한 지표, 최소 실패 사례, 부족한 계측을 외부 조사자에게 전달한다. 외부 조사자는 문제별 대안 두 개 이내, 필요한 관측/실행 계약, 예상 장단점, 반증 실험을 돌려준다. 최종적으로 03_joint_decision_handoff.md의 결정표를 같이 채운다.

논문을 많이 모으거나 로그를 많이 그리는 것이 완료 조건은 아니다. 각 조사 항목에서 ‘이 증거가 나오면 v8의 무엇을 바꿀지, 무엇은 바꾸지 않을지’가 적혀 있어야 한다.


---

# 외부 조사 출발 자료: 확인한 원문·공식 문서

조회일: 2026-09-27 KST. 아래는 서지·공식 소개/원문 HTML에서 주제와 관련성을 확인한 탐색 시작점이다. 모든 본문·보충자료·공식 코드를 재현한 목록이나 최신 연구의 완전한 목록이 아니다. 실제 채택 판단은 외부 조사자가 정정·버전·대조 실험·코드·라이선스를 확인하고, 내부 증거와 연결한 뒤 수행한다.

기본 논문과 2026-09-27까지의 후속 자료를 함께 검색한다. 최종 출처는 논문 원문/출판사/저자 코드/공식 문서를 우선한다. 검색 결과 요약·블로그·게임 시연은 발견용 자료로만 쓰고 근거 원문을 추적한다.

## S01. Gymnasium: Handling Time Limits

공식 문서; 조회 2026-09-27

원문/공식 주소: https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/

관련 요청: E01, E07

확인할 지점: 게임 내재 시간제한과 외부 truncation, bootstrap의 구분. 프로젝트 설치 버전에 적용되는 부분은 별도 대조.

## S02. Unity ML-Agents: Getting started with the LLAPI

공식 문서; 조회 2026-09-27

원문/공식 주소: https://unity-technologies.github.io/ml-agents/Python-LLAPI/

관련 요청: E01, E08

확인할 지점: DecisionSteps/TerminalSteps의 observation/reward/agent_id/interrupted. 대회 loader와 점수 전달 계약까지 보장하지 않음.

## S03. Residual Reinforcement Learning for Robot Control

Johannink et al.; ICRA 2019; preprint 2018

원문/공식 주소: https://arxiv.org/abs/1812.03201

관련 요청: E02

확인할 지점: 고전 제어와 학습 residual의 결합. 로봇 연속 additive 제어를 방향 정규화 게임에 그대로 옮기지 말 것.

## S04. A Closer Look at Invalid Action Masking in Policy Gradient Algorithms

Huang & Ontañón; FLAIRS 2022; preprint 2020

원문/공식 주소: https://doi.org/10.32473/flairs.v35i.130584

관련 요청: E02

확인할 지점: mask 적용·학습/평가 조건 비교. arXiv 2006.14171의 최종 수정판과 저자 코드도 대조.

## S05. Proximal Policy Optimization Algorithms

Schulman et al.; 2017

원문/공식 주소: https://arxiv.org/abs/1707.06347

관련 요청: E02, E03

확인할 지점: behavior policy와 likelihood ratio, clipped surrogate. gate·mask·강제개입의 정확한 joint probability를 정의할 기준.

## S06. The Option-Critic Architecture

Bacon, Harb & Precup; AAAI 2017

원문/공식 주소: https://ojs.aaai.org/index.php/AAAI/article/view/10916

관련 요청: E03

확인할 지점: 옵션 내부 정책, 옵션 선택, 종료조건. action repeat와 고수준 제어의 차이를 읽을 출발점.

## S07. High-Dimensional Continuous Control Using Generalized Advantage Estimation

Schulman et al.; preprint 2015; arXiv 수정 이력 확인

원문/공식 주소: https://arxiv.org/abs/1506.02438

관련 요청: E03

확인할 지점: GAE의 bias/variance와 value bootstrap. 추정 시간척도를 학습 가능 시간의 절대 상한으로 해석하지 말 것.

## S08. The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games

Yu et al.; NeurIPS 2022; preprint 2021

원문/공식 주소: https://arxiv.org/abs/2103.01955

관련 요청: E03, E06

확인할 지점: MAPPO의 구현·하이퍼파라미터·critic 입력 ablation과 저자 on-policy 구현. 본 프로젝트의 team joint actor와 차이 확인.

## S09. Counterfactual Multi-Agent Policy Gradients

Foerster et al.; AAAI 2018

원문/공식 주소: https://ojs.aaai.org/index.php/AAAI/article/view/11794

관련 요청: E03

확인할 지점: 팀 보상에서 individual action의 기여도 baseline. COMA의 decentralized actor를 현재 joint residual과 구별.

## S10. Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs

Ni, Eysenbach & Salakhutdinov; ICML 2022

원문/공식 주소: https://proceedings.mlr.press/v162/ni22a.html

관련 요청: E04

확인할 지점: 기억 기반 baseline의 입력·구조·훈련 설정. 공식 inference가 받지 못하는 reward/metadata는 제외.

## S11. A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning

Ross, Gordon & Bagnell; AISTATS 2011

원문/공식 주소: https://proceedings.mlr.press/v15/ross11a.html

관련 요청: E04

확인할 지점: DAgger, learner-induced distribution과 순차 오류. v3에서 이미 사용한 방법이므로 기존 실패와의 구체적 차이를 요구.

## S12. Transforming a head direction signal into a goal-oriented steering command

Nature 2024; 2025-03-11 Author Correction 표시

원문/공식 주소: https://www.nature.com/articles/s41586-024-07039-2

관련 요청: E04, E05

확인할 지점: heading·goal의 좌표변환과 PFL/steering. 정정과 보충자료를 반영. 신경생물학 결과를 게임 성능 증거로 읽지 말 것.

## S13. Connectome-constrained networks predict neural activity across the fly visual system

Nature 2024

원문/공식 주소: https://doi.org/10.1038/s41586-024-07939-3

관련 요청: E05

확인할 지점: connectome 제약과 task optimization의 분리, cell-type parameterization, random model/구조 ablation. 저자 구현·데이터 확인.

## S14. A Drosophila computational brain model reveals sensorimotor processing

Nature 2024

원문/공식 주소: https://doi.org/10.1038/s41586-024-07763-9

관련 요청: E05

확인할 지점: connectivity/transmitter 기반 모델의 감각·출력·개입 검증과 가정. feeding/grooming 검증을 범용 게임 능력으로 일반화하지 말 것.

## S15. Prioritized Level Replay

Jiang, Grefenstette & Rocktäschel; ICML 2021

원문/공식 주소: https://proceedings.mlr.press/v139/jiang21b.html

관련 요청: E06

확인할 지점: 미래 방문할 level의 학습 가능성 기반 선택. stale PPO transition 재사용과 구분.

## S16. Deep Reinforcement Learning at the Edge of the Statistical Precipice

Agarwal et al.; NeurIPS 2021

원문/공식 주소: https://proceedings.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html

관련 요청: E07

확인할 지점: 소수 학습 run의 통계 불확실성, 구간/aggregate metric. 저자 rliable 도구가 우리 crossed seed×map 구조에 자동 정답인 것은 아님.

## S17. Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO

Engstrom et al.; 2020

원문/공식 주소: https://arxiv.org/abs/2005.12729

관련 요청: E03, E07

확인할 지점: 코드-level 선택과 성능 귀속. 버전/튜닝/정규화 차이와 알고리즘 기여를 분리.

## S18. PyTorch Performance Tuning Guide

공식 문서; 조회 2026-09-27

원문/공식 주소: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide

관련 요청: E08

확인할 지점: CPU/GPU/스레드·전송/동기화 비용 측정의 출발점. 현재 설치 버전/장치에서 재검증하고 무조건 GPU·AMP·업그레이드 처방 금지.

## 문헌별 근거 카드

ID / E·F·D ID:
원문 제목·저자·정식 출판연도·preprint연도:
원문 URL·DOI·버전·정정:
읽은 절·수식·표·그림:
저자 주장:
직접 확인한 근거:
원래 task·observation·action·reward·training budget·hardware:
baseline·ablation·training seeds·불확실성:
코드 URL·commit·실행 확인·라이선스:
BlackOut와 동일한 조건:
BlackOut에서 다른 조건:
제안하는 최소 변경:
기각 가능한 내부 증거:
필요한 신규 계측:
비용(측정/추정):
보류/실험/채택 판단:
