# PREP-13 평가 계약 확인 목록

기준 소스는 고정된 Python API commit `6ba7d9993cf1bdefe1ed480c8efbcabcb923f539`의
`README.md`, `blackout_env/model/{base.py,loader.py}`,
`blackout_env/competition/match.py`, `blackout_env/env/blackout_env.py`다. 아래에서
“확인”은 이 revision의 코드/문서에서 직접 확인됐다는 뜻이고, “미정”은 명시가 없다는 뜻이다.

## 확인된 계약

- [x] 제출물은 `policy.py`의 `nn.Module` architecture와 `checkpoint.pt` weights 두 파일이다.
- [x] 기본 policy signature는 `forward(vector, graphic) -> action`이다.
- [x] 팀별 호출은 5 agent observation dict를 받고, action은 같은 5개 key의 `float32[2]`다.
- [x] model 입력은 vector `(B,96)`, graphic `(B,11,96,96)` float32이며 환경 HWC를 loader가 CHW로 바꾼다.
- [x] action 계약은 `(dx,dy)∈[-1,1]`; upstream `CheckpointModel`은 출력을 clamp한다.
- [x] checkpoint는 raw state dict 또는 기본 key `policy_state`를 지원하고 `weights_only=True`로 읽는다.
- [x] loader의 기본 device는 CPU이며 호출자가 다른 device를 지정할 수 있다.
- [x] `BaseModel` 문서가 inference interface를 명시적으로 “Stateless”라고 정의한다.
- [x] match runner는 episode마다 model instance를 재생성하거나 reset hook을 호출하지 않는다.
- [x] 현재 `run_match`는 `env.agents`를 따라 팀 dict를 만들며 이 revision에서 `env.agents` 초기값은
  canonical `unit_0~9`다.
- [x] upstream `CheckpointModel` 자체는 `list(obs.keys())` 순서를 그대로 batch row로 사용한다.
- [x] observation vector 내부의 10 unit block 순서는 `unit_0~9` absolute order다.

## 공식 확인이 필요한 미정 항목

- [ ] **batch 순서 보장:** evaluator가 항상 각 팀을 local slot `0~4` 순서로 전달하는가? 현재
  구현에서는 그렇게 동작하지만 `BaseModel.act` API 계약에는 순서 보장이 없다.
- [ ] **custom BaseModel 제출 허용:** README는 이를 optional로 설명하지만 제출물이
  `policy.py + checkpoint.pt`라고도 명시한다. agent 이름 기반 canonical adapter를 공식 제출할 수 있는가?
- [ ] **stateful policy 허용 범위:** “Stateless interface” 밖의 RNN hidden state가 금지인지,
  단지 reset API가 없는 것인지 확인이 필요하다.
- [ ] **episode reset 신호:** model instance를 여러 경기에서 재사용할 때 정책에 reset/episode ID를
  전달하는 공식 방법이 있는가?
- [ ] **step inference 제한:** 팀당/step당 timeout 또는 latency budget이 있는가?
- [ ] **match wall-clock 제한:** 1경기 및 series 전체에 별도 timeout이 있는가?
- [ ] **모델 제한:** parameter 수, `checkpoint.pt` byte 크기, architecture/source file 크기 제한이 있는가?
- [ ] **실행 resource:** 공식 CPU/GPU 종류, GPU 제공 여부, VRAM/RAM/thread/process 제한은 무엇인가?
- [ ] **dependency 제한:** 허용 PyTorch/Python version과 추가 package/custom op 허용 범위는 무엇인가?
- [ ] **I/O와 격리:** network, filesystem read/write, subprocess, multi-thread 사용 정책은 무엇인가?
- [ ] **determinism:** 평가에서 sampling 허용 여부와 reproducibility 요구 seed 전달 방식은 무엇인가?
- [ ] **failure 판정:** 누락 action, NaN/Inf, 잘못된 dtype/shape, timeout/OOM 발생 시 몰수 규칙은 무엇인가?
- [ ] **winner 기준:** `competition.match.run_match`는 terminal `infos[agent]["winner"]`가 아니라
  누적 Unity shaping reward를 비교한다. 공식 순위 평가도 이 구현을 쓰는지, 실제 terminal winner를
  쓰는지 확인이 필요하다.
- [ ] **평가 seed/side:** 공개·비공개 seed 수, side swap, 동점/100점 조기 종료 처리 규칙은 무엇인가?

## 확인 전 프로젝트의 fail-closed 결정

| 미정 위험 | 현재 프로젝트 결정 |
| --- | --- |
| batch order | 모든 내부 경로에서 agent 이름으로 canonical batching하고 slot ID를 명시한다. |
| state/reset | RNN과 episode 간 hidden state를 사용하지 않는다. |
| latency/size | CPU-compatible 123,450-parameter reference를 기준으로 latency와 파일 크기를 계속 기록한다. |
| device/dependency | CPU fallback, torch+numpy만으로 실행 가능하게 유지한다. |
| determinism | 평가 action은 categorical argmax; sampling은 학습에만 사용한다. |
| I/O/security | policy inference에서 network, subprocess, 파일 write를 사용하지 않는다. |
| winner | 프로젝트 evaluator는 PREP-08대로 terminal `winner`를 유일한 승패 근거로 사용한다. |
| seed/side | paired seed와 side swap을 기본 평가 단위로 유지한다. |

## 승격 gate

RNN/stateful 모델 또는 batch-row 기반 slot embedding을 최종 제출 후보로 승격하려면 위의
batch/reset 두 질문이 공식 답변으로 닫혀야 한다. 공식 latency/size 수치가 나오면 reference
checkpoint의 약 0.50 MB, CPU 팀 추론 약 0.517 ms를 그 한도와 비교하고 checklist를 갱신한다.
