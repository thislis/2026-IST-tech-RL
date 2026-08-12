# BASE-S13 scripted trajectory recorder

구현은 [`blackout_rl/trajectory.py`](../blackout_rl/trajectory.py)에 있다. 메모리를 누적하지 않고
한 줄씩 flush하는 versioned JSONL 형식이며 schema version은
`blackout.scripted_trajectory.v1`이다.

## 기록 계약

- header: seed, team, policy id, game/Python API commit, executable SHA-256
- step: 팀 5명의 96-vector observation, `(dx,dy)` action, role, target, 위치, 보유 item,
  class, reward, score/time, assignment·pickup·stuck·replan event
- graphic observation: 팀 공통 one-hot semantic map을 `uint8` semantic id로 바꾼 뒤
  `zlib+base64` 압축; shape/encoding을 함께 저장
- footer: record 수, 종료 상태, 완료 step과 검증 결과

synthetic round-trip에서 vector 길이, 필수 field, 역할/목표, semantic map 복원을 검증했다.
실제 seed `240513` 실행은 31개 step record를 생성했으며 파일은 33줄, 147,253 bytes다.
evidence에 저장된 SHA-256과 파일을 다시 계산한 값도 일치한다.

- trajectory: [`logs/base_s13_coordination_trajectory.jsonl`](../logs/base_s13_coordination_trajectory.jsonl)
- 실행 요약: [`logs/base_s04_s07_s13_coordination.json`](../logs/base_s04_s07_s13_coordination.json)
