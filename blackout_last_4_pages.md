# BlackOut Competition PPT 마지막 4페이지 정리

## 8페이지 — 관측 공간 (Observation Space)

각 에이전트는 `vector`와 `graphic` 두 가지 관측값을 받습니다.

### vector — float32[N]

`N = 96`

| 항목 | 크기 | 설명 |
|---|---:|---|
| `pos_x, pos_y` | 2 | 정규화된 위치  (유닛 10개 각각) |
| `team_sign` | 1 | 아군 +1.0 / 적군 −1.0 |
| `holding_item (one-hot)` | 6 | None + 아이템 5종 |
| `Class (one-hot)` | 3 | 이 에이전트의 유닛 클래스 |
| `own_score` | 1 | 0~1 정규화 |
| `opp_score` | 1 | 0~1 정규화 |
| `time_left` | 1 | 0~1 정규화 |

### graphic — float32[H, W, C]

`H×W = 96×96  (resolution_scale=4 × 타일 24개)   C = 11`

| 채널 | 의미 |
|---:|---|
| 채널 0 | 빈 공간 |
| 채널 1 | 벽 |
| 채널 2 | 아군 창고 |
| 채널 3 | 적군 창고 |
| 채널 4 | 아군 유닛 |
| 채널 5 | 적군 유닛 |
| 채널 6 | 배터리 |
| 채널 7 | BuffSpeed |
| 채널 8 | DebuffSpeed |
| 채널 9 | BuffSize |
| 채널 10 | DebuffSize |

## 9페이지 — (모의) 대회 환경

제출 파일 및 매치 방식

### 제출 파일

| 제출 파일 | 설명 |
|---|---|
| `policy.py` | `nn.Module` 서브클래스 (모델 아키텍처) |
| `checkpoint.pt` | 학습된 가중치 |

### 정책 인터페이스

```python
class MyPolicy(nn.Module):
    def forward(self, vector, graphic):
        # vector  : (B, N)       float32
        # graphic : (B, C, H, W) float32
        # 반환값  : (B, 2)       float32
        #           (dx, dy) in [-1, 1]
        ...
```

팀당 다른 아키텍처 사용 가능 (각 팀의 `policy.py`를 동적으로 로드)

### 매치 방식

| 항목 | 내용 |
|---|---|
| 단일 매치 | `run_match()` 호출<br>결과: 0=A 승, 1=B 승, None=무승부 |
| 시리즈 | `run_series(n_matches=N)`<br>매 경기마다 팀 사이드 교체 |
| 사이드 교체 | 맵 위치 유불리를 제거하기 위해<br>매 경기마다 A/B 포지션 스왑 |
| 액션 공간 | `float32[2]  —  (dx, dy) in [−1, 1]`<br>에이전트당 1개 |
| 보상 | 학습 전용으로 반환.<br>대회 점수 집계와는 무관 |

## 10페이지 — Step 1 — 정책 정의

`nn.Module`을 상속하여 `forward(vector, graphic) → action` 구현

### `policy.py`

```python
# policy.py
import torch, torch.nn as nn

class MyPolicy(nn.Module):
    def __init__(self, vector_size: int, n_channels: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.mlp = nn.Sequential(
            nn.Linear(32*4*4 + vector_size, 256), nn.ReLU(),
            nn.Linear(256, 2),
            nn.Tanh(),
        )

    def forward(self, vector, graphic):
        # vector : (B, N)       graphic : (B, C, H, W)
        cnn_out = self.cnn(graphic).flatten(1)
        return self.mlp(torch.cat([vector, cnn_out], dim=1))
```

### 체크포인트 저장

```python
# 저장
torch.save(
    {"policy_state": model.state_dict()},
    "checkpoint.pt"
)

# 또는 순수 state_dict
torch.save(model.state_dict(),
           "checkpoint.pt")
```

### 주의사항

- graphic 입력은 (B, C, H, W) 형식 — env 출력(B,H,W,C)을 blackout-env가 자동 변환
- 기본 설정: vector_size=96,  n_channels=11
- action 출력 범위는 반드시 [−1, 1]

## 11페이지 — Step 2 — 매치 실행

`load_checkpoint  ·  run_match  ·  run_series`

```python
from blackout_env import BlackOutEnv, load_checkpoint, run_match, run_series
from policy import MyPolicy

vector_size, n_channels = 96, 11

# 모델 로드
model_a = load_checkpoint(MyPolicy, "team_a/checkpoint.pt",
                          state_dict_key="policy_state",
                          device="cuda",
                          vector_size=vector_size, n_channels=n_channels)
model_b = load_checkpoint(MyPolicy, "team_b/checkpoint.pt",
                          state_dict_key="policy_state",
                          device="cuda",
                          vector_size=vector_size, n_channels=n_channels)

env = BlackOutEnv(env_path="path/to/BlackOut.x86_64")

# ── 단일 매치 ──────────────────────────────────────────────────────────
result = run_match(env, model_a, model_b)
print(f"승자: {'A' if result.winner==0 else 'B' if result.winner==1 else '무승부'}")

# ── 시리즈 (best-of-10, 매 경기 사이드 교체) ──────────────────────────
series = run_series(env, model_a, model_b, n_matches=10)
print(f"A:{series.model_a_wins}승  B:{series.model_b_wins}승  무:{series.draws}")

env.close()
```
