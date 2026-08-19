# AGENT-14 Navigation Shaping Ablation

AGENT-14 was executed as a live, equal-budget Unity experiment. Both arms
started from `checkpoints/phase3_repr_flatten_legacy.pt`, used train seed 3001,
four 128-step PPO updates (512 environment steps total), and the same five
held-out dev seeds with side swapping (10 games per arm).

| Arm | Dev wins | Win rate | Mean score difference |
| --- | ---: | ---: | ---: |
| score + terminal only | 0/10 | 0.00 | -89.7 |
| navigation potential, linearly annealed | 0/10 | 0.00 | -94.3 |

The navigation weight followed `0.1875, 0.125, 0.0625, 0.0` at the four update
boundaries. Thus the final reward contract contains no residual navigation
weight. PPO remained finite in both arms, but navigation shaping did not improve
held-out sample efficiency at this budget: its mean score difference was 4.6
points worse. It was not promoted, and the selected output is
`checkpoints/phase3_agent14_score_terminal_only.pt`.

Evidence: [`logs/phase3_agent14_navigation.json`](../logs/phase3_agent14_navigation.json)
and [`scripts/run_phase3_agent14_navigation.py`](../scripts/run_phase3_agent14_navigation.py).

