# MAPPO vs `win_70_vs_scripted.pt` Training

## Objective

Train a decentralized MAPPO actor against the immutable
`checkpoints/win_70_vs_scripted.pt` opponent. Evaluation uses the five committed
dev seeds and swaps physical sides, producing ten held-out games. A requested
85% threshold therefore requires at least 9/10 wins; draws do not count.

## Start and resume

```bash
./scripts/train_mappo_vs_win70.sh
```

The foreground process prints one line per PPO update. `Ctrl-C` safely writes
`checkpoints/mappo_vs_win70_latest.pt`. Resume it with:

```bash
./scripts/train_mappo_vs_win70.sh --resume-latest
```

Command-line arguments appended to the wrapper override its defaults. For
example, a shorter engineering run is:

```bash
./scripts/train_mappo_vs_win70.sh \
  --max-env-steps 100000 \
  --eval-every 10000 \
  --rollout-steps 256
```

## Defaults and artifacts

- learner initialization: `checkpoints/phase3_selfplay_gen_08.pt`;
- opponent: frozen SHA-validated `win_70_vs_scripted.pt`, including its planner
  override metadata;
- training sides alternate every update;
- train seeds: 3001–3008; model-selection seeds: 3101–3105;
- reward: competitive score delta + terminal result + `0.05` Unity shaping;
- 512 environment steps per rollout, four PPO epochs, 0.03 target KL;
- evaluation every 25,000 steps and resumable save every 5,000 steps;
- maximum initial budget: 2,000,000 environment steps.

Generated artifacts:

| Artifact | Meaning |
| --- | --- |
| `checkpoints/mappo_vs_win70_latest.pt` | resumable actor, centralized critic, optimizer, and run state |
| `checkpoints/mappo_vs_win70_best.pt` | best dev win rate, then best score difference |
| `checkpoints/mappo_win_85_vs_win70.pt` | written only after the held-out target passes |
| `logs/mappo_vs_win70/training.jsonl` | PPO diagnostics for every update |
| `logs/mappo_vs_win70/eval_step_*.json` | full paired episode evidence |
| `logs/mappo_vs_win70/run_summary.json` | latest/best evaluation and target status |

The ordinary checkpoint loader sees only the decentralized actor. Training
resume additionally restores `mappo_state` and `mappo_optimizer_state`, which
contain the centralized critic and optimizer. Test seeds 3201–3210 are not used
for stopping or model selection. `--eval-seeds` may select a dev subset for a
quick engineering check, but a subset is marked promotion-ineligible and can
never create the 85% target checkpoint.

Exit status is 0 when the target is reached, 2 when the configured budget ends
without reaching it, and 130 after a handled interrupt.
