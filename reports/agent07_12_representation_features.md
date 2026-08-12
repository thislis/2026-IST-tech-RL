# AGENT-07/09/10/12 Representation Features

- Self-relative positions use canonical team/slot identity and return all ten
  relative coordinates.
- Absorption features encode the normalized 300-second clock as a repeating
  20-second sin/cos phase.
- Auxiliary heads and strict JSONL logging cover role, held item, seconds to
  absorption, and score delta.
- `reports/phase3_role_differentiation.svg` visualizes 325 live trajectory slot
  observations. `logs/phase3_role_metrics.json` records paths, collection,
  death, transformation, and storage visits. Slots 0/1/2 collected 3/2/3
  batteries; guard slot 3 transformed once; carrier slot 4 transformed once,
  collected two, and visited storage twice. No death occurred in this episode.

Attention and global/local crop modules are implemented and shape-tested, but
AGENT-06/08 remain open until trained held-out ablations are recorded. Auxiliary
loss selection likewise remains open until an actual win-rate experiment exists.
