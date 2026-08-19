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

Attention, global/local crop, and four independent auxiliary losses were subsequently
trained with a common 1,500-gradient-step budget and evaluated on five side-swapped
dev seeds. None improved closed-loop win rate or score difference, so no candidate was
promoted. Full evidence is in `agent06_08_11_representation_ablation.md` and
`logs/phase3_representation_ablation.json`.
