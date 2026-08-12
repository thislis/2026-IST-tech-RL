#!/usr/bin/env python3
"""Render role differentiation from the recorded live scripted trajectory."""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from blackout_rl.representation import render_role_svg,summarize_roles

source=ROOT/"logs/base_s08_s10_role_fsm_trajectory.jsonl"; records=[]
previous_class={slot:0 for slot in range(5)}
for line in source.read_text().splitlines():
    row=json.loads(line)
    if row.get("record_type") != "step": continue
    events=row.get("events",[])
    for slot in range(5):
        agent=f"unit_{slot}"; state=row["unit_state"][agent]; class_id=int(state["class_id"])
        kinds=[event["kind"] for event in events if event.get("agent")==agent]
        records.append({"slot":slot,"x":state["position_normalized"][0],"y":state["position_normalized"][1],
                        "collected":"battery_pickup" in kinds,"died":"death" in kinds,
                        "transformed":class_id != previous_class[slot],
                        "storage_visit":"battery_deposit" in kinds})
        previous_class[slot]=class_id
metrics=summarize_roles(records); render_role_svg(metrics,ROOT/"reports/phase3_role_differentiation.svg")
(ROOT/"logs/phase3_role_metrics.json").write_text(json.dumps([m.__dict__|{"path":list(m.path)} for m in metrics],indent=2)+"\n")
print("rendered",len(records),"slot observations")
