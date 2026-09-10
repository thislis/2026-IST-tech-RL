from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
import torch

from blackout_rl.representation import (
    AuxiliaryHeads, AuxiliaryTargetLogger, GlobalLocalMapEncoder, UnitEntityAttention, absorption_phase_features,
    auxiliary_losses, render_role_svg, select_auxiliary_targets, self_relative_positions,
    summarize_roles,
)


class RepresentationTests(unittest.TestCase):
    def test_relative_position_uses_team_local_self_slot(self) -> None:
        vector = torch.zeros(2, 96)
        for unit in range(10): vector[:, unit*9:unit*9+2] = torch.tensor([unit/10, unit/20])
        relative = self_relative_positions(vector, torch.tensor([2,2]), torch.tensor([0,1]))
        self.assertTrue(torch.allclose(relative[0,2], torch.zeros(2)))
        self.assertTrue(torch.allclose(relative[1,7], torch.zeros(2)))

    def test_absorption_phase_repeats_every_twenty_seconds(self) -> None:
        time = torch.tensor([1.0, 1.0-20/420, 1.0-5/420])
        phase = absorption_phase_features(time)
        self.assertTrue(torch.allclose(phase[0], phase[1], atol=1e-5))
        self.assertTrue(torch.allclose(phase[2], torch.tensor([1.0, 0.0]), atol=1e-5))

    def test_attention_and_global_local_encoder_shapes(self) -> None:
        attention = UnitEntityAttention(16, heads=4)
        self.assertEqual(attention(torch.randn(3,10,16), torch.tensor([0,4,9])).shape, (3,16))
        encoder = GlobalLocalMapEncoder(crop_size=9, output_dim=24)
        self.assertEqual(encoder(torch.randn(3,11,32,32), torch.tensor([[0.,0.],[.5,.5],[1.,1.]])).shape, (3,24))

    def test_auxiliary_heads_log_all_targets_and_selection_is_conservative(self) -> None:
        predictions = AuxiliaryHeads(8)(torch.randn(4,8))
        targets = {"role": torch.zeros(4), "holding_item": torch.ones(4),
                   "seconds_to_absorption": torch.zeros(4), "score_delta": torch.zeros(4)}
        self.assertEqual(set(auxiliary_losses(predictions, targets)), set(targets))
        selected = select_auxiliary_targets({
            "role": {"win_rate_delta": .1, "score_diff_delta": 1},
            "score_delta": {"win_rate_delta": .1, "score_diff_delta": -1},
        })
        self.assertEqual(selected, ("role",))

    def test_role_summary_and_svg_include_all_slots_and_metrics(self) -> None:
        records = [{"slot":0,"x":.1,"y":.2,"collected":True,"storage_visit":True},
                   {"slot":0,"x":.2,"y":.3,"died":True,"transformed":True}]
        metrics = summarize_roles(records)
        self.assertEqual(len(metrics), 5); self.assertEqual(metrics[0].collected, 1)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"roles.svg"; render_role_svg(metrics, output)
            text = output.read_text(); self.assertIn("slot 4", text); self.assertIn("polyline", text)

    def test_auxiliary_target_logger_round_trips_strict_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"aux.jsonl"; logger=AuxiliaryTargetLogger(path)
            values={"role":torch.tensor([0]),"holding_item":torch.tensor([1]),
                    "seconds_to_absorption":torch.tensor([2.]),"score_delta":torch.tensor([.1])}
            logger.log(step=3,targets=values,predictions=values)
            self.assertIn('"step": 3',path.read_text())


if __name__ == "__main__": unittest.main(verbosity=2)
