import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from agent_league.trace import load_chunk
from blackout_v11.behavior import SituationRecorder
from blackout_v11.io import read_json


NAMES = [f"unit_{i}" for i in range(10)]


def observations():
    entities = np.zeros((10, 9), np.float32)
    entities[:, :2] = [[.15+.02*i, .15] for i in range(5)]+[[.85-.02*i, .85] for i in range(5)]
    entities[:, 3] = 1
    result = {}
    for i, name in enumerate(NAMES):
        side = i//5
        vector = np.zeros(96, np.float32)
        world = entities.copy()
        world[:, 2] = [1 if j//5 == side else -1 for j in range(10)]
        vector[:90] = world.ravel()
        vector[90+i%3] = 1
        vector[93:96] = [.2, .1, .9] if side == 0 else [.1, .2, .9]
        graphic = np.zeros((96, 96, 11), np.float32)
        graphic[..., 0] = 1
        for row, col, owner in ((80, 14, 0), (14, 80, 1)):
            graphic[row:row+4, col:col+4, 0] = 0
            graphic[row:row+4, col:col+4, 2 if side == owner else 3] = 1
        result[name] = dict(vector=vector, graphic=graphic)
    return result


def flags(value=False):
    return {n: value for n in NAMES}


class BehaviorEvidenceTests(unittest.TestCase):
    def test_alignment_missing_terminal_and_teacher_execution(self):
        before = observations()
        for item in before.values():
            e = item["vector"][:90].reshape(10, 9)
            e[1, 3:9] = 0
            e[1, 4] = 1
        after = copy.deepcopy(before)
        for item in after.values():
            e = item["vector"][:90].reshape(10, 9)
            e[0, 0] += .01
            e[0, 3:9] = 0
            e[0, 4] = 1
            e[1, 3:9] = 0
            e[1, 3] = 1
            e[2, :2] = [.8, .8]
            item["vector"][95] -= .01
        actions = {n: np.array([1., 0.], np.float32) for n in NAMES}
        actions["unit_1"][:] = 0
        with tempfile.TemporaryDirectory() as tmp:
            recorder = SituationRecorder(Path(tmp)/"behavior", {"behavior_chunk_steps": 1})
            recorder.start(before)
            recorder.observe(0, before, after, actions, {n: .1 if n == "unit_0" else 0 for n in NAMES},
                flags(), flags(), decision=dict(side=0, event=True, teacher_forced=True,
                    planned_actions=np.zeros((5, 2), np.float32), selected_goal=[1, 2, 3, 4, 5], selected_role=[0, 0, 0, 1, 1]),
                inference_seconds=[.01, .02])
            actions["unit_0"] = np.array([0., 1.], np.float32)
            recorder.observe(1, after, {}, actions, {n: -.2 if n == "unit_0" else 0 for n in NAMES}, flags(True), flags())
            reference = recorder.close(complete=True)
            self.assertEqual(reference, recorder.close())
            summary = read_json(reference["summary_path"])
            self.assertEqual(summary["steps"], 2)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["teacher_forced_team_steps"], 1)
            score = summary["situations"]["side0"]["score:winning"]
            self.assertEqual(score["unit_steps"], 10)
            self.assertEqual(score["movement_observed_steps"], 5)
            self.assertEqual(score["pickup_proxy"], 1)
            self.assertEqual(score["delivery_proxy"], 1)
            self.assertEqual(score["respawn_proxy"], 1)
            self.assertEqual(score["turn"], 1)
            index = read_json(Path(reference["trace_path"])/"index.json")
            first, last = [load_chunk(Path(reference["trace_path"])/x["path"]) for x in index["chunks"]]
            np.testing.assert_array_equal(first["before"][0, 0], before["unit_0"]["vector"])
            np.testing.assert_array_equal(first["after"][0, 0], after["unit_0"]["vector"])
            np.testing.assert_array_equal(first["actions"][0, 0], [1., 0.])
            np.testing.assert_array_equal(first["planned_actions"][0, :5], 0.)
            self.assertTrue(first["teacher_forced"][0, 0])
            self.assertFalse(last["after_present"][0])
            self.assertFalse(last["after_present_mask"].any())
            self.assertTrue(np.isnan(last["after"]).all())
            np.testing.assert_allclose(last["cumulative_reward"][0], [-.1, 0])
            with gzip.open(Path(reference["trace_path"])/"events.jsonl.gz", "rt") as stream:
                events = [json.loads(line) for line in stream]
            event = next(e for e in events if e["kind"] == "policy_decision")
            self.assertEqual(event["data"]["executed_actions"][0], [1., 0.])
            self.assertEqual(event["data"]["planned_actions"][0], [0., 0.])

    def test_trace_cap_keeps_all_step_summaries_and_incomplete_close(self):
        rng = np.random.default_rng(19)
        obs = observations()
        actions = {n: np.array([1., 0.], np.float32) for n in NAMES}
        with tempfile.TemporaryDirectory() as tmp:
            recorder = SituationRecorder(tmp, dict(behavior_trace_max_bytes=16384, behavior_chunk_steps=1))
            recorder.start(obs)
            for step in range(40):
                nxt = copy.deepcopy(obs)
                for item in nxt.values():
                    e = item["vector"][:90].reshape(10, 9)
                    e[:, :2] = rng.uniform(.1, .9, (10, 2)).astype(np.float32)
                    item["vector"][95] -= .001
                recorder.observe(step, obs, nxt, actions, {n: 0. for n in NAMES}, flags(), flags())
                obs = nxt
            reference = recorder.close(error="fixture interruption")
            summary = read_json(reference["summary_path"])
            self.assertEqual(summary["steps"], 40)
            self.assertEqual(sum(summary["league"]["sides"][0]["action_histogram"]), 200)
            self.assertTrue(summary["trace"]["capped"])
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["error"], "fixture interruption")
            self.assertLess(summary["trace"]["steps"], 40)
            self.assertLessEqual(sum(p.stat().st_size for p in Path(reference["trace_path"]).iterdir()), 16384)

    def test_central_path_metadata_and_partial_after(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = dict(id="attempt-1", kind="collect", arm="F", side=1, requested_seed=42,
                        action_seed=53, opponent=dict(id="v4", family="planner"),
                        trajectory=str(root/"run/attempt-1/trajectory"),
                        config=dict(behavior_output_root=str(root/"behavior")))
            recorder = SituationRecorder.for_task(task)
            obs = observations()
            recorder.start(obs)
            partial = {"unit_0": obs["unit_0"]}
            recorder.observe(0, obs, partial, {n: np.zeros(2, np.float32) for n in NAMES},
                             {n: 0. for n in NAMES}, flags(), flags())
            reference = recorder.close(error="partial observation")
            path = Path(reference["summary_path"])
            self.assertEqual(path.parent.parent, root/"behavior/episodes")
            summary = read_json(path)
            self.assertEqual(summary["metadata"]["opponent_id"], "v4")
            self.assertEqual(summary["metadata"]["side"], 1)
            self.assertEqual(summary["invalid_transitions"], 1)
            self.assertEqual(summary["situations"]["side0"]["score:winning"]["movement_observed_steps"], 0)


if __name__ == "__main__":
    unittest.main()
