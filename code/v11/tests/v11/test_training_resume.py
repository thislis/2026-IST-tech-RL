import copy
import json
from pathlib import Path
import tempfile
import unittest
from blackout_v11.io import atomic_json
from blackout_v11.training import account_episode, repair_interruption_accounting


def state(**changes):
    data = dict(step=100, trained_steps=100, updates=1, wave=0, valid_episodes=0,
                invalid_episodes=0, consecutive_invalid=0, attempted_steps=0, discarded_steps=0,
                unmeasured_crash_attempts=0, interrupted_episodes=0, interrupted_steps=0,
                interrupted_completed_episodes=0, pending_wave=None)
    data.update(changes)
    return data


def row(root, serial, *, valid=False, interrupted=False, steps=10, error=None, error_type=None, wave=0):
    directory = Path(root)/"waves"/f"wave-{wave:06d}-123"/f"episode-{serial:08d}"
    directory.mkdir(parents=True)
    result = dict(valid=True, natural_terminal=True) if valid else None
    record = dict(id=directory.name, directory=str(directory), valid=valid, interrupted=interrupted,
                  attempt_steps=steps, step_count_complete=True, result=result, error=error)
    if error_type:
        atomic_json(directory/"error.json", dict(type=error_type, error=error))
        atomic_json(directory/"invalid.json", dict(type=error_type, error=error, steps=steps))
    atomic_json(directory/"attempt.json", record)
    return record


def ledger(root, rows):
    path = Path(root)/"episodes.jsonl"
    path.write_text("".join(json.dumps(item)+"\n" for item in rows))
    return path


class InterruptedEpisodeAccounting(unittest.TestCase):
    def test_deliberate_stop_does_not_add_or_clear_real_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            s = state(invalid_episodes=2, consecutive_invalid=2)
            stopped = row(temp, 0, interrupted=True, steps=7,
                          error="stopped; incomplete episode excluded", error_type="InterruptedError")
            self.assertEqual(account_episode(s, stopped), "interrupted")
            self.assertEqual(s["invalid_episodes"], 2)
            self.assertEqual(s["consecutive_invalid"], 2)
            self.assertEqual(s["interrupted_episodes"], 1)
            self.assertEqual(s["attempted_steps"], 7)
            self.assertEqual(s["discarded_steps"], 7)

    def test_completed_after_stop_is_discarded_once(self):
        with tempfile.TemporaryDirectory() as temp:
            s = state()
            stopped = row(temp, 0, valid=True, interrupted=True, steps=9)
            self.assertEqual(account_episode(s, stopped, discard_complete=True), "interrupted")
            self.assertEqual(s["interrupted_completed_episodes"], 1)
            self.assertEqual(s["discarded_steps"], 9)
            self.assertEqual(s["invalid_episodes"], 0)

    def test_stop_flag_cannot_hide_runtime_error_or_watchdog(self):
        with tempfile.TemporaryDirectory() as temp:
            s = state()
            bad = row(temp, 0, interrupted=True, error="nonfinite action", error_type="ValueError")
            watchdog = row(temp, 1, interrupted=True, error="external heartbeat watchdog", error_type="InterruptedError")
            unknown = row(temp, 2, interrupted=True, error="worker exit -9")
            for item in (bad, watchdog, unknown):
                self.assertEqual(account_episode(s, item), "invalid")
            self.assertEqual(s["invalid_episodes"], 3)
            self.assertEqual(s["consecutive_invalid"], 3)
            self.assertEqual(s["interrupted_episodes"], 0)

    def test_legacy_repair_is_exact_read_only_and_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            rows = [row(temp, 0, valid=True, steps=10),
                    row(temp, 1, valid=True, interrupted=True, steps=7),
                    row(temp, 2, interrupted=True, steps=11, error="stopped; incomplete episode excluded", error_type="InterruptedError"),
                    row(temp, 3, interrupted=True, steps=13, error="invalid action", error_type="ValueError")]
            path = ledger(temp, rows)
            checkpoint = Path(temp)/"original.pt"
            checkpoint.write_bytes(b"immutable saved weights optimizer rng")
            original = state(valid_episodes=1, invalid_episodes=3, consecutive_invalid=3,
                             attempted_steps=41, discarded_steps=48)
            before = copy.deepcopy(original)
            ledger_bytes = path.read_bytes()
            repaired, receipt = repair_interruption_accounting(temp, original, checkpoint)
            self.assertEqual(original, before)
            self.assertEqual(path.read_bytes(), ledger_bytes)
            self.assertEqual(checkpoint.read_bytes(), b"immutable saved weights optimizer rng")
            self.assertTrue(receipt["applied"])
            self.assertEqual(len(receipt["reclassified"]), 2)
            self.assertEqual(receipt["corrected_double_counted_steps"], 7)
            self.assertEqual(repaired["invalid_episodes"], 1)
            self.assertEqual(repaired["consecutive_invalid"], 1)
            self.assertEqual(repaired["interrupted_episodes"], 2)
            self.assertEqual(repaired["interrupted_steps"], 18)
            self.assertEqual(repaired["discarded_steps"], 41)
            for key in ("step", "trained_steps", "updates", "attempted_steps"):
                self.assertEqual(repaired[key], original[key])
            again, second_receipt = repair_interruption_accounting(temp, repaired, checkpoint)
            self.assertEqual(again, repaired)
            self.assertIsNone(second_receipt)

    def test_ambiguous_legacy_ledger_preserves_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            cancelled = row(temp, 0, interrupted=True, error="stopped", error_type="InterruptedError")
            ledger(temp, [cancelled])
            original = state(invalid_episodes=3, consecutive_invalid=3, discarded_steps=30)
            repaired, receipt = repair_interruption_accounting(temp, original)
            self.assertFalse(receipt["applied"])
            self.assertEqual(repaired["invalid_episodes"], 3)
            self.assertEqual(repaired["discarded_steps"], 30)
            self.assertEqual(repaired["interrupted_episodes"], 0)

    def test_pending_wave_is_not_migrated_before_recovery_accounting(self):
        with tempfile.TemporaryDirectory() as temp:
            complete = row(temp, 0, valid=True, wave=0)
            pending = row(temp, 1, interrupted=True, wave=1, error="stopped", error_type="InterruptedError")
            ledger(temp, [complete, pending])
            original = state(valid_episodes=1, wave=1, attempted_steps=10,
                             pending_wave=str(Path(pending["directory"]).parent))
            repaired, receipt = repair_interruption_accounting(temp, original)
            self.assertTrue(receipt["applied"])
            self.assertEqual(repaired["interrupted_episodes"], 0)
            account_episode(repaired, pending, discard_complete=True)
            self.assertEqual(repaired["interrupted_episodes"], 1)
            self.assertEqual(repaired["attempted_steps"], 20)
            self.assertEqual(repaired["discarded_steps"], 10)


if __name__ == "__main__":
    unittest.main()
