"""Small process/registration checks, without game or learning execution."""
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
from blackout_v12 import runner
from blackout_v12.io import read_json
from blackout_v12.runtime import FileLease, managed_popen, process_options, process_tree_guard


class RuntimeTests(unittest.TestCase):
    def test_managed_child_starts_inside_process_guard(self):
        with process_tree_guard():
            child = managed_popen([sys.executable,'-c','print("guarded")'],
                                  stdout=subprocess.PIPE,stderr=subprocess.PIPE,**process_options())
            output,error = child.communicate(timeout=10)
            self.assertEqual(child.returncode,0,error.decode())
            self.assertEqual(output.strip(),b'guarded')

    def test_live_lease_and_stale_file_are_distinguished(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertFalse(runner.is_active(directory))
            with FileLease(directory/'experiment.lock', blocking=False):
                self.assertTrue(runner.is_active(directory))
            self.assertFalse(runner.is_active(directory))

    def test_changed_registration_is_refused_and_original_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with patch.object(runner, 'registration', return_value={'version':1}):
                initial = runner.prepare_registration(directory,{})
                self.assertEqual(initial, runner.prepare_registration(directory,{}))
            with patch.object(runner, 'registration', return_value={'version':2}):
                with self.assertRaises(ValueError):
                    runner.prepare_registration(directory,{})
            self.assertEqual(read_json(directory/'registration.json'), {'version':1})


if __name__ == '__main__':
    unittest.main()
