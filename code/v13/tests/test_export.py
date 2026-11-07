import importlib.util
import tempfile
import unittest
import torch
from blackout_v13.policy import Policy
from blackout_v13.export import export_policy


class ExportTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('onnx') and importlib.util.find_spec('onnxruntime'),'ONNX tools required')
    def test_full_graph(self):
        torch.set_num_threads(1);torch.manual_seed(15)
        with tempfile.TemporaryDirectory() as directory:
            result=export_policy(Policy(),directory)
            self.assertTrue(result['full_graph'])
            self.assertEqual(result['parity_steps'],8)


if __name__=='__main__':unittest.main()
