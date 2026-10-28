"""Regresiones: impedir inferencia silenciosa con pipeline incorrecto y cache de otro teacher."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)

CH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CH.parent))
sys.path.insert(0, str(CH/"model/submission"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trainer = load_module("audit_trainer", CH/"model/official_gpu/train_official_gpu_v1.py")
exporter = load_module("audit_exporter", CH/"model/submission/build_submission.py")
evaluator = load_module("audit_evaluator", CH/"model/submission/eval_ensemble.py")


class PipelineGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def checkpoint(self, span):
        model = nn.ModuleDict({"core": nn.LSTM(112, 4, num_layers=2, batch_first=True),
                               "head": nn.Linear(4, 2)})
        path = self.base/f"span{span}.pt"
        torch.save({"state_dict": model.state_dict(), "config": {"hidden": 4, "layers": 2, "cell": "lstm"},
                    "n_inputs": 112, "pipeline": {"mean": np.zeros(112, np.float32),
                    "std": np.ones(112, np.float32), "adapt_span": span}}, path)
        return path

    def test_export_rejects_adaptive_normalizer(self):
        with self.assertRaisesRegex(ValueError, "adapt_span=8000"):
            exporter.load_any(self.checkpoint(8000))

    def test_teacher_rejects_adaptive_normalizer(self):
        with self.assertRaisesRegex(ValueError, "adapt_span=8000"):
            trainer.load_teacher(self.checkpoint(8000), torch.device("cpu"))

    def test_evaluator_rejects_adaptive_normalizer(self):
        with self.assertRaisesRegex(ValueError, "adapt_span=8000"):
            evaluator.load_model(self.checkpoint(8000), torch.device("cpu"))

    def test_static_pipeline_keeps_export_and_teacher_parity(self):
        path = self.checkpoint(0)
        sd, hidden, mean, std, clip, cell = exporter.load_any(path)
        exported = exporter.make_net(sd, hidden, cell)
        teacher, mu, scale, tc = trainer.load_teacher(path, torch.device("cpu"))
        x = torch.randn(1, 101, 112)
        with torch.no_grad():
            expected = exported["head"](exported["core"](x)[0])
            actual = teacher["head"](teacher["core"]((x-mu)/scale)[0])
        torch.testing.assert_close(actual, expected)

    def test_same_basename_different_content_invalidates_teacher_cache(self):
        a, b = self.base/"a/best_model.pt", self.base/"b/best_model.pt"
        a.parent.mkdir(); b.parent.mkdir()
        a.write_bytes(b"teacher A"); b.write_bytes(b"teacher B")
        self.assertNotEqual(trainer._teacher_cache_key([a]), trainer._teacher_cache_key([b]))

    def test_moved_checkpoint_keeps_cache_identity(self):
        a, b = self.base/"old.pt", self.base/"new.pt"
        a.write_bytes(b"same teacher"); b.write_bytes(a.read_bytes())
        self.assertEqual(trainer._teacher_cache_key([a]), trainer._teacher_cache_key([b]))

    def test_overwritten_checkpoint_invalidates_cache(self):
        a = self.base/"best_model.pt"
        a.write_bytes(b"epoch 1")
        key = trainer._teacher_cache_key([a])
        a.write_bytes(b"epoch 2")
        self.assertNotEqual(key, trainer._teacher_cache_key([a]))


if __name__ == "__main__":
    unittest.main()
