"""Model restoration checks with small synthetic checkpoints; no training."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.data import Data

from SpiderNet.api import build_model
from SpiderNet.config import TrainingConfig
from modules.module4_perturb import service as s


class ModelLoadingTests(unittest.TestCase):
    def setUp(self):
        self.processed = SimpleNamespace(
            genenames_train=np.array(["A", "B", "C"]), lr_list=["A-B"],
            spidernet_data=[{"cell_class_onehot": torch.ones(2, 2)}],
        )
        self.config = TrainingConfig(dim_envir=2)
        self.data = Data(x=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                         edge_index=torch.tensor([[0, 1], [1, 0]]),
                         cell_class_onehot=torch.eye(2))

    def load(self, state_dict, cfg=None):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model.pth"
            torch.save(state_dict, path)
            return s._load_trained_model(self.processed, self.config, cfg or {}, path, "cpu")

    def test_different_widths_restore_all_weights_and_predictions(self):
        for width in (64, 128, 256):
            with self.subTest(width=width):
                original = build_model(self.processed, self.config, "cpu", hidden_channels=width).eval()
                loaded = self.load(original.state_dict())
                self.assertEqual(loaded.hidden_channels, width)
                self.assertFalse(loaded.training)
                for name, value in original.state_dict().items():
                    self.assertTrue(torch.equal(value, loaded.state_dict()[name]), name)
                with torch.inference_mode():
                    expected, actual = original(self.data), loaded(self.data)
                for left, right in zip(expected, actual):
                    self.assertTrue(torch.equal(left, right))
                    self.assertTrue(torch.isfinite(right).all())

    def test_wrapped_checkpoint_and_explicit_configuration(self):
        model = build_model(self.processed, self.config, "cpu", hidden_channels=64)
        loaded = self.load({"model_state_dict": model.state_dict()}, {"hidden_channels": 64})
        self.assertEqual(loaded.hidden_channels, 64)
        with self.assertRaisesRegex(ValueError, "checkpoint requires 64"):
            self.load(model.state_dict(), {"HIDDEN_CHANNELS": 256})

    def test_missing_or_extra_weights_are_rejected(self):
        model = build_model(self.processed, self.config, "cpu", hidden_channels=64)
        state = model.state_dict()
        del state["loading_sender_ori"]
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            self.load(state)
        state = model.state_dict()
        state["unexpected_parameter"] = torch.zeros(1)
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            self.load(state)

    def test_mismatched_features_are_rejected(self):
        model = build_model(self.processed, self.config, "cpu", hidden_channels=64)
        self.processed.genenames_train = np.array(["A", "B", "C", "D"])
        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            self.load(model.state_dict())

    def test_invalid_architecture_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported checkpoint"):
            s._checkpoint_hidden_channels({"enc_factor_envir_pre_receiver.2.weight": torch.zeros(3, 7)}, {})
        with self.assertRaisesRegex(ValueError, "missing the encoder"):
            s._checkpoint_hidden_channels({}, {})


if __name__ == "__main__":
    unittest.main()
