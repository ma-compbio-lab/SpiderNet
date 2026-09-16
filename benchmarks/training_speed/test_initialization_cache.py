"""Small cache equivalence tests; no real NMF or training is performed."""

from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from initialization_cache import capture_rng_state, initialize_with_cache, restore_rng_state


class FakeModule:
    device = "previous-device"

    def __init__(self):
        self.calls = 0

    def Initial_model(self, data, dim_envir, Factor_mode="cell_class", dim_intri=None,
                      n_jobs=10, Initial_regression="Linear", enhance_init_with_gene_coexp=True,
                      threshold_crosscorr=0.3, numtop_crosscorr=10, spearcorr_use_rowmax_threshold=0.3):
        self.calls += 1
        assert self.device == str(data[0]["x"].device)
        return {"sample": np.array([random.random(), np.random.rand(), torch.rand(()).item()]),
                "input": data[0]["x"].cpu().numpy().copy(), "dim_envir": dim_envir}


class InitializationCacheTests(unittest.TestCase):
    def setUp(self):
        random.seed(93)
        np.random.seed(93)
        torch.manual_seed(93)
        self.original_rng = capture_rng_state()
        self.folder = tempfile.TemporaryDirectory()
        self.directory = Path(self.folder.name)
        self.module = FakeModule()
        self.data = [{"x": torch.tensor([[1., 2.], [3., 4.]]),
                      "edge_index": torch.tensor([[0, 1], [1, 0]]),
                      "cellpair_LRpair_neigh": torch.tensor([[0.5], [0.75]]),
                      "num_cells": 2, "cell_class_onehot": torch.eye(2)}]
        self.config = {"dim_envir": 2, "Factor_mode": "cell_class"}

    def tearDown(self):
        self.folder.cleanup()
        restore_rng_state(self.original_rng)

    def run_cache(self, **kwargs):
        return initialize_with_cache(self.module, self.data, config=self.config,
                                     cache_dir=self.directory, **kwargs)

    def next_samples(self):
        return random.random(), np.random.rand(), torch.rand(3)

    def test_hit_matches_output_and_all_following_rng_streams(self):
        initial_rng = capture_rng_state()
        first, miss = self.run_cache()
        expected_next = self.next_samples()
        restore_rng_state(initial_rng)
        second, hit = self.run_cache()
        actual_next = self.next_samples()
        self.assertEqual(self.module.calls, 1)
        self.assertFalse(miss["hit"])
        self.assertTrue(hit["hit"])
        self.assertEqual(miss["key"], hit["key"])
        self.assertEqual(self.module.device, "previous-device")
        self.assertEqual(hit["compute_s"], 0.0)
        np.testing.assert_array_equal(first["sample"], second["sample"])
        self.assertEqual(expected_next[:2], actual_next[:2])
        self.assertTrue(torch.equal(expected_next[2], actual_next[2]))

    def test_data_config_and_incoming_rng_changes_miss(self):
        initial_rng = capture_rng_state()
        _, original = self.run_cache()
        restore_rng_state(initial_rng)
        self.data[0]["x"][0, 0] += 1
        _, changed_input = self.run_cache()
        restore_rng_state(initial_rng)
        self.config["dim_envir"] = 3
        _, changed_config = self.run_cache()
        # No RNG restoration here: this is a different incoming random stream.
        _, changed_rng = self.run_cache()
        self.assertEqual(self.module.calls, 4)
        self.assertEqual(len({row["key"] for row in (original, changed_input, changed_config, changed_rng)}), 4)
        self.assertFalse(any(row["hit"] for row in (original, changed_input, changed_config, changed_rng)))

    def test_disabled_runs_exact_initializer_and_writes_nothing(self):
        initial_rng = capture_rng_state()
        first, metadata = self.run_cache(enabled=False)
        self.assertEqual(list(self.directory.iterdir()), [])
        expected_rng = self.next_samples()
        restore_rng_state(initial_rng)
        cached, _ = self.run_cache()
        actual_rng = self.next_samples()
        np.testing.assert_array_equal(first["sample"], cached["sample"])
        self.assertEqual(expected_rng[:2], actual_rng[:2])
        self.assertTrue(torch.equal(expected_rng[2], actual_rng[2]))
        self.assertFalse(metadata["enabled"])
        self.assertIsNone(metadata["key"])
        self.assertIsNone(metadata["path"])
        self.assertEqual(len(list(self.directory.glob("*.pt"))), 1)

    def test_corrupt_output_is_recomputed_with_original_rng(self):
        initial_rng = capture_rng_state()
        first, miss = self.run_cache()
        path = Path(miss["path"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["initial_dict"]["sample"][0] += 1
        torch.save(payload, path)
        restore_rng_state(initial_rng)
        with self.assertWarnsRegex(RuntimeWarning, "checksum"):
            second, metadata = self.run_cache()
        self.assertFalse(metadata["hit"])
        self.assertEqual(self.module.calls, 2)
        np.testing.assert_array_equal(first["sample"], second["sample"])
        self.assertEqual(list(self.directory.glob("*.tmp")), [])

    def test_source_change_misses_even_with_identical_data_and_rng(self):
        initial_rng = capture_rng_state()
        source = self.directory / "initializer.py"
        source.write_text("# Version one\n", encoding="utf-8")
        self.module.__file__ = str(source)
        _, original = self.run_cache()
        restore_rng_state(initial_rng)
        source.write_text("# Version two\n", encoding="utf-8")
        _, updated = self.run_cache()
        self.assertFalse(updated["hit"])
        self.assertNotEqual(original["key"], updated["key"])
        self.assertEqual(self.module.calls, 2)

    def test_corrupt_rng_is_recomputed_without_changing_next_random_values(self):
        initial_rng = capture_rng_state()
        _, miss = self.run_cache()
        expected_next = self.next_samples()
        path = Path(miss["path"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["post_rng"]["torch_cpu"][0] ^= 1
        torch.save(payload, path)
        restore_rng_state(initial_rng)
        with self.assertWarnsRegex(RuntimeWarning, "RNG state failed its checksum"):
            _, metadata = self.run_cache()
        actual_next = self.next_samples()
        self.assertFalse(metadata["hit"])
        self.assertEqual(self.module.calls, 2)
        self.assertEqual(expected_next[:2], actual_next[:2])
        self.assertTrue(torch.equal(expected_next[2], actual_next[2]))


if __name__ == "__main__":
    unittest.main()
