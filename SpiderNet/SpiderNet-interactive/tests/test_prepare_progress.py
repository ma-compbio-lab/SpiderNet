"""Preparation progress must remain readable while inference is running."""
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from flask import Flask
from torch_geometric.data import Data

from SpiderNet.api import build_model
from SpiderNet.config import TrainingConfig
from modules.module4_perturb import service as s
from modules.module4_perturb.routes import bp
from core import loaders


class PrepareProgressTests(unittest.TestCase):
    def setUp(self):
        self.ds = SimpleNamespace(name="progress-test")
        s._STATE_CACHE.pop(self.ds.name, None)
        s._PREPARE_PROGRESS.pop(self.ds.name, None)

    def tearDown(self):
        s._STATE_CACHE.pop(self.ds.name, None)
        s._PREPARE_PROGRESS.pop(self.ds.name, None)

    def test_progress_endpoint_does_not_wait_for_preparation(self):
        entered, release = threading.Event(), threading.Event()
        state = {"slice_infos": [None, None], "device": "cpu"}
        def slow_build(*args, **kwargs):
            s._prepare_progress(self.ds, "Predicting baseline: slice 1/2", completed=0, total=2)
            entered.set()
            release.wait(3)
            return state
        app = Flask(__name__)
        app.config["DATASETS"] = {self.ds.name: self.ds}
        app.register_blueprint(bp)
        with patch.object(s, "_build_state", side_effect=slow_build) as build:
            worker = threading.Thread(target=s.get_state, args=(self.ds,))
            worker.start()
            try:
                self.assertTrue(entered.wait(2))
                started = time.monotonic()
                response = app.test_client().get(f"/dataset/{self.ds.name}/perturb/api/prepare-progress")
                self.assertLess(time.monotonic() - started, 1)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json["total"], 2)
                self.assertEqual(response.json["state"], "loading")
                self.assertEqual(response.headers["Cache-Control"], "no-store")
            finally:
                release.set()
                worker.join(3)
            self.assertFalse(worker.is_alive())
            self.assertEqual(s.get_prepare_progress(self.ds)["state"], "ready")
            self.assertIs(s.get_state(self.ds), state)
            self.assertEqual(build.call_count, 1)

    def test_failure_is_visible_and_retry_resets_progress(self):
        with patch.object(s, "_build_state", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                s.get_state(self.ds)
        self.assertEqual(s.get_prepare_progress(self.ds)["state"], "error")
        self.assertFalse(s.is_state_loaded(self.ds))
        with patch.object(s, "_build_state", return_value={"slice_infos": [], "device": "cpu"}):
            s.get_state(self.ds)
        self.assertEqual(s.get_prepare_progress(self.ds)["state"], "ready")

    def test_shared_loader_reports_steps_and_reuses_same_objects(self):
        self.ds.processed_dir = self.ds.run_dir = Path("unused-test-path")
        adata, graphs, factors = [object()], [object()], [object()]
        messages = []
        with patch.object(loaders, "_BUNDLE_CACHE", {}), \
             patch.object(loaders.pd, "read_pickle", side_effect=[adata, factors]) as read, \
             patch.object(loaders, "_load_pyg_list", return_value=graphs) as load:
            first = loaders.get_core_bundle(self.ds, progress=messages.append)
            second = loaders.get_core_bundle(self.ds, progress=messages.append)
            self.assertIs(first, second)
            self.assertIs(first["pyg_list"], graphs)
            self.assertEqual(read.call_count, 2)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(len(messages), 4)

    def test_baseline_progress_and_cache_reuse_without_duplicate_data_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.ds.root = self.ds.run_dir = self.ds.processed_dir = self.ds.model_dir = root
            self.ds.version = "V1"
            self.ds.setup = {"SPECIES": "human"}
            self.ds.config = {}
            pd.to_pickle(np.array(["A", "B", "C"]), root / "genenames_train.pkl")
            pd.to_pickle(["A-B"], root / "LR_list.pkl")
            data = Data(x=torch.ones(2, 3), cell_class_onehot=torch.eye(2),
                        edge_index=torch.tensor([[0, 1], [1, 0]]), cell_class=["A", "B"])
            raw = [data, data.clone()]
            processed = SimpleNamespace(genenames_train=np.array(["A", "B", "C"]),
                                        lr_list=["A-B"], spidernet_data=raw)
            model = build_model(processed, TrainingConfig(dim_envir=2), "cpu", hidden_channels=64)
            torch.save(model.state_dict(), root / "model_epoch1.pth")
            bundle = {"pyg_list": raw, "adata_list": None}
            messages = []
            original_report = s._prepare_progress
            def report(ds, message, **kwargs):
                messages.append((message, kwargs))
                original_report(ds, message, **kwargs)
            with patch.object(s, "get_core_bundle", return_value=bundle) as load, \
                 patch.object(s, "_load_loading_bundle", return_value={"mi_list": ["MI-1", "MI-2"]}), \
                 patch.object(s, "_select_device", return_value="cpu"), \
                 patch.object(s, "_prepare_progress", side_effect=report), \
                 patch("SpiderNet.io.load_processed_data", side_effect=AssertionError("Duplicate data load")):
                state = s.get_state(self.ds)
                self.assertIs(state["raw_data_list"], raw)
                self.assertTrue(callable(load.call_args.kwargs["progress"]))
                counts = [kw["completed"] for msg, kw in messages if "completed" in kw]
                self.assertEqual(counts, [0, 1, 1, 2])
                self.assertTrue(any("compressing cache" in msg for msg, _ in messages))
                s._STATE_CACHE.pop(self.ds.name)
                with patch.object(s, "_infer_predictions_for_slices", side_effect=AssertionError("Cache must be reused")):
                    cached = s.get_state(self.ds)
                for idx in range(2):
                    np.testing.assert_array_equal(state["baseline_pred_by_slice"][idx], cached["baseline_pred_by_slice"][idx])
                self.assertTrue(any("from cache" in msg for msg, _ in messages))


if __name__ == "__main__":
    unittest.main()
