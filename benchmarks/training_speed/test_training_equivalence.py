"""Tiny CPU regression tests for the isolated SpiderNet training optimization.

Run with the project's SpiderNet environment:
    python -B benchmarks/training_speed/test_training_equivalence.py

These tests never use real datasets or GPU memory. They compare the original
implementation with the optimized subclass from identical model/input states.
"""

from __future__ import annotations

import contextlib
import copy
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import warnings

# Hide CUDA before importing torch: the baseline uses CUDA autocast/GradScaler
# even when fit(device="cpu") is requested.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch_geometric.data import Data

import spidernet_optimized as optimized_module
from spidernet_optimized import (
    OptimizationOptions,
    load_baseline_module,
    make_optimized_model_class,
)


torch.set_num_threads(1)
with mock.patch.object(torch.cuda, "is_available", return_value=False):
    BASE = load_baseline_module()
OPTIMIZED = make_optimized_model_class(BASE)


def tiny_problem():
    """Different graph sizes/degrees, duplicate receiver indices, isolated nodes."""
    rng = np.random.default_rng(1407)
    edges = (
        np.array([[0, 1], [0, 2], [1, 2], [2, 0], [2, 1]], dtype=np.int64),
        np.array(
            [[0, 1], [1, 0], [1, 2], [2, 3], [3, 1], [3, 4], [4, 0], [0, 4]],
            dtype=np.int64,
        ),
    )
    batches = []
    for num_cells, edge_index in zip((5, 7), edges):
        labels = np.arange(num_cells) % 3
        batches.append(
            Data(
                x=torch.tensor(rng.uniform(0.05, 0.9, (num_cells, 5)), dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                cell_class_onehot=torch.tensor(np.eye(3)[labels], dtype=torch.float32),
                cellpair_LRpair_neigh=torch.tensor(
                    rng.uniform(0.02, 0.8, (len(edge_index), 4)), dtype=torch.float32
                ),
                num_cells=num_cells,
            )
        )
    initial = {
        "loading_intrinsic_init": rng.uniform(0.02, 0.3, (3, 5)).astype(np.float32),
        "loading_receiver_init": rng.uniform(0.02, 0.3, (2, 5)).astype(np.float32),
        "loading_sender_init": rng.uniform(0.02, 0.3, (2, 5)).astype(np.float32),
        "loading_LR_init": rng.uniform(0.02, 0.3, (2, 4)).astype(np.float32),
        "factor_GP_minibatchNMF": [
            rng.uniform(0.1, 0.9, (len(edge), 2)).astype(np.float32) for edge in edges
        ],
        # Provide a complete manual initialization for the NMF model. The old
        # Initial_model implementation is deliberately not part of this test.
        "factor_intrinsic_init": [batch.cell_class_onehot.numpy().copy() for batch in batches],
    }
    return batches, initial


def make_model(cls, factor_mode="cell_class"):
    return cls(
        num_gene=5,
        num_LR=4,
        hidden_channels=8,
        Factor_mode=factor_mode,
        dim_intri=3,
        dim_envir=2,
    ).cpu()


def scalar(value):
    return float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)


def tracing_class(parent):
    class Traced(parent):
        def compute_loss_in_batches(self, *args, **kwargs):
            result = super().compute_loss_in_batches(*args, **kwargs)
            self.test_trace.append(
                {
                    "losses": tuple(scalar(value) for value in result),
                    "requires_grad": {name: p.requires_grad for name, p in self.named_parameters()},
                    "gradients": {
                        name: None if p.grad is None else p.grad.detach().clone()
                        for name, p in self.named_parameters()
                    },
                }
            )
            return result

    return Traced


class TrainingEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batches, cls.initial = tiny_problem()

    def setUp(self):
        # This CUDA build can report is_available=True even with zero visible
        # devices. Make both original and optimized autocast/GradScaler take
        # their genuine disabled CPU path for the duration of each test.
        patcher = mock.patch.object(torch.cuda, "is_available", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def assert_tensor_identical(self, actual, expected, label):
        self.assertEqual(actual.dtype, expected.dtype, label)
        self.assertEqual(actual.shape, expected.shape, label)
        if not torch.equal(actual, expected):
            delta = (actual - expected).abs().max().item()
            self.fail(f"{label}: tensors are not bitwise equal; max_abs_error={delta:.9g}")

    def assert_model_identical(self, actual, expected):
        self.assertEqual(list(actual.state_dict()), list(expected.state_dict()))
        for name, parameter in expected.state_dict().items():
            self.assert_tensor_identical(actual.state_dict()[name], parameter, name)
        actual.eval()
        expected.eval()
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for batch_index, batch in enumerate(self.batches):
                actual_output, expected_output = actual(batch), expected(batch)
                for index, (left, right) in enumerate(zip(actual_output, expected_output)):
                    self.assert_tensor_identical(left, right, f"batch {batch_index} output {index}")
                    self.assertTrue(torch.isfinite(left).all().item())

    def _run_fit(self, model, *, options=None, record_history=True):
        model.test_trace = []
        schedulers = []
        scheduler_type = BASE.ReduceLROnPlateau

        def recording_scheduler(optimizer, *args, **kwargs):
            scheduler = scheduler_type(optimizer, *args, **kwargs)
            record = {
                "initial_lr": optimizer.param_groups[0]["lr"],
                "kwargs": kwargs.copy(),
                "steps": [],
            }
            schedulers.append(record)
            original_step = scheduler.step

            def step(metric, *step_args, **step_kwargs):
                record["steps"].append((len(model.test_trace) - 1, scalar(metric)))
                return original_step(metric, *step_args, **step_kwargs)

            scheduler.step = step
            return scheduler

        with tempfile.TemporaryDirectory(prefix="spidernet_equivalence_") as temporary:
            kwargs = dict(
                SpiderNet_data_pyg_list=self.batches,
                device="cpu",
                optim_type="adam",
                lr=2e-4,
                weight_decay=1e-5,
                LR_loss_weight=2.75,
                warmup=3,
                max_epoch=30,
                loss_fn="mse",
                Initial_dict=copy.deepcopy(self.initial),
                file_savepath_model_main=temporary,
            )
            if options is not None:
                kwargs.update(options=options, record_history=record_history)
            with mock.patch.object(BASE, "ReduceLROnPlateau", side_effect=recording_scheduler), \
                    mock.patch.object(optimized_module, "ReduceLROnPlateau", side_effect=recording_scheduler):
                with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                    warnings.simplefilter("ignore")
                    result = model.fit(**kwargs)
            self.assertIsNone(result, "fit must retain its original return semantics")
            checkpoint_paths = sorted(Path(temporary).glob("model_epoch*.pth"))
            self.assertEqual([path.name for path in checkpoint_paths], ["model_epoch0.pth", "model_epoch29.pth"])
            checkpoints = {path.name: torch.load(path, map_location="cpu") for path in checkpoint_paths}
        return schedulers, checkpoints

    def _assert_full_trajectory(self, factor_mode, options):
        torch.manual_seed(76543)
        baseline = make_model(tracing_class(BASE.SpiderNet_model), factor_mode)
        optimized = make_model(tracing_class(OPTIMIZED), factor_mode)
        optimized.load_state_dict(copy.deepcopy(baseline.state_dict()))
        torch.manual_seed(301)
        baseline_schedulers, baseline_checkpoints = self._run_fit(baseline)
        torch.manual_seed(301)
        optimized_schedulers, optimized_checkpoints = self._run_fit(optimized, options=options)

        self.assertEqual(len(baseline.test_trace), 30)
        self.assertEqual(len(optimized.test_trace), 30)
        for epoch, (left, right) in enumerate(zip(optimized.test_trace, baseline.test_trace)):
            self.assertEqual(left["requires_grad"], right["requires_grad"], f"epoch {epoch} freeze schedule")
            for index, (left_loss, right_loss) in enumerate(zip(left["losses"], right["losses"])):
                self.assertEqual(left_loss, right_loss, f"epoch {epoch} loss {index}")
            for name, expected_gradient in right["gradients"].items():
                actual_gradient = left["gradients"][name]
                if expected_gradient is None:
                    self.assertIsNone(actual_gradient, f"epoch {epoch} {name} must not acquire a gradient")
                else:
                    self.assertIsNotNone(actual_gradient, f"epoch {epoch} {name} lost gradient")
                    self.assert_tensor_identical(actual_gradient, expected_gradient, f"epoch {epoch} gradient {name}")

        self.assertEqual(optimized_schedulers, baseline_schedulers)
        self.assertEqual(len(optimized_schedulers), 2, "scheduler must be rebuilt at the warmup boundary")
        self.assertEqual([item["initial_lr"] for item in optimized_schedulers], [5e-4, 2e-4])
        self.assertEqual([epoch for epoch, _ in optimized_schedulers[1]["steps"]], [10, 20])
        for epoch, metric in optimized_schedulers[1]["steps"]:
            _, exp_loss, lr_loss = optimized.test_trace[epoch]["losses"]
            self.assertEqual(metric, exp_loss + lr_loss, "scheduler must keep the original unweighted sum")
            self.assertNotAlmostEqual(metric, exp_loss + 2.75 * lr_loss, places=8)

        history = optimized.training_history
        self.assertEqual(len(history), 30)
        for epoch, row in enumerate(history):
            self.assertEqual(row["epoch"], epoch)
            warmup_loss, exp_loss, lr_loss = baseline.test_trace[epoch]["losses"]
            for key, expected in (("loss_warmup", warmup_loss), ("loss_exp", exp_loss), ("loss_LR", lr_loss)):
                self.assertEqual(scalar(row[key]), expected, f"history epoch {epoch} {key}")
            self.assertEqual(scalar(row["loss_total"]), warmup_loss if epoch < 3 else exp_loss + lr_loss)
            self.assertEqual(scalar(row["lr"]), 5e-4 if epoch < 3 else 2e-4)
            self.assertEqual(scalar(row["scale"]), 1.0, "CPU GradScaler must be disabled")

        for filename, expected in baseline_checkpoints.items():
            for name, parameter in expected.items():
                self.assert_tensor_identical(optimized_checkpoints[filename][name], parameter, f"{filename}: {name}")
        self.assert_model_identical(optimized, baseline)

    def test_cell_class_complete_training_and_all_gradients(self):
        self._assert_full_trajectory("cell_class", OptimizationOptions())

    def test_nmf_complete_training_and_three_way_freeze_schedule(self):
        self._assert_full_trajectory("NMF", OptimizationOptions())

    def test_zero_target_cache_budget_preserves_training(self):
        self._assert_full_trajectory("cell_class", OptimizationOptions(target_cache_max_bytes=0))

    def test_disabled_optimizations_preserve_original_training(self):
        self._assert_full_trajectory(
            "cell_class",
            OptimizationOptions(defer_metrics=False, warmup_factor_only=False, cache_targets=False, cache_degrees=False),
        )

    def test_production_fit_without_history_preserves_training(self):
        """Default fast fit must preserve training while skipping unused metrics."""
        torch.manual_seed(2026)
        baseline = make_model(tracing_class(BASE.SpiderNet_model))
        optimized = make_model(tracing_class(OPTIMIZED))
        optimized.load_state_dict(copy.deepcopy(baseline.state_dict()))
        expected_schedulers, expected_checkpoints = self._run_fit(baseline)
        actual_schedulers, actual_checkpoints = self._run_fit(
            optimized, options=OptimizationOptions(), record_history=False
        )
        self.assertEqual(optimized.training_history, [])
        self.assertEqual(actual_schedulers, expected_schedulers)
        for epoch, (actual, expected) in enumerate(zip(optimized.test_trace, baseline.test_trace)):
            self.assertEqual(actual["requires_grad"], expected["requires_grad"])
            for name, expected_gradient in expected["gradients"].items():
                if expected_gradient is None:
                    self.assertIsNone(actual["gradients"][name], name)
                else:
                    self.assert_tensor_identical(
                        actual["gradients"][name], expected_gradient, f"fast fit epoch {epoch}: {name}"
                    )
        # Epoch 1 has neither logging nor a scheduler update; its diagnostic
        # loss is intentionally not copied to the host in the optimized path.
        self.assertEqual(optimized.test_trace[1]["losses"], (0.0, 0.0, 0.0))
        for filename, expected in expected_checkpoints.items():
            for name, parameter in expected.items():
                self.assert_tensor_identical(actual_checkpoints[filename][name], parameter, f"fast fit {filename}: {name}")
        self.assert_model_identical(optimized, baseline)

    def test_target_conversion_cache_budget_and_mutation(self):
        """CPU fit uses float32; explicitly exercise exact half target caching."""
        optimized = make_model(OPTIMIZED)
        original = self.batches[0].x.clone()
        exact_half_bytes = original.numel() * 2
        optimized.optimization_options = OptimizationOptions(target_cache_max_bytes=exact_half_bytes)
        first = optimized._target(original, torch.float16)
        second = optimized._target(original, torch.float16)
        self.assert_tensor_identical(first, original.to(torch.float16), "cached half conversion")
        self.assertEqual(first.data_ptr(), second.data_ptr(), "a repeated static target should hit cache")
        self.assertEqual(optimized.optimization_stats["target_cache_bytes"], exact_half_bytes)
        different = original.clone().add_(0.17)
        fallback = optimized._target(different, torch.float16)
        self.assert_tensor_identical(fallback, different.to(torch.float16), "over-budget fallback")
        self.assertEqual(optimized.optimization_stats["target_cache_bytes"], exact_half_bytes)
        self.assertGreater(optimized.optimization_stats["target_cache_budget_skips"], 0)
        original.add_(0.25)
        self.assert_tensor_identical(
            optimized._target(original, torch.float16), original.to(torch.float16), "mutated input invalidates stale target"
        )
        optimized.clear_optimization_cache()
        optimized.optimization_options = OptimizationOptions(target_cache_max_bytes=0)
        self.assert_tensor_identical(
            optimized._target(original, torch.float16), original.to(torch.float16), "zero-budget conversion"
        )
        self.assertEqual(optimized.optimization_stats["target_cache_bytes"], 0)
        self.assertEqual(optimized.optimization_stats["target_cache_entries"], 0)

    def test_direct_warmup_and_main_losses_and_gradients(self):
        """Exercise the public batch-loss API without fit or history recording."""
        for factor_mode in ("cell_class", "NMF"):
            for stage in ("warmup", "main"):
                with self.subTest(factor_mode=factor_mode, stage=stage):
                    torch.manual_seed(892)
                    baseline = make_model(BASE.SpiderNet_model, factor_mode)
                    optimized = make_model(OPTIMIZED, factor_mode)
                    optimized.load_state_dict(copy.deepcopy(baseline.state_dict()))
                    optimized.optimization_options = OptimizationOptions()
                    targets = [torch.tensor(item) for item in self.initial["factor_GP_minibatchNMF"]]
                    kwargs = dict(stage=stage, criterion=torch.nn.MSELoss(), scaler=None, loss_weight=2.75)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        expected = baseline.compute_loss_in_batches(self.batches, targets, **kwargs)
                        actual = optimized.compute_loss_in_batches(self.batches, targets, **kwargs)
                    self.assertEqual(tuple(map(scalar, actual)), tuple(map(scalar, expected)))
                    for (name, left), (_, right) in zip(optimized.named_parameters(), baseline.named_parameters()):
                        if right.grad is None:
                            self.assertIsNone(left.grad, name)
                        else:
                            self.assertIsNotNone(left.grad, name)
                            self.assert_tensor_identical(left.grad, right.grad, f"{factor_mode} {stage}: {name}")

    def test_clear_cache_and_new_graph_preserve_predictions(self):
        torch.manual_seed(195)
        baseline = make_model(BASE.SpiderNet_model)
        optimized = make_model(OPTIMIZED)
        optimized.load_state_dict(copy.deepcopy(baseline.state_dict()))
        optimized.optimization_options = OptimizationOptions()
        self.assert_model_identical(optimized, baseline)
        optimized.clear_optimization_cache()
        self.assert_model_identical(optimized, baseline)
        # A new graph with the same number of cells must not reuse another
        # graph's degrees; the final two nodes remain completely isolated.
        new_graph = self.batches[0].clone()
        new_graph.edge_index = torch.tensor([[0, 1], [0, 2], [1, 0]], dtype=torch.long)
        new_graph.cellpair_LRpair_neigh = new_graph.cellpair_LRpair_neigh[:3].clone()
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for index, (left, right) in enumerate(zip(optimized(new_graph), baseline(new_graph))):
                self.assert_tensor_identical(left, right, f"new graph output {index}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
