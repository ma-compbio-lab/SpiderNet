# SpiderNet training benchmark

An optional optimized subclass loaded from `SpiderNet/SpiderNet/model.py`, preserving parameter names, architecture and checkpoint format. The package model/API and analysis notebooks are independent of this benchmark.

## Usage

Run from this directory in `SpiderNet_env`:

```powershell
python -B benchmark_training.py --synthetic --device cuda --epochs 200 --warmup 20 --repeats 3 --verify-epochs 50
python -B benchmark_training.py --data "D:\SpiderNet\Results\PerturbFISH\ProcessedData" --dim-envir 23 --device cuda --epochs 200 --warmup 20 --repeats 3 --verify-epochs 50
python -B benchmark_training.py --data "D:\SpiderNet\Results\Simulation\SpiderNet\Sd_use3\Experiment_9\ProcessedData" --dim-envir 2 --n-jobs 1 --device cuda --epochs 1000 --warmup 100 --repeats 3 --verify-epochs 200
```

The existing interpreter `C:\Users\junji\miniconda3\envs\SpiderNet_env\python.exe` can be used directly if conda activation is unavailable. Outputs default to a timestamped directory under `runs/`; `--output` must name a new directory. Recorded measurements are under `results/`. No package reinstallation is required. For timing, use an idle GPU and sufficient system RAM; the runner records other GPU processes without terminating them.

Use `--device cpu` for CPU runs. CUDA runs already record CPU RAM and GPU VRAM. Because the baseline constructs a CUDA GradScaler by default, CPU mode disables CUDA AMP symmetrically for both versions and records that setting.

## Implementation and measurement

- `spidernet_optimized.py`: optional fit implementation inheriting the baseline constructor. Loss reads are deferred until scheduling, printing or validation requires them; Python floats accumulate in batch order. Warmup computes only the environmental branch used by its loss, with the same epoch count. Fixed target dtype conversions are cached within a default 256 MiB budget; over-budget conversions use the baseline path, and warmup caches are released afterward. Input precision and target rounding are preserved. Fixed neighbor counts use the existing dtype, scatter_sum and true_divide_; encoder outputs are not cached. Gradients are cleared at epoch starts and training completion.
- `initialization_cache.py`: optional `Initial_model` cache keyed by input content, source, parameters, runtime and pre-initialization RNG state. Hits restore the post-initialization RNG state; computation and readback costs are separate.
- `benchmark_training.py`: process isolation, timing, memory sampling and comparisons. The optimized subprocess clears CUDA_LAUNCH_BLOCKING before importing torch; baseline inherits it by default. Global environment settings are not modified.
- `test_training_equivalence.py` and `test_initialization_cache.py`: equivalence, cache invalidation/corruption and RNG checks. Run with `python -B -m unittest discover -s . -p "test_*.py" -v`.

Both versions use identical data, Initial_dict, starting state_dict and RNG state, batch order, and one optimizer step per epoch. Measurements use fresh processes in alternating AB/BA order (three repeats by default), reporting medians, ranges and raw values. Two baseline forward/backward prewarm passes make no parameter updates; state/RNG are then restored and allocator caches cleared. Optimized cache construction counts toward fit time.

Fit time includes tensor transfer, warmup, training, optimizer/scheduler and checkpoints at the same frequency. It excludes imports, data loading, Initial_model, shared prewarm and comparisons. CPU RSS is sampled for the process tree every 10 ms and may miss brief peaks; OS process-lifetime peak RSS also includes import/loading/prewarm. GPU allocated/reserved peaks describe the PyTorch allocator during fit, including resident data/model, but excluding drivers, CUDA context and other processes.

Initialization timing is reported separately. Synthetic inputs use fixed initialization and do not test NMF/regression or initialization caching. Short benchmarks default to 10% warmup for both versions; a 200-epoch check does not establish 20,000-epoch or biological equivalence.

## Outputs and controls

`report.md`, `report.json` and `runs.csv` report timing, memory, parameters, source hashes, environment and raw measurements. `correctness.json` compares every state_dict tensor and samples each of eight forward outputs per batch (up to 4096 evenly spaced elements each by default), plus per-epoch loss, learning rate and GradScaler scale in verification runs. `preparation.json` records input size and initialization timing. Per-process logs and checkpoint directories are retained.

Default tolerances `rtol=1e-4`, `atol=1e-6` are numerical checks, not proof of equivalent effects. GPU scatter is nondeterministic, so baseline repeats are also compared. Investigate allclose failures using parameters, outputs, loss/LR/scale and baseline variation; do not loosen thresholds simply to pass.

Optional controls are `--no-defer-metrics`, `--no-warmup-pruning`, `--no-target-cache`, `--target-cache-mib`, `--no-degree-cache`, `--no-init-cache`, `--init-cache`, `--baseline-source`, and `--prewarm 0`. `--baseline-launch-blocking 0` isolates code effects; `--keep-launch-blocking` retains the optimized process's inherited setting. `--baseline-launch-blocking 1` creates a diagnostic synchronized baseline and does not describe the original environment. Disabling optimizations is not an exact baseline because unused scalar reads and duplicate Python calls remain removed; the baseline always calls the source model directly.

For longer verification, match the real epochs, warmup, MI dimension, hidden channels, optimizer and loss settings, and increase `--verify-epochs`. This consumes training time and checkpoint space. For direct use, load `load_baseline_module()`, construct the class with `make_optimized_model_class`, and pass `OptimizationOptions` to `fit`. Initialization caching requires an explicit `initialize_with_cache` call (the benchmark supplies it). Do not mutate fixed data during fit; use `model.clear_optimization_cache()` when changing data.

See [recorded measurements and limitations](BENCHMARK_RESULTS.md).
