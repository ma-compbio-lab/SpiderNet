# Training measurements: 2026-09-11

The optional implementation accelerated the multibatch synthetic test. Stable benefit was not established for the single-batch Simulation dataset: the final sequential measurement was 11.63% slower. Synthetic speedup must not be generalized to all real datasets.

## Measurements

Environment: Windows, RTX 4070 12 GB, PyTorch 2.0.0+cu117, interpreter `C:\Users\junji\miniconda3\envs\SpiderNet_env\python.exe`. Fit times are process-isolated medians; memory values are maximum peaks across repeats.

| Data / configuration | Version | Fit seconds | Sampled CPU peak MiB | GPU allocated peak MiB | GPU reserved peak MiB |
|---|---|---:|---:|---:|---:|
| Synthetic: 8 batches, 60 epochs, warmup 6, 2 repeats | Baseline | 5.259 | 2138.0 | 26.5 | 30.0 |
| Same | Optimized | 3.969 | 2135.1 | 27.8 | 32.0 |
| Simulation: 1 batch, 1000 epochs, warmup 100, 2 repeats | Baseline | 10.731 | 2147.6 | 61.2 | 72.0 |
| Same | Optimized | 11.979 | 2144.7 | 61.2 | 72.0 |

Synthetic elapsed time decreased **24.53%** (throughput ratio **1.3249x**). Simulation elapsed time increased **11.63%** (ratio **0.8958x**). CPU memory was similar; caches may increase GPU memory, with a default target-cache limit of 256 MiB.

GPU figures cover only the PyTorch allocator, including data/model and excluding CUDA context, drivers and other processes. CPU figures sample process-tree RSS every 10 ms rather than a continuous peak; JSON/CSV include OS lifetime peaks and individual measurements.

Simulation input: `D:\SpiderNet\Results\Simulation\SpiderNet\Sd_use3\Experiment_9\ProcessedData`, with 2000 cells, 20000 edges, 80 genes, 20 LR pairs, dim_envir=2 and hidden_channels=64. Initialization regression uses n_jobs=1 in both versions to limit RAM use.

**The machine was shared.** Two GPU Python processes already existed; GPU utilization was about 62% at sampling and 91% at completion. No other benchmark test ran concurrently during the final sequential check, but those external processes continued. Baseline fit times were **15.372 / 6.090 seconds**, optimized **16.703 / 7.256 seconds**. Large external-load variation prevents an exclusive-machine performance conclusion. Evaluate real training data on an idle GPU before choosing the optional implementation.

## Numerical consistency

The recorded **14 unit tests passed**, covering per-epoch CPU loss, gradients, frozen parameters, final weights, eight outputs, checkpoints, NMF, nondefault LR-loss weights, cache budgets and RNG restoration. Required bitwise checks passed. A tiny CPU command-line smoke test passed parameter/output/history checks but is not a performance measurement.

GPU final-parameter/output allclose checks **did not all pass** at rtol=1e-4 and atol=1e-6. The independent 200-epoch Simulation trajectory check reported:

- Maximum warmup-loss difference: 0.
- Maximum expression/LR/total-loss differences: approximately 1.073e-5 / 9.060e-6 / 4.053e-6; all passed the preset thresholds.
- Identical learning-rate and GradScaler-scale trajectories.
- At 1000 epochs, the two baseline/optimized weight comparisons had maximum absolute differences around **3.295e-4 / 1.121e-3**; baseline/baseline repeats reached **1.198e-3**.
- Sampled baseline/optimized output relative L2 differences were approximately **0.0337% / 0.0438%**, versus **0.1579%** between baseline repeats.

These observations are compatible with GPU scatter atomic-accumulation nondeterminism, but do not establish that it caused every difference, or guarantee equivalence at 30000 epochs or in biological conclusions. All state_dict tensors were compared; forward outputs were sampled per batch rather than compared in full. Longer checks can increase `--verify-epochs` and examine downstream results.

## Initialization cache

Initial Simulation `Initial_model` computation took **0.4595 seconds**, or **0.4970 seconds** including fingerprints and cache writes. A matching cache read, including content validation and RNG restoration, took **0.0225 seconds**. This repeat-initialization benefit is separate from fit speedup. First computation does not benefit; reuse requires matching input, configuration, source, environment and entering RNG state.

## Raw results

- Synthetic CUDA: report (`results/synthetic_cuda_20260911/report.md`, supplied in `training-benchmark-results`), JSON (`results/synthetic_cuda_20260911/report.json`, supplied in `training-benchmark-results`), CSV (`results/synthetic_cuda_20260911/runs.csv`, supplied in `training-benchmark-results`).
- Final sequential Simulation: report (`results/simulation_cuda_final_20260911/report.md`, supplied in `training-benchmark-results`), JSON (`results/simulation_cuda_final_20260911/report.json`, supplied in `training-benchmark-results`), CSV (`results/simulation_cuda_final_20260911/runs.csv`, supplied in `training-benchmark-results`), numerical comparisons (`results/simulation_cuda_final_20260911/correctness.json`, supplied in `training-benchmark-results`).
- Initial 200-epoch Simulation: report (`results/simulation_cuda_20260911/report.md`, supplied in `training-benchmark-results`); elapsed time increased 11.22%, with baseline times 1.663 / 2.705 seconds.
- Exploratory 1000-epoch Simulation, three repeats: report (`results/simulation_cuda_long_20260911/report.md`, supplied in `training-benchmark-results`); elapsed time decreased 12.91%. Its last measurement partially overlapped the CPU smoke test, so it is diagnostic and excluded from the final performance summary.
- CPU command-line smoke test: report (`results/synthetic_cpu_smoke_20260911/report.md`, supplied in `training-benchmark-results`).

These records include both faster and slower measurements. The measured baseline `D:\SpiderNet\SpiderNet_proj\SpiderNet_Project\SpiderNet\SpiderNet\model.py` had SHA-256 `5FB6D67B9191A971091B754900B00944AB8D9C010ACBF223036FCB847F70B5DA` before and after those benchmarks. See [README.md](README.md) for commands and measurement definitions.
