# Versioned data bundles

The Git repository contains package/source code and documentation. Research data, complete saved figures, UI datasets, and optional detailed tables are supplied as ZIP64 archives. **The archives have been uploaded to a restricted Zenodo draft and are available to editors and reviewers through the private link supplied in the review manuscript and journal submission materials.** The dataset will be made publicly available upon publication of the paper. The dataset DOI and public download URLs will then be added to the repository; automatic downloads are not yet enabled.

## Profiles

Use `python scripts/data.py list` to see exact bundle dependencies. `plot-<study>` restores one command-line workflow; `all-plots` restores all eight default plot-only workflows. `interactive` restores the packaged UI datasets. `optional-details` restores the 15 detailed CSV tables excluded from default plot-only requirements. `all-data` also includes training-benchmark records and caches.

`results-*` bundles preserve each study's saved output directory. `inputs-*` bundles provide additional upstream inputs required by the original default plot-only preflight. A few large graph/AnnData bundles are necessary because the original plotting workflow reads or validates them; no reduced scientific substitutes were created.

## Files

| Bundle ID | Download size | Restored files |
|---|---:|---:|
| `training-benchmark-results` | 0.501 GiB | 341 |
| `interactive-other` | 0.000 GiB | 5 |
| `interactive-agingbrain` | 2.464 GiB | 56 |
| `interactive-hgsoc` | 1.649 GiB | 177 |
| `results-ablation-study` | 0.001 GiB | 32 |
| `results-agingbrain` | 1.285 GiB | 275 |
| `results-coupling-benchmark` | 0.001 GiB | 20 |
| `results-hgsoc` | 2.091 GiB | 630 |
| `results-pancancer` | 4.197 GiB | 863 |
| `results-perturbfish` | 0.103 GiB | 205 |
| `results-robustness-stability` | 0.013 GiB | 1394 |
| `results-simulation` | 0.008 GiB | 81 |
| `details-agingbrain` | 0.090 GiB | 6 |
| `details-hgsoc` | 0.112 GiB | 6 |
| `details-perturbfish` | 1.404 GiB | 3 |
| `inputs-agingbrain` | 1.957 GiB | 29 |
| `inputs-perturbfish` | 0.361 GiB | 3 |
| `inputs-simulation` | 0.003 GiB | 11 |
| `inputs-coupling-benchmark` | 0.000 GiB | 6 |
| `inputs-pancancer` | 0.429 GiB | 475 |

Sizes are binary GiB. Archives may store repeated identical content once; the restore tool places it at every path required by the manifest. All files have SHA-256 checksums. The archives do not contain Git histories, editor state or disposable build products.

## Use

During peer review, open the private link provided with the submission and download the ZIP files listed for the desired profile by `python scripts/data.py list`. For all eight default plot-only workflows, use the `all-plots` profile. Keep the original ZIP filenames in one local directory and pass that directory as `--archive-dir`; use this repository's `data/manifest.json` for restoration. No Zenodo account or individual access request is needed when using the supplied reviewer link.

From the repository root:

```bash
python scripts/data.py restore --profile plot-agingbrain --archive-dir /path/to/archives
python scripts/data.py verify --profile plot-agingbrain
python scripts/run_study.py AgingBrain --check --plot-only
```

For the locally prepared distribution, the archive directory is `../Zenodo/uploads`. Once links are published, download using `python scripts/data.py fetch --profile plot-agingbrain`, then restore from the default local download cache.

Files are restored to `Tutorial/<study>/output`, `SpiderNet/Interactivetool`, benchmark result/cache folders, or `.spidernet/workspace`. Original execution records containing machine-specific paths are preserved under `.spidernet/provenance` rather than activated as the identity of a new run. These locations are ignored by Git.

The restore tool verifies the complete archive hash and every restored file. It skips identical existing files and stops on conflicting data by default. A `verify` failure after running analysis may indicate a newly generated output rather than archive damage; retain the immutable archives as reference results.

## Publication and version matching

`data/manifest.json` contains bundle IDs, immutable archive filenames, byte sizes, SHA-256 values, per-file restore paths, and profile membership. The current bundle status is `private_peer_review`; public URL and DOI fields remain unset until public release. The Zenodo-uploaded manifest has the same archive checksums, file inventory, and restore paths; its publication-status text predates the upload. Use the repository manifest for current availability information.

At public release, set each bundle's public `url`, version-specific `doi`, and `status`, and record the matching repository tag in `code_tag`. Private review links are supplied separately and are not included in the public catalog. Keep archive bytes unchanged: editing a ZIP invalidates its checksum. A data update needs a new release/version and corresponding checksums.

The prepared data supports the documented saved-result replays. It is not a claim that every raw-data source, database, historical notebook alternative or comparator needed for full retraining is included. See [reproduction scope](REPRODUCIBILITY.md) and each study README.
