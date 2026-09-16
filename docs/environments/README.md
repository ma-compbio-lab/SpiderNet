# Historical environment records

`legacy-development-freeze.txt` preserves the dependency versions recorded during development. The previous root and inner `requirements-full-freeze.txt` files contained the same package list, differing only in text formatting; they are consolidated here.

This is an archival record, not a validated lockfile or installation recipe. It includes conflicting optional dependency versions. Do not pass it to `pip install -r`.

For installation, use the [current installation guide](../../README.md#installation). The root `requirements*.txt` files forward to the maintained lists in `SpiderNet/`. PyTorch/PyG still require the separate platform-specific setup described in that guide.
