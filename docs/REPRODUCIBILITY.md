# Reproduction instructions

## 1. Use the matching environment

Install the package from the repository root. Python 3.11 is required. Follow the [root installation guide](../README.md#installation) for PyTorch/PyG and the optional UI and benchmark requirements. `Tutorial/R_requirements.md` describes R dependencies. The recorded full freeze is a historical environment snapshot with conflicting optional dependency versions; do not install from it as a lockfile.

The optional requirements and package extras constrain the NumPy/igraph/PySAL stack to compatible versions. Run `python -m pip check` and the documented import checks after installation. Successful dependency and import checks do not establish numerical agreement of a complete analysis; verify the relevant study outputs separately.

Do not replace method settings, fitted models, random seeds, or normalization to resolve an environment error. Saved pickle, PyG and checkpoint objects require compatible dependencies.

## 2. Restore an explicit data profile

List profiles using `python scripts/data.py list`. During peer review, editors and reviewers can download the required ZIP files through the private Zenodo link supplied in the review manuscript and journal submission materials. Keep the ZIP filenames unchanged and use the directory containing your downloads as `--archive-dir`. Public download URLs will be added when the dataset is publicly released:

```bash
python scripts/data.py restore --profile plot-hgsoc --archive-dir /path/to/archives
python scripts/data.py verify --profile plot-hgsoc
```

After publication, run `python scripts/data.py fetch --profile plot-hgsoc` and then restore without `--archive-dir`. Use the supplied tool instead of manually unpacking: a bundle may store identical bytes once and restore them to more than one required location.

The checksum manifest binds the data to this release. Verification after a workflow may report files changed by intentional output generation; the immutable ZIP retains the saved reference result. Restore refuses conflicting existing files by default. Use a separate checkout when comparing newly generated results with the reference.

## 3. Check and run one study

The portable entry point sets filesystem locations and invokes the unchanged workflow options:

```bash
python scripts/run_study.py HGSOC --check --plot-only
python scripts/run_study.py HGSOC --plot-only
```

`--check` validates the workflow's declared inputs/environment and relevant cache signatures. It is not a complete numerical or rendered-figure comparison. Plot-only can still be expensive, especially for full-resolution spatial images.

Alternatively, activate paths once and keep the original commands. From the repository root in PowerShell:

```powershell
. ./scripts/activate_reproduction.ps1
cd Tutorial/HGSOC
python run_benchmarks.py --check --plot-only
python run_benchmarks.py --plot-only
```

Or from Bash:

```bash
source scripts/activate_reproduction.sh
cd Tutorial/HGSOC
python run_benchmarks.py --check --plot-only
python run_benchmarks.py --plot-only
```

The activation affects the current shell only. It points input/output variables at this checkout and does not edit another analysis directory. Windows path handling is included for long result filenames. Keep checkout paths short when third-party R, plotting, or shell tools impose their own limits.

## 4. Interpret the outputs correctly

The numerical source tables, models and stored figures remain unchanged in the archives. Rendering can vary with fonts, graphics libraries and platform. Some workflows explicitly retain existing figures; their logs distinguish copied artifacts from redraws. The default Simulation replay includes its representative sample, not every raw simulation replicate or every possible `--setting`/`--experiment` combination.

## Full analysis and training

Default plot-only profiles are not complete raw-data training packages. For a full run, follow the relevant study README, obtain the original datasets and reference databases, install external comparators where applicable, and set that workflow's existing path overrides. Some analysis modes train missing fits or contact external services. Use `--check` in the intended mode before starting. This distribution does not silently substitute fresh results for missing published results.

The portable profiles and eight-study checks apply to the command-line workflows in `Tutorial/`. Source notebooks may still require their documented path overrides when executed directly rather than through a study runner.

## Interactive application

Restore `--profile interactive`, install `SpiderNet/requirements-UI.txt`, and follow `SpiderNet/SpiderNet-interactive/README.md`. Dataset discovery uses the restored `SpiderNet/Interactivetool/SpiderNet-interactive_V2/` directory. UI caches and saved inputs are data assets, not part of the Python wheel.
