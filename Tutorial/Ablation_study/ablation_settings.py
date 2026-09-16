"""Paths and unchanged model settings for component-ablation analyses."""
from pathlib import Path
from collections import OrderedDict
import json
import os

ANALYSIS_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = ANALYSIS_DIR / "output"


def workspace_root():
    override = os.environ.get("SPIDERNET_WORKSPACE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    for candidate in (ANALYSIS_DIR, *ANALYSIS_DIR.parents):
        if (candidate / "Results").is_dir():
            return candidate
    # Saved-result plotting also works in a standalone copy without upstream data.
    # Full-run preflight reports the exact missing inputs and workspace override.
    return ANALYSIS_DIR


WORKSPACE_ROOT = workspace_root()
# Optional package checkout (the directory containing SpiderNet/model.py).
SPIDERNET_PROJECT_DIR = os.environ.get("SPIDERNET_PROJECT_DIR")
ORIGINAL_MODEL_PY = os.environ.get("SPIDERNET_MODEL_PY")

DATASET_CONFIG = {
    "HGSOC": {
        "display_name": "HGSOC",
        "dim_envir": 15,
        "version": "V1",
        "output_root": (WORKSPACE_ROOT / "Results/HGSOC"),
        "processed_data_dir": (WORKSPACE_ROOT / "Results/HGSOC/ProcessedData"),
        "cellclass_name": "cell.types",
        "omnipath_regulatory_csv": (WORKSPACE_ROOT / "Data/Database/OmnipathR/interactions_regulatory_human.csv"),
        "targets_json_name": "targets_by_LR.json",
        "min_upstream_regulators": 5,
        # If run_dirs.json exists and points to a valid full run directory, it is used.
        "prefer_run_dirs_json": True,
    },
    "AgingMousebrain": {
        "display_name": "Aging mouse brain",
        "dim_envir": 30,
        "version": "V1",
        "output_root": (WORKSPACE_ROOT / "Results/AgingBrain"),
        "processed_data_dir": (WORKSPACE_ROOT / "Results/AgingBrain/ProcessedData"),
        "cellclass_name": "celltype",
        "omnipath_regulatory_csv": (WORKSPACE_ROOT / "Data/Database/OmnipathR/interactions_regulatory_mouse.csv"),
        "targets_json_name": "targets_by_LR.json",
        "min_upstream_regulators": 1,
        "prefer_run_dirs_json": False,
    },
}

RETRAIN_ABLATIONS = False


RUN_TRAINING_IF_MISSING = True


TRAIN_ABLATION_IDS = ["no_lr_reinit", "no_gene_reinit", "no_intrinsic_reinit"]


ABLATION_SOURCE_NOTEBOOK = {
    "full": "pretrained_full",
    "no_lr_reinit": "sklearn_compat",
    "no_gene_reinit": "sklearn_compat",
    "no_intrinsic_reinit": "reinit_ablation",
}


STUDY_TRAINING_HPARAMS = {
    "HGSOC": {
        "HIDDEN_CHANNELS": 64,
        "WARMUP_EPOCHS": 2000,
        "MAX_EPOCHS": 20000,
        "LEARNING_RATE": 1e-4,
        "WEIGHT_DECAY": 1e-5,
        "LOSS_FN": "mse",
        "OPTIM_TYPE": "adam",
        "ENHANCE_INIT_WITH_GENE_COEXP": False,
        "INITIAL_REGRESSION": "Linear",
        "N_JOBS_INIT": 5,
        "RANDOM_SEED": 123,
    },
    "AgingMousebrain": {
        "HIDDEN_CHANNELS": 256,
        "WARMUP_EPOCHS": 2000,
        "MAX_EPOCHS": 20000,
        "LEARNING_RATE": 1e-4,
        "WEIGHT_DECAY": 1e-5,
        "LOSS_FN": "mse",
        "OPTIM_TYPE": "adam",
        "ENHANCE_INIT_WITH_GENE_COEXP": False,
        "INITIAL_REGRESSION": "Linear",
        "N_JOBS_INIT": 5,
        "RANDOM_SEED": 123,
    },
}


DEFAULT_TRAINING_HPARAMS = {
    "HIDDEN_CHANNELS": 64,
    "WARMUP_EPOCHS": 2000,
    "MAX_EPOCHS": 20000,
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 1e-5,
    "LOSS_FN": "mse",
    "OPTIM_TYPE": "adam",
    "ENHANCE_INIT_WITH_GENE_COEXP": False,
    "INITIAL_REGRESSION": "Linear",
    "N_JOBS_INIT": 5,
    "RANDOM_SEED": 123,
}


SAVE_FIGURES = True


FIG_DPI = 300


BENCHMARK1_EDGE_LR_MAX_EDGES = 200000


VERSION_SPECS = OrderedDict({
    "full": {
        "label": "SpiderNet",
        "short_label": "SpiderNet",
        "description": "Existing trained full SpiderNet model",
        "expr_loss_weight": 1.0,
        "lr_loss_weight": 1.0,
        "shared_role_encoder": False,
        "nonnegative_loadings": True,
        "use_intrinsic_component": True,
        "is_pretrained_full": True,
        "initialization_mode": "pretrained_full",
    },
    "no_lr_reinit": {
        "label": "No LR recon.",
        "short_label": "No LR",
        "description": (
            "LR reconstruction loss weight = 0; edge-level MI warm-up initialization "
            "uses selected gene-pair co-expression only, or uniform [0, 1] if unavailable."
        ),
        "expr_loss_weight": 1.0,
        "lr_loss_weight": 0.0,
        "shared_role_encoder": False,
        "nonnegative_loadings": True,
        "use_intrinsic_component": True,
        "is_pretrained_full": False,
        "initialization_mode": "gene_pair_only_edge_init_v2",
        "require_exact_initialization_cache": True,
    },
    "no_gene_reinit": {
        "label": "No gene recon.",
        "short_label": "No gene",
        "description": (
            "Gene-expression reconstruction loss weight = 0; edge-level MI warm-up "
            "initialization uses LR co-expression only."
        ),
        "expr_loss_weight": 0.0,
        "lr_loss_weight": 1.0,
        "shared_role_encoder": False,
        "nonnegative_loadings": True,
        "use_intrinsic_component": True,
        "is_pretrained_full": False,
        "initialization_mode": "lr_only_edge_init_v2",
        "require_exact_initialization_cache": True,
    },
    "no_intrinsic_reinit": {
        "label": "No intrinsic",
        "short_label": "No intrinsic",
        "description": (
            "Gene-expression reconstruction excludes intrinsic cell-identity term; "
            "initialization removes cell-type one-hot intrinsic regression."
        ),
        "expr_loss_weight": 1.0,
        "lr_loss_weight": 1.0,
        "shared_role_encoder": False,
        "nonnegative_loadings": True,
        "use_intrinsic_component": False,
        "is_pretrained_full": False,
        "initialization_mode": "no_intrinsic_regression_init_v2",
        "require_exact_initialization_cache": True,
    },
})


VERSION_COLORS = {
    "SpiderNet": {"edge": "#9F3B38", "fill": "#E1B6A7"},
    "No LR recon.": {"edge": "#D47C00", "fill": "#F2C382"},
    "No gene recon.": {"edge": "#646491", "fill": "#B9B7D0"},
    "No intrinsic": {"edge": "#E43429", "fill": "#F1B3AE"},
}


SELECTED_GENE_PAIR_EDGE_FEATURE_KEYS = [
    "cellpair_gene_pair_coexpression",
    "cellpair_gene_pair_coexp",
    "cellpair_genepair_coexpression",
    "cellpair_genepair_coexp",
    "cellpair_gene_pair_neigh",
    "cellpair_genepair_neigh",
    "cellpair_selected_gene_pair_coexpression",
    "cellpair_selected_gene_pair_coexp",
    "cellpair_selected_genepair_coexpression",
    "cellpair_selected_genepair_coexp",
    "cellpair_selected_gene_pair_neigh",
    "cellpair_selected_genepair_neigh",
    "gene_pair_coexpression_neigh",
    "gene_pair_coexp_neigh",
    "gene_pair_neigh",
    "selected_gene_pair_coexpression_neigh",
    "selected_gene_pair_coexp_neigh",
]


LR_EDGE_FEATURE_KEY = "cellpair_LRpair_neigh"

VERSION_ORDER = list(VERSION_SPECS)

def get_training_hparams(dataset_key):
    hp = DEFAULT_TRAINING_HPARAMS.copy()
    hp.update(STUDY_TRAINING_HPARAMS.get(dataset_key, {}))
    return hp


def infer_full_run_dir(cfg):
    direct = cfg["output_root"] / cfg["version"] / f"SpiderNet_Result_dim{cfg['dim_envir']}"
    if cfg.get("prefer_run_dirs_json", False):
        run_dirs_path = cfg["output_root"] / "run_dirs.json"
        if run_dirs_path.exists():
            try:
                with open(run_dirs_path, "r", encoding="utf-8") as f:
                    run_dirs = json.load(f)
                cand = Path(run_dirs.get("run_dir", ""))
                if (cand / "Factor_envir_list.pkl").exists():
                    return cand
            except Exception as e:
                print(f"[Warning] Could not use run_dirs.json: {e}")
    return direct


def cache_matches_current_hparams(model_dir, hp, spec=None):
    spec_path = model_dir / "ablation_spec.json"
    if not spec_path.exists():
        return False
    try:
        with open(spec_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        saved_hp = saved.get("training_hparams", {})
        hp_ok = all(saved_hp.get(k) == v for k, v in hp.items())
        if not hp_ok:
            return False

        if spec is not None and spec.get("require_exact_initialization_cache", False):
            saved_spec = saved.get("ablation_spec", {})
            for key in ["initialization_mode", "use_intrinsic_component", "expr_loss_weight", "lr_loss_weight"]:
                if saved_spec.get(key) != spec.get(key):
                    return False
        return True
    except Exception:
        return False


def coupling_output_dir(dataset_key):
    return OUTPUT_ROOT / dataset_key / "Benchmark1_curated_gene_coupling"
