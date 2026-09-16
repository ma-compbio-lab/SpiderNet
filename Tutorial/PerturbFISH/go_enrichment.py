"""Enrich MI programs using loadings bound to the selected checkpoint."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

INPUT_DIR = "GO_MI17_inputs"
GO_FILES = ["GO_MI17_regulator_enrichment.csv", "GO_MI17_target_enrichment.csv"]
SAVED_INPUTS = [f"{INPUT_DIR}/{name}" for name in [
    "loading_sender_use.csv", "loading_receiver_use.csv", "selected_genes.csv",
    "MI_activity_max.csv", "provenance.json",
]]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_saved_enrichment(resolve, checkpoint=None):
    """Reject stale/tampered exports; replay still works without upstream model files."""
    path = Path(resolve(f"{INPUT_DIR}/provenance.json"))
    if not path.is_file():
        raise ValueError("GO provenance missing; run python run_benchmarks.py --stage enrichment")
    provenance = json.loads(path.read_text(encoding="utf-8"))
    required = set(GO_FILES + SAVED_INPUTS[:-1])
    if set(provenance["exports_sha256"]) != required:
        raise ValueError("GO provenance has an incomplete export inventory; rerun --stage enrichment")
    for relative, expected in provenance["exports_sha256"].items():
        candidate = Path(resolve(relative))
        if not candidate.is_file() or sha256(candidate) != expected:
            raise ValueError(f"GO file does not match its recorded inputs: {candidate}; rerun --stage enrichment")
    if checkpoint is not None and Path(checkpoint).is_file():
        if sha256(checkpoint) != provenance["checkpoint"]["sha256"]:
            raise ValueError("GO results belong to a different checkpoint; rerun --stage enrichment")
    return provenance


def checkpoint_loadings(model, factor_envir, genes):
    """Use the original training export's ReLU and MI-wise maximum scaling."""
    import numpy as np
    import pandas as pd
    import torch

    factors = np.asarray(factor_envir, dtype=np.float32)
    if factors.ndim != 2 or not np.isfinite(factors).all() or len(factors) == 0:
        raise ValueError("Expected the finite, unnormalized MI activities of the selected model.")
    maxima = np.max(factors, axis=0)
    rows = [f"MI{i + 1}" for i in range(len(maxima))]
    tables = {}
    for side in ["sender", "receiver"]:
        loading = model.Relu(getattr(model, f"loading_{side}_ori")).detach().cpu().numpy().astype(np.float32)
        if loading.shape != (len(maxima), len(genes)):
            raise ValueError("Model loading dimensions and processed gene names do not agree.")
        tables[side] = pd.DataFrame(loading * maxima[:, np.newaxis], index=rows, columns=genes)
    lr = model.Relu(model.loading_LR_ori).detach().cpu().numpy().astype(np.float32)
    return tables, lr * maxima[:, np.newaxis], maxima


def _run_enrichr_with_rate_limit_retry(*, retry_delays=(30, 60, 120), **kwargs):
    """Retry an unchanged Enrichr request when the service returns HTTP 429."""
    import gseapy as gp
    from gseapy.enrichr import EnrichrAPIError

    for attempt in range(len(retry_delays) + 1):
        try:
            return gp.enrichr(**kwargs)
        except EnrichrAPIError as error:
            is_rate_limit = "status code: 429" in str(error)
            if not is_rate_limit or attempt == len(retry_delays):
                raise
            delay = retry_delays[attempt]
            print(f"Enrichr rate limit reached; retrying unchanged request in {delay} seconds.", flush=True)
            time.sleep(delay)


def run_enrichment(results_dir, processed_root, output_dir, *, model=None, processed=None, factor_envir=None):
    """Reuse the full analysis's model/activity matrix, or infer once without training."""
    import pickle
    from types import SimpleNamespace
    import gseapy as gp
    import numpy as np
    import pandas as pd
    import torch
    from SpiderNet.api import build_model
    from SpiderNet.analysis import predict_batches_spidernet, select_top_features_for_mi
    from SpiderNet.config import TrainingConfig

    results_dir, processed_root, output_dir = map(Path, (results_dir, processed_root, output_dir))
    checkpoint = results_dir / "Model/model_epoch19999.pth"
    state = torch.load(checkpoint, map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if processed is None:
        def read(name):
            with (processed_root / name).open("rb") as stream:
                return pickle.load(stream)
        processed = SimpleNamespace(spidernet_data=read("SpiderNet_data_pyg_list.pkl"),
                                    genenames_train=read("genenames_train.pkl"), lr_list=read("LR_list.pkl"))
    if model is None:
        model = build_model(processed, TrainingConfig(dim_envir=state["loading_sender_ori"].shape[0]), device)
        model.load_state_dict(state)
    else:
        actual = model.state_dict()
        if set(actual) != set(state) or any(not torch.equal(actual[k].detach().cpu(), state[k]) for k in state):
            raise ValueError("The supplied model differs from model_epoch19999.pth; refusing mixed model inputs.")
        device = str(next(model.parameters()).device)
    reused_activities = factor_envir is not None
    if factor_envir is None:
        print("Exporting GO loadings from existing epoch19999; one inference, no training.", flush=True)
        _, factor_envir = predict_batches_spidernet(model, processed.spidernet_data, device)
    tables, lr, maxima = checkpoint_loadings(model, factor_envir, processed.genenames_train)
    features = select_top_features_for_mi("MI17", tables["receiver"], tables["sender"], lr, processed.lr_list, top_n=5)

    folder = output_dir / INPUT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    # A failed request cannot leave old exports certified by an old manifest.
    provenance_path = folder / "provenance.json"
    if provenance_path.exists():
        provenance_path.unlink()
    for side, table in tables.items():
        table.to_csv(folder / f"loading_{side}_use.csv")
    pd.DataFrame({"MI": tables["sender"].index, "MaxActivity": maxima}).to_csv(folder / "MI_activity_max.csv", index=False)
    selected = []
    for side, key in [("sender", "regulatorgene_top"), ("receiver", "targetgene_top")]:
        norm = features[f"loading_{side}_norm"].loc["MI17"]
        selected += [{"Role": side, "Gene": gene, "NormalizedLoading": norm[gene]} for gene in features[key]]
    pd.DataFrame(selected).to_csv(folder / "selected_genes.csv", index=False)

    enriched = {}
    for side, key, label, csv_name in [
        ("sender", "regulatorgene_top", "regulators", GO_FILES[0]),
        ("receiver", "targetgene_top", "targets", GO_FILES[1]),
    ]:
        print(f"GO {side}: {len(features[key])} genes: {features[key]}", flush=True)
        result = _run_enrichr_with_rate_limit_retry(
            gene_list=features[key], gene_sets=["GO_Biological_Process_2021"], organism="mouse",
            outdir=str(output_dir / f"GO_MI17_{label}"), cutoff=0.05,
        )
        enriched[side] = result.results[result.results["Adjusted P-value"] < 0.05].copy()
        enriched[side].to_csv(output_dir / csv_name, index=False)
        if side == "sender":
            time.sleep(10)
    provenance = {
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256(checkpoint)},
        "processed": {name: {"path": str((processed_root / name).resolve()), "sha256": sha256(processed_root / name)}
                      for name in ["genenames_train.pkl", "LR_list.pkl", "SpiderNet_data_pyg_list.pkl"]},
        "method": {"MI": "MI17", "library": "GO_Biological_Process_2021", "organism_argument": "mouse",
                   "enrichr_endpoint": "https://maayanlab.cloud/Enrichr", "gseapy_version": gp.__version__,
                   "loading_export": "ReLU(checkpoint loading) * max(original MI activity across all edges)",
                   "feature_selection": "Original select_top_features_for_mi: gene-wise sum normalization, >0.2; unchanged fallback",
                   "significance": "Adjusted P-value < 0.05", "reused_core_activities": reused_activities},
        "exports_sha256": {name: sha256(output_dir / name) for name in GO_FILES + SAVED_INPUTS[:-1]},
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("Saved checkpoint-bound GO provenance:", provenance_path, flush=True)
    return enriched
