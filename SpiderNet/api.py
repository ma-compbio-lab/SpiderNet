import torch
import time
import json
import numpy as np
from .io import ProcessedData, save_pickle
from .config import TrainingConfig
from pathlib import Path
from typing import Any
import pickle

import pandas as pd
import scanpy as sc

from .config import PathConfig, PreprocessConfig

REQUIRED_PREPROCESSED_FILES = [
    "adata_all.h5ad",
    "SpiderNet_data_pyg_list.pkl or SpiderNet_data_pyg_list.pt",
    "metadata_sample.csv",
    "LR_list.pkl",
    "LR_list_cellchatdb.pkl",
    "LR_meta_cellchatdb.pkl",
    "batch_cell_unique.pkl",
    "batch_cell.pkl",
    "genenames_train.pkl",
    "adata_list.pkl",
]





def select_device(prefer: str | None = None) -> str:
    if prefer is not None:
        if prefer == "cuda" and torch.cuda.is_available():
            return "cuda"
        if prefer in {"cpu", "cuda"}:
            return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"

def _move_spidernet_data_to_device(spidernet_data: list[Any], device: str) -> list[Any]:
    moved = []
    for item in spidernet_data:
        moved.append(item.to(device) if hasattr(item, "to") else item)
    return moved

def build_model(processed: ProcessedData, train_cfg: TrainingConfig, device: str | None = None, hidden_channels: int | None = None):
    from .model import SpiderNet_model
    device = select_device(device)
    dim_intri = processed.spidernet_data[0]['cell_class_onehot'].shape[1]
    if hidden_channels is not None:
        hidden_channels = hidden_channels
    else:
        if train_cfg.dim_envir > 20:
            hidden_channels = 128
        else:
            hidden_channels = 64
    model = SpiderNet_model(
        num_gene=processed.genenames_train.shape[0],
        num_LR=len(processed.lr_list),
        hidden_channels=hidden_channels,
        Factor_mode="cell_class",
        dim_intri=dim_intri,
        dim_envir=train_cfg.dim_envir,
    )
    return model.to(device)

def run_training(
    model,
    processed: ProcessedData,
    train_cfg: TrainingConfig,
    model_dir: str | Path,
    device: str | None = None,
):
    from .model import Initial_model

    device = select_device(device)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    spidernet_data = _move_spidernet_data_to_device(processed.spidernet_data, device)
    num_genes = processed.genenames_train.shape[0]
    if num_genes >= 300:
        enhance_init_with_gene_coexp = False
    else:
        enhance_init_with_gene_coexp = True
    final_model_path = model_dir / f"model_epoch{train_cfg.max_epoch - 1}.pth"

    if final_model_path.exists():
        print("Loading existing trained model...")
        model.load_state_dict(torch.load(final_model_path, map_location=device))

    else:
        print(f"Training on device: {device}")
        print(f"Max epochs: {train_cfg.max_epoch}")
        ########################################################
        # Initialization stage
        ########################################################
        print("========== SpiderNet Initialization Stage ==========")
        init_start = time.time()

        initial_dict = Initial_model(
            spidernet_data,
            dim_envir=train_cfg.dim_envir,
            Factor_mode="cell_class",
            dim_intri=processed.spidernet_data[0]['cell_class_onehot'].shape[1] if "cell_class" == "NMF" else None,
            n_jobs=train_cfg.n_jobs,
            Initial_regression="Linear",
            enhance_init_with_gene_coexp=enhance_init_with_gene_coexp,
            threshold_crosscorr=0.1,
            numtop_crosscorr=10,
            spearcorr_use_rowmax_threshold=0.2,
        )

        init_elapsed = (time.time() - init_start) / 60.0
        print(f"Initialization completed in {init_elapsed:.2f} minutes.")

        ########################################################
        # Main training stage
        ########################################################
        print("========== SpiderNet Main Training Stage ==========")
        train_start = time.time()

        model.fit(
            SpiderNet_data_pyg_list=spidernet_data,
            device=device,
            optim_type=train_cfg.optimizer,
            lr=1e-4,
            weight_decay=1e-5,
            LR_loss_weight=1,
            warmup=int(train_cfg.max_epoch * 0.1),
            max_epoch=train_cfg.max_epoch,
            loss_fn='mse',
            Initial_dict=initial_dict,
            file_savepath_model_main=str(model_dir),
        )

        train_elapsed = (time.time() - train_start) / 60.0
        print(f"Main training completed in {train_elapsed:.2f} minutes.")

        total_elapsed = init_elapsed + train_elapsed
        print(f"Total training time: {total_elapsed:.2f} minutes.")

    config_path = model_dir / "SpiderNet_model_config.json"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(train_cfg.to_dict(), handle, indent=2)

    return model

def infer_meta_interactions(model, processed: ProcessedData, device: str | None = None) -> dict[str, Any]:
    device = select_device(device)
    spidernet_data = _move_spidernet_data_to_device(processed.spidernet_data, device)

    model.eval()
    factor_envir_list = []
    loading_receiver = loading_sender = loading_lr = None

    with torch.no_grad():
        for batch in spidernet_data:
            _, _, _, loading_intrinsic, factor_envir_batch, loading_receiver, loading_sender, loading_lr = model(batch)
            factor_envir_list.append(factor_envir_batch.detach().cpu().numpy().astype(np.float32))

    factor_envir = np.vstack(factor_envir_list)

    return {
        "factor_envir_list": factor_envir_list,
        "factor_envir": factor_envir,
        "loading_receiver": loading_receiver.detach().cpu().numpy().astype(np.float32),
        "loading_sender": loading_sender.detach().cpu().numpy().astype(np.float32),
        "loading_lr": loading_lr.detach().cpu().numpy().astype(np.float32),
        "loading_intrinsic":loading_intrinsic.detach().cpu().numpy().astype(np.float32),
    }

def normalize_outputs(results: dict[str, Any], eps: float = 1e-10) -> dict[str, Any]:
    factor_envir = np.asarray(results["factor_envir"], dtype=np.float32)
    scale = np.max(factor_envir, axis=0)
    scale = np.where(scale > 0, scale, 1.0).astype(np.float32)

    results = dict(results)
    results["loading_lr"] = results["loading_lr"] * scale[:, None]
    results["loading_receiver"] = results["loading_receiver"] * scale[:, None]
    results["loading_sender"] = results["loading_sender"] * scale[:, None]
    results["factor_envir"] = factor_envir / (scale[None, :] + eps)
    results["factor_envir_list"] = [batch / (scale[None, :] + eps) for batch in results["factor_envir_list"]]
    results["scale"] = scale
    return results

def export_results(results: dict[str, Any], processed: ProcessedData, precessed_data_dir: str | Path, output_dir: str | Path) -> None:
    precessed_data_dir = Path(precessed_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    import pickle

    with open(precessed_data_dir / "cellclass_unique.pkl", "rb") as f:
        celltype_unique = pickle.load(f)

    mi_names = [f"MI{i + 1}" for i in range(results["loading_receiver"].shape[0])]
    genes = processed.adata_list[0].var_names

    receiver_df = pd.DataFrame(results["loading_receiver"], index=mi_names, columns=genes)
    sender_df = pd.DataFrame(results["loading_sender"], index=mi_names, columns=genes)

    receiver_df.to_csv(output_dir / "loading_receiver_use.csv", index=True)
    sender_df.to_csv(output_dir / "loading_sender_use.csv", index=True)

    np.save(output_dir / "loading_sender_use.npy", results["loading_sender"])
    np.save(output_dir / "loading_receiver_use.npy", results["loading_receiver"])

    np.save(output_dir / "Factor_envir_use.npy", results["factor_envir"])
    np.save(output_dir / "loading_LR_use.npy", results["loading_lr"])
    save_pickle(results["factor_envir_list"], output_dir / "Factor_envir_list.pkl")

    loading_intrinsic_df = pd.DataFrame(results["loading_intrinsic"], index=celltype_unique, columns=genes)
    loading_intrinsic_df.to_csv(output_dir / "loading_intrinsic_use.csv", index=True)

def export_results_V0(results: dict[str, Any], processed: ProcessedData, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    import pickle

    with open(output_dir / "cellclass_unique.pkl", "rb") as f:
        celltype_unique = pickle.load(f)

    mi_names = [f"MI{i + 1}" for i in range(results["loading_receiver"].shape[0])]
    genes = processed.adata_list[0].var_names

    receiver_df = pd.DataFrame(results["loading_receiver"], index=mi_names, columns=genes)
    sender_df = pd.DataFrame(results["loading_sender"], index=mi_names, columns=genes)

    receiver_df.to_csv(output_dir / "loading_receiver_use.csv", index=True)
    sender_df.to_csv(output_dir / "loading_sender_use.csv", index=True)

    np.save(output_dir / "loading_sender_use.npy", results["loading_sender"])
    np.save(output_dir / "loading_receiver_use.npy", results["loading_receiver"])

    np.save(output_dir / "Factor_envir_use.npy", results["factor_envir"])
    np.save(output_dir / "loading_LR_use.npy", results["loading_lr"])
    save_pickle(results["factor_envir_list"], output_dir / "Factor_envir_list.pkl")

    loading_intrinsic_df = pd.DataFrame(results["loading_intrinsic"], index=celltype_unique, columns=genes)
    loading_intrinsic_df.to_csv(output_dir / "loading_intrinsic_use.csv", index=True)

