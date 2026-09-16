"""Component-specific initialization, model training and factor-cache loading."""
from pathlib import Path
import sys
import json
import pickle
import random
import importlib.util

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import NMF
try:
    from sklearn.decomposition import MiniBatchNMF
except Exception:
    MiniBatchNMF = None
import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_scatter import scatter_mean

from ablation_settings import (
    SPIDERNET_PROJECT_DIR, ORIGINAL_MODEL_PY, DATASET_CONFIG, VERSION_SPECS,
    RETRAIN_ABLATIONS, RUN_TRAINING_IF_MISSING, ABLATION_SOURCE_NOTEBOOK,
    SELECTED_GENE_PAIR_EDGE_FEATURE_KEYS, LR_EDGE_FEATURE_KEY, BENCHMARK1_EDGE_LR_MAX_EDGES,
    get_training_hparams, infer_full_run_dir, cache_matches_current_hparams,
)

if SPIDERNET_PROJECT_DIR and str(Path(SPIDERNET_PROJECT_DIR).resolve()) not in sys.path:
    sys.path.append(str(Path(SPIDERNET_PROJECT_DIR).resolve()))
try:
    from SpiderNet.model import Initial_model
except Exception as import_error:
    if ORIGINAL_MODEL_PY is None:
        raise ImportError("Could not import SpiderNet.model.Initial_model; use the analysis environment "
                          "or set SPIDERNET_PROJECT_DIR / SPIDERNET_MODEL_PY.") from import_error
    spec = importlib.util.spec_from_file_location("spidernet_original_model", ORIGINAL_MODEL_PY)
    original_model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original_model_module)
    Initial_model = original_model_module.Initial_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def edge_index_to_e2(edge_index):
    """Return edge_index as an E x 2 numpy integer array."""
    if torch.is_tensor(edge_index):
        arr = edge_index.detach().cpu().numpy()
    else:
        arr = np.asarray(edge_index)
    if arr.ndim != 2:
        raise ValueError(f"edge_index must be 2D, got {arr.shape}")
    if arr.shape[1] == 2:
        out = arr
    elif arr.shape[0] == 2:
        out = arr.T
    else:
        raise ValueError(f"Cannot interpret edge_index shape as E x 2 or 2 x E: {arr.shape}")
    return out.astype(np.int64, copy=False)


def get_data_item(data, key):
    try:
        return data[key]
    except Exception:
        return getattr(data, key)


def set_data_item(data, key, value):
    try:
        data[key] = value
    except Exception:
        setattr(data, key, value)


def get_num_cells_from_data(data, adata_sub=None):
    try:
        return int(get_data_item(data, "x").shape[0])
    except Exception:
        if adata_sub is None:
            raise
        return int(adata_sub.n_obs)


class SpiderNetAblationModel(nn.Module):
    """SpiderNet model with component-level ablation switches.

    This class mirrors the SpiderNet model architecture, with the following added switches:
      - shared_role_encoder: use the same MLP for sender and receiver pre-encoding.
      - nonnegative_loadings: if False, do not ReLU the loading parameters.
      - use_intrinsic_component: if False, omit intrinsic cell-identity expression reconstruction.
    """
    def __init__(
        self,
        num_gene,
        num_LR,
        hidden_channels,
        Factor_mode,
        dim_intri=10,
        dim_envir=10,
        shared_role_encoder=False,
        nonnegative_loadings=True,
        use_intrinsic_component=True,
    ):
        super().__init__()
        self.num_gene = int(num_gene)
        self.num_LR = int(num_LR)
        self.hidden_channels = int(hidden_channels)
        self.hidden_channels_scale = int(hidden_channels / 2)
        self.Factor_mode = Factor_mode
        self.dim_intri = int(dim_intri)
        self.dim_envir = int(dim_envir)
        self.shared_role_encoder = bool(shared_role_encoder)
        self.nonnegative_loadings = bool(nonnegative_loadings)
        self.use_intrinsic_component = bool(use_intrinsic_component)

        self.Relu = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        scale_factor = 2

        if self.Factor_mode == "NMF":
            self.enc_factor_intrinsic = nn.Sequential(
                nn.Linear(num_gene, hidden_channels),
                nn.Tanh(),
                nn.Linear(hidden_channels, dim_intri),
                nn.Sigmoid(),
            )

        def make_role_encoder():
            return nn.Sequential(
                nn.Linear(num_gene, 2 * scale_factor * self.hidden_channels_scale),
                nn.Tanh(),
                nn.Linear(2 * scale_factor * self.hidden_channels_scale, scale_factor * self.hidden_channels_scale),
                nn.Tanh(),
            )

        self.enc_factor_envir_pre_receiver = make_role_encoder()
        if self.shared_role_encoder:
            self.enc_factor_envir_pre_sender = self.enc_factor_envir_pre_receiver
        else:
            self.enc_factor_envir_pre_sender = make_role_encoder()

        self.enc_factor_envir = nn.Sequential(
            nn.Linear(2 * scale_factor * self.hidden_channels_scale, 2 * scale_factor * self.hidden_channels_scale),
            nn.Tanh(),
            nn.Linear(2 * scale_factor * self.hidden_channels_scale, self.dim_envir),
        )

        self.Loading_intrinsic_ori = nn.Parameter(torch.abs(torch.randn(self.dim_intri, self.num_gene)))
        self.loading_receiver_ori = nn.Parameter(torch.abs(torch.randn(self.dim_envir, self.num_gene)))
        self.loading_sender_ori = nn.Parameter(torch.abs(torch.randn(self.dim_envir, self.num_gene)))
        self.loading_LR_ori = nn.Parameter(torch.abs(torch.randn(self.dim_envir, self.num_LR)))

    def _loadings(self):
        if self.nonnegative_loadings:
            return (
                self.Relu(self.Loading_intrinsic_ori),
                self.Relu(self.loading_receiver_ori),
                self.Relu(self.loading_sender_ori),
                self.Relu(self.loading_LR_ori),
            )
        return (
            self.Loading_intrinsic_ori,
            self.loading_receiver_ori,
            self.loading_sender_ori,
            self.loading_LR_ori,
        )

    def forward(self, SpiderNet_data_pyg):
        exp = SpiderNet_data_pyg.x
        num_cell = exp.shape[0]
        edge_index = SpiderNet_data_pyg["edge_index"]

        if self.Factor_mode == "NMF":
            Factor_intrinsic = self.enc_factor_intrinsic(exp)
            Factor_intrinsic = Factor_intrinsic / Factor_intrinsic.sum(dim=1, keepdim=True).clamp(min=1e-8)
        else:
            Factor_intrinsic = SpiderNet_data_pyg[self.Factor_mode + "_onehot"]

        exp_enc_receiver = self.enc_factor_envir_pre_receiver(exp)
        exp_enc_sender = self.enc_factor_envir_pre_sender(exp)
        exp_enc_center_receiver = exp_enc_receiver[edge_index[:, 1], :]
        exp_enc_neighbor_sender = exp_enc_sender[edge_index[:, 0], :]
        pair_emb = torch.cat((exp_enc_neighbor_sender, exp_enc_center_receiver), dim=1)

        Factor_envir = self.Sigmoid(self.enc_factor_envir(pair_emb))
        edge_receiver = edge_index[:, 1]
        edge_sender = edge_index[:, 0]
        Factor_envir_receiver_neighagg = scatter_mean(Factor_envir, edge_receiver, dim=0, dim_size=num_cell)
        Factor_envir_sender_neighagg = scatter_mean(Factor_envir, edge_sender, dim=0, dim_size=num_cell)

        Loading_intrinsic, loading_receiver, loading_sender, loading_LR = self._loadings()

        if self.use_intrinsic_component:
            exp_recon = torch.matmul(Factor_intrinsic, Loading_intrinsic)
        else:
            exp_recon = torch.zeros_like(exp)
        exp_recon = exp_recon + torch.matmul(Factor_envir_receiver_neighagg, loading_receiver)
        exp_recon = exp_recon + torch.matmul(Factor_envir_sender_neighagg, loading_sender)

        exp_LR_recon = torch.matmul(Factor_envir, loading_LR)
        return exp_recon, exp_LR_recon, Factor_intrinsic, Loading_intrinsic, Factor_envir, loading_receiver, loading_sender, loading_LR

    def compute_loss_in_batches(
        self,
        SpiderNet_data_pyg_list,
        factor_envir_init,
        stage,
        criterion,
        scaler=None,
        expr_loss_weight=1.0,
        lr_loss_weight=1.0,
    ):
        num_batches = len(SpiderNet_data_pyg_list)
        loss_total = 0.0
        loss_warmup = 0.0
        loss_exp = 0.0
        loss_LR = 0.0

        for batch_index_cur in range(num_batches):
            exp_recon_cur, exp_LR_recon_cur, _, _, Factor_envir_cur, _, _, _ = self(SpiderNet_data_pyg_list[batch_index_cur])

            if stage == "warmup":
                target = factor_envir_init[batch_index_cur]
                if Factor_envir_cur.dtype != target.dtype:
                    target = target.to(Factor_envir_cur.dtype)
                loss_batch = criterion(Factor_envir_cur, target) / num_batches
                loss_warmup += float(loss_batch.item())
            elif stage == "main":
                target_exp = SpiderNet_data_pyg_list[batch_index_cur].x
                if exp_recon_cur.dtype != target_exp.dtype:
                    target_exp = target_exp.to(exp_recon_cur.dtype)
                loss_exp_batch = criterion(exp_recon_cur, target_exp) / num_batches

                target_LR = SpiderNet_data_pyg_list[batch_index_cur]["cellpair_LRpair_neigh"]
                if exp_LR_recon_cur.dtype != target_LR.dtype:
                    target_LR = target_LR.to(exp_LR_recon_cur.dtype)
                loss_LR_batch = criterion(exp_LR_recon_cur, target_LR) / num_batches

                loss_batch = expr_loss_weight * loss_exp_batch + lr_loss_weight * loss_LR_batch
                loss_exp += float(loss_exp_batch.item())
                loss_LR += float(loss_LR_batch.item())
            else:
                raise ValueError(f"Unknown stage: {stage}")

            if scaler is not None:
                scaler.scale(loss_batch).backward()
            else:
                loss_batch.backward()
            loss_total += float(loss_batch.item())

        return loss_total, loss_warmup, loss_exp, loss_LR

    def fit_ablation(
        self,
        SpiderNet_data_pyg_list,
        device=DEVICE,
        optim_type="adam",
        lr=1e-3,
        weight_decay=1e-5,
        expr_loss_weight=1.0,
        lr_loss_weight=1.0,
        warmup=10,
        max_epoch=100,
        loss_fn="mse",
        Initial_dict=None,
        file_savepath_model_main=None,
        log_every=100,
    ):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "poisson":
            criterion = nn.PoissonNLLLoss(log_input=False)
        else:
            raise ValueError(f"Unsupported loss_fn: {loss_fn}")

        use_amp = device.type == "cuda"
        scaler = GradScaler(enabled=use_amp)

        loading_params_names = ["Loading_intrinsic_ori", "loading_receiver_ori", "loading_sender_ori", "loading_LR_ori"]
        init_params = {
            "Loading_intrinsic_ori": torch.tensor(Initial_dict["loading_intrinsic_init"], device=device, dtype=torch.float32),
            "loading_receiver_ori": torch.tensor(Initial_dict["loading_receiver_init"], device=device, dtype=torch.float32),
            "loading_sender_ori": torch.tensor(Initial_dict["loading_sender_init"], device=device, dtype=torch.float32),
            "loading_LR_ori": torch.tensor(Initial_dict["loading_LR_init"], device=device, dtype=torch.float32),
        }
        factor_envir_init = [
            torch.tensor(x, device=device, dtype=torch.float32)
            for x in Initial_dict["factor_GP_minibatchNMF"]
        ]

        all_params = dict(self.named_parameters())
        enc_factor_intrinsic_params = [p for n, p in all_params.items() if "enc_factor_intrinsic" in n]
        enc_factor_envir_params = [p for n, p in all_params.items() if "enc_factor_envir" in n]
        loading_params_trainable = [p for n, p in all_params.items() if n in loading_params_names]

        optimizer = None
        scheduler = None
        history = []

        for epoch in range(max_epoch):
            self.train()
            if epoch == 0 or epoch == warmup:
                opt_lr = 5e-4 if epoch == 0 else lr
                if optim_type == "adam":
                    optimizer = optim.Adam(self.parameters(), lr=opt_lr, weight_decay=weight_decay)
                elif optim_type == "SGD":
                    optimizer = optim.SGD(self.parameters(), lr=opt_lr, weight_decay=weight_decay)
                else:
                    raise ValueError(f"Unsupported optim_type: {optim_type}")
                scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=20, min_lr=1e-5)

            for _, param in self.named_parameters():
                param.requires_grad = True

            if epoch < warmup:
                for name, param in self.named_parameters():
                    if name in loading_params_names:
                        param.requires_grad = False
                        # For signed-loading ablation, the initialization is still positive, but training can move it signed later.
                        param.data.copy_(init_params[name].to(param.dtype))
            else:
                if self.Factor_mode == "NMF":
                    if epoch % 3 == 0:
                        for p in enc_factor_intrinsic_params: p.requires_grad = True
                        for p in enc_factor_envir_params + loading_params_trainable: p.requires_grad = False
                    elif epoch % 3 == 1:
                        for p in enc_factor_envir_params: p.requires_grad = True
                        for p in enc_factor_intrinsic_params + loading_params_trainable: p.requires_grad = False
                    else:
                        for p in loading_params_trainable: p.requires_grad = True
                        for p in enc_factor_intrinsic_params + enc_factor_envir_params: p.requires_grad = False
                else:
                    if epoch % 2 < 1 or epoch < warmup + 10:
                        for p in enc_factor_envir_params: p.requires_grad = False
                        for p in loading_params_trainable: p.requires_grad = True
                    else:
                        for p in loading_params_trainable: p.requires_grad = False
                        for p in enc_factor_envir_params: p.requires_grad = True

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                if epoch < warmup:
                    loss_total, loss_warmup, loss_exp, loss_LR = self.compute_loss_in_batches(
                        SpiderNet_data_pyg_list,
                        factor_envir_init,
                        stage="warmup",
                        criterion=criterion,
                        scaler=scaler,
                        expr_loss_weight=expr_loss_weight,
                        lr_loss_weight=lr_loss_weight,
                    )
                else:
                    loss_total, loss_warmup, loss_exp, loss_LR = self.compute_loss_in_batches(
                        SpiderNet_data_pyg_list,
                        factor_envir_init,
                        stage="main",
                        criterion=criterion,
                        scaler=scaler,
                        expr_loss_weight=expr_loss_weight,
                        lr_loss_weight=lr_loss_weight,
                    )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if epoch >= warmup and epoch % 10 == 0:
                scheduler.step(loss_total)

            if epoch == 0 or epoch == warmup - 1 or epoch == warmup or epoch % log_every == 0 or epoch == max_epoch - 1:
                rec = {
                    "epoch": epoch,
                    "stage": "warmup" if epoch < warmup else "main",
                    "loss_total_weighted": loss_total,
                    "loss_warmup": loss_warmup,
                    "loss_exp_unweighted": loss_exp,
                    "loss_LR_unweighted": loss_LR,
                }
                history.append(rec)
                print(rec)

            if file_savepath_model_main is not None and (epoch % 1000 == 0 or epoch == max_epoch - 1):
                save_dir = Path(file_savepath_model_main)
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(self.state_dict(), save_dir / f"model_epoch{epoch}.pth")

        return pd.DataFrame(history)


def canonicalize_graphs_for_model(graph_list, device=DEVICE):
    out = []
    for data in graph_list:
        # Work in place to avoid duplicating large objects.
        e2 = edge_index_to_e2(get_data_item(data, "edge_index"))
        set_data_item(data, "edge_index", torch.as_tensor(e2, dtype=torch.long))
        if hasattr(data, "to"):
            data = data.to(device)
        else:
            for key in ["x", "edge_index", "cellpair_LRpair_neigh"]:
                val = get_data_item(data, key)
                if torch.is_tensor(val):
                    set_data_item(data, key, val.to(device))
        out.append(data)
    return out


def infer_factor_mode_and_dim_intri(graph_list, preferred_factor_mode):
    candidates = [preferred_factor_mode, "celltype", "cell.types", "cell_class", "cellclass", "CellType", "cell_type", "NMF"]
    for mode in candidates:
        if mode == "NMF":
            continue
        key = mode + "_onehot"
        try:
            mat = get_data_item(graph_list[0], key)
            return mode, int(mat.shape[1])
        except Exception:
            pass
    # Fallback to NMF intrinsic factors if no one-hot is available.
    return "NMF", 10


@torch.no_grad()
def extract_factor_envir_list(model, graph_list):
    model.eval()
    factors = []
    for data in graph_list:
        _, _, _, _, Factor_envir, _, _, _ = model(data)
        factors.append(Factor_envir.detach().cpu().numpy().astype(np.float32, copy=False))
    return factors


def save_pickle(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_dataset_context(dataset_key):
    cfg = DATASET_CONFIG[dataset_key].copy()
    run_dir = infer_full_run_dir(cfg)
    processed_data_dir = Path(cfg["processed_data_dir"])
    if not processed_data_dir.exists():
        raise FileNotFoundError(f"Processed data directory does not exist: {processed_data_dir}")
    if not (run_dir / "Factor_envir_list.pkl").exists():
        raise FileNotFoundError(f"Full model Factor_envir_list.pkl not found: {run_dir / 'Factor_envir_list.pkl'}")

    adata_list = pd.read_pickle(processed_data_dir / "adata_list.pkl")
    graph_list = pd.read_pickle(processed_data_dir / "SpiderNet_data_pyg_list.pkl")
    LR_list = pd.read_pickle(processed_data_dir / "LR_list.pkl")
    full_factor_list = pd.read_pickle(run_dir / "Factor_envir_list.pkl")

    adata_all_path = processed_data_dir / "adata_all.h5ad"
    if adata_all_path.exists():
        adata_all = sc.read_h5ad(adata_all_path)
    else:
        adata_all = sc.concat(adata_list, join="inner", merge="same") if len(adata_list) > 1 else adata_list[0].copy()

    context = {
        "dataset_key": dataset_key,
        "cfg": cfg,
        "run_dir": run_dir,
        "processed_data_dir": processed_data_dir,
        "adata_list": adata_list,
        "adata_all": adata_all,
        "graph_list_cpu": graph_list,
        "LR_list": LR_list,
        "factor_lists": {"full": [np.asarray(x, dtype=np.float32) for x in full_factor_list]},
        "ablation_root": run_dir / "Component_ablation",
    }
    context["ablation_root"].mkdir(parents=True, exist_ok=True)
    print(f"[{dataset_key}] run_dir:", run_dir)
    print(f"[{dataset_key}] processed_data_dir:", processed_data_dir)
    print(f"[{dataset_key}] n_slices:", len(adata_list), "full factor first shape:", np.asarray(full_factor_list[0]).shape)
    return context


def data_keys(data):
    try:
        keys = data.keys
        if callable(keys):
            return list(keys())
        return list(keys)
    except Exception:
        pass
    try:
        return list(data._store.keys())
    except Exception:
        return []


def as_numpy_float32(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if sp.issparse(x):
        x = x.toarray()
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def n_edges_in_graph(data):
    try:
        lr = get_data_item(data, LR_EDGE_FEATURE_KEY)
        return int(as_numpy_float32(lr).shape[0])
    except Exception:
        e2 = edge_index_to_e2(get_data_item(data, "edge_index"))
        return int(e2.shape[0])


def get_edge_feature_matrix(data, source):
    """Return an E x P edge-feature matrix for source='lr' or source='gene_pair'.

    For selected gene-pair co-expression, returns None if no usable key exists
    or if the matrix has zero columns.
    """
    if source == "lr":
        mat = as_numpy_float32(get_data_item(data, LR_EDGE_FEATURE_KEY))
        if mat.shape[1] == 0:
            return None
        return mat

    if source == "gene_pair":
        available = set(data_keys(data))
        for key in SELECTED_GENE_PAIR_EDGE_FEATURE_KEYS:
            if key not in available:
                continue
            try:
                mat = as_numpy_float32(get_data_item(data, key))
                if mat.shape[1] > 0:
                    return mat
            except Exception:
                continue
        return None

    raise ValueError(f"Unknown edge-feature source: {source}")


def build_edge_feature_list(graph_list, sources):
    """Build per-slice edge-feature matrices from requested sources.

    Missing selected-gene-pair matrices are treated as absent. If a source is
    absent for all slices, it contributes zero columns. If a source is present
    for some slices but absent in others, absent slices are filled with zeros so
    all slices have consistent feature dimensions.
    """
    per_source = {}
    for source in sources:
        mats = []
        n_cols = 0
        for data in graph_list:
            mat = get_edge_feature_matrix(data, source)
            mats.append(mat)
            if mat is not None:
                n_cols = max(n_cols, int(mat.shape[1]))
        if n_cols > 0:
            fixed = []
            for data, mat in zip(graph_list, mats):
                e = n_edges_in_graph(data)
                if mat is None:
                    fixed.append(np.zeros((e, n_cols), dtype=np.float32))
                elif mat.shape[1] == n_cols:
                    fixed.append(mat.astype(np.float32, copy=False))
                else:
                    padded = np.zeros((mat.shape[0], n_cols), dtype=np.float32)
                    padded[:, :mat.shape[1]] = mat
                    fixed.append(padded)
            per_source[source] = fixed
        else:
            per_source[source] = None

    if all(per_source[source] is None for source in sources):
        return None, {source: 0 for source in sources}

    feature_list = []
    for slice_i, data in enumerate(graph_list):
        mats = []
        for source in sources:
            if per_source[source] is not None:
                mats.append(per_source[source][slice_i])
        if len(mats) == 0:
            e = n_edges_in_graph(data)
            feature_list.append(np.zeros((e, 0), dtype=np.float32))
        elif len(mats) == 1:
            feature_list.append(mats[0].astype(np.float32, copy=False))
        else:
            feature_list.append(np.hstack(mats).astype(np.float32, copy=False))

    ncols_by_source = {
        source: 0 if per_source[source] is None else int(per_source[source][0].shape[1])
        for source in sources
    }
    return feature_list, ncols_by_source


def make_uniform_factor_init(graph_list, dim_envir, seed=123):
    rng = np.random.default_rng(int(seed))
    return [
        rng.uniform(0.0, 1.0, size=(n_edges_in_graph(data), int(dim_envir))).astype(np.float32)
        for data in graph_list
    ]


def sample_rows_from_matrices(matrix_list, max_rows=200000, seed=123):
    rng = np.random.default_rng(int(seed))
    n_total = int(sum(mat.shape[0] for mat in matrix_list))
    if n_total == 0:
        return np.zeros((0, matrix_list[0].shape[1] if matrix_list else 0), dtype=np.float32)

    sampled_parts = []
    for mat in matrix_list:
        n = int(mat.shape[0])
        if n == 0:
            continue
        n_take = int(np.ceil(max_rows * n / n_total))
        n_take = max(1, min(n, n_take))
        if n_take < n:
            idx = rng.choice(n, size=n_take, replace=False)
            sampled_parts.append(mat[idx])
        else:
            sampled_parts.append(mat)
    if len(sampled_parts) == 0:
        return np.zeros((0, matrix_list[0].shape[1]), dtype=np.float32)
    out = np.vstack(sampled_parts).astype(np.float32, copy=False)
    if out.shape[0] > max_rows:
        idx = rng.choice(out.shape[0], size=max_rows, replace=False)
        out = out[idx]
    return out


def fit_transform_edge_nmf(feature_list, dim_envir, seed=123, max_fit_edges=200000, batch_size=8192, source_notebook="sklearn_compat"):
    """Fit NMF/MiniBatchNMF on sampled edge features and transform every slice.

    The returned edge-level MI activity matrices are component-wise scaled and
    clipped to [0, 1], matching the sigmoid range used by the SpiderNet encoder.
    """
    if feature_list is None or len(feature_list) == 0:
        return None
    n_features = int(feature_list[0].shape[1])
    if n_features == 0:
        return None

    x_fit_raw = sample_rows_from_matrices(feature_list, max_rows=max_fit_edges, seed=seed)
    if x_fit_raw.shape[0] == 0 or x_fit_raw.shape[1] == 0:
        return None

    x_fit_raw = np.nan_to_num(x_fit_raw, nan=0.0, posinf=0.0, neginf=0.0)
    x_fit_raw = np.clip(x_fit_raw, 0.0, None)
    if np.nanmax(x_fit_raw) <= 0:
        return None

    scale = np.nanpercentile(x_fit_raw, 99, axis=0).astype(np.float32)
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0

    def preprocess(x):
        x = np.nan_to_num(x.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        x = np.clip(x, 0.0, None)
        return x / scale[None, :]

    x_fit = preprocess(x_fit_raw)
    n_components = int(dim_envir)
    init = "nndsvda" if n_components <= min(x_fit.shape[0], x_fit.shape[1]) else "random"

    source_notebook = str(source_notebook)

    if source_notebook == "reinit_ablation":
        # Original notebook implementation. This intentionally keeps the
        # MiniBatchNMF(..., n_init=1) call used in
        # SpiderNet_component_ablation_study_Benchmark1_edgeLR_panel_reinit_ablation.
        if MiniBatchNMF is not None:
            nmf_model = MiniBatchNMF(
                n_components=n_components,
                init=init,
                random_state=int(seed),
                batch_size=int(batch_size),
                max_iter=500,
                n_init=1,
            )
        else:
            nmf_model = NMF(
                n_components=n_components,
                init=init,
                random_state=int(seed),
                max_iter=500,
            )
    else:
        # sklearn-compatible notebook implementation. MiniBatchNMF has slightly
        # different constructor signatures across scikit-learn versions. Older
        # versions do not support n_init, so only pass supported arguments.
        def _filter_estimator_kwargs(estimator_cls, kwargs):
            try:
                import inspect
                params = inspect.signature(estimator_cls).parameters
                return {k: v for k, v in kwargs.items() if k in params}
            except Exception:
                return kwargs

        if MiniBatchNMF is not None:
            mb_kwargs = {
                "n_components": n_components,
                "init": init,
                "random_state": int(seed),
                "batch_size": int(batch_size),
                "max_iter": 500,
                "n_init": 1,
            }
            mb_kwargs = _filter_estimator_kwargs(MiniBatchNMF, mb_kwargs)
            nmf_model = MiniBatchNMF(**mb_kwargs)
        else:
            nmf_kwargs = {
                "n_components": n_components,
                "init": init,
                "random_state": int(seed),
                "max_iter": 500,
            }
            nmf_kwargs = _filter_estimator_kwargs(NMF, nmf_kwargs)
            nmf_model = NMF(**nmf_kwargs)

    nmf_model.fit(x_fit)

    factor_list = []
    for mat in feature_list:
        x = preprocess(mat)
        w = nmf_model.transform(x).astype(np.float32, copy=False)
        factor_list.append(w)

    # Scale each MI dimension into [0, 1] using a robust global high quantile.
    sample_w = sample_rows_from_matrices(factor_list, max_rows=max_fit_edges, seed=seed + 17)
    comp_scale = np.nanpercentile(sample_w, 99, axis=0).astype(np.float32)
    comp_scale[~np.isfinite(comp_scale) | (comp_scale <= 0)] = 1.0
    factor_list = [np.clip(w / comp_scale[None, :], 0.0, 1.0).astype(np.float32, copy=False) for w in factor_list]
    return factor_list


def aggregate_edge_factors_to_cells(factor, edge_index, n_cells, role):
    e2 = edge_index_to_e2(edge_index)
    if role == "receiver":
        idx = e2[:, 1]
    elif role == "sender":
        idx = e2[:, 0]
    else:
        raise ValueError("role must be 'receiver' or 'sender'")
    out = np.zeros((int(n_cells), factor.shape[1]), dtype=np.float32)
    cnt = np.zeros(int(n_cells), dtype=np.float32)
    np.add.at(out, idx, factor)
    np.add.at(cnt, idx, 1.0)
    out = out / np.maximum(cnt[:, None], 1.0)
    return out


def get_intrinsic_factor_matrix(data, factor_mode, dim_intri):
    if factor_mode == "NMF":
        return None
    key = factor_mode + "_onehot"
    try:
        mat = as_numpy_float32(get_data_item(data, key))
        if mat.shape[1] == int(dim_intri):
            return mat
    except Exception:
        pass
    return None


def ridge_lstsq_nonnegative(design, target, ridge=1e-4):
    design = np.nan_to_num(np.asarray(design, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(np.asarray(target, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if design.ndim != 2 or target.ndim != 2:
        raise ValueError(f"Expected 2D design/target, got {design.shape}, {target.shape}")
    if design.shape[0] == 0 or design.shape[1] == 0:
        return np.zeros((design.shape[1], target.shape[1]), dtype=np.float32)
    xtx = design.T @ design
    xtx = xtx + float(ridge) * np.eye(xtx.shape[0], dtype=np.float32)
    xty = design.T @ target
    try:
        coef = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(design, target, rcond=None)[0]
    coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    coef = np.clip(coef, 0.0, None)
    return coef


def initialize_gene_loadings_from_factors(
    graph_list,
    factor_list,
    factor_mode,
    dim_intri,
    dim_envir,
    use_intrinsic_regression=True,
    seed=123,
    max_cells=200000,
):
    """Initialize intrinsic/receiver/sender gene loadings by linear regression.

    If use_intrinsic_regression=False, the design matrix excludes the intrinsic
    one-hot/NMF term; this is the modified initialization required for the
    No intrinsic ablation.
    """
    rng = np.random.default_rng(int(seed))
    design_parts = []
    target_parts = []
    n_total = int(sum(as_numpy_float32(get_data_item(data, "x")).shape[0] for data in graph_list))

    for data, factor in zip(graph_list, factor_list):
        x = as_numpy_float32(get_data_item(data, "x"))
        n = int(x.shape[0])
        edge_index = get_data_item(data, "edge_index")
        recv_agg = aggregate_edge_factors_to_cells(factor, edge_index, n, role="receiver")
        send_agg = aggregate_edge_factors_to_cells(factor, edge_index, n, role="sender")

        blocks = []
        intrinsic = None
        if use_intrinsic_regression:
            intrinsic = get_intrinsic_factor_matrix(data, factor_mode, dim_intri)
            if intrinsic is not None:
                blocks.append(intrinsic)
        blocks.extend([recv_agg, send_agg])
        design = np.hstack(blocks).astype(np.float32, copy=False)

        n_take = int(np.ceil(max_cells * n / max(n_total, 1)))
        n_take = max(1, min(n, n_take))
        if n_take < n:
            idx = rng.choice(n, size=n_take, replace=False)
            design_parts.append(design[idx])
            target_parts.append(x[idx])
        else:
            design_parts.append(design)
            target_parts.append(x)

    design_all = np.vstack(design_parts).astype(np.float32, copy=False)
    target_all = np.vstack(target_parts).astype(np.float32, copy=False)
    coef = ridge_lstsq_nonnegative(design_all, target_all)

    offset = 0
    if use_intrinsic_regression and get_intrinsic_factor_matrix(graph_list[0], factor_mode, dim_intri) is not None:
        loading_intrinsic = coef[offset:offset + int(dim_intri), :]
        offset += int(dim_intri)
    else:
        num_gene = target_all.shape[1]
        loading_intrinsic = np.zeros((int(dim_intri), num_gene), dtype=np.float32)

    loading_receiver = coef[offset:offset + int(dim_envir), :]
    offset += int(dim_envir)
    loading_sender = coef[offset:offset + int(dim_envir), :]

    return loading_intrinsic.astype(np.float32), loading_receiver.astype(np.float32), loading_sender.astype(np.float32)


def initialize_lr_loading_from_factors(graph_list, factor_list, dim_envir, seed=123, max_edges=200000):
    factor_sample_parts = []
    lr_sample_parts = []
    rng = np.random.default_rng(int(seed))
    n_total = int(sum(f.shape[0] for f in factor_list))

    for data, factor in zip(graph_list, factor_list):
        lr = as_numpy_float32(get_data_item(data, LR_EDGE_FEATURE_KEY))
        n = int(factor.shape[0])
        n_take = int(np.ceil(max_edges * n / max(n_total, 1)))
        n_take = max(1, min(n, n_take))
        if n_take < n:
            idx = rng.choice(n, size=n_take, replace=False)
            factor_sample_parts.append(factor[idx])
            lr_sample_parts.append(lr[idx])
        else:
            factor_sample_parts.append(factor)
            lr_sample_parts.append(lr)

    factor_all = np.vstack(factor_sample_parts).astype(np.float32, copy=False)
    lr_all = np.vstack(lr_sample_parts).astype(np.float32, copy=False)
    loading_lr = ridge_lstsq_nonnegative(factor_all, lr_all)
    if loading_lr.shape[0] != int(dim_envir):
        num_lr = lr_all.shape[1]
        loading_lr = np.random.default_rng(int(seed)).uniform(0.0, 0.1, size=(int(dim_envir), num_lr)).astype(np.float32)
    return loading_lr.astype(np.float32, copy=False)


def random_positive_loading(shape, seed=123, high=0.1):
    return np.random.default_rng(int(seed)).uniform(0.0, high, size=shape).astype(np.float32)


def build_custom_initial_dict(
    context,
    graph_list_device,
    factor_mode,
    dim_intri,
    hp,
    edge_feature_sources,
    use_intrinsic_regression=True,
    uniform_if_no_features=False,
    regress_lr_loading=True,
    seed_offset=0,
    source_notebook="sklearn_compat",
):
    dim_envir = int(context["cfg"]["dim_envir"])
    seed = int(hp["RANDOM_SEED"]) + int(seed_offset)
    num_gene = int(as_numpy_float32(get_data_item(graph_list_device[0], "x")).shape[1])
    num_lr = int(as_numpy_float32(get_data_item(graph_list_device[0], LR_EDGE_FEATURE_KEY)).shape[1])

    feature_list, ncols_by_source = build_edge_feature_list(graph_list_device, edge_feature_sources)
    source_notebook = str(source_notebook)
    print("Modified initialization source notebook:", source_notebook)
    print("Modified initialization edge-feature sources:", edge_feature_sources)
    print("Edge-feature columns by source:", ncols_by_source)

    if feature_list is None or feature_list[0].shape[1] == 0:
        if uniform_if_no_features:
            print("No usable edge features for MI warm-up initialization; using uniform [0, 1] edge-level MI activity.")
            factor_list = make_uniform_factor_init(graph_list_device, dim_envir, seed=seed)
        else:
            raise ValueError(
                "No usable edge features for modified MI warm-up initialization. "
                f"Requested sources: {edge_feature_sources}. Available keys in first graph: {data_keys(graph_list_device[0])}"
            )
    else:
        factor_list = fit_transform_edge_nmf(
            feature_list,
            dim_envir=dim_envir,
            seed=seed,
            max_fit_edges=BENCHMARK1_EDGE_LR_MAX_EDGES,
            source_notebook=source_notebook,
        )
        if factor_list is None:
            if uniform_if_no_features:
                print("NMF input was empty/all-zero; using uniform [0, 1] edge-level MI activity.")
                factor_list = make_uniform_factor_init(graph_list_device, dim_envir, seed=seed)
            else:
                raise ValueError("NMF input was empty/all-zero for modified initialization.")

    loading_intrinsic, loading_receiver, loading_sender = initialize_gene_loadings_from_factors(
        graph_list_device,
        factor_list,
        factor_mode=factor_mode,
        dim_intri=dim_intri,
        dim_envir=dim_envir,
        use_intrinsic_regression=use_intrinsic_regression,
        seed=seed + 101,
        max_cells=200000,
    )

    if not use_intrinsic_regression:
        # Explicitly remove intrinsic loading contribution for No intrinsic initialization.
        loading_intrinsic = np.zeros((int(dim_intri), num_gene), dtype=np.float32)

    if regress_lr_loading:
        loading_lr = initialize_lr_loading_from_factors(
            graph_list_device,
            factor_list,
            dim_envir=dim_envir,
            seed=seed + 203,
            max_edges=BENCHMARK1_EDGE_LR_MAX_EDGES,
        )
    else:
        loading_lr = random_positive_loading((dim_envir, num_lr), seed=seed + 203, high=0.1)

    init = {
        "factor_GP_minibatchNMF": [x.astype(np.float32, copy=False) for x in factor_list],
        "loading_intrinsic_init": loading_intrinsic.astype(np.float32, copy=False),
        "loading_receiver_init": loading_receiver.astype(np.float32, copy=False),
        "loading_sender_init": loading_sender.astype(np.float32, copy=False),
        "loading_LR_init": loading_lr.astype(np.float32, copy=False),
        "modified_initialization_metadata": {
            "edge_feature_sources": list(edge_feature_sources),
            "edge_feature_columns_by_source": ncols_by_source,
            "use_intrinsic_regression": bool(use_intrinsic_regression),
            "uniform_if_no_features": bool(uniform_if_no_features),
            "regress_lr_loading": bool(regress_lr_loading),
            "seed": int(seed),
        },
    }
    return init


def get_initial_dict(context, graph_list_device, factor_mode, dim_intri, hp, ablation_id=None):
    spec = VERSION_SPECS.get(ablation_id, {}) if ablation_id is not None else {}
    init_mode = spec.get("initialization_mode", "standard_original")
    source_notebook = ABLATION_SOURCE_NOTEBOOK.get(ablation_id, "sklearn_compat")

    init_tag = (
        f"init{hp['INITIAL_REGRESSION']}"
        f"_geneCoexp{int(bool(hp['ENHANCE_INIT_WITH_GENE_COEXP']))}"
        f"_{init_mode}"
        f"_source{source_notebook}"
    )
    init_path = context["ablation_root"] / f"Initial_dict_dim{context['cfg']['dim_envir']}_{factor_mode}_{init_tag}.pkl"

    if init_path.exists():
        print("Loading cached Initial_dict:", init_path)
        return load_pickle(init_path)

    print("Computing Initial_dict. This can be slow for large datasets.")
    print("Initialization mode:", init_mode)
    print("Source notebook for this version:", source_notebook)

    if init_mode == "standard_original":
        init = Initial_model(
            graph_list_device,
            dim_envir=context["cfg"]["dim_envir"],
            Factor_mode=factor_mode,
            dim_intri=dim_intri,
            n_jobs=hp["N_JOBS_INIT"],
            Initial_regression=hp["INITIAL_REGRESSION"],
            enhance_init_with_gene_coexp=hp["ENHANCE_INIT_WITH_GENE_COEXP"],
        )
    elif init_mode == "gene_pair_only_edge_init_v2":
        # No LR recon.: edge-level MI activity must not use LR co-expression.
        # Use selected gene-pair co-expression only; if absent/empty, use uniform [0, 1].
        init = build_custom_initial_dict(
            context,
            graph_list_device,
            factor_mode=factor_mode,
            dim_intri=dim_intri,
            hp=hp,
            edge_feature_sources=["gene_pair"],
            use_intrinsic_regression=True,
            uniform_if_no_features=True,
            regress_lr_loading=False,
            seed_offset=1000,
            source_notebook=source_notebook,
        )
    elif init_mode == "lr_only_edge_init_v2":
        # No gene recon.: edge-level MI activity must not use selected gene-pair co-expression.
        # Use LR co-expression only.
        init = build_custom_initial_dict(
            context,
            graph_list_device,
            factor_mode=factor_mode,
            dim_intri=dim_intri,
            hp=hp,
            edge_feature_sources=["lr"],
            use_intrinsic_regression=True,
            uniform_if_no_features=False,
            regress_lr_loading=True,
            seed_offset=2000,
            source_notebook=source_notebook,
        )
    elif init_mode == "no_intrinsic_regression_init_v2":
        # No intrinsic: remove cell-type one-hot intrinsic regression and initialize
        # sender/receiver gene loadings without an intrinsic expression component.
        init = build_custom_initial_dict(
            context,
            graph_list_device,
            factor_mode=factor_mode,
            dim_intri=dim_intri,
            hp=hp,
            edge_feature_sources=["lr", "gene_pair"],
            use_intrinsic_regression=False,
            uniform_if_no_features=False,
            regress_lr_loading=True,
            seed_offset=3000,
            source_notebook=source_notebook,
        )
    else:
        raise ValueError(f"Unknown initialization_mode: {init_mode}")

    save_pickle(init, init_path)
    print("Saved Initial_dict:", init_path)
    return init


def train_or_load_ablation(context, ablation_id):
    spec = VERSION_SPECS[ablation_id]
    source_notebook = ABLATION_SOURCE_NOTEBOOK.get(ablation_id, "sklearn_compat")
    hp = get_training_hparams(context["dataset_key"])
    model_dir = context["ablation_root"] / ablation_id
    factor_path = model_dir / "Factor_envir_list.pkl"
    history_path = model_dir / "training_history.csv"
    model_dir.mkdir(parents=True, exist_ok=True)

    if factor_path.exists() and not RETRAIN_ABLATIONS and cache_matches_current_hparams(model_dir, hp, spec=spec):
        print(f"[{context['dataset_key']} | {spec['label']}] loading cached factors:", factor_path)
        print("  source notebook preference:", source_notebook)
        return load_pickle(factor_path)

    if factor_path.exists() and not RETRAIN_ABLATIONS:
        print(f"[{context['dataset_key']} | {spec['label']}] cached factors exist but hyperparameters/initialization spec changed; retraining.")

    if not RUN_TRAINING_IF_MISSING:
        raise FileNotFoundError(f"Missing valid cached factors and RUN_TRAINING_IF_MISSING=False: {factor_path}")

    print(f"\nTraining ablation: {context['dataset_key']} | {spec['label']} ({ablation_id})")
    print("Source notebook for this ablation:", source_notebook)
    graph_list_device = canonicalize_graphs_for_model(context["graph_list_cpu"], device=DEVICE)
    factor_mode, dim_intri = infer_factor_mode_and_dim_intri(graph_list_device, context["cfg"].get("cellclass_name", "celltype"))
    print("Factor_mode:", factor_mode, "dim_intri:", dim_intri)

    print("Training hyperparameters:", hp)

    init = get_initial_dict(context, graph_list_device, factor_mode, dim_intri, hp, ablation_id=ablation_id)
    num_gene = int(get_data_item(graph_list_device[0], "x").shape[1])
    num_LR = int(get_data_item(graph_list_device[0], "cellpair_LRpair_neigh").shape[1])

    set_seed(hp["RANDOM_SEED"])
    model = SpiderNetAblationModel(
        num_gene=num_gene,
        num_LR=num_LR,
        hidden_channels=hp["HIDDEN_CHANNELS"],
        Factor_mode=factor_mode,
        dim_intri=dim_intri,
        dim_envir=context["cfg"]["dim_envir"],
        shared_role_encoder=spec["shared_role_encoder"],
        nonnegative_loadings=spec["nonnegative_loadings"],
        use_intrinsic_component=spec["use_intrinsic_component"],
    ).to(DEVICE)

    history = model.fit_ablation(
        graph_list_device,
        device=DEVICE,
        optim_type=hp["OPTIM_TYPE"],
        lr=hp["LEARNING_RATE"],
        weight_decay=hp["WEIGHT_DECAY"],
        expr_loss_weight=spec["expr_loss_weight"],
        lr_loss_weight=spec["lr_loss_weight"],
        warmup=hp["WARMUP_EPOCHS"],
        max_epoch=hp["MAX_EPOCHS"],
        loss_fn=hp["LOSS_FN"],
        Initial_dict=init,
        file_savepath_model_main=model_dir,
        log_every=100,
    )
    history.to_csv(history_path, index=False)

    factor_list = extract_factor_envir_list(model, graph_list_device)
    save_pickle(factor_list, factor_path)
    with open(model_dir / "ablation_spec.json", "w", encoding="utf-8") as f:
        json.dump({
            "ablation_id": ablation_id,
            "ablation_spec": {k: v for k, v in spec.items() if isinstance(v, (str, int, float, bool))},
            "training_hparams": hp,
            "source_notebook": source_notebook,
        }, f, indent=2)
    print("Saved ablation factors:", factor_path)
    return factor_list


def aggregate_sender_receiver(edge_features, edge_index, num_cells, device=DEVICE):
    edge_features = np.asarray(edge_features, dtype=np.float32)
    e2 = edge_index_to_e2(edge_index)
    if e2.shape[0] != edge_features.shape[0]:
        raise ValueError(f"edge feature rows {edge_features.shape[0]} != edge_index rows {e2.shape[0]}")
    edge_tensor = torch.as_tensor(edge_features, dtype=torch.float32, device=device)
    sender = torch.as_tensor(e2[:, 0], dtype=torch.long, device=device)
    receiver = torch.as_tensor(e2[:, 1], dtype=torch.long, device=device)
    receiver_agg = scatter_mean(edge_tensor, receiver, dim=0, dim_size=num_cells).detach().cpu().numpy()
    sender_agg = scatter_mean(edge_tensor, sender, dim=0, dim_size=num_cells).detach().cpu().numpy()
    return {"sender": sender_agg, "receiver": receiver_agg}
