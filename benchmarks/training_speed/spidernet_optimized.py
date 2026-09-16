"""Opt-in SpiderNet training implementation; the original source is never edited.

The architecture, parameter names, AMP policy, batch weights, blocked updates,
optimizer resets and scheduler inputs are inherited/preserved. Static inputs
must not be mutated while fit is running. No encoder activations are cached.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_scatter import scatter_mean, scatter_sum


@dataclass(frozen=True)
class OptimizationOptions:
    defer_metrics: bool = True
    warmup_factor_only: bool = True
    cache_targets: bool = True
    cache_degrees: bool = True
    target_cache_max_bytes: int = 256 * 1024 * 1024

    def __post_init__(self):
        if self.target_cache_max_bytes < 0:
            raise ValueError("target_cache_max_bytes must be nonnegative")


def load_baseline_module(path=None):
    """Load the actual original model.py, without importing the package API."""
    source = Path(path) if path else Path(__file__).resolve().parents[2] / "SpiderNet" / "SpiderNet" / "model.py"
    source = source.resolve(strict=True)
    name = "_spidernet_training_baseline_" + str(abs(hash(str(source))))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_optimized_model_class(baseline_module):
    class SpiderNet_model_optimized(baseline_module.SpiderNet_model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.optimization_options = OptimizationOptions()
            self.training_history = []
            self.clear_optimization_cache()
            self._collect_metrics = True

        def clear_optimization_cache(self):
            self._target_cache = {}
            self._degree_cache = {}
            self.optimization_stats = {
                "target_cache_bytes": 0, "target_cache_entries": 0,
                "target_cache_peak_bytes": 0,
                "target_cache_hits": 0, "target_cache_budget_skips": 0,
                "degree_cache_entries": 0,
            }

        @staticmethod
        def _tensor_key(tensor):
            return (tensor.data_ptr(), tensor._version, tuple(tensor.shape),
                    tuple(tensor.stride()), tensor.dtype, tensor.device)

        def _target(self, target, dtype):
            if target.dtype == dtype:
                return target
            options = self.optimization_options
            if not options.cache_targets:
                return target.to(dtype)
            key = (self._tensor_key(target), dtype)
            if key in self._target_cache:
                self.optimization_stats["target_cache_hits"] += 1
                return self._target_cache[key][1]
            converted = target.to(dtype)
            size = converted.numel() * converted.element_size()
            if self.optimization_stats["target_cache_bytes"] + size <= options.target_cache_max_bytes:
                # Retain the source storage: a future tensor must not reuse its
                # address and accidentally hit a cache entry for different data.
                self._target_cache[key] = (target, converted)
                self.optimization_stats["target_cache_bytes"] += size
                self.optimization_stats["target_cache_entries"] += 1
                self.optimization_stats["target_cache_peak_bytes"] = max(
                    self.optimization_stats["target_cache_peak_bytes"],
                    self.optimization_stats["target_cache_bytes"])
            else:
                self.optimization_stats["target_cache_budget_skips"] += 1
            return converted

        def _mean(self, values, index, num_cells):
            if not self.optimization_options.cache_degrees:
                return scatter_mean(values, index, dim=0, dim_size=num_cells)
            # Match torch_scatter.scatter_mean exactly: scatter_sum, SAME-dtype
            # ones/count, clamp empty neighborhoods to one, then true_divide_.
            # Do not replace division with multiplication by a reciprocal.
            key = (self._tensor_key(index), values.dtype, values.device, num_cells)
            entry = self._degree_cache.get(key)
            if entry is None:
                ones = torch.ones(index.size(), dtype=values.dtype, device=values.device)
                count = scatter_sum(ones, index, dim=0, dim_size=num_cells)
                count[count < 1] = 1
                count = count.view(-1, 1)
                self._degree_cache[key] = (index, count)
                self.optimization_stats["degree_cache_entries"] += 1
            else:
                count = entry[1]
            result = scatter_sum(values, index, dim=0, dim_size=num_cells)
            result.true_divide_(count)
            return result

        def _environment(self, data):
            exp = data.x
            edge_index = data['edge_index']
            exp_enc_receiver = self.enc_factor_envir_pre_receiver(exp)
            exp_enc_sender = self.enc_factor_envir_pre_sender(exp)
            receiver = exp_enc_receiver[edge_index[:, 1], :]
            sender = exp_enc_sender[edge_index[:, 0], :]
            combined = torch.cat((sender, receiver), dim=1)
            return self.Sigmoid(self.enc_factor_envir(combined))

        def forward(self, data):
            with autocast():
                exp = data.x
                edge_index = data['edge_index']
                if self.Factor_mode == "NMF":
                    intrinsic = self.enc_factor_intrinsic(exp)
                    intrinsic = intrinsic / intrinsic.sum(dim=1, keepdim=True).clamp(min=1e-8)
                else:
                    intrinsic = data[self.Factor_mode + '_onehot']
                factor = self._environment(data)
                receiver_agg = self._mean(factor, edge_index[:, 1], exp.shape[0])
                sender_agg = self._mean(factor, edge_index[:, 0], exp.shape[0])
                loading_intrinsic = self.Relu(self.Loading_intrinsic_ori)
                loading_receiver = self.Relu(self.loading_receiver_ori)
                loading_sender = self.Relu(self.loading_sender_ori)
                loading_lr = self.Relu(self.loading_LR_ori)
                exp_recon = torch.matmul(intrinsic, loading_intrinsic)
                receiver_recon = torch.matmul(receiver_agg, loading_receiver)
                sender_recon = torch.matmul(sender_agg, loading_sender)
                exp_recon.add_(receiver_recon).add_(sender_recon)
                lr_recon = torch.matmul(factor, loading_lr)
                return (exp_recon, lr_recon, intrinsic, loading_intrinsic, factor,
                        loading_receiver, loading_sender, loading_lr)

        def compute_loss_in_batches(self, SpiderNet_data_pyg_list, factor_envir_init,
                                    stage, criterion, scaler=None, loss_weight=1.0):
            num_batches = len(SpiderNet_data_pyg_list)
            totals = [0.0, 0.0, 0.0]
            pending = [[], [], []]
            collect = self._collect_metrics or not self.optimization_options.defer_metrics
            for batch_index, data in enumerate(SpiderNet_data_pyg_list):
                if stage == 'warmup' and self.optimization_options.warmup_factor_only:
                    with autocast():
                        factor = self._environment(data)
                else:
                    exp_recon, lr_recon, _, _, factor, _, _, _ = self(data)
                with autocast():
                    if stage == 'warmup':
                        target = self._target(factor_envir_init[batch_index], factor.dtype)
                        loss_batch = criterion(factor, target) * (1 / num_batches)
                        metrics = ((0, loss_batch),)
                    elif stage == 'main':
                        target_exp = self._target(data.x, exp_recon.dtype)
                        loss_exp = criterion(exp_recon, target_exp) * (1 / num_batches)
                        target_lr = self._target(data['cellpair_LRpair_neigh'], lr_recon.dtype)
                        loss_lr = criterion(lr_recon, target_lr) * (1 / num_batches)
                        loss_batch = loss_exp + loss_lr * loss_weight
                        metrics = ((1, loss_exp), (2, loss_lr))
                    else:
                        raise ValueError(f"Unsupported stage {stage}")
                if scaler is not None:
                    scaler.scale(loss_batch).backward()
                else:
                    loss_batch.backward()
                if collect:
                    for position, value in metrics:
                        if self.optimization_options.defer_metrics:
                            pending[position].append(value.detach())
                        else:
                            totals[position] += value.item()
            # Transfer scalar VALUES, not computation graphs. Summation remains
            # sequential Python float addition (not a GPU/NumPy reduction).
            if collect and self.optimization_options.defer_metrics:
                lengths = [len(items) for items in pending]
                flat = [value for items in pending for value in items]
                values = torch.stack(flat).cpu().tolist() if flat else []
                offset = 0
                for position, length in enumerate(lengths):
                    for value in values[offset:offset + length]:
                        totals[position] += value
                    offset += length
            return tuple(totals)

        def fit(self, SpiderNet_data_pyg_list, device='cuda', optim_type='adam',
                lr=1e-3, weight_decay=1e-5, LR_loss_weight=1, warmup=10,
                max_epoch=100, loss_fn='mse', Initial_dict=None,
                file_savepath_model_main=None, *, options=None, record_history=False):
            self.optimization_options = options or OptimizationOptions()
            self.clear_optimization_cache()
            self.training_history = []
            if not SpiderNet_data_pyg_list:
                raise ValueError("At least one batch is required")
            if loss_fn == 'mse':
                criterion = nn.MSELoss()
            elif loss_fn == 'poisson':
                criterion = nn.PoissonNLLLoss(log_input=False)
            else:
                raise ValueError(f"Unsupported loss_fn {loss_fn}")
            if optim_type not in ('adam', 'SGD'):
                raise ValueError("optim_type must be 'adam' or 'SGD', as in the original")
            scaler = GradScaler()
            loading_names = ['Loading_intrinsic_ori', 'loading_receiver_ori',
                             'loading_sender_ori', 'loading_LR_ori']
            initial_keys = ['loading_intrinsic_init', 'loading_receiver_init',
                            'loading_sender_init', 'loading_LR_init']
            init_params = {name: torch.tensor(Initial_dict[key]).to(device).to(torch.float32)
                           for name, key in zip(loading_names, initial_keys)}
            factor_init = [torch.tensor(value).to(device).to(torch.float32)
                           for value in Initial_dict['factor_GP_minibatchNMF']]
            named = dict(self.named_parameters())
            intrinsic_params = [p for n, p in named.items() if "enc_factor_intrinsic" in n]
            envir_params = [p for n, p in named.items() if "enc_factor_envir" in n]
            loading_params = [p for n, p in named.items() if n in loading_names]
            optimizer = None
            for epoch in range(max_epoch):
                self.train()
                if epoch == warmup and epoch > 0:
                    # Warmup targets are no longer read. Reuse the bounded
                    # conversion budget for expression/LR targets in main fit.
                    self._target_cache.clear()
                    self.optimization_stats["target_cache_bytes"] = 0
                    self.optimization_stats["target_cache_entries"] = 0
                # Preserve the epoch==0 precedence, including warmup==0.
                if epoch == 0 or epoch == warmup:
                    args = dict(lr=5e-4 if epoch == 0 else lr, weight_decay=weight_decay)
                    optimizer = (optim.Adam(self.parameters(), **args) if optim_type == 'adam'
                                 else optim.SGD(self.parameters(), **args))
                    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9,
                                                 patience=20, min_lr=1e-5)
                for param in named.values():
                    param.requires_grad = True
                if epoch < warmup:
                    for name in loading_names:
                        named[name].requires_grad = False
                        named[name].data.copy_(init_params[name].to(named[name].dtype))
                else:
                    if self.Factor_mode == 'NMF':
                        frozen = (envir_params + loading_params if epoch % 3 == 0 else
                                  intrinsic_params + loading_params if epoch % 3 == 1 else
                                  intrinsic_params + envir_params)
                    else:
                        frozen = (envir_params if epoch % 2 < 1 or epoch < warmup + 10
                                  else loading_params)
                    for param in frozen:
                        param.requires_grad = False
                optimizer.zero_grad(set_to_none=True)
                stage = 'warmup' if epoch < warmup else 'main'
                print_epoch = (epoch == 0 or epoch == warmup - 1) if stage == 'warmup' else epoch % 1000 == 0
                scheduler_epoch = epoch >= warmup and epoch % 10 == 0
                self._collect_metrics = bool(record_history or print_epoch or scheduler_epoch)
                with autocast():
                    warm_loss, exp_loss, lr_loss = self.compute_loss_in_batches(
                        SpiderNet_data_pyg_list, factor_init, stage, criterion, scaler,
                        loss_weight=1 if stage == 'warmup' else LR_loss_weight)
                    # The legacy scheduler deliberately receives UNWEIGHTED
                    # exp+LR, even when backward uses LR_loss_weight != 1.
                    loss_total = warm_loss if stage == 'warmup' else exp_loss + lr_loss
                    if print_epoch:
                        if stage == 'warmup':
                            print(f"Epoch: {epoch} train_loss: {loss_total} loss_factor_intrinsic: 0 loss_factor_envir: {warm_loss}")
                        else:
                            print(f"Epoch: {epoch} train_loss: {loss_total}Loss_exprecon:{exp_loss}Loss_expLRpair_recon:{lr_loss}")
                scaler.step(optimizer)
                scaler.update()
                if scheduler_epoch:
                    scheduler.step(loss_total)
                if record_history:
                    self.training_history.append(dict(
                        epoch=epoch, loss_warmup=warm_loss, loss_exp=exp_loss,
                        loss_LR=lr_loss, loss_total=loss_total,
                        lr=optimizer.param_groups[0]['lr'], scale=scaler.get_scale()))
                if epoch % 1000 == 0 or epoch == max_epoch - 1:
                    save_dir = Path(file_savepath_model_main)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(self.state_dict(), save_dir / f"model_epoch{epoch}.pth")
            # Retain the original post-fit gradient state without clearing it
            # twice on every iteration. No optimizer.step is added or removed.
            if optimizer is not None:
                optimizer.zero_grad()
            self._collect_metrics = True

    SpiderNet_model_optimized.__name__ = "SpiderNet_model_optimized"
    return SpiderNet_model_optimized
