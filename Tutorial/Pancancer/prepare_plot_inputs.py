"""Export missing drawing caches from aligned, saved analyses without inference.

The original notebook cells select edges and count triplets. Only drawing is
omitted here; plot-only subsequently invokes the original renderers. The graph
reader leaves unused expression/LR tensors on disk, preserving tensor strides.
"""
from __future__ import annotations

import io
import json
import pickle
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent


class _StorageRef:
    def __init__(self, path, offset, size):
        self.path, self.offset, self.size = path, offset, size


def _lazy_storage(value):
    return value


class _TensorRef:
    def __init__(self, storage, offset, shape, stride, *rest):
        self.storage, self.offset, self.shape, self.stride = storage, offset, shape, stride

    def numpy(self):
        value = self.storage
        if isinstance(value, _StorageRef):
            with open(value.path, "rb") as handle:
                handle.seek(value.offset)
                value = handle.read(value.size)
        storage = torch.load(io.BytesIO(value), map_location="cpu", weights_only=False)
        return torch._utils._rebuild_tensor(storage, self.offset, self.shape, self.stride).numpy()

    def cpu(self):
        return self


class _GraphUnpickler(pickle._Unpickler):
    """Defer embedded Torch storages in the trusted upstream graph pickle."""
    dispatch = pickle._Unpickler.dispatch.copy()

    def __init__(self, handle):
        super().__init__(handle)
        self.input_file = handle

    def find_class(self, module, name):
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            return _lazy_storage
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _TensorRef
        return super().find_class(module, name)

    def _read_bytes(self, width):
        size = struct.unpack("<I" if width == 4 else "<Q", self.read(width))[0]
        frame = self._unframer.current_frame
        remaining = 0 if frame is None else len(frame.getbuffer()) - frame.tell()
        if self.stack and self.stack[-1] is _lazy_storage and remaining == 0:
            value = _StorageRef(self.input_file.name, self.input_file.tell(), size)
            self.input_file.seek(size, 1)
        else:
            value = self.read(size)
        self.append(value)

    def load_binbytes(self):
        self._read_bytes(4)

    def load_binbytes8(self):
        self._read_bytes(8)

    dispatch[pickle.BINBYTES[0]] = load_binbytes
    dispatch[pickle.BINBYTES8[0]] = load_binbytes8


def _cells(name):
    book = json.loads((HERE / name).read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in book["cells"] if cell["cell_type"] == "code"]


def _one(cells, marker):
    found = [source for source in cells if marker in source]
    if len(found) != 1:
        raise ValueError(f"Expected one original notebook cell containing {marker!r}.")
    return found[0]


def _without_between(source, start, end, replacement=""):
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError(f"Notebook export boundaries changed: {start!r}, {end!r}")
    i, j = source.index(start), source.index(end)
    if i >= j:
        raise ValueError("Notebook export boundaries are out of order.")
    return source[:i] + replacement + source[j:]


def _compact_bundle(processed_dir):
    import h5py
    from anndata._io.specs import read_elem

    with (processed_dir / "SpiderNet_data_pyg_list.pkl").open("rb") as handle:
        graphs = _GraphUnpickler(handle).load()
    batches = np.asarray(pd.read_pickle(processed_dir / "batch_cell_unique.pkl")).astype(str)
    with h5py.File(processed_dir / "adata_all.h5ad", "r") as handle:
        obs = pd.DataFrame({name: read_elem(handle["obs"][name])
                            for name in ("SampleID", "celltype_final")})
        obs.index = read_elem(handle["obs"][handle["obs"].attrs["_index"]])
        coords = read_elem(handle["obsm"]["spatial"])
    groups = obs.groupby("SampleID", observed=True, sort=False).indices
    if len(graphs) != len(batches) or set(groups) != set(batches):
        raise ValueError("Graph, batch and annotation sample inventories differ.")
    adatas = []
    # The upstream exporter builds adata_list by selecting each SampleID from
    # this same adata_all object, preserving within-sample row order.
    for sample, graph in zip(batches, graphs):
        positions = groups[sample]
        sample_obs = obs.iloc[positions].copy()
        spatial = coords[positions]
        # Validate every cell's identity, label and coordinate against the graph;
        # preserve the original float64 spatial coordinates for cascade drawings.
        if not np.array_equal(sample_obs.index.astype(str), np.asarray(graph.cellnames).astype(str)):
            raise ValueError(f"AnnData/graph cell ordering differs for {sample}.")
        if not np.array_equal(sample_obs.celltype_final.astype(str), np.asarray(graph.cell_class).astype(str)):
            raise ValueError(f"AnnData/graph cell annotations differ for {sample}.")
        np.testing.assert_array_equal(spatial.astype(np.float32), graph.pos.numpy())
        adatas.append(SimpleNamespace(obs=sample_obs, obsm={"spatial": spatial}, n_obs=len(sample_obs)))
    print(f"Validated cell identities, labels and coordinates for all {len(batches)} sub-slices.", flush=True)
    return SimpleNamespace(adata_list=adatas, spidernet_data=graphs)


def _release(*names, namespace, **kwargs):
    import gc
    for name in names:
        namespace.pop(name, None)
    gc.collect()


def prepare_spatial(results, processed_dir, bundle):
    cells = _cells("Pancancer_analysis_V2.ipynb")
    env = dict(processed=bundle, processed_data_dir=processed_dir, run_dirs={"run_dir": results},
               save_path_insituMI=str(results / "In_situ_meta_interaction"), MI_index=3, vis_mode=1,
               release_memory=_release)
    source = _one(cells, "def save_insitu_figure(")
    source = _without_between(source, "        norm_edge = Normalize", "    finally:\n        plt.close",
                              "        drawing_cache_frames.append(drawing_cache_path)\n        fig = None\n")
    exec(compile(source, "<original MI4 cache export>", "exec"), env)
    source = _one(cells, "def select_mi2_representative_subslices(")
    source = _without_between(source, "            norm = Normalize", "    write_drawing_manifest(drawing_cache_dir, drawing_cache_frames")
    exec(compile(source, "<original MI2 cache export>", "exec"), env)


def prepare_cascade(results, processed_dir, bundle):
    cells = _cells("Pancancer_MIcascade_analysis_V3.ipynb")
    env = dict(Path=Path, SpiderNet_data_pyg_list=bundle.spidernet_data, adata_list=bundle.adata_list,
               file_savepath_main=str(results), display=lambda *args: None)
    exec(_one(cells, "from matplotlib.patches import Rectangle"), env)
    metadata = _one(cells, "def extract_cancer_type(")
    # The three lists are already loaded or supplied by the exact saved cache.
    metadata = metadata[metadata.index("samples_name_list ="):]
    exec(metadata, env)
    exec(_one(cells, "pvalue_thresold = 1e-3"), env)
    cache_path = results / "MIcascade_colocalization_cache/MIcascade_colocalization_cache_MIthreshold0p5_nperm100.pkl"
    print("Loading saved normalization and adjacency; no permutations are run.", flush=True)
    cache = pd.read_pickle(cache_path)
    raw = pd.read_pickle(results / "Factor_envir_list.pkl")
    saved_norm = cache["Factor_envir_norm_list"]
    if len(raw) != len(bundle.adata_list) or len(saved_norm) != len(raw):
        raise ValueError("Factor and graph sample inventories differ.")
    maxima = np.max(np.concatenate(raw, axis=0), axis=0)
    np.testing.assert_array_equal(maxima, cache["Factor_envir_all_max"])
    for index, (factor, norm, graph) in enumerate(zip(raw, saved_norm, bundle.spidernet_data)):
        np.testing.assert_array_equal(factor / (maxima + 1e-8), norm)
        if len(factor) != graph.edge_index.shape[0]:
            raise ValueError(f"Factor/edge count mismatch in sample {index}.")
    del raw
    env.update(Factor_envir_norm_list=saved_norm, hyper_edge_adj_list=cache["hyper_edge_adj_list"],
               MI_colocal_summary_significant_by_cancertype=pd.read_csv(results / "MI_colocal_summary_significant_by_cancertype.csv"))
    # Check every saved adjacency against the current edge list, one sample at a time.
    import ast
    helpers = ast.parse(_one(cells, "def build_hyper_edge_adj("))
    exec(compile(ast.Module(body=[helpers.body[0]], type_ignores=[]), "<original adjacency helper>", "exec"), env)
    for graph, adjacency in zip(bundle.spidernet_data, env["hyper_edge_adj_list"]):
        edges = graph.edge_index.numpy()
        actual = env["build_hyper_edge_adj"](edges, len(edges))
        if actual.shape != adjacency.shape or (actual != adjacency).nnz:
            raise ValueError("Saved cascade adjacency differs from the current graph.")
    print("Validated saved MI normalization and all cascade adjacency matrices.", flush=True)
    exec(_one(cells, "target_MI_first = 7"), env)
    # The historical significant-pair table supplies an independent comparison
    # for the cancer types where this selected pair was significant.
    old_triplets = pd.read_csv(results / "Celltype_triple_prop_within_cancertype_long.csv")
    selected = pd.read_csv(results / "Celltype_triple_prop_selected_MI-7_MI-4_cancercell.csv")
    old_triplets = old_triplets.loc[
        old_triplets.MI_first.eq(7) & old_triplets.MI_second.eq(4)
        & old_triplets.Celltype_triple.str.contains("cancercell", case=False, na=False)]
    keys = ["CancerType", "Celltype_triple"]
    comparison = old_triplets.merge(selected, on=keys, suffixes=("_old", "_new"), validate="one_to_one")
    if len(comparison) != len(old_triplets):
        raise ValueError("Selected triplets are missing entries from the existing significant-pair table.")
    np.testing.assert_allclose(comparison.prop_within_cancertype_old,
                               comparison.prop_within_cancertype_new, rtol=1e-12, atol=1e-15)
    source = _one(cells, "cascade_count_summary = []")
    # Retain the original function and replace just the rendering call with a no-op.
    cut = source.index("cascade_count_summary = []")
    exec(source[:cut], env)
    env["plot_high_order_MI"] = lambda **kwargs: []
    # Compare every count to the existing analysis before writing summary tables.
    suffix = source[cut:]
    suffix = suffix[:suffix.index("# --------------------------------------------------------------\n# Save summary CSVs")]
    publish_start = suffix.index('drawing_manifest["status"] = "complete"')
    publish_end = suffix.index("# --------------------------------------------------------------\n# Summary tables", publish_start)
    publish = suffix[publish_start:publish_end]
    suffix = suffix[:publish_start] + suffix[publish_end:]
    exec(suffix, env)
    expected_path = results / "Insitu_High_order_MI_AllCancerTypes/ObservedAndPotentialCount_summary_cancercell-Fibroblast-cancercell_MI-7_MI-4.csv"
    expected = pd.read_csv(expected_path)
    actual = env["cascade_count_summary_df"]
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False)
    exec(publish, env)
    print("All cascade spatial counts match the existing analysis exactly.", flush=True)


def run(stages, results_dir, processed_dir):
    bundle = _compact_bundle(processed_dir)
    for stage in stages:
        if stage == "spatial":
            prepare_spatial(results_dir, processed_dir, bundle)
        elif stage == "cascade":
            prepare_cascade(results_dir, processed_dir, bundle)
        else:
            raise ValueError(f"Unsupported drawing-cache preparation: {stage}")
