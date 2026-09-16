"""Curated sender/receiver coupling, summary statistics and saved-result plotting."""
from pathlib import Path
from collections import OrderedDict, defaultdict
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from scipy.stats import spearmanr, mannwhitneyu, t
from ablation_settings import (
    VERSION_ORDER, VERSION_SPECS, VERSION_COLORS, SAVE_FIGURES, FIG_DPI,
    ABLATION_SOURCE_NOTEBOOK, coupling_output_dir,
)

def set_plot_style():
    plt.close("all")
    plt.style.use("default")
    rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "Arial",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "figure.dpi": 150,
        "savefig.dpi": FIG_DPI,
    })


def join_plus(x):
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return "+".join([str(z) for z in x])
    return str(x)


def mean_ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    m = float(np.mean(x))
    if n == 1:
        return m, m, m, 1
    sem = np.std(x, ddof=1) / np.sqrt(n)
    half_width = t.ppf(0.975, df=n - 1) * sem
    return m, float(m - half_width), float(m + half_width), int(n)


def one_sided_mw_greater(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    try:
        return mannwhitneyu(x, y, alternative="greater", method="asymptotic").pvalue
    except TypeError:
        return mannwhitneyu(x, y, alternative="greater").pvalue


def dedup_gene_sets(gene_sets_by_lr):
    setid_by_lr = {}
    repkey_by_setid = {}
    genes_by_setid = OrderedDict()
    lrs_by_setid = defaultdict(list)
    key_to_setid = {}
    for lr_key, genes in gene_sets_by_lr.items():
        genes_clean = tuple(sorted(set([str(g) for g in genes if pd.notna(g)])))
        if len(genes_clean) == 0:
            continue
        if genes_clean not in key_to_setid:
            set_id = f"Set{len(key_to_setid) + 1}"
            key_to_setid[genes_clean] = set_id
            genes_by_setid[set_id] = list(genes_clean)
            repkey_by_setid[set_id] = lr_key
        set_id = key_to_setid[genes_clean]
        setid_by_lr[lr_key] = set_id
        lrs_by_setid[set_id].append(lr_key)
    return genes_by_setid, repkey_by_setid, setid_by_lr, lrs_by_setid


def cell_by_pair_mean_genesets(adata_sub, genesets_by_LR_unique, use_layer=None, dtype=np.float32, fill_value=np.nan):
    X = adata_sub.X if use_layer is None else adata_sub.layers[use_layer]
    var_names = np.asarray(adata_sub.var_names.astype(str))
    gene_to_idx = {g: i for i, g in enumerate(var_names)}
    pair_names = list(genesets_by_LR_unique.keys())
    n_pairs = len(pair_names)
    rows, cols, data = [], [], []
    n_found = np.zeros(n_pairs, dtype=np.int32)
    for j, lr in enumerate(pair_names):
        genes = genesets_by_LR_unique[lr]
        idx = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        if len(idx) == 0:
            continue
        idx = np.unique(idx)
        n_found[j] = len(idx)
        rows.extend(idx.tolist())
        cols.extend([j] * len(idx))
        data.extend([1.0 / float(len(idx))] * len(idx))
    W = sp.csr_matrix((data, (rows, cols)), shape=(adata_sub.n_vars, n_pairs), dtype=dtype)
    if sp.issparse(X):
        scores = (X @ W).toarray().astype(dtype, copy=False)
    else:
        scores = (np.asarray(X) @ W.toarray()).astype(dtype, copy=False)
    if np.any(n_found == 0):
        scores[:, n_found == 0] = fill_value
    return pd.DataFrame(scores, index=adata_sub.obs_names, columns=pair_names), n_found, pair_names


def compute_rowmax_spearman_df(X_df, Y_array, slice_index, prefix="Slice_"):
    """For each representative LR gene-set score column, report max Spearman correlation over MI dimensions."""
    Y = np.asarray(Y_array, dtype=float)
    records = {}
    for col in X_df.columns:
        x = pd.to_numeric(X_df[col], errors="coerce").to_numpy(dtype=float)
        vals = []
        for j in range(Y.shape[1]):
            y = Y[:, j]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 3 or np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
                vals.append(np.nan)
            else:
                vals.append(spearmanr(x[ok], y[ok]).correlation)
        vals = np.asarray(vals, dtype=float)
        records[col] = np.nanmax(vals) if np.any(np.isfinite(vals)) else np.nan
    return pd.DataFrame(records, index=[f"{prefix}{slice_index}"])


def flatten_lr_genes(x):
    """
    Convert one LR side into a clean list of gene symbols.
    Handles string, list/tuple/set/np.ndarray, and nested structures.
    """
    if x is None:
        return []

    if isinstance(x, str):
        x = x.strip()
        if len(x) == 0:
            return []
        # In case some LR entries are stored as "A+B"
        return [g.strip() for g in x.split("+") if len(g.strip()) > 0]

    if isinstance(x, (list, tuple, set, np.ndarray, pd.Index)):
        out = []
        for z in x:
            out.extend(flatten_lr_genes(z))
        return list(dict.fromkeys(out))

    if pd.isna(x):
        return []

    return [str(x).strip()]


def load_benchmark1_annotations(context):
    cfg = context["cfg"]
    adata_all = context["adata_all"]
    gene_names = adata_all.var_names.astype(str).tolist()
    gene_set = set(gene_names)
    LR_list = context["LR_list"]

    targets_path = context["run_dir"] / cfg["targets_json_name"]
    if not targets_path.exists():
        raise FileNotFoundError(f"Missing target annotation JSON: {targets_path}")

    with open(targets_path, "r", encoding="utf-8") as f:
        targets_by_LR = json.load(f)

    targets_by_LR = {
        str(k): [str(g) for g in v if str(g) in gene_set]
        for k, v in targets_by_LR.items()
        if isinstance(v, (list, tuple))
    }
    targets_by_LR = {k: v for k, v in targets_by_LR.items() if len(v) > 0}

    targets_by_setid, repkey_targets, _, _ = dedup_gene_sets(targets_by_LR)
    targets_by_LR_unique = {
        repkey_targets[set_id]: genes
        for set_id, genes in targets_by_setid.items()
    }

    regulatory_csv = Path(cfg["omnipath_regulatory_csv"])
    if not regulatory_csv.exists():
        raise FileNotFoundError(f"Missing OmniPath regulatory CSV: {regulatory_csv}")

    regulatory = pd.read_csv(regulatory_csv)

    # Make boolean filtering robust
    for col in ["is_stimulation", "is_inhibition"]:
        if col in regulatory.columns and regulatory[col].dtype != bool:
            regulatory[col] = (
                regulatory[col]
                .astype(str)
                .str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
                .fillna(False)
                .astype(bool)
            )

    regulatory_stim = regulatory.loc[
        regulatory["is_stimulation"] & (~regulatory["is_inhibition"])
    ].copy()

    regulatory_stim["target_genesymbol"] = (
        regulatory_stim["target_genesymbol"]
        .dropna()
        .astype(str)
    )
    regulatory_stim["source_genesymbol"] = (
        regulatory_stim["source_genesymbol"]
        .dropna()
        .astype(str)
    )

    regulatory_target_set = set(regulatory_stim["target_genesymbol"].dropna().astype(str))

    upstream_regulators = {}

    for LR in LR_list:
        ligand_genes = flatten_lr_genes(LR[0])
        receptor_genes = flatten_lr_genes(LR[1])

        lr_key = f"{join_plus(LR[0])}->{join_plus(LR[1])}"

        # For ligand complexes, use regulators of any ligand subunit.
        matched_ligands = [g for g in ligand_genes if g in regulatory_target_set]
        if len(matched_ligands) == 0:
            continue

        regs = (
            regulatory_stim.loc[
                regulatory_stim["target_genesymbol"].isin(matched_ligands),
                "source_genesymbol",
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        regs = [r for r in regs if r in gene_set]

        if len(regs) >= int(cfg["min_upstream_regulators"]):
            upstream_regulators[lr_key] = regs

    regulators_by_setid, repkey_regulators, _, _ = dedup_gene_sets(upstream_regulators)
    regulators_by_LR_unique = {
        repkey_regulators[set_id]: genes
        for set_id, genes in regulators_by_setid.items()
    }

    print(
        f"[{context['dataset_key']}] target sets:",
        len(targets_by_LR_unique),
        "regulator sets:",
        len(regulators_by_LR_unique),
    )

    return targets_by_LR_unique, regulators_by_LR_unique


def make_benchmark1_tables(target_corr_frames, regulator_corr_frames, edge_lr_corr_frames=None):
    def corr_frames_to_long(frame_dict, feature_type):
        recs = []
        for version_id, df in frame_dict.items():
            tmp = df.copy()
            tmp["Slice"] = tmp.index.astype(str)
            tmp = tmp.melt(id_vars=["Slice"], var_name="RepresentativeLR", value_name="Correlation")
            tmp["Version_ID"] = version_id
            tmp["ModelVersion"] = VERSION_SPECS[version_id]["label"]
            tmp["FeatureType"] = feature_type
            recs.append(tmp)
        if len(recs) == 0:
            return pd.DataFrame(columns=["Slice", "RepresentativeLR", "Correlation", "Version_ID", "ModelVersion", "FeatureType"])
        out = pd.concat(recs, ignore_index=True)
        out["Correlation"] = pd.to_numeric(out["Correlation"], errors="coerce")
        return out

    long_parts = [
        corr_frames_to_long(target_corr_frames, "Receiver targets"),
        corr_frames_to_long(regulator_corr_frames, "Sender regulators"),
    ]
    if edge_lr_corr_frames is not None and len(edge_lr_corr_frames) > 0:
        long_parts.append(corr_frames_to_long(edge_lr_corr_frames, "LR coexpression"))

    long_df = pd.concat(long_parts, ignore_index=True)

    method_order = [VERSION_SPECS[k]["label"] for k in VERSION_ORDER if k in long_df["Version_ID"].unique()]
    feature_type_order = [
        ft for ft in ["Receiver targets", "Sender regulators", "LR coexpression"]
        if ft in set(long_df["FeatureType"].astype(str))
    ]
    summary_records = []
    stats_records = []
    for feature_type in feature_type_order:
        sub_ft = long_df[long_df["FeatureType"].eq(feature_type)].copy()
        full_label = VERSION_SPECS["full"]["label"]
        rep_lr_order = (
            sub_ft[sub_ft["ModelVersion"].eq(full_label)]
            .groupby("RepresentativeLR")["Correlation"].median()
            .sort_values(ascending=False).index.tolist()
        )
        for rep_lr in rep_lr_order:
            full_vals = sub_ft[(sub_ft["RepresentativeLR"].eq(rep_lr)) & (sub_ft["Version_ID"].eq("full"))]["Correlation"].dropna().to_numpy()
            for version_id in VERSION_ORDER:
                if version_id not in long_df["Version_ID"].unique():
                    continue
                label = VERSION_SPECS[version_id]["label"]
                vals = sub_ft[(sub_ft["RepresentativeLR"].eq(rep_lr)) & (sub_ft["Version_ID"].eq(version_id))]["Correlation"].dropna().to_numpy()
                mean_v, ci_low, ci_high, n = mean_ci95(vals)
                summary_records.append({
                    "FeatureType": feature_type,
                    "RepresentativeLR": rep_lr,
                    "Version_ID": version_id,
                    "ModelVersion": label,
                    "mean": mean_v,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "median": float(np.median(vals)) if vals.size > 0 else np.nan,
                    "n": n,
                })
                if version_id != "full":
                    stats_records.append({
                        "FeatureType": feature_type,
                        "RepresentativeLR": rep_lr,
                        "Comparison": f"SpiderNet > {label}",
                        "Version_ID": version_id,
                        "p_one_sided_mw": one_sided_mw_greater(full_vals, vals),
                        "n_SpiderNet": int(full_vals.size),
                        "n_ablation": int(vals.size),
                        "median_SpiderNet": float(np.median(full_vals)) if full_vals.size > 0 else np.nan,
                        "median_ablation": float(np.median(vals)) if vals.size > 0 else np.nan,
                    })
    return long_df, pd.DataFrame(summary_records), pd.DataFrame(stats_records), method_order


def plot_benchmark1(long_df, method_order, out_prefix):
    set_plot_style()

    feature_types = [
        ft for ft in ["Receiver targets", "Sender regulators"]
        if ft in set(long_df["FeatureType"].astype(str))
    ]

    if len(feature_types) == 0:
        raise ValueError("No valid Benchmark 1 feature types found for plotting.")

    rep_lr_orders = {}

    for feature_type in feature_types:
        sub_ft = long_df[long_df["FeatureType"].eq(feature_type)].copy()
        full_label = VERSION_SPECS["full"]["label"]

        rep_lr_orders[feature_type] = (
            sub_ft[sub_ft["ModelVersion"].eq(full_label)]
            .groupby("RepresentativeLR")["Correlation"]
            .median()
            .sort_values(ascending=False)
            .index
            .tolist()
        )

    # Increase this value to further separate x-axis feature groups.
    # Suggested range: 1.25–1.55.
    category_spacing = 1.38

    total_categories = sum(max(1, len(v)) for v in rep_lr_orders.values())

    fig_w = max(
        10.5,
        2.1 * len(feature_types) + 0.52 * total_categories,
    )

    fig, axes = plt.subplots(
        1,
        len(feature_types),
        figsize=(fig_w, 2.9),
        constrained_layout=True,
    )

    if len(feature_types) == 1:
        axes = np.asarray([axes])

    for ax, feature_type in zip(axes, feature_types):
        sub_ft = long_df[long_df["FeatureType"].eq(feature_type)].copy()
        rep_lr_order = rep_lr_orders[feature_type]

        n_methods = len(method_order)
        base_positions = np.arange(len(rep_lr_order)) * category_spacing + 1

        # Compact within-feature spacing, wider between-feature spacing.
        group_width = 0.82
        offsets = np.linspace(-group_width / 2, group_width / 2, n_methods)

        all_lowers = []
        all_uppers = []

        for offset, label in zip(offsets, method_order):
            means = []
            lows = []
            highs = []

            for rep_lr in rep_lr_order:
                vals = sub_ft[
                    sub_ft["RepresentativeLR"].eq(rep_lr)
                    & sub_ft["ModelVersion"].eq(label)
                ]["Correlation"].dropna().to_numpy()

                mean_v, ci_low, ci_high, _ = mean_ci95(vals)

                means.append(mean_v)
                lows.append(ci_low)
                highs.append(ci_high)

            means = np.asarray(means, dtype=float)
            lows = np.asarray(lows, dtype=float)
            highs = np.asarray(highs, dtype=float)

            yerr = np.vstack([
                means - lows,
                highs - means,
            ])

            colors = VERSION_COLORS[label]

            ax.errorbar(
                base_positions + offset,
                means,
                yerr=yerr,
                fmt="o",
                linestyle="none",
                markersize=2.6,
                capsize=2.2,
                color=colors["edge"],
                ecolor=colors["edge"],
                markerfacecolor=colors["fill"],
                markeredgecolor=colors["edge"],
                markeredgewidth=0.8,
                capthick=0.8,
                elinewidth=0.8,
                zorder=3,
            )

            all_lowers.append(lows)
            all_uppers.append(highs)

        ax.set_title(feature_type)
        ax.set_xticks(base_positions)
        ax.set_xticklabels(rep_lr_order, rotation=45, ha="right")
        ax.set_xlabel("Representative LR pairs")
        ax.set_ylabel("Max Spearman correlation")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.yaxis.grid(True, linewidth=0.6, color="0.85")
        ax.axhline(
            0,
            color="0.6",
            linestyle="--",
            linewidth=0.8,
            zorder=1,
        )

        if len(rep_lr_order) > 0:
            ax.set_xlim(
                base_positions[0] - 0.72,
                base_positions[-1] + 0.72,
            )

        all_lowers = np.concatenate(all_lowers) if len(all_lowers) > 0 else np.array([])
        all_uppers = np.concatenate(all_uppers) if len(all_uppers) > 0 else np.array([])

        valid = np.concatenate([
            all_lowers[np.isfinite(all_lowers)],
            all_uppers[np.isfinite(all_uppers)],
        ])

        if valid.size > 0:
            ymin = float(valid.min())
            ymax = float(valid.max())
            pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
            ax.set_ylim(ymin - pad, ymax + pad)

    handles = [
        mpatches.Patch(
            facecolor=VERSION_COLORS[m]["fill"],
            edgecolor=VERSION_COLORS[m]["edge"],
            label=m,
        )
        for m in method_order
    ]

    axes[-1].legend(
        handles=handles,
        frameon=False,
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        fontsize=7,
    )

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    png = str(out_prefix) + ".png"
    pdf = str(out_prefix) + ".pdf"

    if SAVE_FIGURES:
        fig.savefig(png, dpi=FIG_DPI, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")

    plt.show()

    return png, pdf


def run(dataset_key):
    from ablation_training import (
        load_dataset_context, train_or_load_ablation, aggregate_sender_receiver,
        get_data_item, get_num_cells_from_data, DEVICE,
    )
    contexts = {dataset_key: load_dataset_context(dataset_key)}
    BENCHMARK1_VERSION_ORDER = [
        "full",
        "no_lr_reinit",
        "no_gene_reinit",
        "no_intrinsic_reinit",
    ]

    BENCHMARK1_EXCLUDE_VERSION_IDS = set()

    BENCHMARK1_VERSION_ORDER = [
        version_id
        for version_id in BENCHMARK1_VERSION_ORDER
        if version_id in VERSION_SPECS
        and version_id not in BENCHMARK1_EXCLUDE_VERSION_IDS
    ]

    benchmark1_outputs = {}

    for dataset_key, context in contexts.items():
        print("\n==============================")
        print("Benchmark 1 dataset:", dataset_key)

        outdir = coupling_output_dir(dataset_key)
        outdir.mkdir(parents=True, exist_ok=True)

        targets_by_LR_unique, regulators_by_LR_unique = load_benchmark1_annotations(context)

        # Ensure all configured versions are available in context["factor_lists"].
        # If missing, try to load/train them using the notebook's existing helper.
        missing_before = [
            version_id
            for version_id in BENCHMARK1_VERSION_ORDER
            if version_id not in context["factor_lists"]
        ]

        if len(missing_before) > 0:
            print(
                "Benchmark 1 missing requested versions; trying to load/train:",
                [(v, VERSION_SPECS[v]["label"]) for v in missing_before],
            )

            for version_id in missing_before:
                if version_id == "full":
                    raise KeyError("Full model factor list is missing from context['factor_lists'].")

                context["factor_lists"][version_id] = train_or_load_ablation(
                    context,
                    version_id,
                )

        missing_after = [
            version_id
            for version_id in BENCHMARK1_VERSION_ORDER
            if version_id not in context["factor_lists"]
        ]

        if len(missing_after) > 0:
            raise KeyError(
                "The following requested Benchmark 1 versions are still missing from "
                f"context['factor_lists']: {missing_after}. Available keys are: "
                f"{list(context['factor_lists'].keys())}"
            )

        # Filter and order model versions here.
        factor_lists_benchmark1 = OrderedDict(
            (version_id, context["factor_lists"][version_id])
            for version_id in BENCHMARK1_VERSION_ORDER
        )

        print("Available context factor_list keys:", list(context["factor_lists"].keys()))
        print(
            "Benchmark 1 model versions used:",
            [VERSION_SPECS[v]["label"] for v in factor_lists_benchmark1.keys()]
        )
        print(
            "Benchmark 1 source notebook map:",
            {v: ABLATION_SOURCE_NOTEBOOK.get(v, "sklearn_compat") for v in factor_lists_benchmark1.keys()}
        )

        target_corr = {
            version_id: []
            for version_id in factor_lists_benchmark1.keys()
        }

        regulator_corr = {
            version_id: []
            for version_id in factor_lists_benchmark1.keys()
        }

        for slice_index in range(len(context["adata_list"])):
            print(
                f"Benchmark 1 | {dataset_key} | "
                f"slice {slice_index + 1}/{len(context['adata_list'])}"
            )

            adata_sub = context["adata_list"][slice_index]
            graph = context["graph_list_cpu"][slice_index]

            edge_index = get_data_item(graph, "edge_index")
            num_cells = get_num_cells_from_data(graph, adata_sub)

            target_df, _, _ = cell_by_pair_mean_genesets(
                adata_sub,
                targets_by_LR_unique,
            )

            regulator_df, _, _ = cell_by_pair_mean_genesets(
                adata_sub,
                regulators_by_LR_unique,
            )

            for version_id, factor_list in factor_lists_benchmark1.items():
                edge_features = np.asarray(
                    factor_list[slice_index],
                    dtype=np.float32,
                )

                agg = aggregate_sender_receiver(
                    edge_features=edge_features,
                    edge_index=edge_index,
                    num_cells=num_cells,
                    device=DEVICE,
                )

                target_corr[version_id].append(
                    compute_rowmax_spearman_df(
                        target_df,
                        agg["receiver"],
                        slice_index,
                    )
                )

                regulator_corr[version_id].append(
                    compute_rowmax_spearman_df(
                        regulator_df,
                        agg["sender"],
                        slice_index,
                    )
                )

        target_corr_frames = {
            version_id: pd.concat(frames, axis=0)
            for version_id, frames in target_corr.items()
            if len(frames) > 0
        }

        regulator_corr_frames = {
            version_id: pd.concat(frames, axis=0)
            for version_id, frames in regulator_corr.items()
            if len(frames) > 0
        }

        for version_id, df in target_corr_frames.items():
            df.to_csv(outdir / f"Benchmark1_target_corr_{version_id}.csv")

        for version_id, df in regulator_corr_frames.items():
            df.to_csv(outdir / f"Benchmark1_regulator_corr_{version_id}.csv")

        # Temporarily override VERSION_ORDER only for Benchmark 1 table/plot generation,
        # because make_benchmark1_tables may use the global VERSION_ORDER internally.
        _VERSION_ORDER_BACKUP = VERSION_ORDER.copy()
        VERSION_ORDER[:] = list(factor_lists_benchmark1.keys())

        try:
            long_df, plot_summary_df, stats_df, method_order = make_benchmark1_tables(
                target_corr_frames,
                regulator_corr_frames,
            )
        finally:
            VERSION_ORDER[:] = _VERSION_ORDER_BACKUP

        # Enforce display order by labels.
        method_order = [
            VERSION_SPECS[v]["label"]
            for v in factor_lists_benchmark1.keys()
        ]

        keep_feature_types = [
            "Receiver targets",
            "Sender regulators",
        ]

        if "FeatureType" in long_df.columns:
            long_df = long_df[
                long_df["FeatureType"].isin(keep_feature_types)
            ].copy()

        if "FeatureType" in plot_summary_df.columns:
            plot_summary_df = plot_summary_df[
                plot_summary_df["FeatureType"].isin(keep_feature_types)
            ].copy()

        if "FeatureType" in stats_df.columns:
            stats_df = stats_df[
                stats_df["FeatureType"].isin(keep_feature_types)
            ].copy()

        long_df["Dataset"] = dataset_key
        plot_summary_df["Dataset"] = dataset_key
        stats_df["Dataset"] = dataset_key

        long_df.to_csv(outdir / "Benchmark1_long.csv", index=False)
        plot_summary_df.to_csv(outdir / "Benchmark1_plot_summary.csv", index=False)
        stats_df.to_csv(outdir / "Benchmark1_pairwise_stats.csv", index=False)

        png, pdf = plot_benchmark1(
            long_df,
            method_order,
            outdir / f"{dataset_key}_Benchmark1_curated_gene_coupling_byLR",
        )

        benchmark1_outputs[dataset_key] = {
            "long": long_df,
            "plot_summary": plot_summary_df,
            "stats": stats_df,
            "target_corr_frames": target_corr_frames,
            "regulator_corr_frames": regulator_corr_frames,
            "method_order": method_order,
            "included_version_ids": list(factor_lists_benchmark1.keys()),
            "excluded_version_ids": sorted(BENCHMARK1_EXCLUDE_VERSION_IDS),
            "png": png,
            "pdf": pdf,
            "outdir": outdir,
        }

        print("Saved Benchmark 1 outputs under:", outdir)
        print(plot_summary_df.head(12).to_string(index=False))



    return benchmark1_outputs


def plot_only(dataset_key):
    outdir = coupling_output_dir(dataset_key)
    long_df = pd.read_csv(outdir / "Benchmark1_long.csv")
    method_order = [VERSION_SPECS[v]["label"] for v in VERSION_ORDER]
    return plot_benchmark1(long_df, method_order,
                           outdir / f"{dataset_key}_Benchmark1_curated_gene_coupling_byLR")
