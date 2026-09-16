"""Held-out response Spearman evaluation from upstream observed/predicted LFC tables."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from ablation_settings import WORKSPACE_ROOT, OUTPUT_ROOT

RESULT_ROOT = WORKSPACE_ROOT / "Results" / "PerturbFISH" / "V1" / "SpiderNet_Result_dim23"
OUTPUT_DIR = OUTPUT_ROOT / "PerturbFISH_component_ablation"
FIGURE_VALUE_SOURCE = "canonical"  # options: "canonical", "recomputed"

PERTURBATION_ORDER = [
    "PELI1", "MAP2K2", "TRAM1", "IRAK1", "LBP", "IRF7",
    "MYD88", "IRAK4", "MAP2K6", "CHUK", "TAB2",
]
METHOD_ORDER = ["SpiderNet", "No LR recon.", "No gene recon.", "No intrinsic"]
METHOD_COLORS = {
    "SpiderNet": "#B72E70",
    "No LR recon.": "#BE7914",
    "No gene recon.": "#6E6E91",
    "No intrinsic": "#D4483F",
}

# Make the original saved figure's implicit notebook dark-grid style explicit.
# These Matplotlib settings reproduce its PNG exactly without adding seaborn.
SAVED_FIGURE_STYLE = {
    "figure.facecolor": "white", "axes.labelcolor": ".15",
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.color": ".15", "ytick.color": ".15", "text.color": ".15",
    "axes.axisbelow": True, "grid.linestyle": "-",
    "lines.solid_capstyle": "round", "patch.edgecolor": "w",
    "patch.force_edgecolor": True, "xtick.top": False, "ytick.right": False,
    "axes.grid": True, "axes.facecolor": "#EAEAF2", "axes.edgecolor": "white",
    "grid.color": "white", "axes.spines.left": True, "axes.spines.bottom": True,
    "axes.spines.right": True, "axes.spines.top": True,
    "xtick.bottom": False, "ytick.left": False,
}

observed_path = (
    RESULT_ROOT / "Cached_Tcell_LFC_for_scatter_seed10042" / "Observed_Tcell_LFC_golden.csv"
)
prediction_paths = {
    "SpiderNet": (
        RESULT_ROOT / "Cached_Tcell_LFC_for_scatter_seed10042" / "Predicted_Tcell_LFC_SpiderNet.csv"
    ),
    "No LR recon.": (
        RESULT_ROOT / "Cached_Tcell_LFC_component_ablations_seed10042" / "Predicted_Tcell_LFC_no_lr_reinit.csv"
    ),
    "No gene recon.": (
        RESULT_ROOT / "Cached_Tcell_LFC_component_ablations_seed10042" / "Predicted_Tcell_LFC_no_gene_reinit.csv"
    ),
    "No intrinsic": (
        RESULT_ROOT / "Cached_Tcell_LFC_component_ablations_seed10042" / "Predicted_Tcell_LFC_no_intrinsic_reinit.csv"
    ),
}
canonical_matrix_path = RESULT_ROOT / "Correlation_Matrix_SpiderNet_Component_Ablations.csv"


def calculate_spearman_matrix(observed, predictions, perturbation_order):
    records = {}
    for method, predicted in predictions.items():
        method_values = {}
        for perturbation in perturbation_order:
            observed_values = pd.to_numeric(observed[perturbation], errors="coerce").to_numpy(dtype=float)
            predicted_values = pd.to_numeric(predicted[perturbation], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(observed_values) & np.isfinite(predicted_values)
            if valid.sum() < 3 or np.std(observed_values[valid]) == 0 or np.std(predicted_values[valid]) == 0:
                rho = np.nan
            else:
                rho = spearmanr(observed_values[valid], predicted_values[valid]).statistic
            method_values[perturbation] = rho
        records[method] = method_values
    return pd.DataFrame(records).reindex(index=perturbation_order)



def render(figure_matrix):
    plt.rcParams.update(SAVED_FIGURE_STYLE)
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "Arial",
        "font.size": 10,
        "axes.linewidth": 0.9,
    })

    x = np.arange(len(PERTURBATION_ORDER), dtype=float)
    group_width = 0.82
    bar_width = group_width / len(METHOD_ORDER)
    offsets = np.linspace(
        -group_width / 2 + bar_width / 2,
        group_width / 2 - bar_width / 2,
        len(METHOD_ORDER),
    )

    fig, ax = plt.subplots(figsize=(12.2, 4.8), facecolor="white")
    for offset, method in zip(offsets, METHOD_ORDER):
        ax.bar(
            x + offset,
            figure_matrix[method].to_numpy(dtype=float),
            width=bar_width * 0.94,
            color=METHOD_COLORS[method],
            edgecolor="none",
            label=method,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(PERTURBATION_ORDER, rotation=42, ha="right")
    ax.set_ylabel("Observed-predicted T-cell\nperturbation-effect Spearman ρ")
    ax.set_xlim(-0.6, len(PERTURBATION_ORDER) - 0.4)
    ax.set_ylim(0, 0.76)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout()

    png_path = OUTPUT_DIR / "PerturbFISH_component_ablation_response_prediction_spearman_barplot.png"
    pdf_path = OUTPUT_DIR / "PerturbFISH_component_ablation_response_prediction_spearman_barplot.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    method_summary = figure_matrix.agg(["mean", "median", "std"]).T
    method_summary.index.name = "Method"
    method_summary.to_csv(OUTPUT_DIR / "PerturbFISH_component_ablation_method_summary.csv")
    print("Saved PNG:", png_path)
    print("Saved PDF:", pdf_path)


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    required_paths = [observed_path, canonical_matrix_path, *prediction_paths.values()]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(map(str, missing_paths)))

    observed_lfc = pd.read_csv(observed_path, index_col=0)
    predicted_lfc = {method: pd.read_csv(path, index_col=0) for method, path in prediction_paths.items()}

    if list(observed_lfc.columns) != PERTURBATION_ORDER:
        observed_lfc = observed_lfc.reindex(columns=PERTURBATION_ORDER)
    if observed_lfc[PERTURBATION_ORDER].isna().all(axis=0).any():
        raise ValueError("One or more required perturbations are absent from the observed LFC table.")

    for method, table in predicted_lfc.items():
        if not observed_lfc.index.equals(table.index):
            raise ValueError(f"Feature index mismatch for {method}.")
        missing_perturbations = [gene for gene in PERTURBATION_ORDER if gene not in table.columns]
        if missing_perturbations:
            raise ValueError(f"Missing perturbations for {method}: {missing_perturbations}")

    print("Measured response genes:", observed_lfc.shape[0])
    print("Perturbations:", len(PERTURBATION_ORDER))

    recomputed_matrix = calculate_spearman_matrix(observed_lfc, predicted_lfc, PERTURBATION_ORDER)
    recomputed_matrix.index.name = "Gene"
    recomputed_matrix.to_csv(OUTPUT_DIR / "PerturbFISH_component_ablation_spearman_recomputed.csv")

    canonical_matrix = pd.read_csv(canonical_matrix_path).set_index("Gene")
    canonical_matrix = canonical_matrix.reindex(index=PERTURBATION_ORDER, columns=METHOD_ORDER)
    if canonical_matrix.isna().any().any():
        raise ValueError("The canonical matrix is missing a required perturbation or model variant.")

    absolute_difference = (recomputed_matrix[METHOD_ORDER] - canonical_matrix).abs()
    absolute_difference.to_csv(OUTPUT_DIR / "PerturbFISH_component_ablation_canonical_absolute_difference.csv")
    print("Maximum absolute difference from canonical matrix:", absolute_difference.to_numpy().max())
    print(absolute_difference.loc[(absolute_difference > 1e-12).any(axis=1)].to_string())

    if FIGURE_VALUE_SOURCE == "canonical":
        figure_matrix = canonical_matrix.copy()
    elif FIGURE_VALUE_SOURCE == "recomputed":
        figure_matrix = recomputed_matrix[METHOD_ORDER].copy()
    else:
        raise ValueError("FIGURE_VALUE_SOURCE must be 'canonical' or 'recomputed'.")

    figure_matrix.index.name = "Gene"
    figure_matrix.to_csv(OUTPUT_DIR / f"PerturbFISH_component_ablation_spearman_{FIGURE_VALUE_SOURCE}.csv")

    render(figure_matrix)


def plot_only():
    path = OUTPUT_DIR / f"PerturbFISH_component_ablation_spearman_{FIGURE_VALUE_SOURCE}.csv"
    figure_matrix = pd.read_csv(path, index_col=0).reindex(index=PERTURBATION_ORDER, columns=METHOD_ORDER)
    if figure_matrix.isna().any().any():
        raise ValueError(f"Missing required figure values: {path}")
    render(figure_matrix)
