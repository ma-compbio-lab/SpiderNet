"""Spatial annotation overview and discoverable copies of existing response plots."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

# RGB colours sampled from the supplied manuscript's rendered legend.
CELL_COLORS = {"cancer": "#dd454c", "T cells": "#56a6bc", "other": "#aae0a3"}
PERTURB_COLORS = {
    "CHUK": "#3974bb", "IRAK1": "#f46e16", "TAB2": "#db5db6",
    "TRAM1": "#37a53e", "LBP": "#d82327", "IRF7": "#4fc5cf",
    "IRAK4": "#8f56b1", "PELI1": "#874a44", "MAP2K2": "#918f8f",
    "MAP2K6": "#b2c11b", "MYD88": "#403c3d",
}
OTHER_COLOR = "#dfedfa"
BACKGROUND_COLOR = "#d3d2d2"


def spatial_overview(adata_path, output_dir):
    """Plot every supplied cell; retain all guide labels on multi-guide melanoma cells.

    This is a display annotation only. It does not define perturbation/control groups
    for any downstream test. Missing guide labels remain explicitly unassigned.
    """
    import anndata as ad
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.path import Path as MarkerPath
    import numpy as np
    import pandas as pd

    adata_path, output_dir = Path(adata_path), Path(output_dir)
    dest = output_dir / "paper_figures"
    dest.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(adata_path, backed="r")
    try:
        obs = adata.obs.copy()
        xy = np.asarray(adata.obsm["spatial"])[:, :2].copy()
    finally:
        adata.file.close()
    required = {"celltype2", "perturbation"}
    if not required.issubset(obs):
        raise ValueError(f"Missing observation columns: {required - set(obs)}")
    if len(xy) != len(obs) or not np.isfinite(xy).all():
        raise ValueError("Spatial coordinates must be finite and aligned with all cells.")
    celltype = obs["celltype2"].astype(str)
    unknown = set(celltype) - set(CELL_COLORS)
    if unknown:
        raise ValueError(f"Unmapped cell types (not silently pooled): {unknown}")

    # The dataset encodes multiple guides as underscore-separated tokens.
    raw = obs["perturbation"].astype(object).fillna("").astype(str)
    tokens = raw.map(lambda value: tuple(dict.fromkeys(value.split("_"))) if value else ())
    cancer = celltype.eq("cancer")
    guides = tokens.map(lambda values: tuple(v for v in values if v != "Control"))

    def display_groups(values):
        selected = tuple(g for g in PERTURB_COLORS if g in values)
        return selected + (("other perturbations",) if any(g not in PERTURB_COLORS for g in values) else ())

    groups = guides.map(display_groups)
    groups.loc[~cancer] = pd.Series([()] * int((~cancer).sum()), index=groups.index[~cancer], dtype=object)
    table = pd.DataFrame({
        "cell_id": obs.index.astype(str), "x": xy[:, 0], "y": xy[:, 1],
        "celltype2": celltype.to_numpy(), "perturbation_raw": raw.to_numpy(),
        "guide_label_missing": obs["perturbation"].isna().to_numpy(),
        "has_control_token": tokens.map(lambda values: "Control" in values).to_numpy(),
        "n_noncontrol_guide_tokens": guides.map(len).to_numpy(),
        "display_groups": groups.map(lambda values: ";".join(values)).to_numpy(),
    })
    table.to_csv(dest / "spatial_annotation_cells.csv", index=False)
    counts = [{"panel": "celltype", "category": key, "n_cells": int(celltype.eq(key).sum())}
              for key in CELL_COLORS]
    for key in [*PERTURB_COLORS, "other perturbations"]:
        counts.append({"panel": "perturbation", "category": key,
                       "n_cells": int(groups.map(lambda values: key in values).sum())})
    counts += [
        {"panel": "annotation_audit", "category": "melanoma_with_multiple_noncontrol_guides",
         "n_cells": int((cancer & guides.map(len).gt(1)).sum())},
        {"panel": "annotation_audit", "category": "melanoma_missing_guide_label",
         "n_cells": int((cancer & obs["perturbation"].isna()).sum())},
        {"panel": "annotation_audit", "category": "melanoma_control_only",
         "n_cells": int((cancer & raw.eq("Control")).sum())},
    ]
    pd.DataFrame(counts).to_csv(dest / "spatial_annotation_counts.csv", index=False)

    with plt.rc_context({"font.family": "Arial", "font.size": 9, "pdf.fonttype": 42}):
        fig = plt.figure(figsize=(7.8, 6.6))
        axes = [fig.add_axes([0.025, bottom, 0.50, 0.42]) for bottom in [0.54, 0.065]]
        colors = celltype.map(CELL_COLORS).to_numpy()
        axes[0].scatter(xy[:, 0], xy[:, 1], c=colors, s=0.35, linewidths=0, rasterized=True)
        axes[1].scatter(xy[:, 0], xy[:, 1], c=BACKGROUND_COLOR, s=0.35, linewidths=0, rasterized=True)
        for group in dict.fromkeys(groups):
            if not group:
                continue
            mask = groups.map(lambda value: value == group).to_numpy()
            for i, label in enumerate(group):
                if len(group) == 1:
                    marker = "o"
                else:
                    angle = np.linspace(2 * np.pi * i / len(group), 2 * np.pi * (i + 1) / len(group), 20)
                    marker = MarkerPath(np.vstack([[0, 0], np.column_stack([np.cos(angle), np.sin(angle)]), [0, 0]]))
                axes[1].scatter(xy[mask, 0], xy[mask, 1], s=1.1, marker=marker,
                                c=PERTURB_COLORS.get(label, OTHER_COLOR), linewidths=0, rasterized=True)
        for ax, title in zip(axes, ["Cell type", "Perturbation gene"]):
            ax.set_aspect("equal")
            ax.set_axis_off()
            ax.set_title(title, pad=4, fontsize=11)

        def handle(label, color):
            return Line2D([], [], marker="o", linestyle="", markerfacecolor=color,
                          markeredgecolor="none", markersize=5, label=label)

        labels = {"cancer": "melanoma cell", "T cells": "T cell", "other": "others"}
        axes[0].legend(handles=[handle(labels[k], v) for k, v in CELL_COLORS.items()],
                       loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        legend = [handle(k, v) for k, v in PERTURB_COLORS.items()]
        legend += [handle("other perturbations", OTHER_COLOR),
                   handle("unperturbed / unassigned", BACKGROUND_COLOR)]
        axes[1].legend(handles=legend, loc="center left", bbox_to_anchor=(1.02, 0.5),
                       ncol=2, columnspacing=1.0, handletextpad=0.4, frameon=False, fontsize=8.5)
        fig.text(0.55, 0.035, "Multiple guide labels: coloured sectors;\ngray includes cells without an assigned guide.",
                 fontsize=7.5, color="0.3")
        fig.savefig(dest / "Fig4a.pdf", dpi=600, bbox_inches="tight")
        fig.savefig(dest / "Fig4a.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    provenance = {
        "source": str(adata_path.resolve()), "source_bytes": adata_path.stat().st_size,
        "n_cells": len(obs), "n_cells_with_multiple_display_categories": int(groups.map(len).gt(1).sum()),
        "coordinates": "obsm['spatial'][:, :2], equal aspect, increasing y upwards as in manuscript; no subsampling or rotation",
        "cell_type": "obs['celltype2']; perturbation overlay restricted to cancer",
        "guide_display": "Exact underscore tokens; all non-Control tokens retained, multi-category sectors; no priority assignment",
        "background": "All cells; missing labels are not recoded as Control for analysis",
        "palette": "RGB sampled from legend of SpiderNet (39).pdf page 10; manuscript multi-guide drawing rule unavailable",
    }
    (dest / "spatial_annotation_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("Spatial overview:", dest / "Fig4a.pdf", flush=True)


def publish_response_panels(output_dir):
    """Copy the original vector/PDF and PNG outputs byte-for-byte, without replotting."""
    output_dir = Path(output_dir)
    dest = output_dir / "paper_figures"
    dest.mkdir(parents=True, exist_ok=True)
    response = output_dir / "MI17_Tcell_DEG_neighboring_perturbed_melanoma"
    sources = {
        "FigS14a": response / "DEG_dotplot/Dotplot_Tcell_DEG_log2FC_neglog10P_MI17_log2FC0.4_p0.05_sizecap5",
        "FigS14b": response / "MI17_receiver_loading_GSEA/GSEA_MI17_receiver_loading_Tcell_upDEG_enrichment",
    }
    records = []
    for label, source in sources.items():
        for suffix in [".pdf", ".png"]:
            original, target = Path(str(source) + suffix), dest / (label + suffix)
            shutil.copy2(original, target)
            records.append({"source": original.relative_to(output_dir).as_posix(), "copy": target.name,
                            "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    (dest / "response_panel_sources.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print("Response panels:", dest / "FigS14a.pdf", dest / "FigS14b.pdf", flush=True)
