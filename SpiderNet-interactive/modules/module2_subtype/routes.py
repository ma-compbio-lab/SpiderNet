"""Module 2 — Subtype Discovery routes."""
from __future__ import annotations

from flask import Blueprint, abort, current_app, jsonify, render_template, request

from . import service

bp = Blueprint(
    "module2_subtype",
    __name__,
    template_folder="templates",
    url_prefix="/dataset/<dataset_name>/subtype",
)


def _resolve(dataset_name: str):
    datasets = current_app.config["DATASETS"]
    ds = datasets.get(dataset_name)
    if ds is None:
        abort(404)
    return ds


@bp.route("/")
def page(dataset_name: str):
    ds = _resolve(dataset_name)
    return render_template(
        "module2_subtype.html",
        dataset_name=dataset_name,
        dataset=ds,
        module_title="Subtype Discovery",
        module_id="subtype",
    )


@bp.route("/api/meta", methods=["GET"])
def api_meta(dataset_name: str):
    ds = _resolve(dataset_name)
    # Build the embedding store (one-time, slow). Returns lightweight summary.
    store = service.get_embedding_store(ds, agg_mode=service.DEFAULT_AGG_MODE)
    # Cell-type counts (so the UI can show "Fibroblast (20,485 cells)" in dropdowns).
    counts = (
        store.meta.groupby("cell_type", sort=False)
        .size().sort_values(ascending=False).to_dict()
    )
    return jsonify({
        "dataset_name": ds.name,
        "dim_envir": store.summary.get("dim_envir"),
        "agg_mode": store.summary.get("agg_mode"),
        "cell_types": [
            {"name": name, "n_cells": int(count)} for name, count in counts.items()
        ],
        "n_cells": int(store.summary.get("n_cells", 0)),
        "species": ds.config.get("SPECIES"),
    })


@bp.route("/api/run", methods=["POST"])
def api_run(dataset_name: str):
    ds = _resolve(dataset_name)
    payload = request.get_json(silent=True) or {}

    cell_type = payload.get("cell_type")
    if not cell_type or not isinstance(cell_type, str):
        return jsonify({"error": "cell_type is required"}), 400

    try:
        result = service.run_subtype_analysis(
            ds=ds,
            cell_type=cell_type,
            agg_mode=str(payload.get("agg_mode", service.DEFAULT_AGG_MODE)),
            mi_threshold=float(payload.get("mi_threshold", service.DEFAULT_MI_THRESHOLD)),
            n_neighbors=int(payload.get("n_neighbors", service.DEFAULT_N_NEIGHBORS)),
            n_pcs=int(payload.get("n_pcs", service.DEFAULT_N_PCS)),
            louvain_resolution=float(payload.get("louvain_resolution", service.DEFAULT_LOUVAIN_RES)),
            umap_min_dist=float(payload.get("umap_min_dist", service.DEFAULT_UMAP_MIN_DIST)),
            random_state=int(payload.get("random_state", 0)),
            max_cells_for_plot=int(payload.get("max_cells_for_plot", service.DEFAULT_MAX_CELLS_FOR_PLOT)),
        )
    except Exception as e:
        current_app.logger.exception("subtype analysis failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    return jsonify(service.build_full_response(result, theme_mode=str(payload.get("theme", "dark"))))


@bp.route("/api/deg", methods=["POST"])
def api_deg(dataset_name: str):
    """DEG (Wilcoxon) + Enrichr GO/KEGG for a chosen cluster + direction.
    Requires a prior `/api/run` so we can re-use the cluster assignment.
    """
    ds = _resolve(dataset_name)
    payload = request.get_json(silent=True) or {}

    cache_key = payload.get("cache_key")
    if not cache_key or not isinstance(cache_key, str):
        return jsonify({"error": "cache_key from /api/run is required"}), 400
    result = service.get_cached_run(cache_key)
    if result is None:
        return jsonify({"error": "no cached run for this cache_key (re-run /api/run)"}), 410

    cluster = payload.get("cluster")
    if cluster is None:
        return jsonify({"error": "cluster is required"}), 400
    direction = str(payload.get("direction", "up")).lower()

    try:
        return jsonify(service.build_deg_response(
            ds=ds, result=result,
            cluster=str(cluster), direction=direction,
            lfc_thresh=float(payload.get("lfc_thresh", 0.5)),
            padj_thresh=float(payload.get("padj_thresh", 0.05)),
            max_genes_for_enrichment=int(payload.get("max_genes_for_enrichment", 100)),
            top_enrich_terms=int(payload.get("top_enrich_terms", 12)),
            theme_mode=str(payload.get("theme", "dark")),
        ))
    except Exception as e:
        current_app.logger.exception("DEG failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
