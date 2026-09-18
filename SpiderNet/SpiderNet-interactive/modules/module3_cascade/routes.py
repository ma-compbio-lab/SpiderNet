"""Module 3 — MI Cascade routes (sub-milestone 5a)."""
from __future__ import annotations

from flask import Blueprint, abort, current_app, jsonify, render_template, request

from . import service

bp = Blueprint(
    "module3_cascade",
    __name__,
    template_folder="templates",
    url_prefix="/dataset/<dataset_name>/cascade",
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
        "module3_cascade.html",
        dataset_name=dataset_name,
        dataset=ds,
        module_title="MI Cascade",
        module_id="cascade",
    )


@bp.route("/api/run", methods=["POST"])
def api_run(dataset_name: str):
    ds = _resolve(dataset_name)
    payload = request.get_json(silent=True) or {}
    try:
        result = service.run_cascade_analysis(
            ds=ds,
            mi_threshold=float(payload.get("MI_threshold", service.DEFAULT_MI_THRESHOLD)),
            zscore_countcolocal_threshold=float(payload.get("zscore_countcolocal_threshold", service.DEFAULT_ZSCORE_THRESHOLD)),
            pvalue_adjusted_threshold=float(payload.get("pvalue_adjusted_threshold", service.DEFAULT_PADJ_THRESHOLD)),
            nperm=int(payload.get("nperm", service.DEFAULT_NPERM)),
            fdr_alpha=float(payload.get("fdr_alpha", service.DEFAULT_FDR_ALPHA)),
            force_recompute=bool(payload.get("force_recompute", False)),
        )
    except Exception as e:
        current_app.logger.exception("cascade analysis failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    return jsonify(service.build_run_response(result, theme_mode=str(payload.get("theme", "dark"))))


@bp.route("/api/stem", methods=["POST"])
def api_stem(dataset_name: str):
    """Return the celltype-triple stem plot for a chosen pair, using the
    cached run keyed by `cache_key` from `/api/run`."""
    _ = _resolve(dataset_name)
    payload = request.get_json(silent=True) or {}
    cache_key = payload.get("cache_key")
    pair_key = payload.get("pair_key")
    if not cache_key or not pair_key:
        return jsonify({"error": "cache_key and pair_key are required"}), 400
    cached = service.get_cached_analysis(cache_key)
    if cached is None:
        return jsonify({"error": "no cached cascade run for this cache_key (re-run /api/run)"}), 410
    try:
        top_n = int(payload.get("top_n", service.STEM_TOPN))
        if not 1 <= top_n <= 100:
            raise ValueError()
    except (TypeError, ValueError):
        return jsonify({"error": "top_n must be an integer from 1 to 100"}), 400
    return jsonify(service.build_stem_response(
        cached, pair_key=str(pair_key), top_n=top_n,
        theme_mode=str(payload.get("theme", "dark")),
    ))


@bp.route("/api/insitu-options", methods=["POST"])
def api_insitu_options(dataset_name: str):
    ds = _resolve(dataset_name)
    payload = request.get_json(silent=True) or {}
    cache_key = payload.get("cache_key"); pair_key = payload.get("pair_key")
    if not cache_key or not pair_key:
        return jsonify({"error": "cache_key and pair_key are required"}), 400
    cached = service.get_cached_analysis(cache_key)
    if cached is None:
        return jsonify({"error": "no cached cascade run for this cache_key (re-run /api/run)"}), 410
    try:
        return jsonify(service.build_insitu_options_payload(ds, cached, str(pair_key)))
    except Exception as e:
        current_app.logger.exception("insitu options failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@bp.route("/api/insitu", methods=["POST"])
def api_insitu(dataset_name: str):
    ds = _resolve(dataset_name)
    payload = request.get_json(silent=True) or {}
    required = ("cache_key", "pair_key", "cell1_type", "cell2_type", "cell3_type", "slice_index")
    missing = [k for k in required if k not in payload]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400
    cached = service.get_cached_analysis(payload["cache_key"])
    if cached is None:
        return jsonify({"error": "no cached cascade run for this cache_key (re-run /api/run)"}), 410
    try:
        return jsonify(service.build_insitu_response(
            ds=ds, pair_payload=cached,
            pair_key=str(payload["pair_key"]),
            cell1_type=str(payload["cell1_type"]),
            cell2_type=str(payload["cell2_type"]),
            cell3_type=str(payload["cell3_type"]),
            slice_index=int(payload["slice_index"]),
            mi_threshold=float(payload.get("mi_threshold", service.DEFAULT_MI_THRESHOLD)),
            cell_alpha=float(payload.get("cell_alpha", 0.82)),
            cell_size=float(payload.get("cell_size", 6.0)),
            edge_width=float(payload.get("edge_width", 1.2)),
            arrow_size=float(payload.get("arrow_size", 0.55)),
            theme_mode=str(payload.get("theme", "dark")),
        ))
    except Exception as e:
        current_app.logger.exception("insitu render failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@bp.route("/api/deggo", methods=["POST"])
def api_deggo(dataset_name: str):
    ds = _resolve(dataset_name)
    payload = request.get_json(silent=True) or {}
    required = ("cache_key", "pair_key", "cell1_type", "cell2_type", "cell3_type")
    missing = [k for k in required if k not in payload]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400
    cached = service.get_cached_analysis(payload["cache_key"])
    if cached is None:
        return jsonify({"error": "no cached cascade run for this cache_key (re-run /api/run)"}), 410
    try:
        return jsonify(service.run_cascade_deggo(
            ds=ds, pair_payload=cached,
            pair_key=str(payload["pair_key"]),
            cell1_type=str(payload["cell1_type"]),
            cell2_type=str(payload["cell2_type"]),
            cell3_type=str(payload["cell3_type"]),
            mi_threshold=float(payload.get("mi_threshold", service.DEFAULT_MI_THRESHOLD)),
            baseline_upstream_active=bool(payload.get("baseline_upstream_active", False)),
            baseline_downstream_active=bool(payload.get("baseline_downstream_active", True)),
            lfc_thresh=float(payload.get("lfc_thresh", service.DEFAULT_DEG_LFC)),
            padj_thresh=float(payload.get("padj_thresh", service.DEFAULT_DEG_PADJ)),
            gene_direction=str(payload.get("gene_direction", "up")),
            top_n_terms=int(payload.get("top_n_terms", service.DEFAULT_GO_TOPN)),
            theme_mode=str(payload.get("theme", "dark")),
        ))
    except Exception as e:
        current_app.logger.exception("DEG/GO failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
