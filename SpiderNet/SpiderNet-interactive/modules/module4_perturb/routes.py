"""Module 4 — In silico spatial perturbation routes."""
from __future__ import annotations

import time

from flask import Blueprint, abort, current_app, jsonify, render_template, request

from . import service

bp = Blueprint(
    "module4_perturb",
    __name__,
    template_folder="templates",
    url_prefix="/dataset/<dataset_name>/perturb",
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
    bundle = service.get_loading_bundle(ds)
    return render_template(
        "module4_perturb.html",
        dataset_name=dataset_name,
        dataset=ds,
        module_title="Spatial Perturbation",
        module_id="perturb",
        mi_list=bundle["mi_list"],
        species=ds.setup.get("SPECIES") or ds.config.get("SPECIES") or "human",
    )


@bp.route("/api/feature-tables", methods=["POST"])
def api_feature_tables(dataset_name: str):
    ds = _resolve(dataset_name)
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(service.build_feature_panel_response(
            ds=ds,
            mi_name=str(payload.get("mi_name") or ""),
            top_n=int(payload.get("top_n", service.DEFAULT_TOP_N)),
            theme_mode=str(payload.get("theme", "dark")),
        ))
    except Exception as e:
        current_app.logger.exception("M4 feature-tables failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@bp.route("/api/state-info", methods=["GET"])
def api_state_info(dataset_name: str):
    """Lightweight introspection — returns whether the heavy state is loaded
    and the available cell-type list (only available once state is loaded)."""
    ds = _resolve(dataset_name)
    if not service.is_state_loaded(ds):
        return jsonify({
            "loaded": False,
            "available_celltypes": [],
            "n_slices": 0,
        })
    state = service.get_state(ds)
    return jsonify({
        "loaded": True,
        "available_celltypes": state["available_celltypes"],
        "n_slices": len(state["slice_infos"]),
        "n_genes": int(len(state["gene_labels"])),
        "device": state["device"],
        "species": state["species"],
    })


@bp.route("/api/prepare", methods=["POST"])
def api_prepare(dataset_name: str):
    """Eagerly load the trained model + baseline predictions. Long on first call."""
    ds = _resolve(dataset_name)
    payload = request.get_json(silent=True) or {}
    t0 = time.time()
    try:
        state = service.get_state(ds, force_baseline_rebuild=bool(payload.get("force_rebuild", False)))
    except Exception as e:
        current_app.logger.exception("M4 prepare failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    return jsonify({
        "loaded": True,
        "available_celltypes": state["available_celltypes"],
        "n_slices": len(state["slice_infos"]),
        "n_genes": int(len(state["gene_labels"])),
        "device": state["device"],
        "species": state["species"],
        "elapsed_seconds": round(time.time() - t0, 2),
    })


@bp.route("/api/prepare-progress", methods=["GET"])
def api_prepare_progress(dataset_name: str):
    response = jsonify(service.get_prepare_progress(_resolve(dataset_name)))
    response.headers["Cache-Control"] = "no-store"
    return response


@bp.route("/api/run", methods=["POST"])
def api_run(dataset_name: str):
    ds = _resolve(dataset_name)
    payload = request.get_json(silent=True) or {}
    t0 = time.time()
    try:
        state = service.get_state(ds)
        mode = str(payload.get("mode", "knockout"))
        max_cells = payload.get("max_cells_for_de", service.DEFAULT_MAX_CELLS_FOR_DE)
        max_cells = None if max_cells in (None, "", 0) else int(max_cells)

        if mode == "knockout":
            genes = service._collect_knockout_genes(
                lr_rows=payload.get("lr_rows") or [],
                lr_selected=payload.get("lr_selected") or [],
                sender_rows=payload.get("sender_rows") or [],
                sender_selected=payload.get("sender_selected") or [],
                receiver_rows=payload.get("receiver_rows") or [],
                receiver_selected=payload.get("receiver_selected") or [],
                gene_to_idx=state["gene_to_idx"],
            )
            de_df, summary = service.run_knockout_analysis(
                ds=ds, state=state,
                mi_name=str(payload.get("mi_name", "")),
                sender_gene_idx=genes["sender_gene_idx"],
                receiver_gene_idx=genes["receiver_gene_idx"],
                sender_gene_names=genes["sender_gene_names"],
                receiver_gene_names=genes["receiver_gene_names"],
                sender_types=payload.get("sender_types") or [],
                receiver_types=payload.get("receiver_types") or [],
                keep_pct=float(payload.get("keep_pct", service.DEFAULT_KO_PERCENT)),
                target_celltypes=payload.get("target_celltypes") or [],
                max_cells_for_de=max_cells,
            )
        elif mode == "replacement":
            de_df, summary = service.run_replacement_analysis(
                ds=ds, state=state,
                replacement_celltypes=payload.get("replacement_celltypes") or [],
                replaced_celltypes=payload.get("replaced_celltypes") or [],
                target_celltypes=payload.get("target_celltypes") or [],
                max_cells_for_de=max_cells,
                random_seed=int(payload.get("random_seed", 0)),
            )
        else:
            return jsonify({"error": f"Unknown mode: {mode}"}), 400

        elapsed = time.time() - t0
        summary["elapsed_seconds"] = round(elapsed, 2)
        csv_path, json_path = service.export_run(ds, de_df, summary)
        summary["csv_path"] = str(csv_path)
        summary["json_path"] = str(json_path)
        summary["status"] = "ok"

        # Build cache key from compact payload signature.
        sig = {k: payload.get(k) for k in (
            "mode", "mi_name", "keep_pct",
            "sender_types", "receiver_types",
            "replacement_celltypes", "replaced_celltypes",
            "target_celltypes", "random_seed", "max_cells_for_de",
            "lr_selected", "sender_selected", "receiver_selected",
        )}
        cache_key = service._cache_key_for_run(ds.name, sig)
        service.store_run(cache_key, de_df, summary)

        return jsonify(service.build_run_response(
            ds=ds, de_df=de_df, summary=summary, cache_key=cache_key,
            lfc_threshold=float(payload.get("lfc_threshold", service.DEFAULT_DE_LOGFC_THRESHOLD)),
            padj_threshold=float(payload.get("padj_threshold", service.DEFAULT_DE_PADJ_THRESHOLD)),
            enrich_direction=str(payload.get("enrich_direction", service.DEFAULT_ENRICH_DIRECTION)),
            enrich_scope=str(payload.get("enrich_scope", service.DEFAULT_ENRICH_SCOPE)),
            theme_mode=str(payload.get("theme", "dark")),
            top_terms=int(payload.get("top_terms", service.DEFAULT_ENRICH_TOP_TERMS)),
        ))
    except Exception as e:
        current_app.logger.exception("M4 run failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@bp.route("/api/refresh", methods=["POST"])
def api_refresh(dataset_name: str):
    """Re-render volcano / DE table / GO / KEGG using a cached run.
    Only thresholds and theme change — perturbation is not re-run."""
    ds = _resolve(dataset_name)
    payload = request.get_json(silent=True) or {}
    cache_key = payload.get("cache_key")
    if not cache_key:
        return jsonify({"error": "cache_key required"}), 400
    cached = service.get_run(cache_key)
    if cached is None:
        return jsonify({"error": "no cached perturbation run for this cache_key (re-run /api/run)"}), 410
    try:
        return jsonify(service.build_run_response(
            ds=ds,
            de_df=cached["de_df"], summary=cached["summary"], cache_key=cache_key,
            lfc_threshold=float(payload.get("lfc_threshold", service.DEFAULT_DE_LOGFC_THRESHOLD)),
            padj_threshold=float(payload.get("padj_threshold", service.DEFAULT_DE_PADJ_THRESHOLD)),
            enrich_direction=str(payload.get("enrich_direction", service.DEFAULT_ENRICH_DIRECTION)),
            enrich_scope=str(payload.get("enrich_scope", service.DEFAULT_ENRICH_SCOPE)),
            theme_mode=str(payload.get("theme", "dark")),
            top_terms=int(payload.get("top_terms", service.DEFAULT_ENRICH_TOP_TERMS)),
        ))
    except Exception as e:
        current_app.logger.exception("M4 refresh failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
