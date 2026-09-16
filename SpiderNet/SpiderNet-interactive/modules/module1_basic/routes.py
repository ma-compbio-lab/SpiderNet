"""Module 1 — Basic Analysis routes."""
from __future__ import annotations

from flask import Blueprint, abort, current_app, jsonify, render_template, request

from . import service

bp = Blueprint(
    "module1_basic",
    __name__,
    template_folder="templates",
    url_prefix="/dataset/<dataset_name>/basic",
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
    # Ensure the bundle exists. Cheap if cached, slow on first call.
    summary = service.export_in_situ_bundle(ds)
    return render_template(
        "module1_basic.html",
        dataset_name=dataset_name,
        dataset=ds,
        module_title="Basic Analysis",
        module_id="basic",
        n_slices=len(summary["slices"]),
        dim_envir=summary["dim_envir"],
    )


@bp.route("/api/meta", methods=["GET"])
def api_meta(dataset_name: str):
    ds = _resolve(dataset_name)
    summary = service.export_in_situ_bundle(ds)
    palette = service.load_palette(ds)
    return jsonify({
        "dataset_name": ds.name,
        "dim_envir": summary["dim_envir"],
        "mi_list": summary["mi_list"],
        "slices": summary["slices"],
        "palette": palette,
    })


@bp.route("/api/spatial", methods=["POST"])
def api_spatial(dataset_name: str):
    ds = _resolve(dataset_name)
    payload = request.get_json(silent=True) or {}

    try:
        slice_idx = int(payload.get("slice_idx", 0))
        mi_idx = int(payload.get("mi_idx", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "slice_idx and mi_idx must be integers"}), 400

    threshold = payload.get("threshold", None)
    if threshold is not None:
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            return jsonify({"error": "threshold must be numeric"}), 400

    sender_types = payload.get("sender_types") or None
    receiver_types = payload.get("receiver_types") or None
    if sender_types is not None and not isinstance(sender_types, list):
        return jsonify({"error": "sender_types must be a list"}), 400
    if receiver_types is not None and not isinstance(receiver_types, list):
        return jsonify({"error": "receiver_types must be a list"}), 400

    try:
        result = service.build_spatial_figure(
            ds=ds,
            slice_idx=slice_idx,
            mi_idx=mi_idx,
            threshold=threshold,
            sender_types=sender_types,
            receiver_types=receiver_types,
            cell_alpha=float(payload.get("cell_alpha", service.DEFAULT_CELL_ALPHA)),
            max_edges=int(payload.get("max_edges", service.DEFAULT_MAX_EDGES_DISPLAY)),
            marker_size=float(payload.get("marker_size", service.DEFAULT_MARKER_SIZE)),
            edge_width=float(payload.get("edge_width", service.DEFAULT_EDGE_WIDTH)),
            arrow_size=float(payload.get("arrow_size", service.DEFAULT_ARROW_SIZE)),
            theme_mode=str(payload.get("theme", "dark")),
        )
    except Exception as e:  # pragma: no cover - surface server errors to UI
        current_app.logger.exception("spatial figure build failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    return jsonify(result)


@bp.route("/api/enrichment", methods=["GET"])
def api_enrichment(dataset_name: str):
    """Both enrichment heatmaps. Heavy on first call (computes + caches CSV)."""
    ds = _resolve(dataset_name)
    theme_mode = str(request.args.get("theme", "dark"))
    try:
        return jsonify({
            "lr_pathway": service.build_lr_pathway_heatmap(ds, theme_mode=theme_mode),
            "celltype_pair": service.build_celltype_pair_heatmap(ds, theme_mode=theme_mode),
        })
    except Exception as e:
        current_app.logger.exception("enrichment build failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@bp.route("/api/loadings", methods=["POST"])
def api_loadings(dataset_name: str):
    ds = _resolve(dataset_name)
    payload = request.get_json(silent=True) or {}
    try:
        mi_idx = int(payload.get("mi_idx", 0))
        top_n = int(payload.get("top_n", 10))
    except (TypeError, ValueError):
        return jsonify({"error": "mi_idx and top_n must be integers"}), 400
    theme_mode = str(payload.get("theme", "dark"))

    try:
        return jsonify(service.build_loadings_figures(
            ds=ds, mi_idx=mi_idx, top_n=top_n, theme_mode=theme_mode,
        ))
    except Exception as e:
        current_app.logger.exception("loadings build failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@bp.route("/api/circle-summary", methods=["POST"])
def api_circle_summary(dataset_name: str):
    """Return both cell-type interaction summary plots — across all slices
    (slice-mean-then-averaged) and for the currently selected slice."""
    ds = _resolve(dataset_name)
    payload = request.get_json(silent=True) or {}
    try:
        slice_idx = int(payload.get("slice_idx", 0))
        mi_idx = int(payload.get("mi_idx", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "slice_idx and mi_idx must be integers"}), 400
    theme_mode = str(payload.get("theme", "dark"))
    try:
        return jsonify(service.build_circle_summary_response(
            ds=ds, slice_idx=slice_idx, mi_idx=mi_idx, theme_mode=theme_mode,
        ))
    except Exception as e:
        current_app.logger.exception("circle-summary build failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@bp.route("/api/threshold-default", methods=["GET"])
def api_threshold_default(dataset_name: str):
    """Return (vmin, vmax, q90) for the chosen (slice, MI) so the UI can
    initialize the threshold slider sensibly."""
    ds = _resolve(dataset_name)
    try:
        slice_idx = int(request.args.get("slice_idx", 0))
        mi_idx = int(request.args.get("mi_idx", 0))
    except ValueError:
        return jsonify({"error": "slice_idx and mi_idx must be integers"}), 400

    tables = service.load_slice(ds, slice_idx)
    mi_name = service.mi_columns(ds.dim_envir)[mi_idx]
    vmin, vmax, q90 = service.edge_threshold_default(tables.edges, mi_name)
    return jsonify({
        "mi_name": mi_name,
        "vmin": vmin, "vmax": vmax, "default": q90,
        "available_sender_celltypes": tables.summary["available_sender_celltypes"],
        "available_receiver_celltypes": tables.summary["available_receiver_celltypes"],
    })
