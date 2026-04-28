// Module 1 — Basic Analysis client.
//
// Fetches dataset metadata, populates the controls, and re-renders the spatial
// Plotly figure on every "Render" click.

(function () {
  "use strict";

  const BASE = window.M1_BASE_URL;
  const DATASET = window.M1_DATASET;

  let META = null;        // /api/meta payload
  let CURRENT_THRESHOLD = null;

  const $ = (id) => document.getElementById(id);
  const sliceSel = $("m1-slice");
  const miSel = $("m1-mi");
  const thrInput = $("m1-threshold");
  const thrValue = $("m1-threshold-value");
  const maxEdges = $("m1-max-edges");
  const markerSize = $("m1-marker-size");
  const edgeWidth = $("m1-edge-width");
  const arrowSize = $("m1-arrow-size");
  const cellAlpha = $("m1-cell-alpha");
  const senderSel = $("m1-sender");
  const receiverSel = $("m1-receiver");
  const renderBtn = $("m1-render");
  const resetBtn = $("m1-reset");
  const statusEl = $("m1-status");
  const countsEl = $("m1-counts");
  const spinner = $("m1-spinner");
  const topNInput = $("m1-top-n");
  const stemLR = $("m1-stem-lr");
  const stemSender = $("m1-stem-sender");
  const stemReceiver = $("m1-stem-receiver");
  const loadingsMiLabel = $("m1-loadings-mi");

  $("m1-sender-clear").addEventListener("click", (e) => {
    e.preventDefault();
    [...senderSel.options].forEach((o) => (o.selected = false));
  });
  $("m1-receiver-clear").addEventListener("click", (e) => {
    e.preventDefault();
    [...receiverSel.options].forEach((o) => (o.selected = false));
  });

  function setStatus(msg) { statusEl.textContent = msg; }
  function setSpinner(on) { spinner.classList.toggle("active", !!on); }

  function fillSelect(sel, items, formatter) {
    sel.innerHTML = "";
    items.forEach((it, idx) => {
      const opt = document.createElement("option");
      opt.value = formatter ? formatter(it, idx).value : idx;
      opt.textContent = formatter ? formatter(it, idx).label : String(it);
      sel.appendChild(opt);
    });
  }

  function fillMultiselect(sel, items) {
    sel.innerHTML = "";
    items.forEach((label) => {
      const opt = document.createElement("option");
      opt.value = label;
      opt.textContent = label;
      sel.appendChild(opt);
    });
  }

  function selectedValues(sel) {
    return [...sel.selectedOptions].map((o) => o.value);
  }

  async function fetchMeta() {
    setStatus("Loading metadata...");
    const r = await fetch(`${BASE}/api/meta`);
    if (!r.ok) throw new Error(`meta ${r.status}`);
    return r.json();
  }

  async function fetchThresholdDefault(sliceIdx, miIdx) {
    const params = new URLSearchParams({ slice_idx: sliceIdx, mi_idx: miIdx });
    const r = await fetch(`${BASE}/api/threshold-default?${params}`);
    if (!r.ok) throw new Error(`threshold ${r.status}`);
    return r.json();
  }

  async function fetchSpatial(payload) {
    const r = await fetch(`${BASE}/api/spatial`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.error || `spatial ${r.status}`);
    }
    return r.json();
  }

  async function fetchLoadings(miIdx, topN) {
    const r = await fetch(`${BASE}/api/loadings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mi_idx: miIdx, top_n: topN, theme: "dark" }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.error || `loadings ${r.status}`);
    }
    return r.json();
  }

  function renderStem(targetEl, fig) {
    Plotly.react(
      targetEl,
      fig.data,
      Object.assign({ autosize: true }, fig.layout),
      { responsive: true, displaylogo: false, displayModeBar: false }
    );
  }

  let loadingsInflight = 0;
  async function refreshLoadings() {
    const miIdx = parseInt(miSel.value, 10);
    const topN = Math.max(3, Math.min(40, parseInt(topNInput.value, 10) || 10));
    const reqId = ++loadingsInflight;
    try {
      const result = await fetchLoadings(miIdx, topN);
      if (reqId !== loadingsInflight) return;  // superseded
      loadingsMiLabel.textContent = `· ${result.mi_name}`;
      renderStem(stemLR, result.lr.figure);
      renderStem(stemSender, result.sender.figure);
      renderStem(stemReceiver, result.receiver.figure);
    } catch (err) {
      if (reqId !== loadingsInflight) return;
      loadingsMiLabel.textContent = `· error: ${err.message}`;
    }
  }

  topNInput.addEventListener("change", refreshLoadings);

  // Enrichment heatmaps — heavy on first call, lazy-loaded.
  const enrichBtn = $("m1-enrichment-load");
  const enrichStatus = $("m1-enrichment-status");
  const heatmapCelltype = $("m1-heatmap-celltype");
  const heatmapLrPathway = $("m1-heatmap-lr-pathway");

  let enrichmentLoaded = false;
  async function loadEnrichment() {
    enrichBtn.disabled = true;
    enrichStatus.textContent = "Computing — this may take ~30–60 s on first run...";
    try {
      const t0 = performance.now();
      const r = await fetch(`${BASE}/api/enrichment?theme=dark`);
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.error || `enrichment ${r.status}`);
      }
      const data = await r.json();
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      const ct = data.celltype_pair;
      const lr = data.lr_pathway;
      Plotly.react(heatmapCelltype, ct.figure.data, ct.figure.layout,
        { responsive: true, displaylogo: false });
      Plotly.react(heatmapLrPathway, lr.figure.data, lr.figure.layout,
        { responsive: true, displaylogo: false });
      enrichStatus.textContent =
        `Loaded in ${dt}s · ${ct.n_pairs} celltype pairs above z=${ct.threshold.toFixed(2)} · ` +
        `${lr.table.rows.length} pathways`;
      enrichmentLoaded = true;
      enrichBtn.innerHTML = '<i class="fas fa-redo me-1"></i>Reload';
    } catch (err) {
      enrichStatus.textContent = `Error: ${err.message}`;
    } finally {
      enrichBtn.disabled = false;
    }
  }
  enrichBtn.addEventListener("click", loadEnrichment);

  async function refreshSliceBound() {
    const sliceIdx = parseInt(sliceSel.value, 10);
    const miIdx = parseInt(miSel.value, 10);
    const def = await fetchThresholdDefault(sliceIdx, miIdx);
    thrInput.min = def.vmin;
    thrInput.max = Math.max(def.vmax, def.vmin + 1e-6);
    thrInput.step = (def.vmax - def.vmin) / 1000 || 0.001;
    thrInput.value = def.default;
    CURRENT_THRESHOLD = def.default;
    thrValue.textContent = def.default.toFixed(3);

    fillMultiselect(senderSel, def.available_sender_celltypes);
    fillMultiselect(receiverSel, def.available_receiver_celltypes);
  }

  thrInput.addEventListener("input", () => {
    CURRENT_THRESHOLD = parseFloat(thrInput.value);
    thrValue.textContent = CURRENT_THRESHOLD.toFixed(3);
  });

  async function render() {
    setSpinner(true);
    try {
      const payload = {
        slice_idx: parseInt(sliceSel.value, 10),
        mi_idx: parseInt(miSel.value, 10),
        threshold: parseFloat(thrInput.value),
        sender_types: selectedValues(senderSel),
        receiver_types: selectedValues(receiverSel),
        cell_alpha: parseFloat(cellAlpha.value),
        max_edges: parseInt(maxEdges.value, 10),
        marker_size: parseFloat(markerSize.value),
        edge_width: parseFloat(edgeWidth.value),
        arrow_size: parseFloat(arrowSize.value),
        theme: "dark",
      };
      const t0 = performance.now();
      const result = await fetchSpatial(payload);
      const dt = ((performance.now() - t0) / 1000).toFixed(2);
      const fig = result.figure;
      Plotly.react(
        "m1-spatial-canvas",
        fig.data,
        fig.layout,
        { responsive: true, displaylogo: false }
      );
      setStatus(`Rendered ${result.meta.mi_name} · slice ${payload.slice_idx} · ${dt}s`);
      const trunc = result.meta.truncated ? " <span class='text-warning'>(truncated)</span>" : "";
      countsEl.innerHTML = `${result.meta.n_edges_visible.toLocaleString()} visible edges${trunc}`;
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    } finally {
      setSpinner(false);
    }
  }

  renderBtn.addEventListener("click", render);

  // When slice or MI changes, refresh threshold range + celltype lists + loadings.
  let bindBusy = false;
  async function onSliceOrMiChange() {
    if (bindBusy) return;
    bindBusy = true;
    setSpinner(true);
    try {
      await refreshSliceBound();
      // Spatial render and loadings refresh in parallel.
      await Promise.all([render(), refreshLoadings()]);
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    } finally {
      bindBusy = false;
      setSpinner(false);
    }
  }
  sliceSel.addEventListener("change", onSliceOrMiChange);
  miSel.addEventListener("change", onSliceOrMiChange);

  resetBtn.addEventListener("click", async () => {
    [...senderSel.options].forEach((o) => (o.selected = false));
    [...receiverSel.options].forEach((o) => (o.selected = false));
    cellAlpha.value = 0.9;
    maxEdges.value = 12000;
    markerSize.value = 6;
    edgeWidth.value = 1.2;
    arrowSize.value = 0.55;
    await onSliceOrMiChange();
  });

  // Bootstrap.
  (async function init() {
    try {
      META = await fetchMeta();
      fillSelect(sliceSel, META.slices, (s, i) => ({
        value: String(i),
        label: `slice ${String(i).padStart(3, "0")} (${s.n_cells.toLocaleString()} cells, ${s.n_edges.toLocaleString()} edges)`,
      }));
      fillSelect(miSel, META.mi_list, (name, i) => ({
        value: String(i),
        label: name,
      }));
      // Default to MI-10 if available, otherwise first MI.
      const idx10 = META.mi_list.indexOf("MI-10");
      if (idx10 >= 0) miSel.value = String(idx10);

      await onSliceOrMiChange();
    } catch (err) {
      setStatus(`Init failed: ${err.message}`);
    }
  })();
})();
