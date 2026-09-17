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
  function currentTheme() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t === "light" || t === "dark") return t;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
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
    setAllCelltypes(senderSel, false);
  });
  $("m1-receiver-clear").addEventListener("click", (e) => {
    e.preventDefault();
    setAllCelltypes(receiverSel, false);
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
    const previous = new Set(selectedValues(sel));
    sel.replaceChildren();
    items.forEach((label, index) => {
      const row = document.createElement("label");
      row.className = "m1-celltype-option";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.className = "form-check-input";
      input.id = `${sel.id}-option-${index}`;
      input.value = label;
      input.checked = previous.has(label);
      const text = document.createElement("span");
      text.textContent = label;
      row.append(input, text);
      sel.appendChild(row);
    });
    updateCelltypeSummary(sel);
  }

  function selectedValues(sel) {
    return [...sel.querySelectorAll('input[type="checkbox"]:checked')].map((o) => o.value);
  }

  function updateCelltypeSummary(sel) {
    const count = selectedValues(sel).length;
    $(`${sel.id}-summary`).textContent = count ? `${count} selected` : "All types (no filter)";
  }

  function setAllCelltypes(sel, checked) {
    sel.querySelectorAll('input[type="checkbox"]').forEach((input) => { input.checked = checked; });
    updateCelltypeSummary(sel);
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
      body: JSON.stringify({ mi_idx: miIdx, top_n: topN, theme: currentTheme() }),
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

  // Cell-type interaction summary (chord/circle plots).
  const circleAll = $("m1-circle-all");
  const circleSlice = $("m1-circle-slice");
  const circleAllTitle = $("m1-circle-all-title");
  const circleSliceTitle = $("m1-circle-slice-title");
  const circleMiLabel = $("m1-circle-mi");
  const circleSpinner = $("m1-circle-spinner");

  let circleInflight = 0;
  async function refreshCircleSummary() {
    const sliceIdx = parseInt(sliceSel.value, 10);
    const miIdx = parseInt(miSel.value, 10);
    if (Number.isNaN(sliceIdx) || Number.isNaN(miIdx)) return;
    const reqId = ++circleInflight;
    circleSpinner.classList.add("active");
    try {
      const r = await fetch(`${BASE}/api/circle-summary`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ slice_idx: sliceIdx, mi_idx: miIdx, theme: currentTheme() }),
      });
      const data = await r.json().catch(() => ({}));
      if (reqId !== circleInflight) return;
      if (!r.ok) throw new Error(data.error || `circle-summary ${r.status}`);
      circleMiLabel.textContent = `· ${data.mi_name}`;
      circleAllTitle.textContent = data.all_title;
      circleSliceTitle.textContent = data.slice_title;
      Plotly.react(circleAll, data.all_figure.data, data.all_figure.layout,
        { responsive: true, displaylogo: false, displayModeBar: false });
      Plotly.react(circleSlice, data.slice_figure.data, data.slice_figure.layout,
        { responsive: true, displaylogo: false, displayModeBar: false });
    } catch (err) {
      if (reqId !== circleInflight) return;
      circleMiLabel.textContent = `· error: ${err.message}`;
    } finally {
      if (reqId === circleInflight) circleSpinner.classList.remove("active");
    }
  }

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
      const r = await fetch(`${BASE}/api/enrichment?theme=${currentTheme()}`);
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

  let spatialInflight = 0;
  async function render() {
    const reqId = ++spatialInflight;
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
        theme: currentTheme(),
      };
      const t0 = performance.now();
      const result = await fetchSpatial(payload);
      if (reqId !== spatialInflight) return;        // stale, skip
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
      if (reqId === spatialInflight) setStatus(`Error: ${err.message}`);
    } finally {
      if (reqId === spatialInflight) setSpinner(false);
    }
  }

  // Debounced auto-render — every control change triggers a re-render after a
  // short pause, so the user doesn't have to click "Render". The button still
  // works as an explicit refresh.
  let renderTimer = null;
  function scheduleRender(delayMs = 180) {
    clearTimeout(renderTimer);
    renderTimer = setTimeout(render, delayMs);
  }

  renderBtn.addEventListener("click", render);

  // Live value labels for the sliders (decimal places match each slider's step).
  function bindRangeLabel(input, label, decimals) {
    if (!label) return;
    const update = () => { label.textContent = parseFloat(input.value).toFixed(decimals); };
    input.addEventListener("input", update);
    update();
  }
  bindRangeLabel(markerSize, $("m1-marker-size-val"), 1);
  bindRangeLabel(edgeWidth,  $("m1-edge-width-val"),  1);
  bindRangeLabel(arrowSize,  $("m1-arrow-size-val"),  2);
  bindRangeLabel(cellAlpha,  $("m1-cell-alpha-val"),  2);

  // Max edges is still a number input; commit on `change`.
  maxEdges.addEventListener("change", () => scheduleRender(80));
  [senderSel, receiverSel].forEach((el) => {
    el.addEventListener("change", () => {
      updateCelltypeSummary(el);
      scheduleRender(80);
    });
    $(`${el.id}-all`).addEventListener("click", () => {
      setAllCelltypes(el, true);
      scheduleRender(80);
    });
  });

  // Range sliders fire `input` continuously while dragging — debounce harder
  // so the server isn't flooded. The slider value-labels update on every
  // input event for live feedback (handled above).
  thrInput.addEventListener("input", () => scheduleRender(220));
  cellAlpha.addEventListener("input", () => scheduleRender(220));
  markerSize.addEventListener("input", () => scheduleRender(220));
  edgeWidth.addEventListener("input", () => scheduleRender(220));
  arrowSize.addEventListener("input", () => scheduleRender(220));

  // The Clear buttons under the multi-selects need to also fire a re-render.
  $("m1-sender-clear").addEventListener("click", () => scheduleRender(80));
  $("m1-receiver-clear").addEventListener("click", () => scheduleRender(80));

  // When slice or MI changes, refresh threshold range + celltype lists + loadings.
  let bindBusy = false;
  async function onSliceOrMiChange() {
    if (bindBusy) return;
    bindBusy = true;
    setSpinner(true);
    try {
      await refreshSliceBound();
      // Spatial render, loadings refresh, and circle summary refresh run in parallel.
      await Promise.all([render(), refreshLoadings(), refreshCircleSummary()]);
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
    setAllCelltypes(senderSel, false);
    setAllCelltypes(receiverSel, false);
    cellAlpha.value = 0.9;
    maxEdges.value = 12000;
    markerSize.value = 6;
    edgeWidth.value = 1.2;
    arrowSize.value = 0.55;
    await onSliceOrMiChange();
  });

  $("m1-select-all").addEventListener("click", async () => {
    setAllCelltypes(senderSel, true);
    setAllCelltypes(receiverSel, true);
    await render();
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

  // Re-render server-themed Plotly figures when the user toggles theme.
  // Includes the spatial canvas — the figure has the theme baked in by the
  // server, so it must be re-fetched, not just re-skinned client-side.
  window.addEventListener("spn:themechange", () => {
    render();
    refreshLoadings();
    refreshCircleSummary();
    if (enrichmentLoaded) loadEnrichment();
  });
})();
