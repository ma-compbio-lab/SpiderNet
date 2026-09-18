// Module 3 — MI Cascade client.
// /api/run -> heatmap + significant-pair table + default stem
// click a pair row -> /api/stem + /api/insitu-options
// /api/insitu -> spatial cascade scatter
// /api/deggo -> three volcanos + three GO bubbles

(function () {
  "use strict";
  const BASE = window.M3_BASE_URL;
  const $ = (id) => document.getElementById(id);
  function currentTheme() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t === "light" || t === "dark") return t;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  // -- DOM refs -----------------------------------------------------------
  const runBtn = $("m3-run");
  const statusEl = $("m3-status");
  const runtimeEl = $("m3-runtime");
  const spinner = $("m3-spinner");
  const heatmapDiv = $("m3-heatmap");
  const stemDiv = $("m3-stem");
  const pairTbody = $("m3-pair-tbody");
  const stemTopn = $("m3-stem-topn");

  const insituSpinner = $("m3-insitu-spinner");
  const insituStatus = $("m3-insitu-status");
  const insituRunBtn = $("m3-insitu-run");
  const insituDiv = $("m3-insitu");
  const miUpSel = $("m3-insitu-mi-up");
  const miDownSel = $("m3-insitu-mi-down");
  const cell1Sel = $("m3-insitu-cell1");
  const cell2Sel = $("m3-insitu-cell2");
  const cell3Sel = $("m3-insitu-cell3");
  const sliceSel = $("m3-insitu-slice");
  const insituMiThr = $("m3-insitu-mi-thr");
  const insituCellSize = $("m3-insitu-cell-size");
  const insituCellAlpha = $("m3-insitu-cell-alpha");
  const insituEdgeWidth = $("m3-insitu-edge-width");
  const insituArrowSize = $("m3-insitu-arrow-size");

  // Live value labels for the styling sliders — keep them in sync as the user drags.
  function bindRangeLabel(input, label, decimals) {
    const update = () => { label.textContent = parseFloat(input.value).toFixed(decimals); };
    input.addEventListener("input", update);
    update();
  }
  bindRangeLabel(insituMiThr,    $("m3-insitu-mi-thr-val"),    2);
  bindRangeLabel(insituCellSize, $("m3-insitu-cell-size-val"), 1);
  bindRangeLabel(insituCellAlpha,$("m3-insitu-cell-alpha-val"),2);
  bindRangeLabel(insituEdgeWidth,$("m3-insitu-edge-width-val"),1);
  bindRangeLabel(insituArrowSize,$("m3-insitu-arrow-size-val"),2);

  const degSpinner = $("m3-deg-spinner");
  const degStatus = $("m3-deg-status");
  const degRunBtn = $("m3-deg-run");
  const degDirection = $("m3-deg-direction");
  const degLfc = $("m3-deg-lfc");
  const degPadj = $("m3-deg-padj");
  const degBaselineUp = $("m3-deg-baseline-up");
  const degBaselineDown = $("m3-deg-baseline-down");

  // -- state --------------------------------------------------------------
  let CURRENT_CACHE_KEY = null;
  let CURRENT_PAIR_KEY = null;
  let CURRENT_OPTIONS = null;
  let PAIR_OPTIONS = [];           // [{pair_key, mi_first, mi_second, ...}, ...]
  let SUPPRESS_MI_DROPDOWN_EVENTS = false;
  let INSITU_RENDERED = false;
  let DEGGO_RENDERED = false;

  // -- helpers ------------------------------------------------------------
  function setSpinner(on) { spinner.classList.toggle("active", !!on); }
  function setInsituSpinner(on) { insituSpinner.classList.toggle("active", !!on); }
  function setDegSpinner(on) { degSpinner.classList.toggle("active", !!on); }
  function setStatus(msg) { statusEl.textContent = msg; }

  function renderFig(el, fig, opts) {
    if (!fig) return;
    window.spnRenderPlot(el, fig.data, Object.assign({ autosize: true }, fig.layout),
      Object.assign({ responsive: true, displaylogo: false, displayModeBar: false }, opts || {}));
  }

  function fillSelect(sel, items, selectedValue) {
    sel.innerHTML = "";
    items.forEach((it) => {
      const opt = document.createElement("option");
      opt.value = String(it.value !== undefined ? it.value : it);
      opt.textContent = String(it.label !== undefined ? it.label : it);
      if (selectedValue !== undefined && opt.value === String(selectedValue)) opt.selected = true;
      sel.appendChild(opt);
    });
  }

  // -- pair table ---------------------------------------------------------
  function renderPairTable(pairs) {
    if (!pairs || pairs.length === 0) {
      pairTbody.innerHTML = "<tr><td colspan='4' class='text-muted small'>No significant pairs at these thresholds.</td></tr>";
      return;
    }
    pairTbody.innerHTML = pairs.map((p, i) => `
      <tr data-pair-key="${p.pair_key}" data-idx="${i}" class="${i === 0 ? "selected" : ""}">
        <td class="font-monospace small">${p.pair_key}</td>
        <td class="text-end small">${p.zscore.toFixed(3)}</td>
        <td class="text-end small">${p.pvalue_adjusted.toExponential(2)}</td>
        <td class="text-end small">${p.n_celltype_triples}</td>
      </tr>
    `).join("");
    [...pairTbody.querySelectorAll("tr")].forEach((tr) => {
      tr.addEventListener("click", () => {
        const pk = tr.dataset.pairKey;
        if (!pk) return;
        [...pairTbody.querySelectorAll("tr")].forEach((r) => r.classList.remove("selected"));
        tr.classList.add("selected");
        selectPair(pk);
      });
    });
  }

  // -- API calls ----------------------------------------------------------
  async function postJson(path, body) {
    const r = await fetch(`${BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.error || `${path} ${r.status}`);
    }
    return r.json();
  }

  // -- /api/run -----------------------------------------------------------
  async function runAnalysis() {
    if (!window.spnValidateInputs(["m3-mi-threshold", "m3-zscore-thr", "m3-padj-thr", "m3-nperm", "m3-fdr-alpha"], statusEl)) return;
    ++pairVersion;
    ++selectionVersion;
    CURRENT_CACHE_KEY = null;
    PAIR_OPTIONS = [];
    populateMiUpstreamSelect();
    renderPairTable([]);
    Plotly.purge(heatmapDiv);
    CURRENT_PAIR_KEY = null;
    CURRENT_OPTIONS = null;
    insituRunBtn.disabled = degRunBtn.disabled = true;
    [stemDiv, insituDiv, ...[1, 2, 3].flatMap(i => [$( `m3-deg-pos${i}-volcano`), $(`m3-deg-pos${i}-go`)])].forEach(el => Plotly.purge(el));
    [cell1Sel, cell2Sel, cell3Sel, sliceSel].forEach(el => el.replaceChildren());
    insituStatus.textContent = degStatus.textContent = "Run analysis and select a cascade pair.";
    runBtn.disabled = true;
    setSpinner(true);
    setStatus("Running permutation test...");
    runtimeEl.textContent = "";
    try {
      const t0 = performance.now();
      const data = await postJson("/api/run", {
        MI_threshold: parseFloat($("m3-mi-threshold").value),
        zscore_countcolocal_threshold: parseFloat($("m3-zscore-thr").value),
        pvalue_adjusted_threshold: parseFloat($("m3-padj-thr").value),
        nperm: parseInt($("m3-nperm").value, 10),
        fdr_alpha: parseFloat($("m3-fdr-alpha").value),
        force_recompute: $("m3-force").checked,
        theme: currentTheme(),
      });
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      CURRENT_CACHE_KEY = data.cache_key;
      runtimeEl.textContent = `coloc ${data.coloc_seconds.toFixed(1)}s · triples ${data.triples_seconds.toFixed(1)}s · total ${dt}s`;
      setStatus(`${data.n_significant_pairs} significant pairs.`);
      renderFig(heatmapDiv, data.heatmap_figure);
      renderPairTable(data.pair_options);
      PAIR_OPTIONS = data.pair_options || [];
      populateMiUpstreamSelect();
      if (data.stem_figure) {
        renderFig(stemDiv, data.stem_figure);
      }
      if (data.default_pair_key) {
        await selectPair(data.default_pair_key);
      }
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    } finally {
      runBtn.disabled = false;
      setSpinner(false);
    }
  }

  // -- pair selection: refresh stem + in-situ options + MI dropdowns -----
  let pairVersion = 0, stemVersion = 0, selectionVersion = 0;
  async function selectPair(pairKey) {
    ++pairVersion;
    ++selectionVersion;
    [insituDiv, ...[1, 2, 3].flatMap(i => [$(`m3-deg-pos${i}-volcano`), $(`m3-deg-pos${i}-go`)])].forEach(el => Plotly.purge(el));
    INSITU_RENDERED = DEGGO_RENDERED = false;
    degStatus.textContent = "Selection changed. Click Compute DEG + GO to update.";
    CURRENT_PAIR_KEY = pairKey;
    syncMiDropdownsToPair(pairKey);
    await Promise.all([loadStem(pairKey), loadInsituOptions(pairKey)]);
    refreshDegSchematics();
  }

  async function loadStem(pairKey) {
    if (!CURRENT_CACHE_KEY) return;
    const version = ++stemVersion;
    const key = CURRENT_CACHE_KEY;
    setStatus(`Loading triples for ${pairKey}...`);
    try {
      const data = await postJson("/api/stem", {
        cache_key: CURRENT_CACHE_KEY, pair_key: pairKey, theme: currentTheme(), top_n: Number(stemTopn.value),
      });
      if (version !== stemVersion || pairKey !== CURRENT_PAIR_KEY || key !== CURRENT_CACHE_KEY) return;
      renderFig(stemDiv, data.stem_figure);
      setStatus(`${pairKey} loaded.`);
    } catch (err) {
      if (version === stemVersion && pairKey === CURRENT_PAIR_KEY) setStatus(`Stem error: ${err.message}`);
    }
  }

  // -- /api/insitu-options + /api/insitu ----------------------------------
  async function loadInsituOptions(pairKey) {
    if (!CURRENT_CACHE_KEY) return;
    const version = pairVersion;
    [cell1Sel, cell2Sel, cell3Sel, sliceSel].forEach(el => { el.disabled = true; });
    insituStatus.textContent = `Loading in-situ options for ${pairKey}...`;
    insituRunBtn.disabled = true;
    degRunBtn.disabled = true;
    try {
      const data = await postJson("/api/insitu-options", {
        cache_key: CURRENT_CACHE_KEY, pair_key: pairKey,
      });
      if (version !== pairVersion) return;
      [cell1Sel, cell2Sel, cell3Sel, sliceSel].forEach(el => { el.disabled = false; });
      CURRENT_OPTIONS = data;
      fillSelect(cell1Sel, data.cell1_options.map((c) => ({ value: c, label: c })), data.default_cell1);
      fillSelect(cell2Sel, data.cell2_options.map((c) => ({ value: c, label: c })), data.default_cell2);
      fillSelect(cell3Sel, data.cell3_options.map((c) => ({ value: c, label: c })), data.default_cell3);
      updateSliceOptions();
      insituMiThr.value = data.base_threshold.toFixed(2);
      $("m3-insitu-mi-thr-val").textContent = Number(insituMiThr.value).toFixed(2);
      insituStatus.textContent =
        `${data.n_summary_rows} celltype-triple/slice rows · ${data.total_cascade_count} total cascades at threshold ${data.base_threshold.toFixed(2)}`;
    } catch (err) {
      if (version === pairVersion) insituStatus.textContent = `Error: ${err.message}`;
    }
  }

  // -- MI dropdown wiring -------------------------------------------------
  // The pair_options list returned by /api/run already covers every
  // significant (upstream → downstream) pair. We use it to drive two
  // dropdowns: the upstream MI selects an mi_first; the downstream MI
  // updates to show only mi_seconds that exist for that mi_first.
  function uniqueSorted(items) {
    return [...new Set(items)].sort((a, b) => a - b);
  }
  function populateMiUpstreamSelect() {
    const upstreams = uniqueSorted(PAIR_OPTIONS.map((p) => p.mi_first));
    fillSelect(
      miUpSel,
      upstreams.map((m) => ({ value: m, label: `MI-${m}` })),
      upstreams[0],
    );
    populateMiDownstreamSelect(upstreams[0]);
  }
  function populateMiDownstreamSelect(upstream) {
    const downstreams = uniqueSorted(
      PAIR_OPTIONS.filter((p) => p.mi_first === upstream).map((p) => p.mi_second),
    );
    fillSelect(
      miDownSel,
      downstreams.map((m) => ({ value: m, label: `MI-${m}` })),
      downstreams[0],
    );
  }
  // Sync the two dropdowns to a given pair_key, without firing change events.
  function syncMiDropdownsToPair(pairKey) {
    const opt = PAIR_OPTIONS.find((p) => p.pair_key === pairKey);
    if (!opt) return;
    SUPPRESS_MI_DROPDOWN_EVENTS = true;
    miUpSel.value = String(opt.mi_first);
    populateMiDownstreamSelect(opt.mi_first);
    miDownSel.value = String(opt.mi_second);
    SUPPRESS_MI_DROPDOWN_EVENTS = false;
  }
  miUpSel.addEventListener("change", () => {
    if (SUPPRESS_MI_DROPDOWN_EVENTS) return;
    const upstream = parseInt(miUpSel.value, 10);
    populateMiDownstreamSelect(upstream);
    const downstream = parseInt(miDownSel.value, 10);
    const pairKey = `MI-${upstream} -> MI-${downstream}`;
    if (PAIR_OPTIONS.some((p) => p.pair_key === pairKey)) {
      // Reflect the new selection in the heatmap-side pair table too.
      [...pairTbody.querySelectorAll("tr")].forEach((r) => {
        r.classList.toggle("selected", r.dataset.pairKey === pairKey);
      });
      selectPair(pairKey);
    }
  });
  miDownSel.addEventListener("change", () => {
    if (SUPPRESS_MI_DROPDOWN_EVENTS) return;
    const upstream = parseInt(miUpSel.value, 10);
    const downstream = parseInt(miDownSel.value, 10);
    const pairKey = `MI-${upstream} -> MI-${downstream}`;
    if (PAIR_OPTIONS.some((p) => p.pair_key === pairKey)) {
      [...pairTbody.querySelectorAll("tr")].forEach((r) => {
        r.classList.toggle("selected", r.dataset.pairKey === pairKey);
      });
      selectPair(pairKey);
    }
  });

  function updateSliceOptions() {
    if (!CURRENT_OPTIONS) return;
    const c1 = cell1Sel.value, c2 = cell2Sel.value, c3 = cell3Sel.value;
    const matching = CURRENT_OPTIONS.summary_rows
      .filter((r) => r.cell1_type === c1 && r.cell2_type === c2 && r.cell3_type === c3)
      .sort((a, b) => b.count - a.count || a.slice_index - b.slice_index);
    if (matching.length === 0) {
      sliceSel.innerHTML = "<option disabled>no slices contain this triple</option>";
      insituRunBtn.disabled = true;
      degRunBtn.disabled = true;
      return;
    }
    fillSelect(
      sliceSel,
      matching.map((r) => ({
        value: r.slice_index,
        label: `slice ${r.slice_index} · ${r.sample_name} · ${r.count} cascades`,
      })),
      matching[0].slice_index,
    );
    insituRunBtn.disabled = false;
    degRunBtn.disabled = false;
  }

  [cell1Sel, cell2Sel, cell3Sel].forEach((el) => el.addEventListener("change", updateSliceOptions));

  // -- Cascade schematics (Current cascade of interest + Baseline) -------
  function _escapeHtml(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  function _schemaColors() {
    return currentTheme() === "light"
      ? { up: "#D12052", down: "#03AED2", text: "#111111", border: "rgba(17,17,17,0.32)" }
      : { up: "#FF0087", down: "#00F7FF", text: "#ffffff", border: "rgba(255,255,255,0.60)" };
  }
  function _cellFill(name) {
    const palette = (CURRENT_OPTIONS && CURRENT_OPTIONS.palette) || {};
    return palette[String(name)] || "#9ca3af";
  }
  function renderCascadeSchematic(containerId, cfg) {
    const el = document.getElementById(containerId);
    if (!el) return;
    const cell1 = String(cfg.cell1 || ""), cell2 = String(cfg.cell2 || ""), cell3 = String(cfg.cell3 || "");
    const showUp = cfg.showUpstream !== false;
    const showDown = cfg.showDownstream !== false;
    if (!CURRENT_PAIR_KEY || !cell1 || !cell2 || !cell3) {
      el.innerHTML = `<div class="m3-schematic-empty">${_escapeHtml(cfg.emptyMsg || "Select a pair and triple to preview the cascade.")}</div>`;
      return;
    }
    const c = _schemaColors();
    const idSuf = containerId.replace(/[^A-Za-z0-9_-]/g, "-");
    const x1 = 70, x2 = 210, x3 = 350, cy = 62, r = 28, edgeY = 62;
    const miUp = String(cfg.miUp || "");
    const miDown = String(cfg.miDown || "");
    const line1 = showUp
      ? `<line x1="${x1 + r + 8}" y1="${edgeY}" x2="${x2 - r - 12}" y2="${edgeY}" stroke="${c.up}" stroke-width="6" stroke-linecap="round" marker-end="url(#m3-arrow-up-${idSuf})"></line>`
      : "";
    const line2 = showDown
      ? `<line x1="${x2 + r + 8}" y1="${edgeY}" x2="${x3 - r - 12}" y2="${edgeY}" stroke="${c.down}" stroke-width="6" stroke-linecap="round" marker-end="url(#m3-arrow-down-${idSuf})"></line>`
      : "";
    const lab1 = showUp ? `<text x="${(x1 + x2) / 2}" y="26" text-anchor="middle" font-size="22" font-weight="700" fill="${c.up}">${_escapeHtml(miUp)}</text>` : "";
    const lab2 = showDown ? `<text x="${(x2 + x3) / 2}" y="26" text-anchor="middle" font-size="22" font-weight="700" fill="${c.down}">${_escapeHtml(miDown)}</text>` : "";
    el.innerHTML = `
      <svg viewBox="0 0 420 148" role="img" aria-label="Cascade schematic">
        <defs>
          <marker id="m3-arrow-up-${idSuf}" markerWidth="14" markerHeight="14" refX="12" refY="7" orient="auto" markerUnits="userSpaceOnUse">
            <path d="M0,0 L14,7 L0,14 z" fill="${c.up}"></path>
          </marker>
          <marker id="m3-arrow-down-${idSuf}" markerWidth="14" markerHeight="14" refX="12" refY="7" orient="auto" markerUnits="userSpaceOnUse">
            <path d="M0,0 L14,7 L0,14 z" fill="${c.down}"></path>
          </marker>
        </defs>
        ${line1}${line2}${lab1}${lab2}
        <circle cx="${x1}" cy="${cy}" r="${r}" fill="${_cellFill(cell1)}" stroke="${c.border}" stroke-width="2.4"></circle>
        <circle cx="${x2}" cy="${cy}" r="${r}" fill="${_cellFill(cell2)}" stroke="${c.border}" stroke-width="2.4"></circle>
        <circle cx="${x3}" cy="${cy}" r="${r}" fill="${_cellFill(cell3)}" stroke="${c.border}" stroke-width="2.4"></circle>
        <text x="${x1}" y="${cy + 5}" text-anchor="middle" font-size="22" font-weight="700" fill="${c.text}">1</text>
        <text x="${x2}" y="${cy + 5}" text-anchor="middle" font-size="22" font-weight="700" fill="${c.text}">2</text>
        <text x="${x3}" y="${cy + 5}" text-anchor="middle" font-size="22" font-weight="700" fill="${c.text}">3</text>
        <text x="${x1}" y="124" text-anchor="middle" font-size="13" font-weight="600" fill="${c.text}">${_escapeHtml(cell1)}</text>
        <text x="${x2}" y="124" text-anchor="middle" font-size="13" font-weight="600" fill="${c.text}">${_escapeHtml(cell2)}</text>
        <text x="${x3}" y="124" text-anchor="middle" font-size="13" font-weight="600" fill="${c.text}">${_escapeHtml(cell3)}</text>
      </svg>`;
  }

  function _currentMiLabels() {
    if (CURRENT_OPTIONS && CURRENT_OPTIONS.mi_first && CURRENT_OPTIONS.mi_second) {
      return { up: `MI-${CURRENT_OPTIONS.mi_first}`, down: `MI-${CURRENT_OPTIONS.mi_second}` };
    }
    const m = String(CURRENT_PAIR_KEY || "").match(/MI-(\d+)\s*->\s*MI-(\d+)/i);
    return m ? { up: `MI-${m[1]}`, down: `MI-${m[2]}` } : { up: "MI-i", down: "MI-j" };
  }
  function refreshDegSchematics() {
    const labels = _currentMiLabels();
    renderCascadeSchematic("m3-deg-interest-schematic", {
      cell1: cell1Sel.value, cell2: cell2Sel.value, cell3: cell3Sel.value,
      miUp: labels.up, miDown: labels.down,
      showUpstream: true, showDownstream: true,
      emptyMsg: "Select a significant MI pair and a (cell1 → cell2 → cell3) triple in the in-situ panel above.",
    });
    renderCascadeSchematic("m3-deg-baseline-schematic", {
      cell1: cell1Sel.value, cell2: cell2Sel.value, cell3: cell3Sel.value,
      miUp: labels.up, miDown: labels.down,
      showUpstream: degBaselineUp.value === "true",
      showDownstream: degBaselineDown.value === "true",
      emptyMsg: "Inactive selections remove the corresponding edge.",
    });
  }
  // Refresh whenever any input that drives the schematics changes.
  [cell1Sel, cell2Sel, cell3Sel, degBaselineUp, degBaselineDown].forEach((el) => {
    el.addEventListener("change", refreshDegSchematics);
  });
  // And on theme toggle (colors flip).
  window.addEventListener("spn:themechange", refreshDegSchematics);

  async function runInsitu() {
    const version = selectionVersion;
    if (!CURRENT_CACHE_KEY || !CURRENT_PAIR_KEY) {
      insituStatus.textContent = "Run cascade analysis first.";
      return;
    }
    insituRunBtn.disabled = true;
    setInsituSpinner(true);
    insituStatus.textContent = "Rendering...";
    try {
      const t0 = performance.now();
      const data = await postJson("/api/insitu", {
        cache_key: CURRENT_CACHE_KEY,
        pair_key: CURRENT_PAIR_KEY,
        cell1_type: cell1Sel.value,
        cell2_type: cell2Sel.value,
        cell3_type: cell3Sel.value,
        slice_index: parseInt(sliceSel.value, 10),
        mi_threshold: parseFloat(insituMiThr.value),
        cell_size: parseFloat(insituCellSize.value) || 6,
        cell_alpha: parseFloat(insituCellAlpha.value) || 0.82,
        edge_width: parseFloat(insituEdgeWidth.value) || 1.2,
        arrow_size: parseFloat(insituArrowSize.value) || 0.55,
        theme: currentTheme(),
      });
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      if (version !== selectionVersion) return;
      renderFig(insituDiv, data.figure);
      const m = data.meta;
      insituStatus.textContent =
        `Slice ${m.slice_index} (${m.sample_name}) · ${m.n_cascade_instances.toLocaleString()} instances · ` +
        `cells: ${m.n_cell1}/${m.n_cell2}/${m.n_cell3} · ${dt}s`;
      INSITU_RENDERED = true;
    } catch (err) {
      insituStatus.textContent = `Error: ${err.message}`;
    } finally {
      setInsituSpinner(false);
      insituRunBtn.disabled = !sliceSel.value || sliceSel.disabled || !CURRENT_PAIR_KEY;
    }
  }

  // -- /api/deggo ---------------------------------------------------------
  async function runDegGo() {
    const version = selectionVersion;
    if (!window.spnValidateInputs(["m3-deg-lfc", "m3-deg-padj"], degStatus)) return;
    if (!CURRENT_CACHE_KEY || !CURRENT_PAIR_KEY) {
      degStatus.textContent = "Run analysis and pick a triple first.";
      return;
    }
    degRunBtn.disabled = true;
    setDegSpinner(true);
    degStatus.textContent = "Computing DEG + GO across positions...";
    try {
      const t0 = performance.now();
      const data = await postJson("/api/deggo", {
        cache_key: CURRENT_CACHE_KEY,
        pair_key: CURRENT_PAIR_KEY,
        cell1_type: cell1Sel.value,
        cell2_type: cell2Sel.value,
        cell3_type: cell3Sel.value,
        mi_threshold: parseFloat(insituMiThr.value),
        baseline_upstream_active: degBaselineUp.value === "true",
        baseline_downstream_active: degBaselineDown.value === "true",
        lfc_thresh: parseFloat(degLfc.value),
        padj_thresh: parseFloat(degPadj.value),
        gene_direction: degDirection.value,
        theme: currentTheme(),
      });
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      if (version !== selectionVersion) return;
      ["1", "2", "3"].forEach((pos) => {
        const p = data.positions[pos];
        if (!p) return;
        renderFig($(`m3-deg-pos${pos}-volcano`), p.volcano_figure);
        renderFig($(`m3-deg-pos${pos}-go`), p.go_figure);
      });
      const summary = ["1", "2", "3"].map((pos) => {
        const p = data.positions[pos];
        return `${p.label}: up=${p.n_up}, down=${p.n_down}`;
      }).join(" · ");
      degStatus.textContent =
        `${data.gene_direction} genes · ${data.interest_triplets.toLocaleString()} interest triplets across ` +
        `${data.interest_slices} slices · ${summary} · ${dt}s`;
      const errors = Object.values(data.positions).filter(p => p.go_state === "error").map(p => `${p.label}: ${p.go_message}`);
      if (errors.length) degStatus.textContent += ` · ${errors.join(" · ")}`;
      DEGGO_RENDERED = true;
    } catch (err) {
      degStatus.textContent = `Error: ${err.message}`;
    } finally {
      setDegSpinner(false);
      degRunBtn.disabled = !sliceSel.value || sliceSel.disabled || !CURRENT_PAIR_KEY;
    }
  }

  [cell1Sel, cell2Sel, cell3Sel, sliceSel, insituMiThr, degBaselineUp, degBaselineDown,
    degLfc, degPadj, degDirection].forEach(el => el.addEventListener("change", () => { ++selectionVersion; }));

  // -- bindings -----------------------------------------------------------
  runBtn.addEventListener("click", runAnalysis);
  insituRunBtn.addEventListener("click", runInsitu);
  degRunBtn.addEventListener("click", runDegGo);
  stemTopn.addEventListener("change", () => {
    if (CURRENT_PAIR_KEY) loadStem(CURRENT_PAIR_KEY);
  });

  // Theme changes restyle existing plots through the shared helpers in main.js.
})();
