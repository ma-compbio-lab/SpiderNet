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
  const cell1Sel = $("m3-insitu-cell1");
  const cell2Sel = $("m3-insitu-cell2");
  const cell3Sel = $("m3-insitu-cell3");
  const sliceSel = $("m3-insitu-slice");
  const insituMiThr = $("m3-insitu-mi-thr");

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

  // -- helpers ------------------------------------------------------------
  function setSpinner(on) { spinner.classList.toggle("active", !!on); }
  function setInsituSpinner(on) { insituSpinner.classList.toggle("active", !!on); }
  function setDegSpinner(on) { degSpinner.classList.toggle("active", !!on); }
  function setStatus(msg) { statusEl.textContent = msg; }

  function renderFig(el, fig, opts) {
    if (!fig) return;
    Plotly.react(el, fig.data, Object.assign({ autosize: true }, fig.layout),
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

  // -- pair selection: refresh stem + in-situ options ---------------------
  async function selectPair(pairKey) {
    CURRENT_PAIR_KEY = pairKey;
    await Promise.all([loadStem(pairKey), loadInsituOptions(pairKey)]);
  }

  async function loadStem(pairKey) {
    if (!CURRENT_CACHE_KEY) return;
    setStatus(`Loading triples for ${pairKey}...`);
    try {
      const data = await postJson("/api/stem", {
        cache_key: CURRENT_CACHE_KEY, pair_key: pairKey, theme: currentTheme(),
      });
      renderFig(stemDiv, data.stem_figure);
      setStatus(`${pairKey} loaded.`);
    } catch (err) {
      setStatus(`Stem error: ${err.message}`);
    }
  }

  // -- /api/insitu-options + /api/insitu ----------------------------------
  async function loadInsituOptions(pairKey) {
    if (!CURRENT_CACHE_KEY) return;
    insituStatus.textContent = `Loading in-situ options for ${pairKey}...`;
    insituRunBtn.disabled = true;
    degRunBtn.disabled = true;
    try {
      const data = await postJson("/api/insitu-options", {
        cache_key: CURRENT_CACHE_KEY, pair_key: pairKey,
      });
      CURRENT_OPTIONS = data;
      fillSelect(cell1Sel, data.cell1_options.map((c) => ({ value: c, label: c })), data.default_cell1);
      fillSelect(cell2Sel, data.cell2_options.map((c) => ({ value: c, label: c })), data.default_cell2);
      fillSelect(cell3Sel, data.cell3_options.map((c) => ({ value: c, label: c })), data.default_cell3);
      updateSliceOptions();
      insituMiThr.value = data.base_threshold.toFixed(2);
      insituStatus.textContent =
        `${data.n_summary_rows} celltype-triple/slice rows · ${data.total_cascade_count} total cascades at threshold ${data.base_threshold.toFixed(2)}`;
    } catch (err) {
      insituStatus.textContent = `Error: ${err.message}`;
    }
  }

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

  async function runInsitu() {
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
        theme: currentTheme(),
      });
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      renderFig(insituDiv, data.figure);
      const m = data.meta;
      insituStatus.textContent =
        `Slice ${m.slice_index} (${m.sample_name}) · ${m.n_cascade_instances.toLocaleString()} instances · ` +
        `cells: ${m.n_cell1}/${m.n_cell2}/${m.n_cell3} · ${dt}s`;
    } catch (err) {
      insituStatus.textContent = `Error: ${err.message}`;
    } finally {
      setInsituSpinner(false);
      insituRunBtn.disabled = false;
    }
  }

  // -- /api/deggo ---------------------------------------------------------
  async function runDegGo() {
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
    } catch (err) {
      degStatus.textContent = `Error: ${err.message}`;
    } finally {
      setDegSpinner(false);
      degRunBtn.disabled = false;
    }
  }

  // -- bindings -----------------------------------------------------------
  runBtn.addEventListener("click", runAnalysis);
  insituRunBtn.addEventListener("click", runInsitu);
  degRunBtn.addEventListener("click", runDegGo);
  stemTopn.addEventListener("change", () => {
    if (CURRENT_PAIR_KEY) loadStem(CURRENT_PAIR_KEY);
  });

  // Re-render server-themed Plotly figures when the user toggles theme.
  window.addEventListener("spn:themechange", () => {
    if (CURRENT_CACHE_KEY) runAnalysis();
  });
})();
