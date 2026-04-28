// Module 4 — In silico spatial perturbation client.
// Lifecycle:
//   1. /api/state-info  -> already loaded? cell-type list?
//   2. /api/prepare     -> blocking model + baseline load (first time only)
//   3. /api/feature-tables -> populate stem plots + 3 feature tables (knockout mode)
//   4. /api/run         -> perturb + compare; returns volcano + DE + GO/KEGG + cache_key
//   5. /api/refresh     -> re-render with new thresholds using cached run

(function () {
  "use strict";
  const BASE = window.M4_BASE_URL;
  const $ = (id) => document.getElementById(id);

  // -- DOM refs ----------------------------------------------------------
  const root = $("m4-root");
  const runBtn = $("m4-run");
  const runHint = $("m4-run-hint");

  const modeKO = $("m4-mode-knockout");
  const modeRP = $("m4-mode-replacement");

  const miSel = $("m4-mi");
  const topnInput = $("m4-topn");
  const keepPctInput = $("m4-keep-pct");
  const senderTypesSel = $("m4-sender-types");
  const receiverTypesSel = $("m4-receiver-types");

  const replacementTypesSel = $("m4-replacement-types");
  const replacedTypesSel = $("m4-replaced-types");
  const seedInput = $("m4-seed");

  const targetTypesSel = $("m4-target-types");
  const maxCellsInput = $("m4-max-cells");

  const lfcInput = $("m4-lfc");
  const padjInput = $("m4-padj");
  const directionSel = $("m4-direction");
  const scopeSel = $("m4-scope");

  const lrStem = $("m4-lr-stem");
  const senderStem = $("m4-sender-stem");
  const receiverStem = $("m4-receiver-stem");
  const lrTbody = $("m4-lr-tbody");
  const senderTbody = $("m4-sender-tbody");
  const receiverTbody = $("m4-receiver-tbody");
  const featureStatus = $("m4-feature-status");
  const featureSpinner = $("m4-feature-spinner");

  const summaryDiv = $("m4-summary");
  const volcanoDiv = $("m4-volcano");
  const volcanoSpinner = $("m4-volcano-spinner");
  const goDiv = $("m4-go");
  const keggDiv = $("m4-kegg");
  const deTbody = $("m4-de-tbody");

  // -- state -------------------------------------------------------------
  let CURRENT_CACHE_KEY = null;
  let LR_ROWS = [], SENDER_ROWS = [], RECEIVER_ROWS = [];
  let LR_SELECTED = new Set(), SENDER_SELECTED = new Set(), RECEIVER_SELECTED = new Set();
  let CELLTYPES = [];
  let REFRESH_TIMER = null;
  let FETCHED_FEATURES_FOR_MI = null;

  // -- helpers -----------------------------------------------------------
  function setMode(mode) {
    root.classList.toggle("m4-mode-knockout", mode === "knockout");
    root.classList.toggle("m4-mode-replacement", mode === "replacement");
  }

  function renderFig(el, fig) {
    if (!fig) return;
    Plotly.react(el, fig.data, Object.assign({ autosize: true }, fig.layout),
      { responsive: true, displaylogo: false, displayModeBar: false });
  }

  async function postJson(path, body) {
    const r = await fetch(`${BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const json = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(json.error || `${r.status} ${r.statusText}`);
    return json;
  }

  async function getJson(path) {
    const r = await fetch(`${BASE}${path}`);
    const json = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(json.error || `${r.status} ${r.statusText}`);
    return json;
  }

  function fillMultiSelect(sel, items, defaultSelected = []) {
    const prevSelected = new Set(
      [...sel.selectedOptions].map((o) => o.value).concat(defaultSelected)
    );
    sel.innerHTML = "";
    items.forEach((it) => {
      const opt = document.createElement("option");
      opt.value = String(it);
      opt.textContent = String(it);
      if (prevSelected.has(opt.value)) opt.selected = true;
      sel.appendChild(opt);
    });
  }

  function selectedValues(sel) {
    return [...sel.selectedOptions].map((o) => o.value);
  }

  function currentMode() {
    return modeRP.checked ? "replacement" : "knockout";
  }

  // -- feature tables (knockout) ----------------------------------------
  function renderFeatureTable(tbody, rows, kind, selectedSet) {
    if (!rows || rows.length === 0) {
      tbody.innerHTML = "<tr><td colspan='4' class='text-muted small'>No rows.</td></tr>";
      return;
    }
    tbody.innerHTML = rows.map((r, i) => {
      const checked = selectedSet.has(i) ? "checked" : "";
      const label = kind === "lr" ? r.label : r.gene;
      return `
        <tr data-idx="${i}" class="${selectedSet.has(i) ? "selected" : ""}">
          <td><input type="checkbox" class="form-check-input m4-row-check" ${checked}></td>
          <td class="small">${r.rank}</td>
          <td class="small font-monospace">${label}</td>
          <td class="text-end small">${(+r.normalized_loading).toFixed(4)}</td>
        </tr>`;
    }).join("");
    [...tbody.querySelectorAll("tr")].forEach((tr) => {
      tr.addEventListener("click", (ev) => {
        const idx = parseInt(tr.dataset.idx, 10);
        if (Number.isNaN(idx)) return;
        if (ev.target.tagName === "INPUT") return; // checkbox handles itself
        const cb = tr.querySelector(".m4-row-check");
        cb.checked = !cb.checked;
        toggleSelect(selectedSet, idx, cb.checked);
        tr.classList.toggle("selected", cb.checked);
      });
      tr.querySelector(".m4-row-check").addEventListener("change", (ev) => {
        const idx = parseInt(tr.dataset.idx, 10);
        if (Number.isNaN(idx)) return;
        toggleSelect(selectedSet, idx, ev.target.checked);
        tr.classList.toggle("selected", ev.target.checked);
      });
    });
  }

  function toggleSelect(set, idx, on) {
    if (on) set.add(idx); else set.delete(idx);
  }

  async function loadFeatureTables() {
    if (currentMode() !== "knockout") return;
    const miName = miSel.value;
    if (!miName) return;
    const topN = Math.max(5, Math.min(20, parseInt(topnInput.value, 10) || 10));
    featureSpinner.classList.add("active");
    try {
      const data = await postJson("/api/feature-tables", {
        mi_name: miName, top_n: topN, theme: "dark",
      });
      LR_ROWS = data.lr_rows || [];
      SENDER_ROWS = data.sender_rows || [];
      RECEIVER_ROWS = data.receiver_rows || [];
      LR_SELECTED = new Set(LR_ROWS.map((_, i) => i));
      SENDER_SELECTED = new Set(SENDER_ROWS.map((_, i) => i));
      RECEIVER_SELECTED = new Set(RECEIVER_ROWS.map((_, i) => i));
      renderFeatureTable(lrTbody, LR_ROWS, "lr", LR_SELECTED);
      renderFeatureTable(senderTbody, SENDER_ROWS, "sender", SENDER_SELECTED);
      renderFeatureTable(receiverTbody, RECEIVER_ROWS, "receiver", RECEIVER_SELECTED);
      renderFig(lrStem, data.lr_figure);
      renderFig(senderStem, data.sender_figure);
      renderFig(receiverStem, data.receiver_figure);
      featureStatus.textContent = `${miName} · top ${topN}`;
      FETCHED_FEATURES_FOR_MI = `${miName}|${topN}`;
    } catch (e) {
      featureStatus.textContent = `Error: ${e.message}`;
    } finally {
      featureSpinner.classList.remove("active");
    }
  }

  // -- run + refresh -----------------------------------------------------
  function buildRunPayload() {
    const mode = currentMode();
    const payload = {
      mode,
      target_celltypes: selectedValues(targetTypesSel),
      max_cells_for_de: parseInt(maxCellsInput.value, 10) || 0,
      lfc_threshold: parseFloat(lfcInput.value) || 0,
      padj_threshold: parseFloat(padjInput.value) || 0,
      enrich_direction: directionSel.value,
      enrich_scope: scopeSel.value,
      theme: "dark",
    };
    if (mode === "knockout") {
      payload.mi_name = miSel.value;
      payload.keep_pct = parseFloat(keepPctInput.value) || 0;
      payload.sender_types = selectedValues(senderTypesSel);
      payload.receiver_types = selectedValues(receiverTypesSel);
      payload.lr_rows = LR_ROWS;
      payload.sender_rows = SENDER_ROWS;
      payload.receiver_rows = RECEIVER_ROWS;
      payload.lr_selected = [...LR_SELECTED];
      payload.sender_selected = [...SENDER_SELECTED];
      payload.receiver_selected = [...RECEIVER_SELECTED];
    } else {
      payload.replacement_celltypes = selectedValues(replacementTypesSel);
      payload.replaced_celltypes = selectedValues(replacedTypesSel);
      payload.random_seed = parseInt(seedInput.value, 10) || 0;
    }
    return payload;
  }

  function renderDETable(records) {
    if (!records || records.length === 0) {
      deTbody.innerHTML = "<tr><td colspan='6' class='text-muted small'>No genes pass thresholds.</td></tr>";
      return;
    }
    deTbody.innerHTML = records.map((r) => `
      <tr>
        <td class="small font-monospace">${r.gene}</td>
        <td class="text-end small">${(+r.mean_before).toExponential(2)}</td>
        <td class="text-end small">${(+r.mean_after).toExponential(2)}</td>
        <td class="text-end small">${(+r.log2FC).toFixed(3)}</td>
        <td class="text-end small">${(+r.padj).toExponential(2)}</td>
        <td class="small">${r.perturbation_role}</td>
      </tr>`).join("");
  }

  function renderSummary(summary) {
    if (!summary) { summaryDiv.textContent = ""; return; }
    if (summary.status === "error") {
      summaryDiv.innerHTML = `<div class="text-danger small">Run failed: ${summary.error || "unknown error"}</div>`;
      return;
    }
    const rows = [];
    if (summary.mode === "knockout") {
      rows.push(`<div>Mode: <b>knockout</b></div>`);
      rows.push(`<div>MI: ${summary.mi_name || "—"}</div>`);
      rows.push(`<div>Keep expression at: ${summary.keep_pct ?? "—"}%</div>`);
      rows.push(`<div>Sender types: ${(summary.sender_types || []).join(", ") || "All"}</div>`);
      rows.push(`<div>Receiver types: ${(summary.receiver_types || []).join(", ") || "All"}</div>`);
      rows.push(`<div>Sender genes (n=${summary.n_sender_genes ?? 0}): ${(summary.sender_gene_names || []).join(", ") || "—"}</div>`);
      rows.push(`<div>Receiver genes (n=${summary.n_receiver_genes ?? 0}): ${(summary.receiver_gene_names || []).join(", ") || "—"}</div>`);
    } else if (summary.mode === "cell_replacement") {
      rows.push(`<div>Mode: <b>cell-type replacement</b></div>`);
      rows.push(`<div>Replacement types: ${(summary.replacement_celltypes || []).join(", ")}</div>`);
      rows.push(`<div>Replaced types: ${(summary.replaced_celltypes || []).join(", ")}</div>`);
      rows.push(`<div>Cells replaced: ${(summary.cells_replaced ?? 0).toLocaleString()}</div>`);
    }
    rows.push(`<div>Affected slices: ${(summary.affected_slices || []).length}</div>`);
    rows.push(`<div>Comparison cell types: ${(summary.target_celltypes || []).join(", ")}</div>`);
    rows.push(`<div>Cells used for DE: ${(summary.n_target_cells_used ?? 0).toLocaleString()} / ${(summary.n_target_cells_total ?? 0).toLocaleString()}${summary.subsampled_target_cells ? " (subsampled)" : ""}</div>`);
    rows.push(`<div>Genes shown at thresholds: ${summary.n_display_genes ?? 0}</div>`);
    rows.push(`<div>Genes used for enrichment: ${summary.n_enrichment_genes ?? 0}</div>`);
    rows.push(`<div>|log2FC| ≥ ${(summary.lfc_threshold ?? 0).toFixed(2)}, adj P &lt; ${(summary.padj_threshold ?? 0).toFixed(3)}</div>`);
    rows.push(`<hr class="my-2">`);
    if (summary.elapsed_seconds !== undefined) rows.push(`<div>Elapsed: ${summary.elapsed_seconds}s</div>`);
    if (summary.csv_path) rows.push(`<div class="font-monospace text-truncate" title="${summary.csv_path}">CSV: ${summary.csv_path.split("/").pop()}</div>`);
    summaryDiv.innerHTML = rows.join("");
  }

  function applyRunResponse(data) {
    if (!data) return;
    CURRENT_CACHE_KEY = data.cache_key || null;
    renderSummary(data.summary);
    renderFig(volcanoDiv, data.volcano_figure);
    renderFig(goDiv, data.go_figure);
    renderFig(keggDiv, data.kegg_figure);
    renderDETable(data.de_records || []);
  }

  async function runPerturbation() {
    runBtn.disabled = true;
    volcanoSpinner.classList.add("active");
    try {
      const data = await postJson("/api/run", buildRunPayload());
      applyRunResponse(data);
    } catch (e) {
      renderSummary({ status: "error", error: e.message });
    } finally {
      volcanoSpinner.classList.remove("active");
      runBtn.disabled = false;
    }
  }

  async function refreshFromCache() {
    if (!CURRENT_CACHE_KEY) return;
    try {
      const data = await postJson("/api/refresh", {
        cache_key: CURRENT_CACHE_KEY,
        lfc_threshold: parseFloat(lfcInput.value) || 0,
        padj_threshold: parseFloat(padjInput.value) || 0,
        enrich_direction: directionSel.value,
        enrich_scope: scopeSel.value,
        theme: "dark",
      });
      applyRunResponse(data);
    } catch (e) {
      console.warn("M4 refresh failed", e);
    }
  }

  function scheduleRefresh() {
    if (!CURRENT_CACHE_KEY) return;
    clearTimeout(REFRESH_TIMER);
    REFRESH_TIMER = setTimeout(refreshFromCache, 250);
  }

  // -- bootstrap ---------------------------------------------------------
  function applyCelltypeOptions(types) {
    CELLTYPES = types || [];
    fillMultiSelect(senderTypesSel, CELLTYPES);
    fillMultiSelect(receiverTypesSel, CELLTYPES);
    fillMultiSelect(replacementTypesSel, CELLTYPES, CELLTYPES.slice(0, 1));
    fillMultiSelect(replacedTypesSel, CELLTYPES, CELLTYPES.slice(1, 2));
    fillMultiSelect(targetTypesSel, CELLTYPES, CELLTYPES.slice(0, 1));
  }

  async function bootstrap() {
    // Load LR/sender/receiver feature tables right away (no model needed).
    await loadFeatureTables();

    // Check whether heavy state is already loaded.
    let info;
    try { info = await getJson("/api/state-info"); }
    catch { info = { loaded: false, available_celltypes: [] }; }

    if (info.loaded) {
      applyCelltypeOptions(info.available_celltypes);
      runBtn.disabled = false;
      runHint.textContent = `Model + baseline cache ready · ${info.n_slices} slices · device=${info.device}`;
      return;
    }

    // Lazy-load the model + baseline.
    runBtn.disabled = true;
    runHint.textContent = "Loading trained model & baseline predictions… (one-time, may take a minute)";
    try {
      const prepared = await postJson("/api/prepare", {});
      applyCelltypeOptions(prepared.available_celltypes);
      runBtn.disabled = false;
      runHint.textContent = `Model + baseline ready in ${prepared.elapsed_seconds}s · ${prepared.n_slices} slices · device=${prepared.device}`;
    } catch (e) {
      runHint.textContent = `Failed to load model: ${e.message}`;
    }
  }

  // -- listeners ---------------------------------------------------------
  modeKO.addEventListener("change", () => setMode("knockout"));
  modeRP.addEventListener("change", () => setMode("replacement"));

  miSel.addEventListener("change", loadFeatureTables);
  topnInput.addEventListener("change", loadFeatureTables);

  runBtn.addEventListener("click", runPerturbation);

  [lfcInput, padjInput, directionSel, scopeSel].forEach((el) => {
    el.addEventListener("change", scheduleRefresh);
  });

  // -- go ----------------------------------------------------------------
  bootstrap();
})();
