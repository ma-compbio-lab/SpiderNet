// Module 2 — Subtype Discovery client.
(function () {
  "use strict";

  const BASE = window.M2_BASE_URL;
  const $ = (id) => document.getElementById(id);
  function currentTheme() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t === "light" || t === "dark") return t;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  const cellSel = $("m2-cell-type");
  const miThr = $("m2-mi-threshold");
  const miThrVal = $("m2-mi-threshold-value");
  const nNeighbors = $("m2-n-neighbors");
  const nPcs = $("m2-n-pcs");
  const louvainRes = $("m2-louvain-res");
  const umapMinDist = $("m2-umap-min-dist");
  const randomState = $("m2-random-state");
  const aggMode = $("m2-agg-mode");
  const runBtn = $("m2-run");
  const statusEl = $("m2-status");
  const runtimeEl = $("m2-runtime");
  const spinner = $("m2-spinner");
  const summaryTable = $("m2-summary-table");

  const umapDiv = $("m2-umap");
  const sizeDiv = $("m2-cluster-size");
  const senderDiv = $("m2-sender-heatmap");
  const receiverDiv = $("m2-receiver-heatmap");

  // DEG panel
  const degCluster = $("m2-deg-cluster");
  const degDirection = $("m2-deg-direction");
  const degLfc = $("m2-deg-lfc");
  const degPadj = $("m2-deg-padj");
  const degRunBtn = $("m2-deg-run");
  const degStatus = $("m2-deg-status");
  const degSpinner = $("m2-deg-spinner");
  const volcanoDiv = $("m2-volcano");
  const markerTableDiv = $("m2-marker-table");
  const goDiv = $("m2-go-bubble");
  const keggDiv = $("m2-kegg-bubble");

  let CURRENT_CACHE_KEY = null;

  miThr.addEventListener("input", () => {
    miThrVal.textContent = parseFloat(miThr.value).toFixed(2);
  });

  function setStatus(msg) { statusEl.textContent = msg; }
  function setSpinner(on) { spinner.classList.toggle("active", !!on); }

  async function fetchMeta() {
    const r = await fetch(`${BASE}/api/meta`);
    if (!r.ok) throw new Error(`meta ${r.status}`);
    return r.json();
  }

  async function fetchRun(payload) {
    const r = await fetch(`${BASE}/api/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.error || `run ${r.status}`);
    }
    return r.json();
  }

  function renderFig(el, fig, opts) {
    Plotly.react(el, fig.data, Object.assign({ autosize: true }, fig.layout),
      Object.assign({ responsive: true, displaylogo: false, displayModeBar: false }, opts || {}));
  }

  function renderSummary(rows) {
    if (!rows || rows.length === 0) {
      summaryTable.innerHTML = "";
      return;
    }
    const cols = ["cluster", "n_cells", "fraction", "n_samples", "dominant_sample"];
    const head = cols.map((c) => `<th class="text-uppercase small text-muted">${c.replaceAll("_", " ")}</th>`).join("");
    const body = rows.map((r) => {
      return "<tr>" + cols.map((c) => {
        let v = r[c];
        if (c === "fraction" && typeof v === "number") v = (v * 100).toFixed(1) + "%";
        if (typeof v === "number" && c === "n_cells") v = v.toLocaleString();
        return `<td>${v ?? ""}</td>`;
      }).join("") + "</tr>";
    }).join("");
    summaryTable.innerHTML = `<table class="table table-sm table-borderless mb-0"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  }

  async function runAnalysis() {
    setSpinner(true);
    runBtn.disabled = true;
    setStatus("Running...");
    runtimeEl.textContent = "";
    try {
      const t0 = performance.now();
      const data = await fetchRun({
        cell_type: cellSel.value,
        agg_mode: aggMode.value,
        mi_threshold: parseFloat(miThr.value),
        n_neighbors: parseInt(nNeighbors.value, 10),
        n_pcs: parseInt(nPcs.value, 10),
        louvain_resolution: parseFloat(louvainRes.value),
        umap_min_dist: parseFloat(umapMinDist.value),
        random_state: parseInt(randomState.value, 10),
        theme: currentTheme(),
      });
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      const s = data.summary;
      setStatus(
        `${s.cell_type} · ${s.n_cells_used.toLocaleString()} cells · ` +
        `${s.n_clusters} clusters · ${s.n_selected_features} MI features` +
        (s.feature_selection.fallback ? " (fallback: all features)" : "")
      );
      runtimeEl.textContent = `${dt}s`;
      renderFig(umapDiv, data.umap_figure);
      renderFig(sizeDiv, data.cluster_size_figure);
      renderFig(senderDiv, data.sender_heatmap_figure);
      renderFig(receiverDiv, data.receiver_heatmap_figure);
      renderSummary(data.summary_table);
      // Populate DEG cluster dropdown.
      CURRENT_CACHE_KEY = data.cache_key;
      degCluster.innerHTML = "";
      s.cluster_order.forEach((cid) => {
        const opt = document.createElement("option");
        opt.value = cid;
        opt.textContent = `Cluster ${cid}`;
        degCluster.appendChild(opt);
      });
      degRunBtn.disabled = false;
      degStatus.textContent = "Pick a cluster + direction, then run DEG.";
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    } finally {
      setSpinner(false);
      runBtn.disabled = false;
    }
  }

  runBtn.addEventListener("click", runAnalysis);

  // DEG + Enrichment.
  function setDegSpinner(on) { degSpinner.classList.toggle("active", !!on); }

  function renderMarkerTable(rows) {
    if (!rows || rows.length === 0) {
      markerTableDiv.innerHTML = "<div class='text-muted small p-2'>No marker genes at these thresholds.</div>";
      return;
    }
    const cols = ["gene", "logfoldchange", "padj", "score"];
    const head = cols.map((c) => `<th class="text-uppercase small text-muted">${c}</th>`).join("");
    const body = rows.map((r) => {
      return "<tr>" + cols.map((c) => {
        let v = r[c];
        if (typeof v === "number") {
          if (c === "padj") v = v.toExponential(2);
          else v = v.toFixed(3);
        }
        return `<td>${v ?? ""}</td>`;
      }).join("") + "</tr>";
    }).join("");
    markerTableDiv.innerHTML =
      `<table class="table table-sm table-hover table-borderless mb-0"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  }

  async function runDeg() {
    if (!CURRENT_CACHE_KEY) {
      degStatus.textContent = "Run subtype discovery first.";
      return;
    }
    degRunBtn.disabled = true;
    setDegSpinner(true);
    degStatus.textContent = "Computing DEG + Enrichr (first run may take minutes)...";
    try {
      const t0 = performance.now();
      const r = await fetch(`${BASE}/api/deg`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cache_key: CURRENT_CACHE_KEY,
          cluster: degCluster.value,
          direction: degDirection.value,
          lfc_thresh: parseFloat(degLfc.value),
          padj_thresh: parseFloat(degPadj.value),
          theme: currentTheme(),
        }),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.error || `deg ${r.status}`);
      }
      const data = await r.json();
      const dt = ((performance.now() - t0) / 1000).toFixed(1);
      const overview = data.deg_overview.find((o) => o.cluster === data.cluster);
      const counts = overview ? `up=${overview.n_up} · down=${overview.n_down}` : "";
      degStatus.textContent =
        `Cluster ${data.cluster} (${data.direction}) · ${data.expression_summary.n_cells.toLocaleString()} cells · ` +
        `${data.expression_summary.n_genes.toLocaleString()} genes · ${counts} · ${dt}s`;
      renderFig(volcanoDiv, data.volcano_figure);
      renderFig(goDiv, data.go_figure);
      renderFig(keggDiv, data.kegg_figure);
      renderMarkerTable(data.marker_table);
    } catch (err) {
      degStatus.textContent = `Error: ${err.message}`;
    } finally {
      setDegSpinner(false);
      degRunBtn.disabled = false;
    }
  }
  degRunBtn.addEventListener("click", runDeg);

  // Bootstrap: load cell-type list.
  (async function init() {
    try {
      setStatus("Loading metadata (first call may take ~30 s)...");
      const meta = await fetchMeta();
      cellSel.innerHTML = "";
      meta.cell_types.forEach((ct) => {
        const opt = document.createElement("option");
        opt.value = ct.name;
        opt.textContent = `${ct.name} (${ct.n_cells.toLocaleString()} cells)`;
        cellSel.appendChild(opt);
      });
      setStatus(`Ready · ${meta.n_cells.toLocaleString()} cells across ${meta.cell_types.length} cell types · species: ${meta.species || "?"}`);
    } catch (err) {
      setStatus(`Init failed: ${err.message}`);
    }
  })();

  // Re-render server-themed Plotly figures when the user toggles theme.
  window.addEventListener("spn:themechange", () => {
    if (CURRENT_CACHE_KEY) runAnalysis();
  });
})();
