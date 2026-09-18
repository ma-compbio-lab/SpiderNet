// Run from the application directory: node tests/test_prepare_reconnect.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../static/js/m4_perturb.js"), "utf8");

async function scenario(mode) {
  const elements = new Map();
  function makeElement(id = "") {
    return {
      id, dataset: {}, children: [],
      value: "", checked: false, textContent: "", hidden: true,
      classList: { add() {}, remove() {}, toggle() {} }, handlers: {},
      addEventListener(name, callback) { this.handlers[name] = callback; },
      appendChild(child) { this.children.push(child); },
      append(...children) { this.children.push(...children); },
      replaceChildren(...children) { this.children = children; },
      querySelectorAll(selector) {
        const descend = node => node.children.flatMap(child => [child, ...descend(child)]);
        return descend(this).filter(child => child.type === "checkbox" && (!selector.includes(":checked") || child.checked));
      },
      setAttribute(key, value) { this[key] = value; },
      removeAttribute(key) { delete this[key]; },
    };
  }
  function element(id) {
    if (!elements.has(id)) elements.set(id, makeElement(id));
    return elements.get(id);
  }
  let submissions = 0, infoCalls = 0, runs = 0, refreshes = 0;
  const timers = new Set();
  const reply = value => ({ ok: true, json: async () => value });
  const ready = { loaded: true, available_celltypes: ["A", "B"], n_slices: 2, device: "cpu" };
  const context = {
    TypeError, AbortController, performance, console, Plotly: { react() {} },
    document: { getElementById: element, createElement: () => makeElement(),
      documentElement: { getAttribute: () => "light" } },
    window: { M4_BASE_URL: "/dataset/test/perturb", addEventListener() {}, spnRenderPlot() {}, spnValidateInputs: () => true },
    setTimeout(callback, ms) {
      const id = setTimeout(() => { timers.delete(id); callback(); }, ms === 2000 ? 5 : ms);
      timers.add(id); return id;
    },
    clearTimeout(id) { clearTimeout(id); timers.delete(id); },
    async fetch(url, options) {
      if (url.endsWith("/api/run")) {
        runs++;
        return reply({ cache_key: "saved-run", summary: {}, enrichment_status: {
          go: {state: "ok", n_genes: 4}, kegg: {state: "error", n_genes: 4}
        }});
      }
      if (url.endsWith("/api/refresh")) {
        refreshes++;
        assert.equal(JSON.parse(options.body).cache_key, "saved-run");
        return reply({ cache_key: "saved-run", summary: {}, enrichment_status: {
          go: {state: "ok", n_genes: 4}, kegg: {state: "ok", n_genes: 4}
        }});
      }
      if (url.endsWith("/api/state-info")) {
        infoCalls++;
        if (mode === "enrichment-retry") return reply(ready);
        if (infoCalls === 1) return reply({ loaded: false });
        if (mode === "offline") throw new TypeError("Failed to fetch");
        if (mode === "restarted") return reply({ loaded: false });
        if (mode === "still-running" && infoCalls === 2) return reply({ loaded: false });
        return reply(ready);
      }
      if (url.endsWith("/api/prepare-progress")) return reply({
        state: mode === "restarted" ? "idle" : "loading",
        message: "Predicting baseline: slice 1/2", elapsed_seconds: 42,
        seconds_since_update: 31, completed: 0, total: 2,
      });
      if (url.endsWith("/api/prepare")) {
        submissions++;
        if (mode === "model-error") return { ok: false, json: async () => ({ error: "Checkpoint mismatch" }) };
        throw new TypeError("Failed to fetch");
      }
      throw new Error(`Unexpected request: ${url}`);
    },
  };
  try {
    vm.runInNewContext(source, context);
    for (let i = 0; i < 100; i++) {
      await new Promise(resolve => setTimeout(resolve, 5));
      if (element("m4-run").disabled === false || element("m4-prepare-retry").hidden === false) break;
    }
    if (mode === "enrichment-retry") {
      element("m4-mode-replacement").checked = true;
      await element("m4-run").handlers.click();
      assert.equal(element("m4-enrichment-retry").hidden, false);
      assert.match(element("m4-enrichment-status").textContent, /Request failed/);
      await element("m4-enrichment-retry").handlers.click();
      assert.equal(runs, 1, "Enrichment retry must not rerun perturbation");
      assert.equal(refreshes, 1);
      assert.equal(element("m4-enrichment-retry").hidden, true);
      assert.equal(element("m4-enrichment-retry").disabled, false);
      assert.match(element("m4-enrichment-status").textContent, /Results available/);
      console.log(`PASS: ${mode}`);
      return;
    }
    assert.equal(submissions, 1, "Recovery must not resubmit the task");
    assert.equal(element("m4-prepare-progress").hidden, true);
    if (["ready", "still-running"].includes(mode)) {
      assert.equal(element("m4-run").disabled, false);
      assert.match(element("m4-run-hint").textContent, /ready in 42s/);
      assert.equal(element("m4-prepare-retry").hidden, true);
    } else {
      assert.equal(element("m4-prepare-retry").hidden, false);
      const expected = mode === "restarted" ? /no active preparation task/
        : mode === "offline" ? /Connection to the application server was lost/ : /Checkpoint mismatch/;
      assert.match(element("m4-run-hint").textContent, expected);
      if (mode === "model-error") assert.equal(infoCalls, 1);
    }
    console.log(`PASS: ${mode}`);
  } finally {
    timers.forEach(clearTimeout);
  }
}

(async () => {
  for (const mode of ["ready", "still-running", "restarted", "offline", "model-error", "enrichment-retry"]) await scenario(mode);
})().catch(error => { console.error(error); process.exitCode = 1; });
