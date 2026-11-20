// SQLocal main thread: UI, DuckDB-WASM (which runs in its own worker), and the
// generate -> check -> run -> repair loop. The model runs in model-worker.js.
import * as duckdb from "https://cdn.jsdelivr.net/npm/@duckdb/duckdb-wasm@1.32.0/+esm";
import {
  MAX_REPAIRS, buildMessages, buildRepairMessages, buildSchema, cleanSql, quoteIdent, readOnlyError, safeColumnName,
  sqliteCompat, tableNameFromFile, tidyError,
} from "./prompt.js";
import {
  chartSpec, explainSql, filteredNumberColumns, filteredTextColumns, looksEmpty, readableName, replaceTextLiteral, textEqualities,
} from "./insights.js";
import { renderChart } from "./chart.js";

// bytes per dtype: q4f16 runs on WebGPU; q4 (fp32 activations) is the CPU/WASM
// fallback, exported for the 1.5B model only (3B on a CPU is too slow to use).
const MODELS = [
  { id: "Tehma/text2sql-qwen2.5-coder-1.5b-onnx", name: "Standard: 1.5B", license: "Apache 2.0", bytes: { q4f16: 877e6, q4: 982e6 } },
  { id: "Tehma/text2sql-qwen2.5-coder-3b-onnx", name: "High accuracy: 3B", license: "non-commercial license", bytes: { q4f16: 1745e6 } },
];
const MAX_HINT_COLUMNS = 3;
const MAX_HINT_VALUES = 8;
const PREVIEW_ROWS = 5;
const PROMPT_SAMPLE_ROWS = 3;
const MAX_SHOWN_ROWS = 100;
const SAMPLE_QUESTIONS = [
  "What is the total revenue for each region?",
  "Which 3 products sold the most units?",
  "How many orders came from new customers?",
  "What is the average order revenue per category?",
];

const $ = (id) => document.getElementById(id);
const state = {
  modelId: null, worker: null, modelReady: false, nextRequest: 1, pending: new Map(),
  db: null, conn: null, dbWorker: null, table: null, schema: null, labels: new Map(),
  busy: false, lastResult: null,
  backend: { device: "webgpu", dtype: "q4f16" }, // switched to wasm/q4 when WebGPU is missing
  isolated: window.crossOriginIsolated, // true: multi-threaded WASM (coi-sw.js)
};
window.__sqlocal = state; // for the end-to-end test driver

// ---------- small UI helpers ----------
function banner(text, isError = false) {
  const el = $("banner");
  el.hidden = !text;
  el.textContent = text ?? "";
  el.classList.toggle("err", isError);
}
const fmtMB = (bytes) => `${Math.round(bytes / 1e6)} MB`;
function setPill(text, state) {
  const pill = $("model-pill");
  pill.textContent = text;
  pill.dataset.state = state;
}
const fmtSec = (ms) => `${(ms / 1000).toFixed(1)} s`;
function storedGet(key) { try { return localStorage.getItem(key); } catch { return null; } }
function storedSet(key, value) { try { localStorage.setItem(key, value); } catch { /* storage unavailable */ } }

function updateAskEnabled() {
  const ready = state.modelReady && state.table && !state.busy;
  $("ask").disabled = !ready;
  $("run").disabled = !state.table || state.busy;
  $("ask-hint").hidden = state.modelReady && !!state.table;
}

// ---------- values ----------
function isDateType(type) { return /^Date|^Timestamp/.test(String(type)); }
function formatValue(v, type) {
  if (v === null || v === undefined) return null;
  if (isDateType(type)) {
    const ms = v instanceof Date ? v.getTime() : Number(v);
    const iso = new Date(ms).toISOString();
    return ms % 86400000 === 0 ? iso.slice(0, 10) : iso.slice(0, 19).replace("T", " ");
  }
  if (typeof v === "bigint") return v.toString();
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(Math.round(v * 1e4) / 1e4);
  if (typeof v === "object") return JSON.stringify(v, (_, x) => (typeof x === "bigint" ? x.toString() : x));
  return String(v);
}

/** Arrow table -> {columns, rows (formatted strings or null), numeric flags}. */
function tableToRows(table, limit = Infinity) {
  const fields = table.schema.fields;
  const vectors = fields.map((_, i) => table.getChildAt(i));
  const n = Math.min(table.numRows, limit);
  const rows = [];
  for (let r = 0; r < n; r++) rows.push(fields.map((f, i) => formatValue(vectors[i].get(r), f.type)));
  const numeric = fields.map((f) => /^(Int|Uint|Float|Decimal)/.test(String(f.type)));
  return { columns: fields.map((f) => f.name), rows, numeric, total: table.numRows };
}

function renderTable(container, { columns, rows, numeric }) {
  const tableEl = document.createElement("table");
  const head = tableEl.createTHead().insertRow();
  columns.forEach((c, i) => {
    const th = document.createElement("th");
    th.textContent = state.labels.get(c) ?? c; // original CSV header for renamed columns
    if (state.labels.has(c)) th.title = `${c} in SQL`;
    if (numeric[i]) th.className = "num"; // header lines up with the right-aligned numbers
    head.appendChild(th);
  });
  const body = tableEl.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    row.forEach((v, i) => {
      const td = tr.insertCell();
      if (v === null) { td.textContent = "NULL"; td.className = "null"; }
      else { td.textContent = v; if (numeric[i]) td.className = "num"; }
    });
  }
  container.replaceChildren(tableEl);
}

// ---------- model ----------
function modelInfo(id) { return MODELS.find((m) => m.id === id); }
const dataFileUrl = (id) => `https://huggingface.co/${id}/resolve/main/onnx/model_${state.backend.dtype}.onnx_data`;
const modelBytes = (info) => info.bytes[state.backend.dtype];

async function isModelCached(id) {
  try {
    const cache = await caches.open("transformers-cache");
    return !!(await cache.match(dataFileUrl(id)));
  } catch {
    return false;
  }
}

async function checkWebGPU() {
  if (!navigator.gpu) return "This browser has no WebGPU. SQLocal needs Chrome or Edge on a desktop computer.";
  const adapter = await navigator.gpu.requestAdapter().catch(() => null);
  if (!adapter) return "WebGPU is turned off or not supported by this graphics card, so the model cannot run here.";
  if (!adapter.features.has("shader-f16")) return "This graphics card does not support 16-bit WebGPU shaders (shader-f16).";
  if (new URLSearchParams(location.search).get("device") === "wasm") return "CPU mode was requested (?device=wasm).";
  return null;
}

async function warnIfStorageTight(bytes) {
  try {
    await navigator.storage?.persist?.();
    const { quota = Infinity, usage = 0 } = (await navigator.storage?.estimate?.()) ?? {};
    if (quota - usage < bytes * 1.1) {
      banner(`This browser only allows about ${fmtMB(quota - usage)} of storage for this site (private windows allow much less), ` +
             `so the model cannot be saved and will be downloaded again on your next visit.`);
    }
  } catch { /* estimate() unavailable: nothing to warn about */ }
}

function request(type, payload) {
  const id = state.nextRequest++;
  return new Promise((resolve, reject) => {
    state.pending.set(id, { resolve, reject });
    state.worker.postMessage({ type, id, ...payload });
  });
}

function startWorker() {
  state.worker?.terminate();
  for (const p of state.pending.values()) p.reject(new Error("The model was switched."));
  state.pending.clear();
  state.modelReady = false;
  const worker = new Worker(new URL("./model-worker.js", import.meta.url), { type: "module" });
  const files = new Map();
  worker.onmessage = ({ data }) => {
    if (data.type === "progress") {
      files.set(data.file, data);
      let loaded = 0, total = 0;
      for (const f of files.values()) { loaded += f.loaded ?? 0; total += f.total ?? 0; }
      const bar = $("model-progress");
      bar.hidden = false;
      bar.max = total || 1;
      bar.value = loaded;
      $("model-status").textContent = `Downloading the model: ${fmtMB(loaded)} of ${fmtMB(total)}...`;
    } else if (data.type === "ready") {
      state.modelReady = true;
      setPill(`Model ready · ${state.backend.device === "wasm" ? "CPU" : "WebGPU"}`, "ready");
      $("model-progress").hidden = true;
      const where = state.backend.device === "wasm" ? " on the CPU" : "";
      $("model-status").textContent = `Model ready${where} (loaded in ${fmtSec(data.loadMs)}). It is saved in your browser for next time.`;
      $("model-load").hidden = true;
      updateAskEnabled();
    } else if (data.type === "result" || data.type === "token") {
      if (data.type === "result") {
        state.pending.get(data.id)?.resolve(data);
        state.pending.delete(data.id);
      }
    } else if (data.type === "error") {
      if (data.id && state.pending.has(data.id)) {
        state.pending.get(data.id).reject(new Error(data.message));
        state.pending.delete(data.id);
      } else {
        $("model-progress").hidden = true;
        $("model-status").textContent = "";
        $("model-load").hidden = false;
        $("model-load").disabled = false;
        banner(`The model could not be loaded: ${data.message}`, true);
        setPill("Model failed to load", "error");
      }
    }
  };
  state.worker = worker;
}

async function loadModel() {
  const info = modelInfo($("model-select").value);
  state.modelId = info.id;
  storedSet("sqlocal.model", info.id);
  $("model-load").disabled = true;
  banner(null);
  const cached = await isModelCached(info.id);
  if (!cached) await warnIfStorageTight(modelBytes(info));
  $("model-status").textContent = cached ? "Loading the model from your browser's storage..." : "Starting the download...";
  setPill(cached ? "Loading model" : "Downloading model", "loading");
  startWorker();
  state.worker.postMessage({ type: "load", modelId: info.id, ...state.backend });
  updateAskEnabled();
}

async function refreshModelChoice() {
  const info = modelInfo($("model-select").value);
  state.worker?.terminate();
  state.worker = null;
  state.modelReady = false;
  $("model-progress").hidden = true;
  $("model-load").hidden = false;
  $("model-load").disabled = false;
  updateAskEnabled();
  if (await isModelCached(info.id)) {
    $("model-load").textContent = "Load model";
    await loadModel(); // already downloaded: no need to ask
  } else {
    $("model-load").textContent = `Download model (${fmtMB(modelBytes(info))})`;
    $("model-status").textContent = `One-time download of ${fmtMB(modelBytes(info))}; after that the model loads from your browser's storage.`;
    setPill("Model not downloaded", "idle");
  }
}

// ---------- data ----------
async function freshDatabase() {
  // A new database per file: external access is switched off after the file
  // is read, and DuckDB cannot switch it back on.
  try { await state.conn?.close(); await state.db?.terminate(); } catch { /* already gone */ }
  state.dbWorker?.terminate();
  const bundle = await duckdb.selectBundle(duckdb.getJsDelivrBundles());
  const workerUrl = URL.createObjectURL(new Blob([`importScripts("${bundle.mainWorker}");`], { type: "text/javascript" }));
  const worker = new Worker(workerUrl);
  const db = new duckdb.AsyncDuckDB(new duckdb.VoidLogger(), worker);
  await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
  URL.revokeObjectURL(workerUrl);
  await db.open({ query: { castBigIntToDouble: true, castDecimalToDouble: true, castTimestampToDate: true } });
  Object.assign(state, { db, conn: await db.connect(), dbWorker: worker });
}

async function loadFile(file) {
  state.table = null;
  updateAskEnabled();
  $("data-status").textContent = `Reading ${file.name}...`;
  $("schema-box").hidden = true;
  $("examples").hidden = true;
  try {
    await freshDatabase();
    const fileName = `upload_${Date.now()}.csv`;
    await state.db.registerFileHandle(fileName, file, duckdb.DuckDBDataProtocol.BROWSER_FILEREADER, true);
    const table = tableNameFromFile(file.name);
    const t = quoteIdent(table);
    // Rename columns the model could not write unquoted (spaces, accents,
    // reserved words); the original headers are still shown in the tables.
    const source = `read_csv_auto('${fileName}')`;
    const raw = tableToRows(await state.conn.query(`DESCRIBE SELECT * FROM ${source}`));
    const used = new Set();
    const columns = raw.rows.map((r) => ({ original: r[0], name: safeColumnName(r[0], used), type: r[1] }));
    const select = columns.map((c) => `${quoteIdent(c.original)} AS ${quoteIdent(c.name)}`).join(", ");
    await state.conn.query(`CREATE TABLE ${t} AS SELECT ${select} FROM ${source}`);
    await state.conn.query("SET enable_external_access = false");
    await state.db.dropFile(fileName);
    state.labels = new Map(columns.filter((c) => c.name !== c.original).map((c) => [c.name, c.original]));
    state.columns = columns;
    const count = (await state.conn.query(`SELECT count(*) FROM ${t}`)).getChildAt(0).get(0);
    const preview = tableToRows(await state.conn.query(`SELECT * FROM ${t} LIMIT ${PREVIEW_ROWS}`));

    state.table = table;
    state.schema = buildSchema(table, columns, preview.rows.slice(0, PROMPT_SAMPLE_ROWS));
    const renamed = state.labels.size ? ` ${state.labels.size} column name${state.labels.size > 1 ? "s were" : " was"} simplified for SQL (see below).` : "";
    $("data-status").textContent = `Loaded table ${table}: ${Number(count).toLocaleString()} rows, ${columns.length} columns.${renamed}`;
    $("columns").replaceChildren(...columns.map((c) => {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = c.name;
      const type = document.createElement("span");
      type.className = "type";
      type.textContent = c.name === c.original ? c.type.toLowerCase() : `${c.type.toLowerCase()} · “${c.original}”`;
      chip.appendChild(type);
      return chip;
    }));
    renderTable($("preview"), preview);
    $("schema-box").hidden = false;
  } catch (e) {
    $("data-status").textContent = "";
    banner(`Could not read ${file.name} as CSV: ${e.message}`, true);
  }
  updateAskEnabled();
}

async function loadSample() {
  const blob = await (await fetch("sample/sales.csv")).blob();
  await loadFile(new File([blob], "sales.csv", { type: "text/csv" }));
  if (!state.table) return;
  $("examples").replaceChildren(...SAMPLE_QUESTIONS.map((q) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "chip";
    b.textContent = q;
    b.addEventListener("click", () => { $("question").value = q; $("ask-form").requestSubmit(); });
    return b;
  }));
  $("examples").hidden = false;
  $("data-status").textContent += " This is made-up sample data: drop your own CSV to replace it.";
}

// ---------- ask / run ----------
/** Runs a checked query and renders it. Returns an error message, or null on success. */
async function runSql(sql) {
  const blocked = readOnlyError(sql);
  if (blocked) return blocked;
  const t0 = performance.now();
  try {
    const result = tableToRows(await state.conn.query(sql));
    state.lastResult = { ...result, ms: performance.now() - t0 };
    return null;
  } catch (e) {
    return tidyError(e?.message ?? e);
  }
}

function showResult(prefix, sql, notes = []) {
  const r = state.lastResult;
  const shown = { ...r, rows: r.rows.slice(0, MAX_SHOWN_ROWS) };
  renderTable($("result"), shown);
  const rowsText = r.total === 0
    ? "0 rows: the query ran, but no row matched."
    : `${r.total.toLocaleString()} row${r.total === 1 ? "" : "s"}${r.total > MAX_SHOWN_ROWS ? ` (showing the first ${MAX_SHOWN_ROWS}; download the CSV for all)` : ""}.`;
  $("result-status").textContent = `${prefix}ran in ${Math.round(r.ms)} ms · ${rowsText}`;
  $("download").hidden = r.total === 0;
  // The rest only helps the user check the result; a bug in it must not hide the result.
  try {
    let explanation = explainSql(sql);
    for (const [safe, original] of state.labels) {
      explanation = explanation?.replace(new RegExp(`\b${safe}\b`, "g"), `“${original}”`) ?? null;
    }
    const lines = [...notes, ...(explanation ? [`In words: ${explanation}`] : [])];
    $("explain").hidden = !lines.length;
    $("explain").textContent = lines.join(" ");
    const spec = r.total <= MAX_SHOWN_ROWS ? chartSpec(r) : null;
    if (spec) {
      const name = (col) => state.labels.get(col) ?? readableName(col).replace(/\b\w+\b/g, (w) => state.labels.get(w) ?? w);
      spec.labelName = name(spec.labelName);
      spec.valueName = name(spec.valueName);
    }
    renderChart($("chart"), spec);
  } catch (e) {
    console.error("result extras failed", e);
  }
  if (looksEmpty(r)) return showValueHints(sql).catch((e) => console.error("value hints failed", e));
}

/**
 * A text literal that is not in the data but matches one value apart from
 * case ('Japan' vs 'japan') is replaced by that value. Returns the new SQL and
 * notes for the user; the SQL box shows what actually ran.
 */
async function fixValueCase(sql) {
  const notes = [];
  const t = quoteIdent(state.table);
  for (const { column, literal } of textEqualities(sql, state.columns)) {
    const c = quoteIdent(column);
    const lit = `'${literal.replaceAll("'", "''")}'`;
    const exact = await state.conn.query(`SELECT 1 FROM ${t} WHERE ${c} = ${lit} LIMIT 1`);
    if (exact.numRows) continue;
    const near = tableToRows(await state.conn.query(`SELECT DISTINCT ${c} FROM ${t} WHERE lower(${c}) = lower(${lit}) LIMIT 2`));
    if (near.rows.length !== 1) continue;
    const actual = near.rows[0][0];
    sql = replaceTextLiteral(sql, column, literal, actual);
    notes.push(`Changed “${literal}” to “${actual}” to match the spelling in ${state.labels.get(column) ?? column}.`);
  }
  return { sql, notes };
}

/** Nothing matched: show real values of the text columns, and the range of the number columns, the query filters on. */
async function showValueHints(sql) {
  const textNames = filteredTextColumns(sql, state.columns).slice(0, MAX_HINT_COLUMNS);
  const numberNames = filteredNumberColumns(sql, state.columns).slice(0, MAX_HINT_COLUMNS);
  if (!textNames.length && !numberNames.length) return;
  const t = quoteIdent(state.table);
  const lines = [];
  for (const name of textNames) {
    const c = quoteIdent(name);
    const top = tableToRows(await state.conn.query(
      `SELECT ${c}, count(*) AS n FROM ${t} WHERE ${c} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT ${MAX_HINT_VALUES}`));
    const distinct = Number((await state.conn.query(`SELECT count(DISTINCT ${c}) FROM ${t}`)).getChildAt(0).get(0));
    const more = distinct > top.rows.length ? `, and ${distinct - top.rows.length} more` : "";
    lines.push(`${state.labels.get(name) ?? name}: ${top.rows.map((row) => `“${row[0]}”`).join(", ")}${more}`);
  }
  for (const name of numberNames) {
    const c = quoteIdent(name);
    const [lo, hi] = tableToRows(await state.conn.query(`SELECT min(${c}), max(${c}) FROM ${t}`)).rows[0];
    lines.push(`${state.labels.get(name) ?? name}: numbers from ${lo} to ${hi}`);
  }
  const box = $("hints");
  box.replaceChildren(document.createTextNode("Nothing matched the filter. The values in the data are:"));
  const list = document.createElement("ul");
  for (const line of lines) { const li = document.createElement("li"); li.textContent = line; list.appendChild(li); }
  box.append(list, document.createTextNode("Check the values in the SQL, or ask again using these values."));
  box.hidden = false;
}

function showError(message) {
  $("error").hidden = false;
  $("error").textContent = message;
  $("result").replaceChildren();
  $("download").hidden = true;
  renderChart($("chart"), null);
  $("explain").hidden = true;
}

function resetResult() {
  $("result-card").hidden = false;
  $("error").hidden = true;
  $("attempts").hidden = true;
  $("attempts").replaceChildren();
  $("result").replaceChildren();
  $("download").hidden = true;
  $("hints").hidden = true;
  $("explain").hidden = true;
  renderChart($("chart"), null);
  state.lastResult = null;
}

function addAttempt(sql, error) {
  const li = document.createElement("li");
  const code = document.createElement("code");
  code.textContent = sql || "(empty)";
  li.append(code, document.createTextNode(` failed: ${error}`));
  $("attempts").appendChild(li);
}

async function ask(question) {
  state.busy = true;
  updateAskEnabled();
  resetResult();
  const report = { question, attempts: [] };
  state.lastAsk = report;
  const base = buildMessages(question, state.schema);
  let messages = base;
  try {
    for (let attempt = 0; attempt <= MAX_REPAIRS; attempt++) {
      $("result-status").textContent = attempt ? `The query failed; asking the model to fix it (${attempt} of ${MAX_REPAIRS})...` : "Writing SQL...";
      const gen = await request("generate", { messages });
      const { sql, notes } = await fixValueCase(sqliteCompat(cleanSql(gen.text), state.table, state.columns));
      $("sql").value = sql;
      const error = await runSql(sql);
      report.attempts.push({ sql, error, ttftMs: gen.ttftMs, totalMs: gen.totalMs });
      if (!error) {
        const fixed = attempt ? ` after ${attempt} fix${attempt > 1 ? "es" : ""}` : "";
        await showResult(`SQL written in ${fmtSec(gen.totalMs)}${fixed} · `, sql, notes);
        return;
      }
      addAttempt(sql, error);
      $("attempts").hidden = false;
      if (attempt === MAX_REPAIRS) {
        showError(`The query still fails after ${MAX_REPAIRS} automatic fixes:\n${error}\n\nYou can edit the SQL above and press “Run SQL”.`);
        $("result-status").textContent = "";
        return;
      }
      messages = buildRepairMessages(base, sql, error);
    }
  } catch (e) {
    showError(`The model failed: ${e.message}`);
    $("result-status").textContent = "";
  } finally {
    report.done = true;
    state.busy = false;
    updateAskEnabled();
  }
}

async function runManual() {
  state.busy = true;
  updateAskEnabled();
  resetResult();
  const { sql, notes } = await fixValueCase(sqliteCompat($("sql").value.trim(), state.table, state.columns));
  $("sql").value = sql;
  const error = await runSql(sql);
  if (error) { showError(error); $("result-status").textContent = ""; }
  else await showResult("", sql, notes);
  state.busy = false;
  updateAskEnabled();
}

function downloadCsv() {
  const r = state.lastResult;
  const esc = (v) => (v === null ? "" : /[",\n\r]/.test(v) ? `"${v.replaceAll('"', '""')}"` : v);
  const header = r.columns.map((c) => esc(state.labels.get(c) ?? c)).join(",");
  const text = [header, ...r.rows.map((row) => row.map(esc).join(","))].join("\r\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  a.download = `${state.table}_result.csv`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// ---------- wiring ----------
async function init() {
  const gpuProblem = await checkWebGPU();
  if (gpuProblem) {
    state.backend = { device: "wasm", dtype: "q4" };
    const pace = state.isolated ? "about half a minute" : "one to three minutes (this browser blocked the faster multi-threaded mode)";
    banner(`${gpuProblem} SQLocal will run the model on the CPU instead. That works, but each question takes ${pace} ` +
           "instead of about a second. Chrome or Edge with WebGPU is much faster.");
  }
  const select = $("model-select");
  for (const m of MODELS) {
    const bytes = m.bytes[state.backend.dtype];
    const option = new Option(bytes ? `${m.name}, ${fmtMB(bytes)} (${m.license})` : `${m.name} (needs WebGPU)`, m.id);
    option.disabled = !bytes;
    select.add(option);
  }
  const saved = storedGet("sqlocal.model");
  if (MODELS.some((m) => m.id === saved && m.bytes[state.backend.dtype])) select.value = saved;
  select.addEventListener("change", refreshModelChoice);
  $("model-load").addEventListener("click", loadModel);
  await refreshModelChoice();

  $("file-input").addEventListener("change", (e) => { const f = e.target.files?.[0]; if (f) loadFile(f); e.target.value = ""; });
  const dz = $("dropzone");
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("over"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("over");
    const f = e.dataTransfer.files?.[0];
    if (f) loadFile(f);
  });
  $("sample-load").addEventListener("click", loadSample);
  $("ask-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("question").value.trim();
    if (q && !$("ask").disabled) ask(q);
  });
  $("run").addEventListener("click", runManual);
  $("download").addEventListener("click", downloadCsv);
  updateAskEnabled();
  if (!state.table) await loadSample(); // first view shows a loaded example, clearly marked as sample data
}

init();
