// Live next-token prediction client.
//
// The textarea has transparent text stacked over a "backdrop" that renders the
// same text as surprisal-colored token spans, followed by a faded "ghost" of
// the model's #1 next-token guess. Keeping the two perfectly aligned is the
// whole trick, so the backdrop is updated to plain text instantly on every
// keystroke and only *upgraded* to colored tokens once the matching server
// result arrives.

const input = document.getElementById("input");
const highlights = document.getElementById("highlights");
const ghost = document.getElementById("ghost");
const backdrop = document.getElementById("backdrop");
const predictionsEl = document.getElementById("predictions");
const accuracyEl = document.getElementById("accuracy");
const statusEl = document.getElementById("status");
const tokenCountEl = document.getElementById("tokencount");
const topkInput = document.getElementById("topk");
const modelSelect = document.getElementById("model");
const modelNow = document.getElementById("modelnow");
const adminPass = document.getElementById("adminpass");
const adminUnlock = document.getElementById("adminunlock");
const adminStatus = document.getElementById("adminstatus");
const maxTokensInput = document.getElementById("maxtokens");
const maxTokensApply = document.getElementById("maxtokensapply");
const viewerToggle = document.getElementById("viewertoggle");
const rebootBtn = document.getElementById("rebootbtn");

let seq = 0;          // monotonically increasing request id
let lastRenderedSeq = -1;
let socket = null;
let lastTokens = [];  // last server-confirmed token render, reused while typing
let truncBoundary = Infinity;  // char index past which text was dropped at the token cap (Infinity = nothing truncated)
let currentModel = null;   // the globally-loaded model
let adminPassword = null;  // set once the admin password is verified
let lastServerId = null;   // detect a server restart across reconnects
let banned = false;        // stop reconnecting if we've been IP-banned
let inactive = false;      // stop reconnecting after an inactivity disconnect (refresh to rejoin)

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = "status" + (cls ? " " + cls : "");
}

// --- surprisal -> highlight --------------------------------------------
// Each *token* gets its own background chip: green (model nailed it) -> red
// (you surprised it), keyed off bits of surprisal: -log2(prob). Highlighting
// the token (not the word) makes subword splits visible, and the alpha grows
// with surprise so unexpected tokens pop. Background-only (no padding/border)
// keeps the backdrop glyph-aligned with the transparent textarea above it.
function tokenBg(prob) {
  if (prob == null) return "transparent";
  const bits = -Math.log2(Math.max(prob, 1e-9));
  const t = Math.min(bits / 12, 1); // 0 bits = certain, >=12 bits = shock
  const hue = 120 * (1 - t);        // 120 green -> 0 red
  const alpha = (0.18 + 0.34 * t).toFixed(2);
  return `hsla(${hue.toFixed(0)}, 85%, 50%, ${alpha})`;
}

function tokenSpan(t) {
  return `<span class="tok" style="background:${tokenBg(t.prob)}">${escapeHtml(
    t.text
  )}</span>`;
}

// Markup for the not-yet-tokenized tail starting at char index `from`: the part
// before the truncation boundary is "pending" (recolors when the next result
// arrives); the part at/after it is "truncated" — text dropped at the token cap
// that the model never saw, painted distinctly (red) so the cut point is clear.
function tailMarkup(text, from) {
  const b = Math.max(from, Math.min(truncBoundary, text.length));
  let html = "";
  if (b > from) html += `<span class="pending">${escapeHtml(text.slice(from, b))}</span>`;
  if (text.length > b) html += `<span class="truncated">${escapeHtml(text.slice(b))}</span>`;
  return html;
}

function renderTyping(text) {
  // Instant keystroke feedback that DOESN'T throw away the highlighting:
  // keep the colored tokens for the prefix the last server result already
  // covered (cached values), and render only the changed / not-yet-computed
  // tail as plain text. The tail re-colors when the next result arrives.
  ghost.textContent = "";
  if (!lastTokens.length) {
    highlights.innerHTML = tailMarkup(text, 0);
    return;
  }
  const cached = lastTokens.map((t) => t.text).join("");
  // longest common character prefix between the cached text and what's typed now
  let cp = 0;
  const max = Math.min(cached.length, text.length);
  while (cp < max && cached[cp] === text[cp]) cp++;
  // reuse only whole cached tokens that fit entirely inside the common prefix
  let used = 0;
  let html = "";
  for (const t of lastTokens) {
    if (used + t.text.length > cp) break;
    html += tokenSpan(t);
    used += t.text.length;
  }
  if (used < text.length) {
    html += tailMarkup(text, used);
  }
  highlights.innerHTML = html;
}

function renderResult(data) {
  // Build the colored token markup and verify it reconstructs exactly what is
  // currently in the textarea; if not (stale result, race), leave plain text.
  const tokens = data.tokens || [];
  const reconstructed = tokens.map((t) => t.text).join("");
  const current = input.value;

  // The server scores only the first max_tokens tokens; when you typed more, the
  // scored tokens reconstruct just a prefix and everything past it was dropped.
  // Record that boundary (in chars) so the tail renders as truncated, not pending.
  const truncated = data.n_tokens_total > data.max_tokens;
  truncBoundary = truncated ? reconstructed.length : Infinity;

  if (reconstructed === current) {
    highlights.innerHTML = tokens.map(tokenSpan).join("");
    lastTokens = tokens;
  } else {
    // Stale result (you typed more since this request) or a tokenizer
    // round-trip mismatch. If it's a clean prefix of the current text, adopt
    // it so more stays highlighted; either way keep the cached highlighting
    // for the matching prefix instead of flashing back to plain text.
    if (current.startsWith(reconstructed)) lastTokens = tokens;
    renderTyping(current);
  }

  // Ghost prediction: only when the caret sits at the very end with no
  // selection (otherwise it would appear in the wrong place).
  const atEnd =
    input.selectionStart === current.length &&
    input.selectionEnd === current.length;
  // When the text was truncated, the prediction is for the cut point, not the
  // real end — don't show a misleading ghost.
  const preds = data.predictions || [];
  ghost.textContent =
    !truncated && atEnd && current.length && preds.length ? preds[0].token : "";

  renderPredictions(preds);
  renderAccuracy(tokens);
  renderTokenCount(data);
}

// Persistent token readout: current / max, flagged when truncated.
function renderTokenCount(data) {
  if (typeof data.n_tokens !== "number" || !data.max_tokens) return;
  const truncated = data.n_tokens_total > data.max_tokens;
  tokenCountEl.textContent =
    `${data.n_tokens_total} / ${data.max_tokens} tokens` + (truncated ? " · truncated" : "");
  tokenCountEl.className = "tokens" + (truncated ? " warn" : "");
}

function renderPredictions(preds) {
  const max = preds.reduce((m, p) => Math.max(m, p.prob), 0) || 1;
  predictionsEl.innerHTML = preds
    .map((p) => {
      const pct = ((p.prob / max) * 100).toFixed(1);
      const label = p.token.replace(/\n/g, "\\n").replace(/ /g, "·");
      return `<li><span class="bar" style="width:${pct}%"></span>` +
        `<span class="tokname">${escapeHtml(label)}</span>` +
        `<span class="prob">${(p.prob * 100).toFixed(1)}%</span></li>`;
    })
    .join("");
}

function renderAccuracy(tokens) {
  // Word-level accuracy: group tokens into words (a new word begins at a token
  // that starts with whitespace, or at the very start), then a word counts as
  // correct only if EVERY scored token in it was the model's #1 guess. A
  // multi-token word has to be guessed right all the way through.
  const words = [];
  let cur = null;
  for (const t of tokens) {
    if (cur === null || /^\s/.test(t.text)) {
      cur = [];
      words.push(cur);
    }
    cur.push(t);
  }
  let total = 0;
  let hits = 0;
  for (const w of words) {
    const hasText = w.some((t) => /\S/.test(t.text));
    const scored = w.filter((t) => t.rank != null);
    if (!hasText || !scored.length) continue; // skip whitespace / unscorable
    total += 1;
    if (scored.every((t) => t.rank === 0)) hits += 1;
  }
  accuracyEl.textContent = total ? Math.round((hits / total) * 100) + "%" : "—";
}

// --- networking ---------------------------------------------------------
// Predict requests no longer carry a model: the server runs whatever single
// model is globally loaded, so all users' requests batch into one forward pass.
function send() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  seq += 1;
  socket.send(
    JSON.stringify({
      text: input.value,
      k: Math.max(1, Math.min(40, parseInt(topkInput.value, 10) || 10)),
      seq,
    })
  );
}

// Changing the dropdown switches the model for EVERYONE (one model in VRAM).
// Gated by the admin password, which the server verifies.
function requestModel(name) {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify({ set_model: name, password: adminPassword }));
}

function setCurrentModel(name) {
  currentModel = name;
  if (modelNow) modelNow.textContent = name;
  if (modelSelect) modelSelect.value = name;
}

function populateModels(models, current) {
  if (modelSelect.options.length) return; // only build once
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.label;
    if (m.id === current) opt.selected = true;
    modelSelect.appendChild(opt);
  }
}

let debounce = null;
function onInput() {
  renderTyping(input.value);           // instant feedback, keeps cached highlight
  syncScroll();
  clearTimeout(debounce);
  debounce = setTimeout(send, 70);     // ~14 req/s ceiling while typing
}

function syncScroll() {
  backdrop.scrollTop = input.scrollTop;
  backdrop.scrollLeft = input.scrollLeft;
}

// Per-browser id (persisted) so an admin can ban THIS browser specifically,
// rather than the whole IP. Cleared storage / incognito gets a fresh id.
function getClientId() {
  try {
    let cid = localStorage.getItem("nfw_cid");
    if (!cid) {
      cid = (window.crypto && crypto.randomUUID && crypto.randomUUID()) ||
            Math.random().toString(36).slice(2);
      localStorage.setItem("nfw_cid", cid);
    }
    return cid;
  } catch {
    return Math.random().toString(36).slice(2);
  }
}
const clientId = getClientId();

function connect() {
  if (banned) return; // don't reconnect once banned
  const proto = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${proto}://${location.host}/ws?cid=${encodeURIComponent(clientId)}`);

  socket.onopen = () => setStatus("connected", "ok");
  socket.onclose = () => {
    if (banned) { setStatus("you have been banned", "err"); return; }
    if (inactive) { setStatus("disconnected for inactivity — refresh the page to rejoin", "err"); return; }
    setStatus("disconnected · retrying…", "err");
    setTimeout(connect, 1500);
  };
  socket.onerror = () => setStatus("connection error", "err");

  socket.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
    if (data.banned) {
      banned = true;
      setStatus("you have been banned", "err");
      input.disabled = true;
      clearEditor();
      return;
    }
    if (data.inactive) {
      // Server dropped us for inactivity. Stop auto-reconnecting and tell the
      // user to refresh; keep their text on screen so they can read/copy it.
      inactive = true;
      setStatus("disconnected for inactivity — refresh the page to rejoin", "err");
      input.disabled = true;
      return;
    }
    if (data.error) {
      setStatus(data.error, "err");
      if (currentModel) modelSelect.value = currentModel; // undo a rejected switch
      return;
    }
    if (data.models) {
      // If the server restarted (new PID) since we last connected, the old text
      // is dead — clear the editor instead of re-predicting it.
      if (lastServerId !== null && data.server_id !== lastServerId) clearEditor();
      lastServerId = data.server_id;
      populateModels(data.models, data.current || data.default);
      setCurrentModel(data.current || data.default);
      if (data.max_tokens) maxTokensInput.value = data.max_tokens;
      if (typeof data.viewer_enabled === "boolean") viewerToggle.checked = data.viewer_enabled;
      setStatus("connected", "ok");
      send(); // prime predictions for whatever is already in the box
      return;
    }
    if (data.max_tokens_changed) {
      maxTokensInput.value = data.max_tokens_changed;
      setStatus(`token cap: ${data.max_tokens_changed}`, "ok");
      return;
    }
    if (typeof data.viewer_enabled === "boolean") {
      // Admin toggled the live conversation viewer (maybe from another panel).
      viewerToggle.checked = data.viewer_enabled;
      setStatus(`live viewer ${data.viewer_enabled ? "enabled" : "disabled"}`, "ok");
      return;
    }
    if (data.cleared) {
      // An admin cleared all conversations — wipe this editor too.
      input.value = "";
      lastTokens = [];
      truncBoundary = Infinity;
      renderTyping("");
      predictionsEl.innerHTML = "";
      accuracyEl.textContent = "—";
      ghost.textContent = "";
      setStatus("conversations cleared", "ok");
      return;
    }
    if (typeof data.admin_ok === "boolean") {
      const unlocked = data.admin_ok;
      adminPassword = unlocked ? adminPass.value : null;
      modelSelect.disabled = !unlocked;
      maxTokensInput.disabled = !unlocked;
      maxTokensApply.disabled = !unlocked;
      viewerToggle.disabled = !unlocked;
      rebootBtn.disabled = !unlocked;
      adminStatus.textContent = unlocked
        ? "unlocked — model, token cap, live viewer & reboot"
        : "wrong password";
      adminStatus.className = "admin-status " + (unlocked ? "ok" : "err");
      return;
    }
    if (data.model_switched) {
      // Someone (maybe another user) switched the global model — sync up.
      setCurrentModel(data.model_switched);
      setStatus(`model: ${data.model_switched}`, "ok");
      send();
      return;
    }
    if (data.status === "loading") {
      setStatus(`loading ${data.model}… (shared by all users; first load downloads weights)`, "");
      return;
    }
    if (data.status === "rebooting") {
      // Admin kicked a server reboot; the socket will drop and auto-reconnect.
      setStatus("server rebooting… reconnecting when it's back", "");
      return;
    }
    // Drop out-of-order/stale responses.
    if (typeof data.seq === "number" && data.seq < lastRenderedSeq) return;
    lastRenderedSeq = data.seq ?? lastRenderedSeq;
    setStatus(`model: ${data.model}`, "ok");
    renderResult(data);
  };
}

// Switching model changes it for every connected user (one model in VRAM).
modelSelect.addEventListener("change", () => requestModel(modelSelect.value));

// Admin unlock: verify the password with the server, then enable the dropdown.
function tryUnlock() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify({ admin_check: adminPass.value }));
}
adminUnlock.addEventListener("click", tryUnlock);
adminPass.addEventListener("keydown", (e) => {
  if (e.key === "Enter") tryUnlock();
});

// Admin: apply a new per-request token cap (affects everyone).
function applyMaxTokens() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  const n = parseInt(maxTokensInput.value, 10);
  if (!n) return;
  socket.send(JSON.stringify({ set_max_tokens: n, password: adminPassword }));
}
maxTokensApply.addEventListener("click", applyMaxTokens);
maxTokensInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") applyMaxTokens();
});

// Admin: enable/disable the live conversation viewer (affects everyone + the
// /viewer.html feed). Sent immediately when the checkbox is flipped.
viewerToggle.addEventListener("change", () => {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  socket.send(
    JSON.stringify({ set_viewer_enabled: viewerToggle.checked, password: adminPassword })
  );
});

// Admin: kill + restart the server process (reloads code + model, drops everyone
// briefly). Guarded by a confirm since it disrupts all connected users.
rebootBtn.addEventListener("click", () => {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  if (!confirm("Reboot the server? This kills and restarts the process — it reloads the code and model, and briefly disconnects everyone while the model reloads.")) return;
  socket.send(JSON.stringify({ reboot: true, password: adminPassword }));
});

input.addEventListener("input", onInput);
input.addEventListener("scroll", syncScroll);
// Caret moves (arrows/click) change whether the ghost should show.
input.addEventListener("keyup", () => {
  if (input.selectionStart !== input.value.length) ghost.textContent = "";
});
input.addEventListener("click", () => {
  if (input.selectionStart !== input.value.length) ghost.textContent = "";
});

// --- live server-log terminal ------------------------------------------
const logterm = document.getElementById("logterm");
const logstatus = document.getElementById("logstatus");
let logText = "";

function appendLog(extra) {
  logText += extra;
  // Collapse \r progress bars (tqdm) to their latest state; cap the backlog.
  const lines = logText.split("\n").map((ln) => {
    ln = ln.replace(/\r+$/, ""); // ignore trailing CRs (mid-update progress)
    const i = ln.lastIndexOf("\r");
    return i >= 0 ? ln.slice(i + 1) : ln; // keep only the latest CR segment
  });
  logText = lines.slice(-600).join("\n");
  const atBottom =
    logterm.scrollHeight - logterm.scrollTop - logterm.clientHeight < 40;
  logterm.textContent = logText;
  if (atBottom) logterm.scrollTop = logterm.scrollHeight; // stick to bottom
}

function connectLogs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ls = new WebSocket(`${proto}://${location.host}/logs`);
  ls.onopen = () => (logstatus.textContent = "streaming app.log");
  ls.onmessage = (ev) => appendLog(ev.data);
  ls.onclose = () => {
    logstatus.textContent = "disconnected · retrying…";
    setTimeout(connectLogs, 1500);
  };
  ls.onerror = () => (logstatus.textContent = "log stream error");
}

// --- live GPU monitor: compact per-GPU readout + a bar chart of util over time.
//     The server streams a JSON snapshot ~1/s; we keep a rolling history of util
//     per GPU and redraw the panel each frame.
const gpuPanel = document.getElementById("gpupanel");
const gpustatus = document.getElementById("gpustatus");
const GPU_BAR_W = 22;                // width of the MEM fill bar, in characters
const GPU_HIST = 32;                 // fixed number of bars in the util time chart
const SPARK = "▁▂▃▄▅▆▇█";            // 8 levels for the util sparkline
const gpuHist = {};                  // gpu index -> last GPU_HIST utils (prefilled 0)

function asciiBar(pct) {
  const v = Math.max(0, Math.min(100, pct || 0));
  const filled = Math.round((v / 100) * GPU_BAR_W);
  return "█".repeat(filled) + "░".repeat(GPU_BAR_W - filled);
}

function sparkline(hist) {
  return hist
    .map((u) => {
      const v = Math.max(0, Math.min(100, u || 0));
      return SPARK[Math.min(SPARK.length - 1, Math.round((v / 100) * (SPARK.length - 1)))];
    })
    .join("");
}

function renderGpu(data) {
  if (!data || data.error) {
    gpuPanel.innerHTML = `<span class="gpu-err">GPU monitor unavailable${data && data.error ? ": " + escapeHtml(data.error) : ""}</span>`;
    return;
  }
  const giB = (x) => (x / 1024).toFixed(1);
  const blocks = [];
  for (const g of data.gpus || []) {
    const util = g.util == null ? 0 : Math.round(g.util);
    const memPct = g.mem_total ? Math.round((g.mem_used / g.mem_total) * 100) : 0;
    const pow = g.power != null ? Math.round(g.power) : "?";
    const powMax = g.power_max != null ? Math.round(g.power_max) : "?";
    const nproc = (g.procs || []).length;

    // Fixed-size util history (prefilled with zeros so the chart is always 32 wide).
    const hist = gpuHist[g.index] || (gpuHist[g.index] = Array(GPU_HIST).fill(0));
    hist.push(util);
    if (hist.length > GPU_HIST) hist.shift();

    // All GPU info on one line.
    const head = `GPU ${g.index} · ${g.name || ""} · ${g.temp == null ? "?" : Math.round(g.temp)}°C · ${pow}/${powMax} W · ${nproc} proc`;
    // MEM: current-usage ASCII fill bar. UTIL: current-usage fill bar PLUS a
    // fixed 32-bar time sparkline of recent utilisation.
    const uHue = Math.round(120 * (1 - util / 100));
    const memBar = `<span class="gbar gbar-mem">${asciiBar(memPct)}</span>`;
    const utilBar = `<span class="gbar" style="color:hsl(${uHue},65%,55%)">${asciiBar(util)}</span>`;
    const utilSpark = `<span class="gbar" style="color:hsl(${uHue},65%,55%)">${sparkline(hist)}</span>`;
    const memLine = `mem  ${memBar} ${String(memPct).padStart(3)}%  ${giB(g.mem_used)}/${giB(g.mem_total)} GiB`;
    const utilLine = `util ${utilBar} ${String(util).padStart(3)}%`;
    const sparkLine = `     ${utilSpark}`; // own line, indented to align under the util bar
    blocks.push(escapeHtml(head) + "\n" + memLine + "\n" + utilLine + "\n" + sparkLine);
  }
  gpuPanel.innerHTML = blocks.join("\n\n") || `<span class="gpu-err">(no GPU found)</span>`;
}

function connectGpu() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const gs = new WebSocket(`${proto}://${location.host}/gpu`);
  gs.onopen = () => (gpustatus.textContent = "live");
  gs.onmessage = (ev) => {
    try {
      renderGpu(JSON.parse(ev.data));
    } catch {
      /* ignore malformed frame */
    }
  };
  gs.onclose = () => {
    gpustatus.textContent = "disconnected · retrying…";
    setTimeout(connectGpu, 2000);
  };
  gs.onerror = () => (gpustatus.textContent = "gpu stream error");
}

// --- "are we running the latest code?" indicator -----------------------
// Green = page + server match disk. Yellow = something changed: a static change
// needs only a page reload; a backend (.py) change needs a server restart.
const reloadBtn = document.getElementById("reloadbtn");
let loadedStatic = null; // disk_static at the moment this page loaded

function setReload(state, label) {
  reloadBtn.className = "reload " + state; // ok | stale
  reloadBtn.textContent = (state === "ok" ? "● " : "● ") + label;
}

async function checkVersion() {
  try {
    const v = await (await fetch("/api/version", { cache: "no-store" })).json();
    if (loadedStatic === null) loadedStatic = v.disk_static;
    const serverStale = v.disk_py !== v.running_py;       // backend changed → restart
    const uiStale = v.disk_static !== loadedStatic;       // static changed → reload
    if (serverStale) setReload("stale", "server changed — restart needed");
    else if (uiStale) setReload("stale", "new UI — click to reload");
    else setReload("ok", "up to date");
  } catch {
    setReload("stale", "offline");
  }
}

reloadBtn.addEventListener("click", () => location.reload());
checkVersion();
setInterval(checkVersion, 4000);

// Start blank on every load — don't let the browser resurrect old text on a
// refresh or back/forward (bfcache). pageshow fires after any restoration.
function clearEditor() {
  input.value = "";
  lastTokens = [];
  truncBoundary = Infinity;
  renderTyping("");
  predictionsEl.innerHTML = "";
  accuracyEl.textContent = "—";
  ghost.textContent = "";
}
window.addEventListener("pageshow", clearEditor);

clearEditor();
connect();
connectLogs();
connectGpu();
