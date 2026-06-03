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

let seq = 0;          // monotonically increasing request id
let lastRenderedSeq = -1;
let socket = null;
let lastTokens = [];  // last server-confirmed token render, reused while typing
let currentModel = null;   // the globally-loaded model
let adminPassword = null;  // set once the admin password is verified

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

function renderTyping(text) {
  // Instant keystroke feedback that DOESN'T throw away the highlighting:
  // keep the colored tokens for the prefix the last server result already
  // covered (cached values), and render only the changed / not-yet-computed
  // tail as plain text. The tail re-colors when the next result arrives.
  ghost.textContent = "";
  if (!lastTokens.length) {
    highlights.textContent = text;
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
    html += `<span class="pending">${escapeHtml(text.slice(used))}</span>`;
  }
  highlights.innerHTML = html;
}

function renderResult(data) {
  // Build the colored token markup and verify it reconstructs exactly what is
  // currently in the textarea; if not (stale result, race), leave plain text.
  const tokens = data.tokens || [];
  const reconstructed = tokens.map((t) => t.text).join("");
  const current = input.value;

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
  const truncated = data.n_tokens_total > data.max_tokens;
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

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${proto}://${location.host}/ws`);

  socket.onopen = () => setStatus("connected", "ok");
  socket.onclose = () => {
    setStatus("disconnected · retrying…", "err");
    setTimeout(connect, 1500);
  };
  socket.onerror = () => setStatus("connection error", "err");

  socket.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
    if (data.error) {
      setStatus(data.error, "err");
      if (currentModel) modelSelect.value = currentModel; // undo a rejected switch
      return;
    }
    if (data.models) {
      populateModels(data.models, data.current || data.default);
      setCurrentModel(data.current || data.default);
      if (data.max_tokens) maxTokensInput.value = data.max_tokens;
      setStatus("connected", "ok");
      send(); // prime predictions for whatever is already in the box
      return;
    }
    if (data.max_tokens_changed) {
      maxTokensInput.value = data.max_tokens_changed;
      setStatus(`token cap: ${data.max_tokens_changed}`, "ok");
      return;
    }
    if (data.cleared) {
      // An admin cleared all conversations — wipe this editor too.
      input.value = "";
      lastTokens = [];
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
      adminStatus.textContent = unlocked
        ? "unlocked — you can switch the model & token cap"
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

// --- live GPU monitor (replace each frame; it's a gauge, not a log) -----
const gputerm = document.getElementById("gputerm");
const gpustatus = document.getElementById("gpustatus");

function connectGpu() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const gs = new WebSocket(`${proto}://${location.host}/gpu`);
  gs.onopen = () => (gpustatus.textContent = "live");
  gs.onmessage = (ev) => (gputerm.textContent = ev.data); // overwrite each frame
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
