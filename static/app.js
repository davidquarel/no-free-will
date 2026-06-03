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
const topkInput = document.getElementById("topk");
const modelSelect = document.getElementById("model");

let seq = 0;          // monotonically increasing request id
let lastRenderedSeq = -1;
let socket = null;

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

// --- surprisal -> color -------------------------------------------------
// Token color goes green (model nailed it) -> red (you surprised it),
// keyed off bits of surprisal: -log2(prob).
function tokenColor(prob, rank) {
  if (prob == null) return "var(--fg)";
  const bits = -Math.log2(Math.max(prob, 1e-9));
  const t = Math.min(bits / 12, 1); // 0 bits = certain, >=12 bits = shock
  const hue = 120 * (1 - t);        // 120 green -> 0 red
  return `hsl(${hue.toFixed(0)}, 75%, 68%)`;
}

function plainBackdrop(text) {
  // Instant, uncolored render so the caret never floats over invisible text.
  highlights.textContent = text;
  ghost.textContent = "";
}

function renderResult(data) {
  // Build the colored token markup and verify it reconstructs exactly what is
  // currently in the textarea; if not (stale result, race), leave plain text.
  const tokens = data.tokens || [];
  const reconstructed = tokens.map((t) => t.text).join("");
  const current = input.value;

  if (reconstructed === current) {
    highlights.innerHTML = tokens
      .map(
        (t) =>
          `<span class="tok" style="color:${tokenColor(t.prob, t.rank)}">${escapeHtml(
            t.text
          )}</span>`
      )
      .join("");
  } else {
    // Tokenizer round-trip didn't match the raw text exactly; stay safe.
    highlights.textContent = current;
  }

  // Ghost prediction: only when the caret sits at the very end with no
  // selection (otherwise it would appear in the wrong place).
  const atEnd =
    input.selectionStart === current.length &&
    input.selectionEnd === current.length;
  const preds = data.predictions || [];
  ghost.textContent = atEnd && current.length && preds.length ? preds[0].token : "";

  renderPredictions(preds);
  renderAccuracy(tokens);
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
  const scored = tokens.filter((t) => t.rank != null);
  if (!scored.length) {
    accuracyEl.textContent = "—";
    return;
  }
  const hits = scored.filter((t) => t.rank === 0).length;
  accuracyEl.textContent = Math.round((hits / scored.length) * 100) + "%";
}

// --- networking ---------------------------------------------------------
function send() {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  seq += 1;
  socket.send(
    JSON.stringify({
      text: input.value,
      k: Math.max(1, Math.min(40, parseInt(topkInput.value, 10) || 10)),
      model: modelSelect.value,
      seq,
    })
  );
}

function populateModels(models, def) {
  if (modelSelect.options.length) return; // only build once
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.label;
    if (m.id === def) opt.selected = true;
    modelSelect.appendChild(opt);
  }
}

let debounce = null;
function onInput() {
  plainBackdrop(input.value);          // instant feedback
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
      return;
    }
    if (data.models) {
      populateModels(data.models, data.default);
      setStatus("connected", "ok");
      send(); // prime predictions for whatever is already in the box
      return;
    }
    if (data.status === "loading") {
      setStatus(`loading ${data.model}… (first load downloads weights)`, "");
      return;
    }
    // Drop out-of-order/stale responses.
    if (typeof data.seq === "number" && data.seq < lastRenderedSeq) return;
    lastRenderedSeq = data.seq ?? lastRenderedSeq;
    setStatus(`model: ${data.model}`, "ok");
    renderResult(data);
  };
}

// Switching model: re-run on the current text immediately.
modelSelect.addEventListener("change", send);

input.addEventListener("input", onInput);
input.addEventListener("scroll", syncScroll);
// Caret moves (arrows/click) change whether the ghost should show.
input.addEventListener("keyup", () => {
  if (input.selectionStart !== input.value.length) ghost.textContent = "";
});
input.addEventListener("click", () => {
  if (input.selectionStart !== input.value.length) ghost.textContent = "";
});

plainBackdrop(input.value);
connect();
