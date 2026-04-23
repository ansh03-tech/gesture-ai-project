/**
 * GestureScript V1.5 — script.js
 * All frontend logic: camera, prediction loop, training, profiles.
 */

const API = "https://gesture-ai-project.onrender.com"; // change if hosted elsewhere

// ── Page navigation ───────────────────────────────────────────────────────
document.querySelectorAll(".nav-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`page-${btn.dataset.page}`).classList.add("active");
  });
});

// ── Backend status check ──────────────────────────────────────────────────
const statusDot  = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");

async function checkBackend() {
  try {
    const res  = await fetch(`${API}/status`);
    const data = await res.json();
    statusDot.className = "online";
    statusText.textContent = `Online · ${data.active_profile}`;
    updateModelStatusBox(data);
    populateProfileSwitcher(data.active_profile);
    updateMappingTable(data.profile_mapping);
  } catch {
    statusDot.className = "offline";
    statusText.textContent = "Backend offline";
  }
}
checkBackend();
setInterval(checkBackend, 8000);

function updateModelStatusBox(data) {
  const box = document.getElementById("model-status-box");
  if (data.model_trained) {
    box.textContent = `✅ Model trained\nGestures: ${data.gestures.join(", ")}`;
  } else {
    box.textContent = "⚠️  No model yet — go to Train tab.";
  }
}

function updateMappingTable(mapping) {
  const body = document.getElementById("mapping-body");
  body.innerHTML = "";
  for (const [g, a] of Object.entries(mapping || {})) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td style="color:var(--cyan)">${g}</td><td style="color:var(--amber)">${a}</td>`;
    body.appendChild(tr);
  }
}

// ── Live camera ───────────────────────────────────────────────────────────
let liveStream   = null;
let predicting   = false;
let holdProgress = 0;   // 0–100
const videoEl    = document.getElementById("video");

document.getElementById("btn-start-cam").addEventListener("click", startLiveCam);
document.getElementById("btn-stop-cam").addEventListener("click", stopLiveCam);

async function startLiveCam() {
  try {
    liveStream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoEl.srcObject = liveStream;
    predicting = true;
    document.getElementById("btn-start-cam").classList.add("hidden");
    document.getElementById("btn-stop-cam").classList.remove("hidden");
    predictLoop();
  } catch (e) {
    showToast("Camera access denied: " + e.message);
  }
}

function stopLiveCam() {
  predicting = false;
  liveStream?.getTracks().forEach(t => t.stop());
  liveStream = null;
  videoEl.srcObject = null;
  document.getElementById("btn-start-cam").classList.remove("hidden");
  document.getElementById("btn-stop-cam").classList.add("hidden");
}

// Grab a single frame from a video element as base64 JPEG
function captureFrame(vid) {
  const canvas = document.createElement("canvas");
  canvas.width  = 320;
  canvas.height = 240;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(vid, 0, 0, 320, 240);
  return canvas.toDataURL("image/jpeg", 0.7).split(",")[1];
}

// ── Prediction loop (runs ~15 fps to keep server load low) ────────────────
async function predictLoop() {
  if (!predicting) return;

  try {
    const frame = captureFrame(videoEl);
    const res   = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame }),
    });
    const d = await res.json();
    updateHUD(d);
  } catch { /* backend might be restarting */ }

  setTimeout(predictLoop, 66); // ~15 fps
}

function updateHUD(d) {
  // Hand detection indicator
  const noHand = document.getElementById("no-hand-msg");
  if (!d.hand_detected) {
    noHand.classList.remove("hidden");
    holdProgress = 0;
  } else {
    noHand.classList.add("hidden");
  }

  // Gesture + confidence
  document.getElementById("hud-gesture").textContent = d.gesture || "—";
  document.getElementById("hud-conf").textContent     = `${d.confidence}%`;
  document.getElementById("hud-profile").textContent  = d.profile || "—";
  document.getElementById("hud-action").textContent   = d.last_action || "—";

  // Confidence bar colour
  const confBar = document.getElementById("conf-bar");
  confBar.style.width = `${d.confidence}%`;
  confBar.style.background =
    d.confidence >= 85 ? "var(--green)" :
    d.confidence >= 60 ? "var(--amber)" : "var(--red)";

  // Hold timer bar
  if (d.hand_detected && d.gesture) {
    holdProgress = d.stable ? 100 : Math.min(holdProgress + 7, 95);
  } else {
    holdProgress = 0;
  }
  document.getElementById("hold-bar").style.width = `${holdProgress}%`;

  // Toast when action fires
  if (d.action) {
    showToast(`⚡ ${d.last_action}`);
    holdProgress = 0;
  }
}

// ── Profile switcher in live panel ───────────────────────────────────────
async function populateProfileSwitcher(active) {
  try {
    const res  = await fetch(`${API}/profiles`);
    const data = await res.json();
    const sel  = document.getElementById("profile-switcher");
    sel.innerHTML = data.profiles.map(p =>
      `<option value="${p}" ${p === active ? "selected" : ""}>${p}</option>`
    ).join("");
  } catch {}
}

document.getElementById("btn-switch-profile").addEventListener("click", async () => {
  const name = document.getElementById("profile-switcher").value;
  const res  = await fetch(`${API}/switch_profile/${name}`, { method: "POST" });
  const data = await res.json();
  updateMappingTable(data.mapping);
  showToast(`Switched to: ${name}`);
  document.getElementById("hud-profile").textContent = name;
});

// ── Training: camera ──────────────────────────────────────────────────────
let trainStream    = null;
let recording      = false;
let recordInterval = null;
let sampleCount    = 0;
const TARGET_SAMPLES = 60;
const trainVideo   = document.getElementById("train-video");

document.getElementById("btn-train-cam").addEventListener("click", async () => {
  trainStream = await navigator.mediaDevices.getUserMedia({ video: true });
  trainVideo.srcObject = trainStream;
  document.getElementById("btn-record-start").disabled = false;
  document.getElementById("btn-train-cam").disabled = true;
});

document.getElementById("btn-record-start").addEventListener("click", () => {
  const name = document.getElementById("gesture-name").value.trim();
  if (!name) { showToast("Enter a gesture name first!"); return; }
  recording   = true;
  sampleCount = 0;
  document.getElementById("btn-record-start").classList.add("hidden");
  document.getElementById("btn-record-stop").classList.remove("hidden");

  recordInterval = setInterval(async () => {
    if (!recording) return;
    const frame = captureFrame(trainVideo);
    const res   = await fetch(`${API}/collect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame, label: name }),
    });
    const d = await res.json();
    if (d.saved) {
      sampleCount++;
      const pct = Math.min((sampleCount / TARGET_SAMPLES) * 100, 100);
      document.getElementById("record-bar").style.width = `${pct}%`;
      document.getElementById("record-count").textContent = `${sampleCount} frames`;
      if (sampleCount >= TARGET_SAMPLES) stopRecording();
    }
  }, 100); // 10 fps collection
});

document.getElementById("btn-record-stop").addEventListener("click", stopRecording);

function stopRecording() {
  recording = false;
  clearInterval(recordInterval);
  document.getElementById("btn-record-start").classList.remove("hidden");
  document.getElementById("btn-record-stop").classList.add("hidden");
  refreshGestures();
  showToast(`Saved ${sampleCount} samples ✔`);
}

// ── Training: model ───────────────────────────────────────────────────────
document.getElementById("btn-train").addEventListener("click", async () => {
  const box = document.getElementById("train-result");
  box.classList.remove("hidden", "success", "error");
  box.textContent = "Training… please wait…";

  try {
    const res  = await fetch(`${API}/train`, { method: "POST" });
    const data = await res.json();
    if (data.success) {
      box.classList.add("success");
      box.textContent =
        `✅ Training complete!\n` +
        `Gestures: ${data.labels.join(", ")}\n` +
        `Samples: ${data.n_samples}\n` +
        `Train accuracy: ${data.train_accuracy}%\n` +
        `CV accuracy: ${data.cv_accuracy_mean}% ± ${data.cv_accuracy_std}%\n` +
        (data.overfit_warning ? `\n${data.overfit_warning}` : "");
    } else {
      box.classList.add("error");
      box.textContent = `❌ ${data.error}`;
    }
  } catch {
    box.classList.add("error");
    box.textContent = "❌ Could not reach backend.";
  }
  checkBackend();
});

async function refreshGestures() {
  try {
    const res  = await fetch(`${API}/gestures`);
    const data = await res.json();
    const ul   = document.getElementById("collected-list");
    if (data.gestures.length === 0) {
      ul.innerHTML = `<li class="muted">None yet</li>`;
    } else {
      ul.innerHTML = data.gestures.map(g => `<li>✋ ${g}</li>`).join("");
    }
  } catch {}
}
document.getElementById("btn-refresh-gestures").addEventListener("click", refreshGestures);
refreshGestures();

// ── Profiles page ─────────────────────────────────────────────────────────
async function loadProfilesPage() {
  const res  = await fetch(`${API}/profiles`);
  const data = await res.json();
  const list = document.getElementById("profile-list");
  list.innerHTML = data.profiles.map(p =>
    `<span class="profile-chip" data-name="${p}">${p}</span>`
  ).join("");
  list.querySelectorAll(".profile-chip").forEach(chip => {
    chip.addEventListener("click", () => loadProfileForEdit(chip.dataset.name));
  });
}

async function loadProfileForEdit(name) {
  document.getElementById("new-profile-name").value = name;
  const res  = await fetch(`${API}/profile/${name}`);
  const data = await res.json();
  const rows = document.getElementById("gesture-action-rows");
  rows.innerHTML = "";
  for (const [g, a] of Object.entries(data)) addGestureActionRow(g, a);
}

function addGestureActionRow(gesture = "", action = "") {
  const actions = [
    "scroll_up","scroll_down","screenshot","volume_up","volume_down",
    "play_pause","next_tab","prev_tab","zoom_in","zoom_out","none"
  ];
  const div = document.createElement("div");
  div.className = "gesture-action-row";
  div.innerHTML = `
    <input type="text" class="inp row-gesture" placeholder="gesture" value="${gesture}"/>
    <select class="row-action">
      ${actions.map(a => `<option value="${a}" ${a === action ? "selected" : ""}>${a}</option>`).join("")}
    </select>
    <button onclick="this.parentElement.remove()">✕</button>
  `;
  document.getElementById("gesture-action-rows").appendChild(div);
}

document.getElementById("btn-add-row").addEventListener("click", () => addGestureActionRow());

document.getElementById("btn-save-profile").addEventListener("click", async () => {
  const name = document.getElementById("new-profile-name").value.trim();
  if (!name) { showToast("Enter a profile name"); return; }
  const mapping = {};
  document.querySelectorAll(".gesture-action-row").forEach(row => {
    const g = row.querySelector(".row-gesture").value.trim();
    const a = row.querySelector(".row-action").value;
    if (g) mapping[g] = a;
  });
  await fetch(`${API}/profile/${name}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(mapping),
  });
  showToast(`Profile "${name}" saved ✔`);
  loadProfilesPage();
});

document.getElementById("btn-export-profile").addEventListener("click", async () => {
  const name = document.getElementById("new-profile-name").value.trim() || "default";
  const res  = await fetch(`${API}/profile/${name}`);
  const data = await res.json();
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url; a.download = `${name}.json`; a.click();
});

document.getElementById("btn-import-profile").addEventListener("click", async () => {
  const json = document.getElementById("import-json").value.trim();
  const name = document.getElementById("import-name").value.trim() || "imported";
  const box  = document.getElementById("import-result");
  try {
    JSON.parse(json); // validate
    await fetch(`${API}/profile/${name}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: json,
    });
    box.classList.remove("hidden", "error");
    box.classList.add("success");
    box.textContent = `✅ Profile "${name}" imported!`;
    loadProfilesPage();
  } catch {
    box.classList.remove("hidden", "success");
    box.classList.add("error");
    box.textContent = "❌ Invalid JSON";
  }
});

// Load profiles page when tab is opened
document.querySelector('[data-page="profiles"]').addEventListener("click", loadProfilesPage);

// ── Toast helper ─────────────────────────────────────────────────────────
function showToast(msg) {
  const toast = document.getElementById("toast");
  toast.textContent = msg;
  toast.classList.remove("hidden");
  toast.classList.add("show");
  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => toast.classList.add("hidden"), 350);
  }, 2500);
}
