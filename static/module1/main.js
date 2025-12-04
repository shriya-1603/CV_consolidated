// Module 1 — Calibrate & Measure (single-user)
// Endpoints are namespaced under /measure in app.py

const API_BASE = "/measure";

// UI elements
const toggleBtn        = document.getElementById("toggleBtn");
const captureCalibBtn  = document.getElementById("captureCalibBtn");
const captureImgBtn    = document.getElementById("captureImgBtn");
const videoBox         = document.getElementById("videoBox");
const videoFeed        = document.getElementById("videoFeed");
const statusBar        = document.getElementById("statusBar");
const lastSave         = document.getElementById("lastSave");

const calibrateBtn     = document.getElementById("calibrateBtn");
const calibStatus      = document.getElementById("calibStatus");
const rowsInput        = document.getElementById("rows");
const colsInput        = document.getElementById("cols");
const squareInput      = document.getElementById("square");

const refreshListBtn   = document.getElementById("refreshListBtn");
const imageSelect      = document.getElementById("imageSelect");
const loadImageBtn     = document.getElementById("loadImageBtn");
const measureCanvas    = document.getElementById("measureCanvas");
const clearPointsBtn   = document.getElementById("clearPointsBtn");
const measureBtn       = document.getElementById("measureBtn");
const measureResult    = document.getElementById("measureResult");

const zInput           = document.getElementById("zInput");

// state
let cameraOn         = false;
let points           = [];
let loadedImageName  = null;
let loadedImageBitmap = null;

// ---- helpers ----
function measureMode() { return "perspective"; }

function refreshStream() {
  // bust cache param so <img> always refreshes
  videoFeed.src = API_BASE + "/video_feed?ts=" + Date.now();
}

function setUI() {
  toggleBtn.textContent         = cameraOn ? "Turn Camera Off" : "Turn Camera On";
  statusBar.textContent         = cameraOn ? "Camera is ON" : "Camera is OFF";
  videoBox.style.display        = cameraOn ? "block" : "none";
  captureCalibBtn.disabled      = !cameraOn;
  captureImgBtn.disabled        = !cameraOn;

  if (cameraOn) refreshStream(); else videoFeed.src = "";
}

async function getStatus() {
  try {
    const res = await fetch(API_BASE + "/status");
    const data = await res.json();
    cameraOn = !!data.on;
    setUI();
  } catch (e) {
    console.error(e);
  }
}

async function toggleCamera() {
  try {
    const res = await fetch(API_BASE + "/toggle", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    });
    const data = await res.json();
    cameraOn = !!data.on;
    setUI();
  } catch (e) {
    console.error(e);
    alert("Failed to toggle camera");
  }
}

async function capture(mode) {
  try {
    lastSave.textContent = "";
    const res = await fetch(API_BASE + `/capture?mode=${encodeURIComponent(mode)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" }
    });
    const data = await res.json();
    if (!data.ok) { alert(data.error || "Capture failed"); return; }
    lastSave.textContent = `Saved: ${data.path}`;
  } catch (e) {
    console.error(e);
    alert("Capture failed");
  }
}

async function runCalibration() {
  try {
    calibStatus.textContent = "Calibrating...";
    const rows   = parseInt(rowsInput.value || "6", 10);
    const cols   = parseInt(colsInput.value || "9", 10);
    const square = parseFloat(squareInput.value || "25");

    const res = await fetch(API_BASE + "/calibrate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rows, cols, square_size: square })
    });
    const data = await res.json();
    if (!data.ok) {
      calibStatus.textContent = "❌ " + (data.error || "Calibration failed");
      return;
    }
    calibStatus.textContent = "✅ Calibrated. Saved: " + data.npz;
  } catch (e) {
    console.error(e);
    calibStatus.textContent = "❌ Calibration failed";
  }
}

async function refreshList() {
  try {
    imageSelect.innerHTML = "";
    const res = await fetch(API_BASE + "/list_captures");
    const data = await res.json();
    if (!data.ok) { alert(data.error || "Failed to list captures"); return; }
    data.files.forEach(f => {
      const opt = document.createElement("option");
      opt.value = f;
      opt.textContent = f;
      imageSelect.appendChild(opt);
    });
  } catch (e) {
    console.error(e);
    alert("Failed to list captures");
  }
}

async function loadImage() {
  try {
    const sel = imageSelect.value;
    if (!sel) { alert("Pick an image first"); return; }
    loadedImageName = sel;

    const url = API_BASE + `/captures/${encodeURIComponent(sel)}`;

    const img = new Image();
    img.crossOrigin = "Anonymous";
    img.onload = async () => {
      loadedImageBitmap = await createImageBitmap(img);
      drawCanvas();
    };
    img.src = url;
  } catch (e) {
    console.error(e);
    alert("Failed to load image");
  }
}

function drawCanvas() {
  const ctx = measureCanvas.getContext("2d");
  ctx.clearRect(0, 0, measureCanvas.width, measureCanvas.height);

  if (loadedImageBitmap) {
    const cw = measureCanvas.width, ch = measureCanvas.height;
    const iw = loadedImageBitmap.width, ih = loadedImageBitmap.height;
    const scale = Math.min(cw / iw, ch / ih);
    const w = iw * scale, h = ih * scale;
    const x = (cw - w) / 2, y = (ch - h) / 2;
    ctx.drawImage(loadedImageBitmap, x, y, w, h);
    measureCanvas._imgX = x; measureCanvas._imgY = y;
    measureCanvas._imgW = w; measureCanvas._imgH = h;
    measureCanvas._imgScale = scale;
  }

  ctx.fillStyle = "red";
  ctx.strokeStyle = "red";
  ctx.lineWidth = 2;

  points.forEach(p => {
    ctx.beginPath();
    ctx.arc(p[0], p[1], 4, 0, 2 * Math.PI);
    ctx.fill();
  });

  if (points.length === 2) {
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    ctx.lineTo(points[1][0], points[1][1]);
    ctx.stroke();
  }
}

measureCanvas.addEventListener("click", (e) => {
  const rect = measureCanvas.getBoundingClientRect();
  // Map from CSS pixels -> canvas pixels
  const scaleX = measureCanvas.width  / rect.width;
  const scaleY = measureCanvas.height / rect.height;

  const cx = (e.clientX - rect.left) * scaleX;
  const cy = (e.clientY - rect.top)  * scaleY;

  if (points.length >= 2) points = [];
  points.push([cx, cy]);
  drawCanvas();
});

function canvasToImageCoords(p) {
  const [cx, cy] = p;
  const x = (cx - (measureCanvas._imgX || 0)) / (measureCanvas._imgScale || 1);
  const y = (cy - (measureCanvas._imgY || 0)) / (measureCanvas._imgScale || 1);
  return [Math.max(0, x), Math.max(0, y)];
}

async function doMeasure() {
  try {
    const mode = measureMode(); // 'perspective'
    measureResult.textContent = "";
    if (!loadedImageName) { alert("Load an image first"); return; }
    if (points.length !== 2) { alert("Click exactly two points on the image"); return; }

    const p1 = canvasToImageCoords(points[0]);
    const p2 = canvasToImageCoords(points[1]);
    const url = mode === "plane" ? (API_BASE + "/measure") : (API_BASE + "/measure_perspective");
    const body = mode === "plane"
      ? { image_name: loadedImageName, p1, p2 }
      : { image_name: loadedImageName, p1, p2, z_world: parseFloat(zInput.value || "0") };

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (!data.ok) { alert(data.error || "Measurement failed"); return; }
    measureResult.textContent =
      `Perspective: |ΔX|≈ ${data.dX_world.toFixed(2)}, length ≈ ${data.length_world.toFixed(2)} (same units as Z)`;
  } catch (e) {
    console.error(e);
    alert("Measurement failed");
  }
}

function clearPoints() {
  points = [];
  drawCanvas();
}

// ---- event wiring ----
toggleBtn.addEventListener("click", toggleCamera);
captureCalibBtn.addEventListener("click", () => capture("calib"));
captureImgBtn.addEventListener("click", () => capture("img"));
calibrateBtn.addEventListener("click", runCalibration);
refreshListBtn.addEventListener("click", refreshList);
loadImageBtn.addEventListener("click", loadImage);
clearPointsBtn.addEventListener("click", clearPoints);
measureBtn.addEventListener("click", doMeasure);

// init
getStatus();
drawCanvas();