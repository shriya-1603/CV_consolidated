// ====== Browser camera helpers ======
let camStream = null;

async function initCamera() {
  const video = document.getElementById("cam");
  if (!video) return; // not on login/register page

  try {
    camStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false
    });
    video.srcObject = camStream;
  } catch (err) {
    console.error("Camera error:", err);
    alert("Cannot access camera. Please allow camera permission.");
  }
}

function captureFrameBlob() {
  const video = document.getElementById("cam");
  if (!video || !video.videoWidth) {
    return Promise.resolve(null);
  }

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.9);
  });
}

// If you already have a DOMContentLoaded listener, just add this inside it.
// Otherwise, this alone is fine:
document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("cam")) {
    initCamera();
  }
});

async function startCapture() {
  const nameEl   = document.getElementById("username");
  const cntEl    = document.getElementById("count");
  const numEl    = document.getElementById("num");
  const statusEl = document.getElementById("status");

  const username = (nameEl.value || "").trim();
  const target   = Math.max(1, Math.min(25, parseInt(numEl.value || "10", 10)));

  if (!username) {
    alert("Enter a username first.");
    return;
  }

  cntEl.textContent = "0";
  statusEl.textContent = "Capturing...";

  for (let i = 0; i < target; i++) {
    const blob = await captureFrameBlob();
    if (!blob) {
      statusEl.textContent = "Could not capture frame from camera.";
      return;
    }

    const form = new FormData();
    form.append("username", username);
    form.append("frame", blob, `frame_${i}.jpg`);

    const r  = await fetch("/api/capture", { method: "POST", body: form });
    const js = await r.json();

    if (!js.ok) {
      statusEl.textContent = js.msg || "Capture failed.";
      break;
    }

    cntEl.textContent = String(js.count);
    await new Promise((res) => setTimeout(res, 400)); // small delay
  }

  statusEl.textContent = "Finished capturing. Now click 'Train Model'.";
}


async function trainModel() {
  const msgEl = document.getElementById("trainMsg");
  msgEl.textContent = "Training...";
  const r = await fetch("/api/train", { method: "POST" });
  const js = await r.json();
  msgEl.textContent = js.msg || (js.ok ? "Trained." : "Train failed.");
}

async function authenticate() {
  const msgEl = document.getElementById("authMsg");
  msgEl.textContent = "Checking...";

  const blob = await captureFrameBlob();
  if (!blob) {
    msgEl.textContent = "Cannot capture frame from camera.";
    return;
  }

  const form = new FormData();
  form.append("frame", blob, "auth.jpg");

  const r  = await fetch("/api/auth", { method: "POST", body: form });
  const js = await r.json();

  if (!js.ok) {
    msgEl.textContent = js.msg || "Auth failed.";
    return;
  }

  if (js.matched && js.redirect) {
    msgEl.textContent = `✅ Welcome, ${js.username}. Redirecting…`;
    window.location.href = js.redirect;
  } else if (js.matched) {
    msgEl.textContent = `✅ Welcome, ${js.username} (conf: ${js.confidence.toFixed(2)})`;
  } else {
    msgEl.textContent = `❌ Not recognized (${js.username}) — conf: ${js.confidence.toFixed?.(2)}`;
  }
}
