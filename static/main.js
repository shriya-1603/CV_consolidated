async function startCapture() {
  const nameEl = document.getElementById("username");
  const cntEl = document.getElementById("count");
  const numEl = document.getElementById("num");
  const statusEl = document.getElementById("status");

  const username = (nameEl.value || "").trim();
  const target = Math.max(1, Math.min(25, parseInt(numEl.value || "10", 10)));

  if (!username) {
    alert("Enter a username first.");
    return;
  }

  cntEl.textContent = "0";
  for (let i = 0; i < target; i++) {
    const form = new FormData();
    form.append("username", username);
    const r = await fetch("/api/capture", { method: "POST", body: form });
    const js = await r.json();
    if (!js.ok) {
      statusEl.textContent = js.msg || "Capture failed.";
      break;
    }
    cntEl.textContent = String(js.count);
    // Small delay so you can slightly move your face for variety
    await new Promise(res => setTimeout(res, 400));
  }
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

  const r = await fetch("/api/auth", { method: "POST" });
  const js = await r.json();

  if (!js.ok) { msgEl.textContent = js.msg || "Auth failed."; return; }

  if (js.matched && js.redirect) {
    window.location.href = js.redirect;
  } else if (js.matched) {
    msgEl.textContent = `✅ Welcome, ${js.username} (conf: ${js.confidence.toFixed(2)})`;
  } else {
    msgEl.textContent = `❌ Not recognized (${js.username}) — conf: ${js.confidence.toFixed?.(2)}`;
  }
}