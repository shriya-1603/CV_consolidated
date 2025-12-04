// ============================================
// Utility: parse "x,y" lines into [[x,y], ...]
// ============================================
function m7_parsePoints(text) {
  const lines = text
    .split(/[\r\n]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);

  const pts = [];
  for (const line of lines) {
    const parts = line.split(",");
    if (parts.length !== 2) {
      throw new Error("Each line must be 'x,y'");
    }
    const x = parseFloat(parts[0].trim());
    const y = parseFloat(parts[1].trim());
    if (Number.isNaN(x) || Number.isNaN(y)) {
      throw new Error("Invalid number in: " + line);
    }
    pts.push([x, y]);
  }
  return pts;
}

// ============================================
// DOM wiring (only if Task-1 elements exist)
// ============================================
document.addEventListener("DOMContentLoaded", () => {
  const leftImg = document.getElementById("m7-img-left");
  const rightImg = document.getElementById("m7-img-right");
  const overlayLeft = document.getElementById("m7-overlay-left");
  const overlayRight = document.getElementById("m7-overlay-right");

  // If we're not on Task 1 page, do nothing.
  if (!leftImg || !rightImg || !overlayLeft || !overlayRight) {
    return;
  }

  let leftPoints = [];
  let rightPoints = [];

  function m7_updateTextarea(id, pts) {
    const text = pts.map((p) => `${p[0]},${p[1]}`).join("\n");
    const el = document.getElementById(id);
    if (el) el.value = text;
  }

  // ---------- LEFT image click ----------
  leftImg.addEventListener("click", (e) => {
    const rect = leftImg.getBoundingClientRect();
    const x = Math.round(e.clientX - rect.left);
    const y = Math.round(e.clientY - rect.top);

    leftPoints.push([x, y]);
    m7_updateTextarea("m7-points-left", leftPoints);
    document.getElementById("m7-left-clicks").textContent =
      `${leftPoints.length} points selected`;

    const dot = document.createElement("div");
    dot.className = "m7-point-dot";
    dot.style.left = x + "px";
    dot.style.top = y + "px";
    overlayLeft.appendChild(dot);
  });

  // ---------- RIGHT image click ----------
  rightImg.addEventListener("click", (e) => {
    const rect = rightImg.getBoundingClientRect();
    const x = Math.round(e.clientX - rect.left);
    const y = Math.round(e.clientY - rect.top);

    rightPoints.push([x, y]);
    m7_updateTextarea("m7-points-right", rightPoints);
    document.getElementById("m7-right-clicks").textContent =
      `${rightPoints.length} points selected`;

    const dot = document.createElement("div");
    dot.className = "m7-point-dot";
    dot.style.left = x + "px";
    dot.style.top = y + "px";
    overlayRight.appendChild(dot);
  });

  // ---------- Reset ----------
  const resetBtn = document.getElementById("m7-btn-reset");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      leftPoints = [];
      rightPoints = [];
      m7_updateTextarea("m7-points-left", []);
      m7_updateTextarea("m7-points-right", []);

      document.getElementById("m7-left-clicks").textContent =
        "0 points selected";
      document.getElementById("m7-right-clicks").textContent =
        "0 points selected";

      overlayLeft.innerHTML = "";
      overlayRight.innerHTML = "";

      const box = document.getElementById("m7-task1-result");
      if (box) {
        box.innerHTML =
          `<p class="muted">Select points and click <strong>Compute Size</strong>.</p>`;
      }

      const err = document.getElementById("m7-task1-error");
      if (err) {
        err.style.display = "none";
        err.textContent = "";
      }
    });
  }

  // ---------- Compute ----------
  const computeBtn = document.getElementById("m7-btn-measure");
  if (computeBtn) {
    computeBtn.addEventListener("click", async () => {
      const resultBox = document.getElementById("m7-task1-result");
      const errorBox = document.getElementById("m7-task1-error");

      if (!resultBox || !errorBox) return;

      errorBox.style.display = "none";
      errorBox.textContent = "";
      resultBox.innerHTML = `<p>Processing...</p>`;

      try {
        const ptsLeft = m7_parsePoints(
          document.getElementById("m7-points-left").value
        );
        const ptsRight = m7_parsePoints(
          document.getElementById("m7-points-right").value
        );
        const shape = document.getElementById("m7-shape").value;

        const body = {
          shape: shape,
          units: "mm",
          points_left: ptsLeft,
          points_right: ptsRight,
        };

        const res = await fetch("/api/module7/task1/measure", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });

        const data = await res.json();

        if (!data.ok) {
          errorBox.style.display = "block";
          errorBox.textContent = data.error || "Unknown error.";
          resultBox.innerHTML = "";
          return;
        }

        let html = `<h3>Measurement Result</h3>`;

        if (typeof data.width === "number") {
          html += `<p><strong>Width:</strong> ${data.width.toFixed(2)} ${data.units}</p>`;
        }
        if (typeof data.height === "number") {
          html += `<p><strong>Height:</strong> ${data.height.toFixed(2)} ${data.units}</p>`;
        }
        if (typeof data.Z_mean === "number") {
          html += `<p><strong>Approximate Z distance:</strong> ${data.Z_mean.toFixed(2)} ${data.units}</p>`;
        }

        resultBox.innerHTML = html;
      } catch (err) {
        errorBox.style.display = "block";
        errorBox.textContent = err.message || String(err);
        resultBox.innerHTML = "";
      }
    });
  }
});