const upStatus       = document.getElementById("uploadStatus");
const baseFile       = document.getElementById("baseFile");
const tmplFile       = document.getElementById("tmplFile");
const uploadBaseBtn  = document.getElementById("uploadBaseBtn");
const uploadTmplBtn  = document.getElementById("uploadTmplBtn");
const detectBtn      = document.getElementById("detectBtn");
const blurBtn        = document.getElementById("blurBtn");
const deblurBtn      = document.getElementById("deblurBtn");
const sigmaInput     = document.getElementById("sigmaInput");
const detPreview     = document.getElementById("detPreview");
const procPreview    = document.getElementById("procPreview");

// state
let base_path_rel = null;
let template_path_rel = null;
let roi = null;               // {x,y,w,h}
let blurred_path_rel = null;

function setStatus(msg) { upStatus.textContent = msg || ""; }

async function upload(kind, fileInput) {
  const f = fileInput.files && fileInput.files[0];
  if (!f) { alert("Pick a file first"); return null; }
  const fd = new FormData();
  fd.append("file", f);
  const res = await fetch(`/mod2/upload?type=${encodeURIComponent(kind)}`, {
    method: "POST",
    body: fd
  });
  const data = await res.json();
  if (!data.ok) { alert(data.error || "upload failed"); return null; }
  return data.path_rel;
}

uploadBaseBtn.addEventListener("click", async () => {
  setStatus("Uploading base...");
  const rel = await upload("base", baseFile);
  if (rel) { base_path_rel = rel; setStatus("Base uploaded ✓"); }
});

uploadTmplBtn.addEventListener("click", async () => {
  setStatus("Uploading template...");
  const rel = await upload("template", tmplFile);
  if (rel) { template_path_rel = rel; setStatus("Template uploaded ✓"); }
});

detectBtn.addEventListener("click", async () => {
  if (!base_path_rel || !template_path_rel) { alert("Upload base & template first"); return; }
  setStatus("Detecting...");
  const res = await fetch("/mod2/detect", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ base_path_rel, template_path_rel })
  });
  const data = await res.json();
  if (!data.ok) { alert(data.error || "detect failed"); return; }
  roi = { x: data.x, y: data.y, w: data.w, h: data.h };
  detPreview.src = `/mod2/file/${encodeURIComponent(data.annotated_rel)}`;
  detPreview.style.display = "block";
  setStatus("Detected ✓");
});

blurBtn.addEventListener("click", async () => {
  if (!base_path_rel || !roi) { alert("Detect first"); return; }
  const sigma = parseFloat(sigmaInput.value || "2.0");
  setStatus("Blurring...");
  const res = await fetch("/mod2/blur", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({
      base_path_rel,
      x: roi.x, y: roi.y, w: roi.w, h: roi.h,
      sigma
    })
  });
  const data = await res.json();
  if (!data.ok) { alert(data.error || "blur failed"); return; }
  blurred_path_rel = data.blurred_rel;
  procPreview.src = `/mod2/file/${encodeURIComponent(blurred_path_rel)}`;
  procPreview.style.display = "block";
  setStatus("Blurred ✓");
});

deblurBtn.addEventListener("click", async () => {
  if (!blurred_path_rel || !roi) { alert("Blur first"); return; }
  setStatus("Deblurring...");
  const res = await fetch("/mod2/deblur", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({
      blurred_path_rel,
      x: roi.x, y: roi.y, w: roi.w, h: roi.h
      // sigma ignored here by backend on purpose; it reuses saved blur params
    })
  });
  const data = await res.json();
  if (!data.ok) { alert(data.error || "deblur failed"); return; }
  procPreview.src = `/mod2/file/${encodeURIComponent(data.deblurred_rel)}`;
  procPreview.style.display = "block";
  setStatus("Deblurred ✓");
});