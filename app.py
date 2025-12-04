import os, json, time, io
from pathlib import Path
from functools import wraps
from datetime import timedelta
from werkzeug.utils import secure_filename
from datetime import datetime
import threading
import atexit

import cv2 as cv
import numpy as np

import csv

try:
    import mediapipe as mp
    HAVE_MEDIAPIPE = True
except ImportError:
    HAVE_MEDIAPIPE = False

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, Response, abort, flash, session, abort
)

# ----------------------------
# Paths & Config
# ----------------------------
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
FACES_DIR  = DATA_DIR / "faces"
MODELS_DIR = DATA_DIR / "models"
RUNTIME    = BASE_DIR / "runtime"

for p in (DATA_DIR, FACES_DIR, MODELS_DIR, RUNTIME / "uploads", RUNTIME / "cache"):
    p.mkdir(parents=True, exist_ok=True)

# Use your existing threshold if you already store one in model_meta.json; fallback to safe default.
META_PATH = MODELS_DIR / "model_meta.json"
DEFAULT_THRESHOLD = 60.0
if META_PATH.exists():
    try:
        _meta = json.loads(META_PATH.read_text())
        LBPH_THRESHOLD = float(_meta.get("threshold", DEFAULT_THRESHOLD))
    except Exception:
        LBPH_THRESHOLD = DEFAULT_THRESHOLD
else:
    LBPH_THRESHOLD = DEFAULT_THRESHOLD

# === Module 1 (Calibrate & Measure) runtime paths ===
BASE_DIR = Path(__file__).resolve().parent  # (skip if already defined earlier)
RUNTIME  = BASE_DIR / "runtime"             # (skip if already defined earlier)

M1_DIR      = RUNTIME / "module1"
M1_CAPT_DIR = M1_DIR / "captures"     # object images
M1_CAL_DIR  = M1_DIR / "calib"        # chessboard captures + calib.npz

# === Module 2 (Detect → Blur → Deblur) paths ===
M2_DIR        = RUNTIME / "module2"
M2_SESSIONS   = M2_DIR / "sessions"      # per-session isolation
for p in (M2_DIR, M2_SESSIONS):
    p.mkdir(parents=True, exist_ok=True)

for p in (RUNTIME, M1_DIR, M1_CAPT_DIR, M1_CAL_DIR):
    p.mkdir(parents=True, exist_ok=True)

#======== Module 3 ========#

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# (A) keep your internal intermediates
INTERNAL_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "module3")

# (B) public outputs for <img src=...> — now namespaced under module3/
STATIC_OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs", "module3")

# Reuse a single base folder for module3 uploads, but split per task
M3_UPLOAD_BASE = os.path.join(BASE_DIR, "uploads", "module3")

# Internal intermediates (not served)
M3_INTERNAL_DIR = os.path.join(BASE_DIR, "outputs", "module3")
# Public outputs for <img src=...> in templates
M3_STATIC_DIR = os.path.join(BASE_DIR, "static", "outputs", "module3")

# Create per-task upload and output dirs
for t in ["task1", "task2", "task3", "task4"]:
    os.makedirs(os.path.join(M3_UPLOAD_BASE, t), exist_ok=True)

for t in ["task1", "task2", "task3", "task4", "task5"]:
    os.makedirs(os.path.join(M3_INTERNAL_DIR, t), exist_ok=True)
    os.makedirs(os.path.join(M3_STATIC_DIR, t), exist_ok=True)


def m3_upload_dir(task_num: int) -> str:
    """Return the upload dir for a given task number (1–4)."""
    return os.path.join(M3_UPLOAD_BASE, f"task{task_num}")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INTERNAL_OUTPUT_DIR, exist_ok=True)
for t in ["task1", "task2", "task3", "task4", "task5"]:
    os.makedirs(os.path.join(INTERNAL_OUTPUT_DIR, t), exist_ok=True)
    os.makedirs(os.path.join(STATIC_OUTPUT_DIR, t), exist_ok=True)


# --- Module 4 runtime (sessions) ---
M4_DIR = RUNTIME / "module4"
M4_SESSIONS = M4_DIR / "sessions"
for p in (M4_DIR, M4_SESSIONS):
    p.mkdir(parents=True, exist_ok=True)

import uuid  # ensure imported once
import time  # needed for unique file names

def m4_rid() -> str:
    rid = session.get("m4_rid")
    if not rid:
        rid = uuid.uuid4().hex[:8]
        session["m4_rid"] = rid
    return rid

def m4_session_dir():
    sd = M4_SESSIONS / m4_rid()
    (sd / "uploads").mkdir(parents=True, exist_ok=True)
    (sd / "outputs").mkdir(parents=True, exist_ok=True)
    return sd

# --- Module 7 runtime dirs ---
# --- Module 7 runtime dirs ---
M7_DIR = RUNTIME / "module7"
M7_TASK1_DIR = M7_DIR / "task1"
M7_TASK1_UPLOADS = M7_TASK1_DIR / "uploads"
M7_TASK3_DIR = M7_DIR / "task3_data"

for p in (M7_DIR, M7_TASK1_DIR, M7_TASK1_UPLOADS, M7_TASK3_DIR):
    p.mkdir(parents=True, exist_ok=True)

if HAVE_MEDIAPIPE:
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    M7_POSE = mp_pose.Pose(model_complexity=1, enable_segmentation=False)
    M7_HANDS = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

M7_T3_FRAME_IDX = 0
M7_T3_LOCK = threading.Lock()
M7_T3_CSV_PATH = M7_TASK3_DIR / f"pose_hand_{int(time.time())}.csv"
   

# Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("SECRET_KEY", "dev-change-me")
app.permanent_session_lifetime = timedelta(hours=8)


# ----------------------------
# Camera (global singleton)
# ----------------------------
# Use the default webcam (0). If you need IP cam, replace with VideoCapture(url).
CAM_INDEX = int(os.environ.get("CAM_INDEX", "0"))
_camera = cv.VideoCapture(CAM_INDEX)

if not _camera.isOpened():
    # Try to open anyway on first request; don’t crash app init.
    _camera = None

def get_camera():
    global _camera
    if _camera is None or (hasattr(_camera, "isOpened") and not _camera.isOpened()):
        cam = cv.VideoCapture(CAM_INDEX)
        if not cam.isOpened():
            raise RuntimeError("Camera not available.")
        _camera = cam
    return _camera

# ----------------------------
# Face detection (Haar) & helpers
# ----------------------------
# Use default frontal face detector; you can swap for your own cascade if you already ship one.
CASCADE = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_largest_face(gray):
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    # pick largest by area
    x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
    return (x, y, w, h)

def prepare_face_roi(frame_bgr):
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    box = detect_largest_face(gray)
    if box is None:
        return None, None, None
    x, y, w, h = box
    roi = gray[y:y+h, x:x+w]
    roi = cv.resize(roi, (200, 200), interpolation=cv.INTER_AREA)
    return roi, (x, y, w, h), gray

# === Module 1 state & helpers ===
M1_CAMERA_ON = False  # UI toggle only; we reuse the same VideoCapture

def jpeg_bytes(frame_bgr):
    ok, buf = cv.imencode(".jpg", frame_bgr, [cv.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else None

import uuid  # if not already imported

def m2_rid() -> str:
    rid = session.get("m2_rid")
    if not rid:
        rid = uuid.uuid4().hex[:8]
        session["m2_rid"] = rid
    return rid

def m2_session_dir():
    sd = M2_SESSIONS / m2_rid()
    (sd / "uploads").mkdir(parents=True, exist_ok=True)
    (sd / "outputs").mkdir(parents=True, exist_ok=True)
    return sd

def _m2_safe(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in ("-", "_", ".", " ")).strip()

def _m2_ksize_from_sigma(sigma: float) -> int:
    k = int(6 * max(0.1, sigma) + 1)
    if k % 2 == 0:
        k += 1
    return max(3, k)

# === /measure/video_feed (MJPEG) ===
def m1_mjpeg_stream():
    cam = get_camera()
    while True:
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.05)
            continue
        # Optional watermark so you know this is Module 1 stream
        cv.putText(frame, "Module 1", (8, 24), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv.LINE_AA)
        jpg = jpeg_bytes(frame)
        if jpg is None:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

# ----------------------------
# LBPH Model utils
# ----------------------------
def lbph_create():
    # Requires opencv-contrib-python
    return cv.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

MODEL_PATH = MODELS_DIR / "lbph.yml"
LABELS_PATH = DATA_DIR / "labels.json"

def load_labels():
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text())
    return {}

def save_labels(mapping):
    LABELS_PATH.write_text(json.dumps(mapping, indent=2))

def ensure_model_trained():
    if not MODEL_PATH.exists():
        return None
    model = lbph_create()
    model.read(str(MODEL_PATH))
    return model

# ----------------------------
# Auth guard
# ----------------------------
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper

# ----------------------------
# Routes: Public
# ----------------------------
@app.route("/")
def index():
    # Simple landing — link to register/login (your base navbar already has links)
    return redirect(url_for("login"))

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/register")
def register():
    return render_template("register.html")

# ----------------------------
# Routes: Dashboard & Modules (protected after face-auth success)
# ----------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=session.get("user"))

@app.route("/module/<int:n>")
@login_required
def module_n(n):
    if n not in (1,2,3,4):
        abort(404)
    return render_template(f"modules/module{n}.html", user=session.get("user"), n=n)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ----------------------------
# Video stream (MJPEG) for <img src="{{ url_for('video_feed') }}">
# ----------------------------
def mjpeg_stream():
    cam = get_camera()
    while True:
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.05)
            continue
        # draw face box for feedback (optional)
        roi, box, gray = prepare_face_roi(frame)
        if box is not None:
            x, y, w, h = box
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        # encode frame
        ret, buf = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        jpg = buf.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ----------------------------
# API: Capture — /api/capture (POST)  [register flow]
# expects: username; saves one cropped face image
# ----------------------------
@app.post("/api/capture")
def api_capture():
    username = (request.form.get("username") or "").strip()
    if not username:
        return jsonify(ok=False, msg="username required"), 400

    # 1) Try to read frame from browser upload
    img_file = request.files.get("frame")
    frame = None
    if img_file:
        data = np.frombuffer(img_file.read(), np.uint8)
        frame = cv.imdecode(data, cv.IMREAD_COLOR)

    # 2) Fallback: server-side camera (for local dev)
    if frame is None:
        cam = get_camera()
        ok, frame = cam.read()
        if not ok:
            return jsonify(ok=False, msg="camera read failed"), 500

    roi, box, _ = prepare_face_roi(frame)
    if roi is None:
        return jsonify(ok=False, msg="no face detected"), 200

    user_dir = FACES_DIR / username
    user_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted([p for p in user_dir.glob("img_*.jpg")])
    count = len(existing) + 1
    out_path = user_dir / f"img_{count:04d}.jpg"
    cv.imwrite(str(out_path), roi)

    return jsonify(ok=True, count=count)

# ----------------------------
# API: Train — /api/train (POST)  [register flow]
# trains LBPH on all folders under data/faces/<username>/*.jpg
# ----------------------------
@app.post("/api/train")
def api_train():
    # Build dataset
    images, labels = [], []
    label_map = {}
    next_label = 0

    for person_dir in sorted(FACES_DIR.glob("*")):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        if name not in label_map:
            label_map[name] = next_label
            next_label += 1

        for img_path in sorted(person_dir.glob("*.jpg")):
            img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # Ensure consistent size
            img = cv.resize(img, (200, 200), interpolation=cv.INTER_AREA)
            images.append(img)
            labels.append(label_map[name])

    if not images:
        return jsonify(ok=False, msg="no training images found")

    model = lbph_create()
    model.train(images, np.array(labels, dtype=np.int32))
    model.write(str(MODEL_PATH))
    save_labels({k: int(v) for k, v in label_map.items()})

    # Persist threshold if you already had one configured; otherwise keep default
    META_PATH.write_text(json.dumps({"threshold": LBPH_THRESHOLD}, indent=2))

    return jsonify(ok=True, msg=f"trained on {len(images)} images, users={len(label_map)}")

# ----------------------------
# API: Auth — /api/auth (POST)  [login flow]
# reads current frame, predicts via LBPH; on success, sets session and returns redirect=/dashboard
# ----------------------------
@app.post("/api/auth")
def api_auth():
    model = ensure_model_trained()
    if model is None:
        return jsonify(ok=False, msg="model not trained yet"), 200

    labels = load_labels()
    inv_labels = {v: k for k, v in labels.items()}

    # 1) Try frame uploaded from browser
    img_file = request.files.get("frame")
    frame = None
    if img_file:
        data = np.frombuffer(img_file.read(), np.uint8)
        frame = cv.imdecode(data, cv.IMREAD_COLOR)

    # 2) Fallback: server-side camera (local dev)
    if frame is None:
        cam = get_camera()
        ok, frame = cam.read()
        if not ok:
            return jsonify(ok=False, msg="camera read failed"), 500

    roi, box, _ = prepare_face_roi(frame)
    if roi is None:
        return jsonify(ok=True, matched=False, username="(no face)", confidence=999.0)

    pred_label, conf = model.predict(roi)
    name = inv_labels.get(int(pred_label), "unknown")

    matched = conf <= LBPH_THRESHOLD and name != "unknown"
    if matched:
        session.permanent = True
        session["user"] = name
        return jsonify(
            ok=True,
            matched=True,
            username=name,
            confidence=float(conf),
            redirect=url_for("dashboard"),
        )

    return jsonify(ok=True, matched=False, username=name, confidence=float(conf))
# ----------------------------
@app.route("/measure/video_feed")
@login_required
def m1_video_feed():
    return Response(m1_mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ===== Module 1 API =====

@app.get("/measure/status")
@login_required
def m1_status():
    return jsonify(ok=True, on=bool(M1_CAMERA_ON))

@app.post("/measure/toggle")
@login_required
def m1_toggle():
    global M1_CAMERA_ON
    M1_CAMERA_ON = not M1_CAMERA_ON
    # Camera object stays open; toggle affects UI stream.
    return jsonify(ok=True, on=bool(M1_CAMERA_ON))

@app.post("/measure/capture")
@login_required
def m1_capture():
    """
    ?mode=calib | img
    Saves the current frame to calib/ or captures/ respectively.
    """
    mode = (request.args.get("mode") or "").lower()
    if mode not in ("calib", "img"):
        return jsonify(ok=False, error="invalid mode"), 400

    cam = get_camera()
    ok, frame = cam.read()
    if not ok:
        return jsonify(ok=False, error="camera read failed"), 500

    ts = int(time.time() * 1000)
    if mode == "calib":
        out_path = M1_CAL_DIR / f"calib_{ts}.jpg"
    else:
        out_path = M1_CAPT_DIR / f"capture_{ts}.jpg"

    cv.imwrite(str(out_path), frame)
    return jsonify(ok=True, path=str(out_path.relative_to(BASE_DIR)))

@app.post("/measure/calibrate")
@login_required
def m1_calibrate():
    """
    Body: { rows:int, cols:int, square_size:float }
    Finds chessboard corners in ALL images in M1_CAL_DIR and saves calib.npz
    """
    try:
        payload = request.get_json(force=True) or {}
        rows = int(payload.get("rows", 6))
        cols = int(payload.get("cols", 9))
        square = float(payload.get("square_size", 25.0))
        pattern_size = (cols, rows)  # OpenCV expects (cols, rows)
    except Exception as e:
        return jsonify(ok=False, error=f"bad args: {e}"), 400

    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square

    objpoints, imgpoints = [], []
    image_size = None

    images = sorted(M1_CAL_DIR.glob("*.jpg"))
    if not images:
        return jsonify(ok=False, error="no calibration images found")

    for imgp in images:
        img = cv.imread(str(imgp))
        if img is None:
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])

        ret, corners = cv.findChessboardCorners(
            gray, pattern_size,
            flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret:
            corners2 = cv.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            objpoints.append(objp)
            imgpoints.append(corners2)

    if not objpoints:
        return jsonify(ok=False, error="no valid chessboard detections"), 400

    reproj_err, K, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    out_npz = M1_CAL_DIR / "calib.npz"
    np.savez_compressed(
        str(out_npz),
        camera_matrix=K,
        dist_coeffs=dist,
        rvecs=rvecs,
        tvecs=tvecs,
        rows=rows, cols=cols, square_size=square,
        image_width=image_size[0], image_height=image_size[1],
        reprojection_error=reproj_err
    )
    return jsonify(ok=True, npz=str(out_npz.relative_to(BASE_DIR)), reprojection_error=float(reproj_err))

@app.get("/measure/list_captures")
@login_required
def m1_list_captures():
    files = [p.name for p in sorted(M1_CAPT_DIR.glob("*.jpg"))]
    return jsonify(ok=True, files=files)

@app.get("/measure/captures/<path:name>")
@login_required
def m1_get_capture(name):
    p = (M1_CAPT_DIR / name).resolve()
    if not str(p).startswith(str(M1_CAPT_DIR.resolve())) or not p.exists():
        abort(404)
    return Response(open(p, "rb").read(), mimetype="image/jpeg")

@app.post("/measure/measure_perspective")
@login_required
def m1_measure_perspective():
    """
    Body: { image_name:str, p1:[x,y], p2:[x,y], z_world:float }
    Uses pinhole approximation: world dX ≈ Z * Δx / fx, world dY ≈ Z * Δy / fy.
    """
    try:
        payload = request.get_json(force=True) or {}
        name = payload["image_name"]
        p1 = payload["p1"]; p2 = payload["p2"]
        Z  = float(payload.get("z_world", 0.0))
    except Exception as e:
        return jsonify(ok=False, error=f"bad args: {e}"), 400

    npz_path = M1_CAL_DIR / "calib.npz"
    if not npz_path.exists():
        return jsonify(ok=False, error="no calibration (calib.npz) found"), 400

    data = np.load(str(npz_path))
    K   = data["camera_matrix"].astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])

    dx_px = float(p2[0]) - float(p1[0])
    dy_px = float(p2[1]) - float(p1[1])

    dX_world = Z * ((dx_px / fx))/2
    dY_world = Z * ((dy_px / fy))/2
    length_world = float(np.hypot(dX_world, dY_world))

    return jsonify(
        ok=True,
        dX_world=float(dX_world),
        dY_world=float(dY_world),
        length_world=length_world
    )
    

@app.route("/module/1")
@login_required
def module_1():
    return render_template("modules/module1.html", user=session.get("user"))

# ===== Module 2 API =====

@app.post("/mod2/upload")
@login_required
def mod2_upload():
    """
    ?type=base|template
    form field: 'file'
    Saves to runtime/module2/sessions/<rid>/uploads/
    """
    ftype = (request.args.get("type") or "").lower()
    if ftype not in ("base", "template"):
        return jsonify(ok=False, error="invalid type (use base|template)"), 400
    if "file" not in request.files:
        return jsonify(ok=False, error="no file"), 400

    sd = m2_session_dir()
    up_dir = sd / "uploads"
    fs = request.files["file"]
    name = _m2_safe(fs.filename or f"{ftype}.jpg")
    if not name:
        name = f"{ftype}.jpg"

    out_name = f"{ftype}_{int(time.time()*1000)}_{name}"
    out_path = up_dir / out_name
    fs.save(str(out_path))

    # return a path relative to BASE_DIR so we can fetch via /mod2/file/<rel>
    return jsonify(ok=True, path_rel=str(out_path.relative_to(BASE_DIR)))

@app.post("/mod2/detect")
@login_required
def mod2_detect():
    payload = request.get_json(force=True) or {}
    base_rel = payload.get("base_path_rel")
    tmpl_rel = payload.get("template_path_rel")
    if not base_rel or not tmpl_rel:
        return jsonify(ok=False, error="missing base_path_rel or template_path_rel"), 400

    base_path = (BASE_DIR / base_rel).resolve()
    tmpl_path = (BASE_DIR / tmpl_rel).resolve()
    if not base_path.exists() or not tmpl_path.exists():
        return jsonify(ok=False, error="file(s) not found"), 404

    img  = cv.imread(str(base_path), cv.IMREAD_COLOR)
    templ = cv.imread(str(tmpl_path), cv.IMREAD_COLOR)
    if img is None or templ is None:
        return jsonify(ok=False, error="failed to read images"), 400

    img_g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    tmp_g = cv.cvtColor(templ, cv.COLOR_BGR2GRAY)
    th, tw = tmp_g.shape[:2]
    if th >= img_g.shape[0] or tw >= img_g.shape[1]:
        return jsonify(ok=False, error="template must be smaller than base"), 400

    res = cv.matchTemplate(img_g, tmp_g, cv.TM_CCORR_NORMED)
    _minVal, maxVal, _minLoc, maxLoc = cv.minMaxLoc(res)
    x, y = maxLoc
    w, h = tw, th

    # draw box & save annotated
    vis = img.copy()
    cv.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    sd = m2_session_dir()
    out_anno = sd / "outputs" / f"det_{int(time.time()*1000)}.jpg"
    cv.imwrite(str(out_anno), vis)

    # stash ROI + score for this session (and clear stored blur params)
    session["m2_box"] = (int(x), int(y), int(w), int(h))
    session.pop("m2_blur_params", None)

    return jsonify(ok=True,
                   x=int(x), y=int(y), w=int(w), h=int(h),
                   annotated_rel=str(out_anno.relative_to(BASE_DIR)))

@app.post("/mod2/blur")
@login_required
def mod2_blur():
    payload = request.get_json(force=True) or {}
    base_rel = payload.get("base_path_rel")
    x = int(payload.get("x", 0)); y = int(payload.get("y", 0))
    w = int(payload.get("w", 0)); h = int(payload.get("h", 0))
    sigma = float(payload.get("sigma", 2.0))

    if not base_rel or w <= 0 or h <= 0:
        return jsonify(ok=False, error="bad params"), 400

    base_path = (BASE_DIR / base_rel).resolve()
    if not base_path.exists():
        return jsonify(ok=False, error="base image not found"), 404

    img = cv.imread(str(base_path), cv.IMREAD_COLOR)
    if img is None:
        return jsonify(ok=False, error="failed to read base image"), 400

    H, W = img.shape[:2]
    x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
    w = max(1, min(w, W - x)); h = max(1, min(h, H - y))

    ksize = _m2_ksize_from_sigma(sigma)
    m = ksize // 2
    X0 = max(0, x - m); Y0 = max(0, y - m)
    X1 = min(W, x + w + m); Y1 = min(H, y + h + m)

    roi_pad = img[Y0:Y1, X0:X1].copy()
    blurred = cv.GaussianBlur(roi_pad, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv.BORDER_REPLICATE)

    out = img.copy()
    out[y:y+h, x:x+w] = blurred[(y - Y0):(y - Y0 + h), (x - X0):(x - X0 + w)]

    sd = m2_session_dir()
    out_path = sd / "outputs" / f"blur_{int(time.time()*1000)}.jpg"
    cv.imwrite(str(out_path), out)

    # persist for deblur
    session["m2_blur_params"] = {"ksize": int(ksize), "sigma": float(sigma), "pad": int(m), "base_rel": base_rel}

    return jsonify(ok=True, blurred_rel=str(out_path.relative_to(BASE_DIR)))

def _psf2otf(psf, out_shape):
    H, W = out_shape
    kh, kw = psf.shape
    pad = np.zeros((H, W), np.float32)
    pad[:kh, :kw] = psf
    pad = np.roll(pad, -kh//2, axis=0)
    pad = np.roll(pad, -kw//2, axis=1)
    return np.fft.fft2(pad)

def _wiener_deconv(gray_roi, ksize=9, sigma=2.0, K=1e-3):
    g = gray_roi.astype(np.float32) / 255.0
    if ksize % 2 == 0:
        ksize += 1
    gk = cv.getGaussianKernel(ksize, sigma)
    psf = (gk @ gk.T).astype(np.float32)
    H = _psf2otf(psf, g.shape)
    G = np.fft.fft2(g)
    X = (np.conj(H) / ((np.abs(H) ** 2) + K + 1e-12)) * G
    x = np.real(np.fft.ifft2(X))
    x = np.clip(x, 0, 1)
    return (x * 255.0).astype(np.uint8)

@app.post("/mod2/deblur")
@login_required
def mod2_deblur():
    payload = request.get_json(force=True) or {}
    bl_rel = payload.get("blurred_path_rel")
    x = int(payload.get("x", 0)); y = int(payload.get("y", 0))
    w = int(payload.get("w", 0)); h = int(payload.get("h", 0))

    if not bl_rel or w <= 0 or h <= 0:
        return jsonify(ok=False, error="bad params"), 400

    bp = session.get("m2_blur_params")
    if not bp:
        return jsonify(ok=False, error="no stored blur params; run Blur first"), 400

    ksize = int(bp.get("ksize", 9))
    sigma = float(bp.get("sigma", 2.0))
    m = int(bp.get("pad", ksize // 2))

    bl_path = (BASE_DIR / bl_rel).resolve()
    if not bl_path.exists():
        return jsonify(ok=False, error="blurred image not found"), 404

    img = cv.imread(str(bl_path), cv.IMREAD_COLOR)
    if img is None:
        return jsonify(ok=False, error="failed to read blurred image"), 400

    H, W = img.shape[:2]
    x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
    w = max(1, min(w, W - x)); h = max(1, min(h, H - y))

    X0 = max(0, x - m); Y0 = max(0, y - m)
    X1 = min(W, x + w + m); Y1 = min(H, y + h + m)
    roi_pad_bgr = img[Y0:Y1, X0:X1].copy()

    ycc = cv.cvtColor(roi_pad_bgr, cv.COLOR_BGR2YCrCb)
    Yc, Cr, Cb = cv.split(ycc)

    deY = _wiener_deconv(Yc, ksize=ksize, sigma=sigma, K=1e-3)
    ycc_de = cv.merge([deY, Cr, Cb])
    roi_pad_deb = cv.cvtColor(ycc_de, cv.COLOR_YCrCb2BGR)

    out = img.copy()
    out[y:y+h, x:x+w] = roi_pad_deb[(y - Y0):(y - Y0 + h), (x - X0):(x - X0 + w)]

    sd = m2_session_dir()
    out_path = sd / "outputs" / f"deblur_{int(time.time()*1000)}.jpg"
    cv.imwrite(str(out_path), out)

    return jsonify(ok=True, deblurred_rel=str(out_path.relative_to(BASE_DIR)))

@app.get("/mod2/file/<path:rel>")
@login_required
def mod2_file(rel):
    p = (BASE_DIR / rel).resolve()
    sd = m2_session_dir().resolve()
    # only allow serving from this user's session folder
    if not str(p).startswith(str(sd)):
        abort(403)
    if not p.exists():
        abort(404)
    return Response(open(p, "rb").read(), mimetype="image/jpeg")

@app.route("/module/2")
@login_required
def module_2():
    return render_template("modules/module2.html", user=session.get("user"))


# ===== Module 3 API =====

from flask import session, abort  # if not already imported

@app.route("/module/3")
@login_required
def module3_dashboard():
    tasks = [
        {"id": 1, "name": "Task 1 – Gradients & LoG"},
        {"id": 2, "name": "Task 2 – Edges & Corners"},
        {"id": 3, "name": "Task 3 – Boundary Detection"},
        {"id": 4, "name": "Task 4 – ArUco Segmentation"},
        {"id": 5, "name": "Task 5 – SAM 2 Demo (External)"},
    ]
    return render_template("modules/module3_dashboard.html", tasks=tasks)

@app.route("/module/3/task/<int:num>", methods=["GET"])
@login_required
def module3_task(num):
    """
    Small dispatcher so the dashboard can use:
      url_for('module3_task', num=1..4)
    """
    if num == 1:
        return redirect(url_for("module3_task1"))
    elif num == 2:
        return redirect(url_for("module3_task2"))
    elif num == 3:
        return redirect(url_for("module3_task3"))
    elif num == 4:
        return redirect(url_for("module3_task4"))
    else:
        abort(404)


def _list_images_in_uploads(task_num: int):
    """List images only from uploads/module3/task{task_num}/."""
    paths = []
    up_dir = m3_upload_dir(task_num)
    if not os.path.isdir(up_dir):
        return paths
    for name in os.listdir(up_dir):
        if os.path.splitext(name)[1].lower() in IMG_EXTS:
            paths.append(os.path.join(up_dir, name))
    return paths

@app.route("/module/3/task/1", methods=["GET", "POST"])
@login_required
def module3_task1():
    """
    Upload or re-run: compute Gradient Magnitude, Gradient Angle, and Laplacian-of-Gaussian.
    Saves to static/outputs/module3/task1/.
    """
    results = []

    if request.method == "POST":
        action = request.form.get("action", "upload")

        # Collect images
        images_to_process = []
        if action == "upload":
            files = request.files.getlist("images")
            if not files or files[0].filename == "":
                flash("Please select at least one image.")
                return redirect(url_for("module3_task1"))

            up_dir = m3_upload_dir(1)
            for f in files:
                up = os.path.join(up_dir, f.filename)
                f.save(up)
                images_to_process.append(up)
            flash(f"Uploaded {len(images_to_process)} image(s). Now processing…")
        else:
            images_to_process = _list_images_in_uploads(1)
            if not images_to_process:
                flash("No images found in uploads/module3/task1/ to re-run on.")
                return redirect(url_for("module3_task1"))
            flash(f"Re-running on {len(images_to_process)} existing image(s).")

        # Process
        for img_path in images_to_process:
            filename = os.path.basename(img_path)
            img = cv.imread(img_path)
            if img is None:
                continue

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Gradients
            gx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
            gy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(gx * gx + gy * gy)
            ang = np.degrees(np.arctan2(gy, gx))
            # Normalize for display
            mag_norm = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
            ang_norm = cv.normalize((ang + 360) % 360, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
            # LoG
            blur = cv.GaussianBlur(gray, (7, 7), 1.2)
            log = cv.Laplacian(blur, cv.CV_64F)
            log_norm = cv.normalize(log, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

            ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
            base, _ = os.path.splitext(filename)
            mag_name = f"{base}_{ts}_gradmag.png"
            ang_name = f"{base}_{ts}_gradang.png"
            log_name = f"{base}_{ts}_log.png"

            # Save public outputs
            out_dir = os.path.join(M3_STATIC_DIR, "task1")
            cv.imwrite(os.path.join(out_dir, mag_name), mag_norm)
            cv.imwrite(os.path.join(out_dir, ang_name), ang_norm)
            cv.imwrite(os.path.join(out_dir, log_name), log_norm)

            results.append({
                "original_name": filename,
                "grad_mag_rel": f"outputs/module3/task1/{mag_name}",
                "grad_ang_rel": f"outputs/module3/task1/{ang_name}",
                "log_rel":      f"outputs/module3/task1/{log_name}",
            })

        return render_template("modules/module3/task1.html", results=results)

    return render_template("modules/module3/task1.html", results=None)


# --------------------
# Task 2 – Edges & Corners
# --------------------
@app.route("/module/3/task/2", methods=["GET", "POST"])
@login_required
def module3_task2():
    """
    Edge points (red) via gradient magnitude threshold.
    Corner points (green) via Harris.
    """
    results = []

    if request.method == "POST":
        action = request.form.get("action", "rerun")

        images_to_process = []
        if action == "upload":
            files = request.files.getlist("images")
            if not files or files[0].filename == "":
                flash("Please select at least one image for Task 2.")
                return redirect(url_for("module3_task2"))

            up_dir = m3_upload_dir(2)
            for f in files:
                up = os.path.join(up_dir, f.filename)
                f.save(up)
                images_to_process.append(up)
            flash(f"Uploaded {len(images_to_process)} image(s) for Task 2.")
        else:
            images_to_process = _list_images_in_uploads(2)
            if not images_to_process:
                flash("No images found in uploads/module3/task2/.")
                return redirect(url_for("module3_task2"))
            flash(f"Running Task 2 on {len(images_to_process)} existing image(s).")

        for img_path in images_to_process:
            filename = os.path.basename(img_path)
            img = cv.imread(img_path)
            if img is None:
                continue

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gray_blur32 = np.float32(cv.GaussianBlur(gray, (3, 3), 1))

            # Edges
            Ix = cv.Sobel(gray_blur32, cv.CV_64F, 1, 0, ksize=3)
            Iy = cv.Sobel(gray_blur32, cv.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(Ix**2 + Iy**2)
            edge_thresh = 50  # simple fixed threshold (tune if needed)
            ey, ex = np.where(grad_mag > edge_thresh)
            edge_vis = img.copy()
            for (x, y) in zip(ex, ey):
                cv.circle(edge_vis, (x, y), 1, (0, 0, 255), -1)  # red

            # Corners (Harris)
            harris = cv.cornerHarris(gray_blur32, blockSize=2, ksize=3, k=0.04)
            harris = cv.dilate(harris, None)
            corner_thresh = 0.02 * harris.max()
            corner_vis = img.copy()
            pts = np.argwhere(harris > corner_thresh)
            for (y, x) in pts:
                cv.circle(corner_vis, (x, y), 4, (0, 255, 0), 1)
                cv.circle(corner_vis, (x, y), 2, (0, 255, 0), -1)

            ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
            base, _ = os.path.splitext(filename)
            edge_name = f"{base}_{ts}_edges.png"
            corner_name = f"{base}_{ts}_corners.png"

            out_dir = os.path.join(M3_STATIC_DIR, "task2")
            cv.imwrite(os.path.join(out_dir, edge_name), edge_vis)
            cv.imwrite(os.path.join(out_dir, corner_name), corner_vis)

            results.append({
                "original_name": filename,
                "edges_rel":   f"outputs/module3/task2/{edge_name}",
                "corners_rel": f"outputs/module3/task2/{corner_name}",
            })

        return render_template("modules/module3/task2.html", results=results)

    return render_template("modules/module3/task2.html", results=None)


# --------------------
# Task 3 – Object Boundary Extraction
# --------------------
@app.route("/module/3/task/3", methods=["GET", "POST"])
@login_required
def module3_task3():
    """
    blur → Canny → morphology → largest contour → draw overlay (blue).
    """
    results = []

    if request.method == "POST":
        action = request.form.get("action", "rerun")

        images_to_process = []
        if action == "upload":
            files = request.files.getlist("images")
            if not files or files[0].filename == "":
                flash("Please select at least one image for Task 3.")
                return redirect(url_for("module3_task3"))

            up_dir = m3_upload_dir(3)
            for f in files:
                up = os.path.join(up_dir, f.filename)
                f.save(up)
                images_to_process.append(up)
            flash(f"Uploaded {len(images_to_process)} image(s) for Task 3.")
        else:
            images_to_process = _list_images_in_uploads(3)
            if not images_to_process:
                flash("No images found in uploads/module3/task3/.")
                return redirect(url_for("module3_task3"))
            flash(f"Running Task 3 on {len(images_to_process)} existing image(s).")

        for img_path in images_to_process:
            filename = os.path.basename(img_path)
            img = cv.imread(img_path)
            if img is None:
                continue

            draw_vis = img.copy()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(gray, (5, 5), 1)

            edges = cv.Canny(blur, 50, 150)
            kernel = np.ones((3, 3), np.uint8)
            edges_closed = cv.dilate(edges, kernel, iterations=1)
            edges_closed = cv.erode(edges_closed, kernel, iterations=1)

            cnts, _ = cv.findContours(edges_closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            if len(cnts) > 0:
                cnts = sorted(cnts, key=cv.contourArea, reverse=True)
                main = cnts[0]
                peri = cv.arcLength(main, True)
                approx = cv.approxPolyDP(main, 0.01 * peri, True)

                # Blue outline (BGR)
                cv.drawContours(draw_vis, [approx], -1, (255, 0, 0), 2)

            ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
            base, _ = os.path.splitext(filename)
            boundary_name = f"{base}_{ts}_boundary.png"
            edges_name = f"{base}_{ts}_edges.png"

            out_dir = os.path.join(M3_STATIC_DIR, "task3")
            cv.imwrite(os.path.join(out_dir, boundary_name), draw_vis)
            cv.imwrite(os.path.join(out_dir, edges_name), edges_closed)

            results.append({
                "original_name": filename,
                "boundary_rel": f"outputs/module3/task3/{boundary_name}",
                "edges_rel":    f"outputs/module3/task3/{edges_name}",
            })

        return render_template("modules/module3/task3.html", results=results)

    return render_template("modules/module3/task3.html", results=None)


# --------------------
# Task 4 – ArUco-based Segmentation
# --------------------
@app.route("/module/3/task/4", methods=["GET", "POST"])
@login_required
def module3_task4():
    """
    Detect ArUco (try 4X4_1000 → 250 → 100 → 50). Fallback to QR if none.
    Build convex hull of all detected corners → mask + overlay.
    """
    results = []

    if request.method == "POST":
        action = request.form.get("action", "rerun")

        # Collect images
        images_to_process = []
        if action == "upload":
            files = request.files.getlist("images")
            if not files or files[0].filename == "":
                flash("Please select at least one image for Task 4.")
                return redirect(url_for("module3_task4"))

            up_dir = m3_upload_dir(4)
            for f in files:
                up = os.path.join(up_dir, f.filename)
                f.save(up)
                images_to_process.append(up)
            flash(f"Uploaded {len(images_to_process)} image(s) for Task 4.")
        else:
            images_to_process = _list_images_in_uploads(4)
            if not images_to_process:
                flash("No images found in uploads/module3/task4/.")
                return redirect(url_for("module3_task4"))
            flash(f"Running Task 4 on {len(images_to_process)} existing image(s).")

        aruco_mod = getattr(cv, "aruco", None)
        dict_candidates = ["DICT_4X4_1000", "DICT_4X4_250", "DICT_4X4_100", "DICT_4X4_50"]

        for img_path in images_to_process:
            filename = os.path.basename(img_path)
            img0 = cv.imread(img_path)
            if img0 is None:
                continue

            h0, w0 = img0.shape[:2]
            scale = 1.5 if min(h0, w0) < 900 else 1.0
            img = cv.resize(img0, None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC) if scale != 1.0 else img0.copy()

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            markers_vis = img.copy()
            hull_vis = img.copy()
            all_pts = []
            used_dict = None

            # 1) ArUco
            if aruco_mod is not None:
                for name in dict_candidates:
                    try:
                        dict_id = getattr(aruco_mod, name)
                        dictionary = aruco_mod.getPredefinedDictionary(dict_id)
                    except Exception:
                        continue

                    try:
                        if hasattr(aruco_mod, "DetectorParameters"):   # new API
                            params = aruco_mod.DetectorParameters()
                            params.cornerRefinementMethod = aruco_mod.CORNER_REFINE_SUBPIX
                            params.adaptiveThreshWinSizeMin = 3
                            params.adaptiveThreshWinSizeMax = 53
                            params.adaptiveThreshWinSizeStep = 10
                            params.minMarkerPerimeterRate = 0.02
                            params.maxMarkerPerimeterRate = 4.0
                            params.minCornerDistanceRate = 0.01
                            params.minOtsuStdDev = 5.0
                            detector = aruco_mod.ArucoDetector(dictionary, params)
                            corners_list, ids, _ = detector.detectMarkers(gray)
                        else:  # legacy API
                            params = aruco_mod.DetectorParameters_create()
                            params.cornerRefinementMethod = aruco_mod.CORNER_REFINE_SUBPIX
                            params.adaptiveThreshWinSizeMin = 3
                            params.adaptiveThreshWinSizeMax = 53
                            params.adaptiveThreshWinSizeStep = 10
                            params.minMarkerPerimeterRate = 0.02
                            params.maxMarkerPerimeterRate = 4.0
                            params.minCornerDistanceRate = 0.01
                            params.minOtsuStdDev = 5.0
                            corners_list, ids, _ = aruco_mod.detectMarkers(gray, dictionary, parameters=params)
                    except Exception:
                        corners_list, ids = None, None

                    if ids is not None and len(corners_list) > 0:
                        used_dict = name
                        try:
                            aruco_mod.drawDetectedMarkers(markers_vis, corners_list, ids)
                        except Exception:
                            pass
                        for c in corners_list:
                            for pt in c[0]:
                                all_pts.append(pt)
                        break

            # 2) Fallback to QR
            if len(all_pts) == 0:
                qr = cv.QRCodeDetector()
                ok, decoded, points, _ = qr.detectAndDecodeMulti(gray)
                if points is not None and len(points) > 0:
                    quads = points.astype(np.int32)
                    for quad in quads:
                        cv.polylines(markers_vis, [quad], True, (0, 255, 255), 2)
                        for pt in quad:
                            all_pts.append(pt)

            # 3) Hull/Mask/Overlay
            ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
            base, _ = os.path.splitext(filename)

            # remap to original resolution if upscaled
            if scale != 1.0 and len(all_pts) > 0:
                all_pts = (np.array(all_pts, dtype=np.float32) / scale).astype(np.int32)
                markers_vis = cv.resize(markers_vis, (w0, h0), interpolation=cv.INTER_AREA)
                hull_vis = cv.resize(hull_vis, (w0, h0), interpolation=cv.INTER_AREA)
                h, w = h0, w0
            else:
                all_pts = np.array(all_pts, dtype=np.int32)
                h, w = img0.shape[:2]

            if len(all_pts) >= 3:
                hull = cv.convexHull(all_pts)

                mask = np.zeros((h, w), dtype=np.uint8)
                cv.fillConvexPoly(mask, hull, 255)

                overlay = hull_vis.copy()
                cv.polylines(overlay, [hull], True, (255, 0, 0), 2)  # blue outline (BGR)
                color_fill = np.zeros_like(hull_vis); color_fill[:] = (255, 200, 0)
                fill = cv.bitwise_and(color_fill, color_fill, mask=mask)
                hull_vis = cv.addWeighted(hull_vis, 1.0, fill, 0.35, 0)
                hull_vis = cv.addWeighted(hull_vis, 1.0, overlay, 1.0, 0)

                out_dir = os.path.join(M3_STATIC_DIR, "task4")
                markers_name = f"{base}_{ts}_markers.png"
                hull_name    = f"{base}_{ts}_hull.png"
                mask_name    = f"{base}_{ts}_mask.png"

                cv.imwrite(os.path.join(out_dir, markers_name), markers_vis)
                cv.imwrite(os.path.join(out_dir, hull_name), hull_vis)
                cv.imwrite(os.path.join(out_dir, mask_name), mask)

                if used_dict:
                    flash(f"{filename}: ArUco detected ({used_dict}).")
                elif len(all_pts) > 0:
                    flash(f"{filename}: QR markers used (Aruco not found).")

                results.append({
                    "original_name": filename,
                    "markers_rel": f"outputs/module3/task4/{markers_name}",
                    "hull_rel":    f"outputs/module3/task4/{hull_name}",
                    "mask_rel":    f"outputs/module3/task4/{mask_name}",
                })
            else:
                # Save just a context image so UI shows something
                out_dir = os.path.join(M3_STATIC_DIR, "task4")
                markers_name = f"{base}_{ts}_markers.png"
                vis_to_save = img0 if scale != 1.0 else markers_vis
                cv.imwrite(os.path.join(out_dir, markers_name), vis_to_save)

                flash(f"{filename}: No markers detected (Aruco or QR).")
                results.append({
                    "original_name": filename,
                    "markers_rel": f"outputs/module3/task4/{markers_name}",
                    "hull_rel": None,
                    "mask_rel": None,
                })

        return render_template("modules/module3/task4.html", results=results)

    return render_template("modules/module3/task4.html", results=None)


# --------------------
# Task 5 – External SAM2 Demo (redirect)
# --------------------
@app.route("/module/3/task/5")
def module3_task5():
    return redirect("https://segment-anything.com/demo#")


# ===== Module 4 API =====

@app.get("/mod4/file/<path:rel>")
@login_required
def mod4_file(rel):
    p = (BASE_DIR / rel).resolve()
    sd = (M4_SESSIONS / m4_rid()).resolve()
    if not str(p).startswith(str(sd)) or not p.exists():
        abort(403)
    # adjust mimetype if you serve PNG/WEBP too; JPEG is fine as default for .jpg
    return Response(open(p, "rb").read(), mimetype="image/jpeg")

@app.route("/module/4")
@login_required
def module_4():
    return render_template("modules/module4/index.html", user=session.get("user"))

ALLOWED_IMG = {"png", "jpg", "jpeg", "webp"}

def m4_allowed(name: str) -> bool:
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_IMG

def m4_uniq(stem: str, ext: str) -> str:
    return f"{stem}-{uuid.uuid4().hex[:8]}.{ext.lower()}"

def m4_imread_color_path(path: Path):
    # robust imread (Windows-safe)
    arr = np.fromfile(str(path), dtype=np.uint8)
    return cv.imdecode(arr, cv.IMREAD_COLOR)

def m4_resize_max_w(img, max_w=1600):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / float(w)
    return cv.resize(img, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)

@app.route("/module/4/task/1", methods=["GET", "POST"])
@login_required
def module_4_task1():
    notes = []
    result_rel = None

    if request.method == "POST":
        files = request.files.getlist("images")
        imgs_to_process = [f for f in files if f and f.filename]
        if len(imgs_to_process) == 0:
            notes.append("Upload at least 4 landscape or 8 portrait images for stitching.")
            return render_template("modules/module4/task1.html", notes=notes, result_rel=result_rel)

        sd = m4_session_dir()
        up_dir = sd / "uploads"
        out_dir = sd / "outputs"

        # save uploads
        saved = []
        for f in imgs_to_process:
            if not m4_allowed(f.filename):
                notes.append(f"Unsupported type: {f.filename}")
                return render_template("modules/module4/task1.html", notes=notes, result_rel=result_rel)
            stem = secure_filename(f.filename.rsplit(".", 1)[0]) or "img"
            ext = f.filename.rsplit(".", 1)[1].lower()
            name = m4_uniq(stem, ext)
            path = up_dir / name
            f.save(str(path))
            saved.append(path)

        # read + downscale
        images = []
        for p in saved:
            img = m4_imread_color_path(p)
            if img is None:
                notes.append(f"Could not read {p.name}; skipping.")
                continue
            images.append(m4_resize_max_w(img, max_w=1600))

        if len(images) < 4:
            notes.append("Need at least 4 usable images after reading (or 8 if portrait capture).")
            return render_template("modules/module4/task1.html", notes=notes, result_rel=result_rel)

        # stitch
        try:
            try:
                stitcher = cv.Stitcher_create(cv.Stitcher_PANORAMA)
            except Exception:
                stitcher = cv.Stitcher_create()
            status, pano = stitcher.stitch(images)
            if status != cv.Stitcher_OK or pano is None:
                notes.append(f"Stitch failed (status={status}). Try more overlap and consistent exposure.")
                return render_template("modules/module4/task1.html", notes=notes, result_rel=result_rel)

            pano = m4_resize_max_w(pano, max_w=4000)
            out_name = m4_uniq("stitched", "jpg")
            out_path = out_dir / out_name
            cv.imwrite(str(out_path), pano)
            result_rel = str(out_path.relative_to(BASE_DIR))
            notes.append(f"Stitched {len(images)} images ✓")
        except Exception as e:
            notes.append(f"Exception during stitching: {e}")

    return render_template("modules/module4/task1.html", notes=notes, result_rel=result_rel)

# ---- SIFT-lite helpers (namespaced) ----
def m4_to_gray(img_bgr):
    return cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0

def m4_gblur(img, sigma):
    k = int(round(6 * sigma + 1))
    if k % 2 == 0: k += 1
    return cv.GaussianBlur(img, (k, k), sigma)

def m4_build_gauss_and_dog(g, sigmas=(1.0, 1.4, 2.0, 2.8, 4.0)):
    G = [m4_gblur(g, s) for s in sigmas]
    D = [G[i+1] - G[i] for i in range(len(G)-1)]
    return G, D

def m4_hessian_edge_filter(D, y, x, r=10.0):
    dxx = D[y, x+1] + D[y, x-1] - 2 * D[y, x]
    dyy = D[y+1, x] + D[y-1, x] - 2 * D[y, x]
    dxy = (D[y+1, x+1] - D[y+1, x-1] - D[y-1, x+1] + D[y-1, x-1]) * 0.25
    tr = dxx + dyy
    det = dxx * dyy - dxy * dxy
    if det <= 0: return False
    return (tr * tr) / det < ((r + 1) ** 2) / r

def m4_detect_kps(D, contrast=0.03, border=8):
    kps = []
    for s in range(1, len(D) - 1):
        H, W = D[s].shape
        for y in range(border, H - border):
            for x in range(border, W - border):
                v = D[s][y, x]
                if abs(v) < contrast:
                    continue
                patch = []
                for ds in (-1, 0, 1):
                    patch.append(D[s + ds][y-1:y+2, x-1:x+2].reshape(-1))
                nb = np.concatenate(patch)
                if (v == nb.max() and (nb[:-1] <= v).all()) or (v == nb.min() and (nb[:-1] >= v).all()):
                    if m4_hessian_edge_filter(D[s], y, x):
                        kps.append((s, y, x))
    return kps

def m4_grad_mag_ori(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=1)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=1)
    mag = np.sqrt(gx * gx + gy * gy)
    ori = (np.rad2deg(np.arctan2(gy, gx)) + 360.0) % 360.0
    return mag, ori

def m4_assign_orientation(Gs, kps, s_sigmas=(1.0, 1.4, 2.0, 2.8, 4.0)):
    out = []
    for (s, y, x) in kps:
        img = Gs[s]
        mag, ori = m4_grad_mag_ori(img)
        sigma = 1.5 * s_sigmas[s]
        rad = int(3 * sigma)
        y0, y1 = max(0, y - rad), min(img.shape[0], y + rad + 1)
        x0, x1 = max(0, x - rad), min(img.shape[1], x + rad + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        w = np.exp(-((yy - y)**2 + (xx - x)**2) / (2 * sigma * sigma))
        m = mag[y0:y1, x0:x1] * w
        o = ori[y0:y1, x0:x1]
        hist = np.zeros(36, np.float32)
        bins = (o / 10).astype(int) % 36
        for b in range(36):
            hist[b] = m[bins == b].sum()
        theta = float(np.argmax(hist) * 10.0)
        out.append((float(y), float(x), theta, int(s)))
    return out

def m4_sift128(img, y, x, theta_deg, win=16, cells=4, bins=8):
    mag, ori = m4_grad_mag_ori(img)
    half = win // 2
    ys = np.linspace(-half + 0.5, half - 0.5, win)
    xs = np.linspace(-half + 0.5, half - 0.5, win)
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    t = np.deg2rad(theta_deg)
    Yr =  Y * np.cos(t) + X * np.sin(t) + y
    Xr = -Y * np.sin(t) + X * np.cos(t) + x
    Yr = np.clip(np.round(Yr).astype(int), 0, img.shape[0] - 1)
    Xr = np.clip(np.round(Xr).astype(int), 0, img.shape[1] - 1)
    m = mag[Yr, Xr]
    o = (ori[Yr, Xr] - theta_deg) % 360.0
    sigma = 0.5 * win
    yy, xx = np.mgrid[0:win, 0:win]
    w = np.exp(-((yy - half)**2 + (xx - half)**2) / (2 * sigma * sigma))
    m *= w
    step = win // cells
    desc = np.zeros((cells, cells, bins), np.float32)
    binw = 360.0 / bins
    for cy in range(cells):
        for cx in range(cells):
            block_m = m[cy*step:(cy+1)*step, cx*step:(cx+1)*step]
            block_o = o[cy*step:(cy+1)*step, cx*step:(cx+1)*step]
            h = np.zeros(bins, np.float32)
            bi = (block_o / binw).astype(int) % bins
            for b in range(bins):
                h[b] = block_m[bi == b].sum()
            desc[cy, cx, :] = h
    v = desc.reshape(-1)
    v /= (np.linalg.norm(v) + 1e-7)
    v = np.clip(v, 0, 0.2)
    v /= (np.linalg.norm(v) + 1e-7)
    return v.astype(np.float32)

def m4_extract_sift_lite(gray):
    sigmas = (1.0, 1.4, 2.0, 2.8, 4.0)
    G, D = m4_build_gauss_and_dog(gray, sigmas)
    kps_idx = m4_detect_kps(D, contrast=0.03, border=8)
    kps = m4_assign_orientation(G, kps_idx, sigmas)
    if not kps:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    pts, descs = [], []
    for (y, x, theta, sidx) in kps:
        pts.append([x, y])
        descs.append(m4_sift128(G[sidx], y, x, theta))
    return np.asarray(pts, np.float32), np.vstack(descs).astype(np.float32)

def m4_ratio_match(D1, D2, ratio=0.75):
    if len(D1) == 0 or len(D2) == 0:
        return []
    d = np.sqrt(((D1[:, None, :] - D2[None, :, :])**2).sum(axis=2))
    out = []
    for i in range(d.shape[0]):
        idx = np.argsort(d[i])[:2]
        if len(idx) == 2 and d[i, idx[0]] < ratio * d[i, idx[1]]:
            out.append((i, int(idx[0])))
    return out

def m4_draw_matches_side_aligned(img1, pts1, img2, pts2, inliers=None):
    h = max(img1.shape[0], img2.shape[0])
    w = img1.shape[1] + img2.shape[1]
    canvas = np.zeros((h, w, 3), np.uint8)
    canvas[:img1.shape[0], :img1.shape[1]] = img1
    canvas[:img2.shape[0], img1.shape[1]:] = img2
    off = img1.shape[1]
    K = min(len(pts1), len(pts2))
    for k in range(K):
        p1 = tuple(np.round(pts1[k]).astype(int))
        p2 = tuple(np.round(pts2[k]).astype(int))
        ok = (inliers is not None and k < len(inliers) and bool(inliers[k]))
        color = (0, 255, 0) if ok else (0, 0, 255)  # green inliers, red outliers
        cv.circle(canvas, p1, 3, color, -1)
        cv.circle(canvas, (int(p2[0]) + off, int(p2[1])), 3, color, -1)
        cv.line(canvas, p1, (int(p2[0]) + off, int(p2[1])), color, 1)
    return canvas

@app.route("/module/4/task/2", methods=["GET", "POST"])
@login_required
def module_4_task2():
    notes, metrics = [], {}
    result_ours = result_cv = None
    sd = m4_session_dir()
    out_dir = sd / "outputs"

    if request.method == "POST":
        f1 = request.files.get("img1")
        f2 = request.files.get("img2")
        if not f1 or not f2 or f1.filename == "" or f2.filename == "":
            notes.append("Upload two images.")
            return render_template("modules/module4/task2.html", notes=notes)

        for f in (f1, f2):
            if not m4_allowed(f.filename):
                notes.append(f"Unsupported type: {f.filename}")
                return render_template("modules/module4/task2.html", notes=notes)

        # read directly (no need to persist uploads)
        img1 = cv.imdecode(np.frombuffer(f1.read(), np.uint8), cv.IMREAD_COLOR)
        img2 = cv.imdecode(np.frombuffer(f2.read(), np.uint8), cv.IMREAD_COLOR)
        if img1 is None or img2 is None:
            notes.append("Could not read one of the images.")
            return render_template("modules/module4/task2.html", notes=notes)

        img1s = m4_resize_max_w(img1, 1000)
        img2s = m4_resize_max_w(img2, 1000)
        g1, g2 = m4_to_gray(img1s), m4_to_gray(img2s)

        # our SIFT-lite
        t0 = time.time()
        P1, D1 = m4_extract_sift_lite(g1)
        P2, D2 = m4_extract_sift_lite(g2)
        t_sift = time.time() - t0

        matches = m4_ratio_match(D1, D2, ratio=0.75)
        if len(matches) >= 4:
            pts1 = np.array([P1[i] for (i, _) in matches], np.float32)
            pts2 = np.array([P2[j] for (_, j) in matches], np.float32)
            H, mask = cv.findHomography(pts1, pts2, cv.RANSAC, 3.0)
            inmask = mask.ravel().astype(bool) if mask is not None else np.zeros(len(matches), bool)

            vis = m4_draw_matches_side_aligned(img1s, pts1, img2s, pts2, inmask)
            out1 = out_dir / f"siftours_{int(time.time()*1000)}.jpg"
            cv.imwrite(str(out1), vis)
            result_ours = str(out1.relative_to(BASE_DIR))

            metrics.update({
                "ours_keypoints_1": int(len(P1)),
                "ours_keypoints_2": int(len(P2)),
                "ours_matches": int(len(matches)),
                "ours_inliers": int(inmask.sum()),
                "ours_inlier_ratio": round(float(inmask.sum()) / max(1, len(matches)), 3),
                "time_sift_sec": round(t_sift, 3),
            })
        else:
            notes.append("Too few matches after ratio test (ours).")

        # OpenCV SIFT comparison (optional)
        try:
            sift = cv.SIFT_create()
        except Exception:
            notes.append("OpenCV SIFT not available. Install 'opencv-contrib-python'.")
        else:
            g1_cv = cv.cvtColor(img1s, cv.COLOR_BGR2GRAY)
            g2_cv = cv.cvtColor(img2s, cv.COLOR_BGR2GRAY)
            k1, d1 = sift.detectAndCompute(g1_cv, None)
            k2, d2 = sift.detectAndCompute(g2_cv, None)
            if d1 is None or d2 is None or len(k1) == 0 or len(k2) == 0:
                notes.append("OpenCV SIFT: no descriptors found.")
            else:
                bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
                knn = bf.knnMatch(d1, d2, k=2)
                good = []
                for pair in knn:
                    if len(pair) < 2: continue
                    m, n = pair
                    if m.distance < 0.75 * n.distance:
                        good.append(m)
                if len(good) >= 4:
                    pts1_cv = np.float32([k1[m.queryIdx].pt for m in good])
                    pts2_cv = np.float32([k2[m.trainIdx].pt for m in good])
                    Hcv, maskcv = cv.findHomography(pts1_cv, pts2_cv, cv.RANSAC, 3.0)
                    inmask_cv = maskcv.ravel().astype(bool) if maskcv is not None else np.zeros(len(good), bool)
                    vis_cv = m4_draw_matches_side_aligned(img1s, pts1_cv, img2s, pts2_cv, inmask_cv)
                    out2 = out_dir / f"opencv_matches_{int(time.time()*1000)}.jpg"
                    cv.imwrite(str(out2), vis_cv)
                    result_cv = str(out2.relative_to(BASE_DIR))
                    metrics.update({
                        "cv_keypoints_1": len(k1),
                        "cv_keypoints_2": len(k2),
                        "cv_matches": len(good),
                        "cv_inliers": int(inmask_cv.sum()),
                        "cv_inlier_ratio": round(float(inmask_cv.sum()) / max(1, len(good)), 3),
                    })
                else:
                    notes.append("OpenCV SIFT: too few good matches after ratio test.")

    return render_template("modules/module4/task2.html",
                           result_ours=result_ours,
                           result_cv=result_cv,
                           metrics=metrics,
                           notes=notes)

# -------- Module 5: PDF viewer --------

@app.route("/module/5")
@login_required
def module_5():
    # mini dashboard with two PDF tiles
    return render_template("modules/module5.html", user=session.get("user"))


@app.route("/module/5/doc/<doc_id>")
@login_required
def module_5_doc(doc_id):
    if doc_id == "1":
        pdf_url = url_for("static", filename="module5/one.pdf")
        title = "Module 5 – Document 1"
    elif doc_id == "2":
        pdf_url = url_for("static", filename="module5/two.pdf")
        title = "Module 5 – Document 2"
    else:
        abort(404)

    return render_template(
        "modules/module5_viewer.html",
        user=session.get("user"),
        page_title=title,
        pdf_url=pdf_url,
    )

# ==========================
# Module 6 – Tracker helpers
# ==========================

# 1) ArUco marker-based tracker
def marker_tracker(frame):
    import cv2  # uses the same cv2 already imported at top

    # Use a 4x4, 50-ID ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    return frame


# 2) Markerless tracker (goodFeaturesToTrack + optical flow)
class MarkerlessTracker:
    def __init__(self):
        self.prev_gray = None
        self.prev_pts = None

    def update(self, frame):
        import cv2
        import numpy as np

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # First frame → detect interest points
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=200,
                qualityLevel=0.01,
                minDistance=7,
                blockSize=7
            )
            return frame

        # If there are no points, reinitialize
        if self.prev_pts is None or len(self.prev_pts) < 10:
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=200,
                qualityLevel=0.01,
                minDistance=7,
                blockSize=7
            )
            self.prev_gray = gray
            return frame

        # Calculate optical flow
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None
        )

        good_new = new_pts[status == 1]
        good_old = self.prev_pts[status == 1]

        # Draw tracks
        for (new, old) in zip(good_new, good_old):
            x1, y1 = new.ravel()
            x2, y2 = old.ravel()
            cv2.circle(frame, (int(x1), int(y1)), 4, (0, 255, 0), -1)

        # Update
        self.prev_gray = gray
        self.prev_pts = good_new.reshape(-1, 1, 2)

        return frame


# 3) SAM2 tracker (mask overlay + bbox + center + label)
def sam2_tracker(frame, masks, frame_id):
    """
    Applies SAM2 segmentation mask to the frame and draws:
      - colored overlay
      - bounding box
      - center point
      - label
    """
    import cv2
    import numpy as np

    # Clamp frame_id to prevent index errors
    frame_id = min(frame_id, masks.shape[0] - 1)

    # Extract mask for this frame
    mask = masks[frame_id]

    # Resize mask if needed
    if mask.shape != frame.shape[:2]:
        mask = cv2.resize(
            mask,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    mask_bin = (mask > 0).astype(np.uint8)

    # Create green overlay on top of mask
    overlay = frame.copy()
    overlay[mask_bin == 1] = (0, 255, 0)

    # Blend with original image
    blended = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    # Compute bounding box
    ys, xs = np.where(mask_bin == 1)

    if len(xs) > 0:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # Draw bounding box (red)
        cv2.rectangle(
            blended,
            (x_min, y_min),
            (x_max, y_max),
            (0, 0, 255),
            2
        )

        # Draw center point (blue)
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        cv2.circle(blended, (cx, cy), 6, (255, 0, 0), -1)

        # Label
        cv2.putText(
            blended,
            "SAM2 Object",
            (x_min, max(0, y_min - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    return blended

# ==========================
# Module 6 – globals
# ==========================

# static/module6/data/input_video.mp4, sam2_masks_fixed.npz
MODULE6_DIR = BASE_DIR / "static" / "module6"
MODULE6_DATA_DIR = MODULE6_DIR / "data"

MODULE6_SAM2_MASKS = np.load(MODULE6_DATA_DIR / "sam2_masks_fixed.npz")["masks"]
MODULE6_VIDEO_PATH = MODULE6_DATA_DIR / "input_video.mp4"

# Pre-open the video used for SAM2 playback
MODULE6_VIDEO = cv.VideoCapture(str(MODULE6_VIDEO_PATH))

MODULE6_MODE = "marker"          # "marker" | "markerless" | "sam2"
MODULE6_MODE_LOCK = threading.Lock()
MODULE6_MARKERLESS = MarkerlessTracker()
MODULE6_FRAME_ID = 0
MODULE6_TOTAL_FRAMES = MODULE6_SAM2_MASKS.shape[0]  # e.g. 199

def module6_gen_frames():
    """
    Streaming generator for /module/6/video_feed
    Uses the main webcam (get_camera) for marker/markerless,
    and the pre-recorded video + SAM2 masks for 'sam2' mode.
    """
    global MODULE6_MODE, MODULE6_FRAME_ID, MODULE6_MARKERLESS

    while True:
        with MODULE6_MODE_LOCK:
            mode = MODULE6_MODE

        # --- Choose source ---
        if mode in ("marker", "markerless"):
            cam = get_camera()
            ret, frame = cam.read()

        elif mode == "sam2":
            ret, frame = MODULE6_VIDEO.read()
            if not ret:
                # loop the video
                MODULE6_VIDEO.set(cv.CAP_PROP_POS_FRAMES, 0)
                MODULE6_FRAME_ID = 0
                continue

            # apply SAM2 first for live preview
            frame = sam2_tracker(frame, MODULE6_SAM2_MASKS, MODULE6_FRAME_ID)
            MODULE6_FRAME_ID = (MODULE6_FRAME_ID + 1) % MODULE6_TOTAL_FRAMES

        else:
            ret, frame = False, None

        if not ret or frame is None:
            continue

        # --- Apply trackers for webcam modes ---
        if mode == "markerless":
            frame = MODULE6_MARKERLESS.update(frame)
        elif mode == "marker":
            frame = marker_tracker(frame)

        # --- Encode and yield MJPEG chunk ---
        ok, buffer = cv.imencode(".jpg", frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

def _module6_cleanup():
    try:
        MODULE6_VIDEO.release()
    except Exception:
        pass

atexit.register(_module6_cleanup)

# ==========================
# Module 6 – routes
# ==========================

@app.route("/module/6")
@login_required
def module_6():
    # main page with buttons + video + slider
    return render_template("modules/module6.html", user=session.get("user"))


@app.route("/module/6/video_feed")
@login_required
def module6_video_feed():
    return Response(
        module6_gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/module/6/set_mode/<mode>")
@login_required
def module6_set_mode(mode):
    """
    Switch between:
      - marker
      - markerless
      - sam2
    """
    global MODULE6_MODE, MODULE6_MARKERLESS, MODULE6_FRAME_ID

    if mode not in ("marker", "markerless", "sam2"):
        abort(404)

    with MODULE6_MODE_LOCK:
        MODULE6_MODE = mode

    # Reset internals on switch
    if mode == "markerless":
        MODULE6_MARKERLESS = MarkerlessTracker()
    if mode == "sam2":
        MODULE6_FRAME_ID = 0
        MODULE6_VIDEO.set(cv.CAP_PROP_POS_FRAMES, 0)

    return "OK"


@app.route("/module/6/sam2_frame/<int:frame_id>")
@login_required
def module6_sam2_frame(frame_id: int):
    """
    Random-access SAM2 frame (for slider + Prev/Next).
    Uses the same input_video.mp4 and masks, but seeks directly.
    """
    # clamp
    frame_id = max(0, min(frame_id, MODULE6_TOTAL_FRAMES - 1))

    cap = cv.VideoCapture(str(MODULE6_DATA_DIR / "input_video.mp4"))
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    frame = sam2_tracker(frame, MODULE6_SAM2_MASKS, frame_id)
    ok, buffer = cv.imencode(".jpg", frame)
    if not ok:
        buffer = cv.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))[1]

    return Response(buffer.tobytes(), mimetype="image/jpeg")

# ==========================
# Module 7 – Stereo + Pose
# ==========================

@app.route("/module/7")
@login_required
def module_7():
    return render_template("modules/module7_dashboard.html", user=session.get("user"))

@app.route("/module/7/task1/file/<path:name>")
@login_required
def module7_task1_file(name):
    """
    Serve left/right images saved under runtime/module7/task1/uploads
    """
    p = (M7_TASK1_UPLOADS / name).resolve()
    if not str(p).startswith(str(M7_TASK1_UPLOADS.resolve())) or not p.exists():
        abort(404)

    # Guess mimetype – images are fine as jpeg/png; using jpeg as default is ok
    return Response(open(p, "rb").read(), mimetype="image/jpeg")

from werkzeug.utils import secure_filename

ALLOWED_STEREO_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

@app.route("/module/7/task/1", methods=["GET", "POST"])
@login_required
def module7_task1():
    """
    Task 1: User uploads Left & Right stereo images.
    We save them under runtime/module7/task1/uploads and
    pass their URLs to the template so JS can let the user
    click points and call /api/module7/task1/measure.
    """
    user = session.get("user")
    left_url = None
    right_url = None
    msg = None

    if request.method == "POST":
        left_file = request.files.get("left_image")
        right_file = request.files.get("right_image")

        if not left_file or left_file.filename == "" or not right_file or right_file.filename == "":
            msg = "Please upload both Left and Right images."
        else:
            # basic extension check
            def _ext_ok(filename: str) -> bool:
                return os.path.splitext(filename)[1].lower() in ALLOWED_STEREO_EXT

            if not _ext_ok(left_file.filename) or not _ext_ok(right_file.filename):
                msg = "Unsupported file type. Please use jpg, png, bmp, or tiff."
            else:
                ts = datetime.now().strftime("%Y%m%d%H%M%S%f")

                # Left
                left_name_safe = secure_filename(left_file.filename)
                left_name = f"left_{ts}_{left_name_safe}"
                left_path = M7_TASK1_UPLOADS / left_name
                left_file.save(str(left_path))

                # Right
                right_name_safe = secure_filename(right_file.filename)
                right_name = f"right_{ts}_{right_name_safe}"
                right_path = M7_TASK1_UPLOADS / right_name
                right_file.save(str(right_path))

                # URLs for <img src=...> in template
                left_url = url_for("module7_task1_file", name=left_name)
                right_url = url_for("module7_task1_file", name=right_name)

                msg = "Stereo pair uploaded. You can now click points and measure."

    return render_template(
        "modules/module7/task1.html",
        user=user,
        left_url=left_url,
        right_url=right_url,
        message=msg,
    )

@app.post("/api/module7/task1/measure")
@login_required
def module7_task1_measure():
    SIZE_SCALE = 2.0
    """
    Body JSON:
    {
      "shape": "rectangle" | "circle" | "polygon",
      "units": "mm",
      "points_left":  [[x,y], ...],
      "points_right": [[x,y], ...]
    }
    """
    data = request.get_json(force=True) or {}
    shape = (data.get("shape") or "rectangle").lower()
    units = data.get("units", "mm")
    pts_left = data.get("points_left") or []
    pts_right = data.get("points_right") or []

    if not pts_left or not pts_right:
        return jsonify(ok=False, error="No points provided.")

    if len(pts_left) != len(pts_right):
        return jsonify(ok=False, error="Left and right point counts must match.")

    L = np.array(pts_left, dtype=np.float32)
    R = np.array(pts_right, dtype=np.float32)
    if L.ndim != 2 or L.shape[1] != 2 or R.ndim != 2 or R.shape[1] != 2:
        return jsonify(ok=False, error="Points must be [x,y] pairs.")

    # Simple stereo: disparity along x
    disparities = L[:, 0] - R[:, 0]
    valid = np.abs(disparities) > 1e-3
    if not np.any(valid):
        return jsonify(ok=False, error="Disparities too small; make sure you click matching points in left and right images.")

    disparities = disparities[valid]

    # Approximate parameters (tunable):
    f_px = 1000.0          # focal length in pixels
    baseline_mm = 60.0     # camera baseline (mm)

    Z_vals = f_px * baseline_mm / disparities
    Z_mean = float(np.mean(Z_vals))

    # Approx pixel→mm scale at mean depth
    scale_mm_per_px = Z_mean / f_px

    width_mm = None
    height_mm = None

    if shape == "circle" and L.shape[0] >= 2:
        # user clicks two opposite points on circle
        d_px = float(np.linalg.norm(L[0] - L[1]))
        diameter = d_px * scale_mm_per_px * SIZE_SCALE
        width_mm = diameter
        height_mm = None
    else:
        # rectangle or polygon → bounding box
        xs = L[:, 0]
        ys = L[:, 1]
        width_px = float(xs.max() - xs.min())
        height_px = float(ys.max() - ys.min())
        width_mm = width_px * scale_mm_per_px * SIZE_SCALE
        height_mm = height_px * scale_mm_per_px * SIZE_SCALE

    return jsonify(
        ok=True,
        shape=shape,
        units=units,
        width=width_mm,
        height=height_mm,
        Z_mean=Z_mean,
    )

@app.route("/module/7/task/2")
@login_required
def module7_task2():
    return render_template("modules/module7/task2.html", user=session.get("user"))

@app.route("/module/7/task/3")
@login_required
def module7_task3():
    return render_template("modules/module7/task3.html", user=session.get("user"))

def m7_log_landmarks(frame_idx, kind, hand_label, landmarks):
    """
    kind: "pose" or "hand"
    hand_label: "Left"/"Right"/"body"/"unknown"
    landmarks: iterable of objects with x, y, z, visibility
    """
    with M7_T3_LOCK:
        new_file = not M7_T3_CSV_PATH.exists()
        with M7_T3_CSV_PATH.open("a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(
                    ["frame_idx", "type", "hand_label",
                     "landmark_idx", "x", "y", "z", "visibility"]
                )
            for idx, lm in enumerate(landmarks):
                writer.writerow([
                    int(frame_idx),
                    kind,
                    hand_label,
                    idx,
                    float(getattr(lm, "x", 0.0)),
                    float(getattr(lm, "y", 0.0)),
                    float(getattr(lm, "z", 0.0)),
                    float(getattr(lm, "visibility", 1.0)),
                ])

def module7_task3_gen():
    """
    MJPEG stream with pose + hand landmarks.
    Requires 'mediapipe'. If not installed, just shows a message overlay.
    """
    global M7_T3_FRAME_IDX

    cam = get_camera()

    while True:
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.03)
            continue

        if HAVE_MEDIAPIPE:
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            pose_res = M7_POSE.process(rgb)
            hands_res = M7_HANDS.process(rgb)

            # Pose
            if pose_res.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    pose_res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                )
                m7_log_landmarks(
                    M7_T3_FRAME_IDX,
                    "pose",
                    "body",
                    pose_res.pose_landmarks.landmark,
                )

            # Hands
            if hands_res.multi_hand_landmarks:
                for lm, handed in zip(
                    hands_res.multi_hand_landmarks,
                    hands_res.multi_handedness
                ):
                    label = handed.classification[0].label if handed.classification else "unknown"
                    mp_drawing.draw_landmarks(
                        frame,
                        lm,
                        mp_hands.HAND_CONNECTIONS,
                    )
                    m7_log_landmarks(
                        M7_T3_FRAME_IDX,
                        "hand",
                        label,
                        lm.landmark,
                    )
        else:
            cv.putText(
                frame,
                "Install 'mediapipe' for Task 3 demo",
                (20, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )

        M7_T3_FRAME_IDX += 1

        ok, buffer = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() +
               b"\r\n")
        
@app.route("/module/7/task3/video_feed")
@login_required
def module7_task3_video_feed():
    return Response(
        module7_task3_gen(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )



# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    # For local dev
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)