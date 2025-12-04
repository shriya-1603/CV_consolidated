document.addEventListener("DOMContentLoaded", () => {
  const modeButtons = document.querySelectorAll(".m6-controls [data-mode]");
  const webcamImg   = document.getElementById("m6_webcam");
  const sam2Img     = document.getElementById("m6_sam2");
  const sam2Controls = document.getElementById("m6_sam2_controls");
  const slider      = document.getElementById("m6_slider");

  const SAM2_BASE = "/module/6/sam2_frame/";

  let currentFrame = 0;
  let totalFrames  = 199;   // matches your NPZ (0..198)
  let playing      = false;
  let playTimer    = null;

  // ------------- MODE SWITCHING -------------

  function setMode(mode) {
    // ping backend
    fetch(`/module/6/set_mode/${mode}`).catch(() => {});

    if (mode === "sam2") {
      // show SAM2 image + controls
      webcamImg.style.display = "none";
      sam2Img.style.display = "block";
      sam2Controls.style.display = "flex";
      loadFrame(currentFrame);
    } else {
      // show webcam stream
      sam2Img.style.display = "none";
      sam2Controls.style.display = "none";
      webcamImg.style.display = "block";

      // stop playback if we were playing
      if (playing) togglePlay();
    }

    // button styling
    modeButtons.forEach(btn => {
      const isActive = btn.dataset.mode === mode;
      btn.classList.toggle("ghost", !isActive);
    });
  }

  // ------------- SAM2 FRAME HELPERS -------------

  function loadFrame(n) {
    currentFrame = Math.max(0, Math.min(totalFrames - 1, n));
    sam2Img.src = `${SAM2_BASE}${currentFrame}?t=${Date.now()}`;
    slider.value = currentFrame;
  }

  function nextFrame() {
    if (currentFrame < totalFrames - 1) {
      loadFrame(currentFrame + 1);
    }
  }

  function prevFrame() {
    if (currentFrame > 0) {
      loadFrame(currentFrame - 1);
    }
  }

  function restart() {
    loadFrame(0);
  }

  function togglePlay() {
    playing = !playing;
    const playBtn = document.querySelector(".m6-sam2-controls [data-action='play']");
    if (playBtn) {
      playBtn.textContent = playing ? "⏸️ Pause" : "▶️ Play";
    }

    if (playing) {
      playTimer = setInterval(() => {
        if (currentFrame < totalFrames - 1) {
          nextFrame();
        } else {
          // reached end -> stop
          togglePlay();
        }
      }, 100); // 10 fps
    } else {
      clearInterval(playTimer);
      playTimer = null;
    }
  }

  // ------------- EVENT WIRING -------------

  // mode buttons
  modeButtons.forEach(btn => {
    btn.addEventListener("click", () => setMode(btn.dataset.mode));
  });

  // sam2 controls (prev / play / next / restart)
  sam2Controls.addEventListener("click", (e) => {
    const action = e.target.dataset.action;
    if (!action) return;

    if (action === "prev")     prevFrame();
    if (action === "next")     nextFrame();
    if (action === "restart")  restart();
    if (action === "play")     togglePlay();
  });

  // slider scrub
  slider.addEventListener("input", (e) => {
    const n = parseInt(e.target.value, 10);
    if (!Number.isNaN(n)) {
      loadFrame(n);
    }
  });

  // default mode on load
  setMode("marker");
});