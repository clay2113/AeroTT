const $ = (id) => document.getElementById(id);

const els = {
  serverStatus: $("serverStatus"),
  imageUpload: $("imageUpload"),
  uploadButton: $("uploadButton"),
  analyzeImageButton: $("analyzeImageButton"),
  startButton: $("startButton"),
  stopButton: $("stopButton"),
  sourceVideo: $("sourceVideo"),
  sourceCanvas: $("sourceCanvas"),
  segmentationCanvas: $("segmentationCanvas"),
  classifierCanvas: $("classifierCanvas"),
  cameraPlaceholder: $("cameraPlaceholder"),
  segmentationPlaceholder: $("segmentationPlaceholder"),
  classifierPlaceholder: $("classifierPlaceholder"),
  videoSize: $("videoSize"),
  detectionStatus: $("detectionStatus"),
  classifierStatus: $("classifierStatus"),
  liveDescription: $("liveDescription"),
  areaValue: $("areaValue"),
  cdaValue: $("cdaValue"),
  pixelValue: $("pixelValue"),
  referenceValue: $("referenceValue"),
  fpsValue: $("fpsValue"),
  autoCalibration: $("autoCalibration"),
  referenceMeters: $("referenceMeters"),
  referencePixels: $("referencePixels"),
  calibrationState: $("calibrationState"),
  calibrateButton: $("calibrateButton"),
  clearCalibrationButton: $("clearCalibrationButton"),
  dragCoefficient: $("dragCoefficient"),
  confidence: $("confidence"),
  confidenceValue: $("confidenceValue"),
  imageSize: $("imageSize"),
  includeBicycle: $("includeBicycle"),
  modelDevice: $("modelDevice"),
  detectionsValue: $("detectionsValue"),
  bestBoxValue: $("bestBoxValue"),
  processingValue: $("processingValue"),
  scaleValue: $("scaleValue"),
  estimateView: $("estimateView"),
  speedView: $("speedView"),
  speedCda: $("speedCda"),
  speedPower: $("speedPower"),
  speedTemperature: $("speedTemperature"),
  speedAltitude: $("speedAltitude"),
  speedCrr: $("speedCrr"),
  speedEfficiency: $("speedEfficiency"),
  speedMass: $("speedMass"),
  speedCdaSource: $("speedCdaSource"),
  projectedSpeed: $("projectedSpeed"),
  speedAeroPower: $("speedAeroPower"),
  speedRollingPower: $("speedRollingPower"),
  speedAirDensity: $("speedAirDensity"),
};

const sourceCtx = els.sourceCanvas.getContext("2d", { alpha: false });
const segmentationCtx = els.segmentationCanvas.getContext("2d", { alpha: false });
const classifierCtx = els.classifierCanvas.getContext("2d", { alpha: false });
const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d", { alpha: false });

let stream = null;
let drawFrame = 0;
let analysisTimer = 0;
let analyzing = false;
let calibrationArmed = false;
let calibrationPoints = [];
let smoothedArea = null;
let smoothedCda = null;
let lastMetrics = null;
let activeSource = "none";
let uploadedImageName = "";
let analysisEpoch = 0;
let pendingStillAnalysis = false;

const formatArea = (value) => (Number.isFinite(value) && value > 0 ? value.toFixed(4) : "--");
const formatInteger = (value) => (Number.isFinite(value) ? Math.round(value).toLocaleString() : "--");
const formatFps = (value) => (Number.isFinite(value) && value > 0 ? value.toFixed(1) : "--");
const formatPhysics = (value, digits = 2) => (Number.isFinite(value) ? value.toFixed(digits) : "--");

function readNumber(element, fallback) {
  const value = Number.parseFloat(element.value);
  return Number.isFinite(value) ? value : fallback;
}

function calculateSpeedProjection(cda = smoothedCda) {
  const selectedCda = Number.isFinite(cda) && cda > 0 ? cda : readNumber(els.speedCda, 0.2729);
  const temperature = readNumber(els.speedTemperature, 20);
  const altitude = readNumber(els.speedAltitude, 0);
  const crr = Math.max(0, readNumber(els.speedCrr, 0.00366));
  const efficiency = Math.max(0.01, readNumber(els.speedEfficiency, 96.5) / 100);
  const mass = Math.max(1, readNumber(els.speedMass, 80));
  const riderPower = Math.max(0, readNumber(els.speedPower, 200));
  const kelvin = Math.max(180, temperature + 273.15);
  const pressure = 101325 * Math.pow(Math.max(0.2, 1 - 2.25577e-5 * altitude), 5.25588);
  const density = pressure / (287.05 * kelvin);
  const wheelPower = riderPower * efficiency;
  let low = 0;
  let high = 100;
  for (let index = 0; index < 48; index += 1) {
    const speed = (low + high) / 2;
    const requiredPower = 0.5 * density * selectedCda * speed ** 3 + crr * mass * 9.80665 * speed;
    if (requiredPower > wheelPower) high = speed;
    else low = speed;
  }
  const rollingPower = crr * mass * 9.80665 * low;
  const aeroPower = 0.5 * density * selectedCda * low ** 3;
  els.speedCda.value = selectedCda.toFixed(4);
  els.speedCdaSource.textContent = Number.isFinite(smoothedCda) && smoothedCda > 0 ? "Using latest camera CdA" : "Using manual CdA";
  els.projectedSpeed.textContent = formatPhysics(low * 3.6, 1);
  els.speedAeroPower.textContent = formatPhysics(aeroPower, 0);
  els.speedRollingPower.textContent = formatPhysics(rollingPower, 0);
  els.speedAirDensity.textContent = formatPhysics(density, 3);
}

function setStatus(text, kind = "") {
  els.serverStatus.textContent = text;
  els.serverStatus.className = `status-pill ${kind}`.trim();
}

function setPlaceholders(cameraVisible, segmentationVisible, classifierVisible = segmentationVisible) {
  els.cameraPlaceholder.classList.toggle("hidden", cameraVisible);
  els.segmentationPlaceholder.classList.toggle("hidden", segmentationVisible);
  els.classifierPlaceholder.classList.toggle("hidden", classifierVisible);
}

function syncCalibrationControls() {
  const isAuto = els.autoCalibration.checked;
  const hasFrame = activeSource !== "none";
  els.referencePixels.disabled = isAuto;
  els.calibrationState.textContent = isAuto ? "Auto" : "Manual";
  els.calibrateButton.disabled = isAuto || !hasFrame;
  els.analyzeImageButton.disabled = activeSource !== "image" || analyzing;
}

function updateReferencePixelsFromPoints() {
  if (calibrationPoints.length !== 2) return;
  const [a, b] = calibrationPoints;
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const pixels = Math.sqrt(dx * dx + dy * dy);
  els.referencePixels.value = pixels.toFixed(0);
  els.referenceValue.textContent = pixels.toFixed(0);
}

function getCanvasPoint(event) {
  const rect = els.sourceCanvas.getBoundingClientRect();
  const scaleX = els.sourceCanvas.width / rect.width;
  const scaleY = els.sourceCanvas.height / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

function drawCalibration() {
  if (!calibrationPoints.length) return;

  sourceCtx.save();
  sourceCtx.lineWidth = Math.max(2, els.sourceCanvas.width / 320);
  sourceCtx.strokeStyle = "#f0b84e";
  sourceCtx.fillStyle = "#f0b84e";

  if (calibrationPoints.length === 2) {
    sourceCtx.beginPath();
    sourceCtx.moveTo(calibrationPoints[0].x, calibrationPoints[0].y);
    sourceCtx.lineTo(calibrationPoints[1].x, calibrationPoints[1].y);
    sourceCtx.stroke();
  }

  for (const point of calibrationPoints) {
    sourceCtx.beginPath();
    sourceCtx.arc(point.x, point.y, Math.max(5, els.sourceCanvas.width / 120), 0, Math.PI * 2);
    sourceCtx.fill();
  }

  sourceCtx.restore();
}

function drawSource() {
  if (activeSource !== "camera" || !stream || els.sourceVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    drawFrame = requestAnimationFrame(drawSource);
    return;
  }

  const { videoWidth, videoHeight } = els.sourceVideo;
  if (videoWidth && videoHeight) {
    if (els.sourceCanvas.width !== videoWidth || els.sourceCanvas.height !== videoHeight) {
      els.sourceCanvas.width = videoWidth;
      els.sourceCanvas.height = videoHeight;
      els.segmentationCanvas.width = videoWidth;
      els.segmentationCanvas.height = videoHeight;
      els.classifierCanvas.width = videoWidth;
      els.classifierCanvas.height = videoHeight;
      els.videoSize.textContent = `${videoWidth} x ${videoHeight}`;
    }

    sourceCtx.drawImage(els.sourceVideo, 0, 0, videoWidth, videoHeight);
    drawCalibration();
  }

  drawFrame = requestAnimationFrame(drawSource);
}

function captureFrame() {
  const sourceWidth = els.sourceCanvas.width;
  const sourceHeight = els.sourceCanvas.height;
  if (!sourceWidth || !sourceHeight) return null;

  const maxWidth = 640;
  const scale = Math.min(1, maxWidth / sourceWidth);
  captureCanvas.width = Math.max(1, Math.round(sourceWidth * scale));
  captureCanvas.height = Math.max(1, Math.round(sourceHeight * scale));
  captureCtx.drawImage(els.sourceCanvas, 0, 0, captureCanvas.width, captureCanvas.height);
  return {
    image: captureCanvas.toDataURL("image/jpeg", 0.78),
    scale,
  };
}

function getAnalysisPayload(capture) {
  const manualReference = Number.parseFloat(els.referencePixels.value || "0") * capture.scale;
  return {
    image: capture.image,
    calibrationMode: els.autoCalibration.checked ? "auto" : "manual",
    referenceRealWidthM: Number.parseFloat(els.referenceMeters.value || "0.42"),
    referencePixelWidth: manualReference,
    dragCoefficient: Number.parseFloat(els.dragCoefficient.value || "0.72"),
    confidence: Number.parseFloat(els.confidence.value || "0.25"),
    imageSize: Number.parseInt(els.imageSize.value, 10),
    includeBicycle: els.includeBicycle.checked,
  };
}

function resetSmoothedMetrics() {
  smoothedArea = null;
  smoothedCda = null;
}

function drawSegmentation(dataUrl) {
  const image = new Image();
  image.onload = () => {
    segmentationCtx.clearRect(0, 0, els.segmentationCanvas.width, els.segmentationCanvas.height);
    segmentationCtx.drawImage(image, 0, 0, els.segmentationCanvas.width, els.segmentationCanvas.height);
    setPlaceholders(true, true);
  };
  image.src = dataUrl;
}

function drawClassifier(dataUrl) {
  const image = new Image();
  image.onload = () => {
    classifierCtx.clearRect(0, 0, els.classifierCanvas.width, els.classifierCanvas.height);
    classifierCtx.drawImage(image, 0, 0, els.classifierCanvas.width, els.classifierCanvas.height);
    els.classifierPlaceholder.classList.add("hidden");
  };
  image.src = dataUrl;
}

function updateMetrics(metrics, description) {
  const alpha = 0.34;
  smoothedArea = smoothedArea == null ? metrics.frontalAreaM2 : smoothedArea * (1 - alpha) + metrics.frontalAreaM2 * alpha;
  smoothedCda = smoothedCda == null ? metrics.cdaM2 : smoothedCda * (1 - alpha) + metrics.cdaM2 * alpha;

  els.areaValue.textContent = formatArea(smoothedArea);
  els.cdaValue.textContent = formatArea(smoothedCda);
  els.pixelValue.textContent = formatInteger(metrics.maskPixels);
  els.referenceValue.textContent = formatInteger(metrics.referencePixelWidth / (lastMetrics?.captureScale || 1));
  els.fpsValue.textContent = formatFps(metrics.backendFps);
  els.processingValue.textContent = `${Math.round(metrics.processingMs)} ms`;
  els.scaleValue.textContent = metrics.metersPerPixel > 0 ? `${(metrics.metersPerPixel * 1000).toFixed(2)} mm/px` : "--";
  els.modelDevice.textContent = String(metrics.device || "cpu").toUpperCase();

  const detectionText = metrics.detections?.length
    ? metrics.detections.map((item) => `${item.label} ${(item.confidence * 100).toFixed(0)}%`).join(", ")
    : "No mask";
  els.detectionsValue.textContent = detectionText;
  els.detectionStatus.textContent = detectionText;

  const bestBox = metrics.detections?.[0]?.box;
  els.bestBoxValue.textContent = bestBox ? `${bestBox.x2 - bestBox.x1} x ${bestBox.y2 - bestBox.y1}` : "--";
  els.classifierStatus.textContent = metrics.detections?.length ? `${metrics.detections.length} box${metrics.detections.length === 1 ? "" : "es"}` : "No boxes";
  els.liveDescription.textContent = description || "No live description yet.";
  if (Number.isFinite(smoothedCda) && smoothedCda > 0) {
    localStorage.setItem("aerott.latestCda", smoothedCda.toFixed(6));
  }
  calculateSpeedProjection(smoothedCda);
}

async function analyzeCurrentFrame() {
  if (activeSource !== "camera" || !stream || analyzing) return;
  const capture = captureFrame();
  if (!capture) return;

  await analyzeCapture(capture, "Live", analysisEpoch);
}

async function analyzeStillFrame() {
  if (activeSource !== "image") return;
  if (analyzing) {
    pendingStillAnalysis = true;
    return;
  }
  const capture = captureFrame();
  if (!capture) return;

  resetSmoothedMetrics();
  await analyzeCapture(capture, "Image analyzed", analysisEpoch);
}

async function analyzeCapture(capture, successText, epoch) {
  analyzing = true;
  syncCalibrationControls();
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(getAnalysisPayload(capture)),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "Analysis failed");
    }
    if (epoch !== analysisEpoch) {
      return;
    }

    lastMetrics = { ...data.metrics, captureScale: capture.scale };
    drawSegmentation(data.overlayImage);
    drawClassifier(data.classifierImage);
    updateMetrics(lastMetrics, data.description);
    setStatus(successText, "ready");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    analyzing = false;
    syncCalibrationControls();
    if (pendingStillAnalysis && activeSource === "image") {
      pendingStillAnalysis = false;
      window.setTimeout(analyzeStillFrame, 0);
    }
  }
}

function startAnalysisLoop() {
  window.clearInterval(analysisTimer);
  analysisTimer = window.setInterval(analyzeCurrentFrame, 650);
  analyzeCurrentFrame();
}

async function startCamera() {
  if (stream) return;

  try {
    const cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: "user",
      },
    });

    activeSource = "camera";
    analysisEpoch += 1;
    pendingStillAnalysis = false;
    uploadedImageName = "";
    resetSmoothedMetrics();
    stream = cameraStream;
    els.sourceVideo.srcObject = stream;
    await els.sourceVideo.play();

    els.startButton.disabled = true;
    els.stopButton.disabled = false;
    setPlaceholders(true, false);
    setStatus("Camera live", "ready");
    syncCalibrationControls();
    drawSource();
    startAnalysisLoop();
  } catch (error) {
    setStatus(error.message || "Camera blocked", "error");
  }
}

function stopCamera() {
  window.clearInterval(analysisTimer);
  window.cancelAnimationFrame(drawFrame);
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
  }

  stream = null;
  activeSource = "none";
  analysisEpoch += 1;
  pendingStillAnalysis = false;
  uploadedImageName = "";
  els.sourceVideo.srcObject = null;
  els.startButton.disabled = false;
  els.stopButton.disabled = true;
  els.analyzeImageButton.disabled = true;
  calibrationArmed = false;
  syncCalibrationControls();
  setPlaceholders(false, false);
  els.liveDescription.textContent = "Start the camera or upload a photo to generate a rider summary.";
  els.classifierStatus.textContent = "No boxes";
  setStatus("Stopped", "warn");
}

function stopCameraTracksOnly() {
  window.clearInterval(analysisTimer);
  window.cancelAnimationFrame(drawFrame);
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
  }
  stream = null;
  els.sourceVideo.srcObject = null;
  els.startButton.disabled = false;
  els.stopButton.disabled = true;
  activeSource = "none";
  analysisEpoch += 1;
}

function loadUploadedImage(file) {
  if (!file) return;

  if (!file.type.startsWith("image/")) {
    setStatus("Choose an image file", "error");
    return;
  }

  stopCameraTracksOnly();
  const reader = new FileReader();
  reader.onload = () => {
    const image = new Image();
    image.onload = () => {
      activeSource = "image";
      analysisEpoch += 1;
      pendingStillAnalysis = false;
      uploadedImageName = file.name;
      calibrationArmed = false;
      calibrationPoints = [];
      resetSmoothedMetrics();

      els.sourceCanvas.width = image.naturalWidth;
      els.sourceCanvas.height = image.naturalHeight;
      els.segmentationCanvas.width = image.naturalWidth;
      els.segmentationCanvas.height = image.naturalHeight;
      els.classifierCanvas.width = image.naturalWidth;
      els.classifierCanvas.height = image.naturalHeight;

      sourceCtx.clearRect(0, 0, els.sourceCanvas.width, els.sourceCanvas.height);
      sourceCtx.drawImage(image, 0, 0, image.naturalWidth, image.naturalHeight);
      segmentationCtx.clearRect(0, 0, els.segmentationCanvas.width, els.segmentationCanvas.height);
      classifierCtx.clearRect(0, 0, els.classifierCanvas.width, els.classifierCanvas.height);

      els.videoSize.textContent = `${image.naturalWidth} x ${image.naturalHeight}`;
      els.detectionStatus.textContent = "Ready";
      els.classifierStatus.textContent = "Ready";
      els.liveDescription.textContent = `Loaded ${uploadedImageName}. Running segmentation now.`;
      setPlaceholders(true, false, false);
      setStatus("Image loaded", "ready");
      syncCalibrationControls();
      analyzeStillFrame();
    };
    image.onerror = () => setStatus("Could not read image", "error");
    image.src = reader.result;
  };
  reader.onerror = () => setStatus("Could not load file", "error");
  reader.readAsDataURL(file);
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();
    if (!data.modelExists) {
      setStatus("Missing model", "error");
      return;
    }
    els.modelDevice.textContent = String(data.device || "cpu").toUpperCase();
    setStatus(data.modelLoaded ? "Model ready" : "Model found", "ready");
  } catch {
    setStatus("Server offline", "error");
  }
}

els.startButton.addEventListener("click", startCamera);
els.stopButton.addEventListener("click", stopCamera);
els.uploadButton.addEventListener("click", () => {
  els.imageUpload.value = "";
  els.imageUpload.click();
});
els.imageUpload.addEventListener("change", () => loadUploadedImage(els.imageUpload.files?.[0]));
els.analyzeImageButton.addEventListener("click", analyzeStillFrame);

els.autoCalibration.addEventListener("change", () => {
  calibrationArmed = false;
  syncCalibrationControls();
});

els.referencePixels.addEventListener("input", () => {
  els.autoCalibration.checked = false;
  syncCalibrationControls();
});

els.confidence.addEventListener("input", () => {
  els.confidenceValue.textContent = Number.parseFloat(els.confidence.value).toFixed(2);
});

[els.speedCda, els.speedPower, els.speedTemperature, els.speedAltitude, els.speedCrr, els.speedEfficiency, els.speedMass]
  .forEach((input) => input.addEventListener("input", () => calculateSpeedProjection()));

document.querySelectorAll(".view-link").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".view-link").forEach((item) => item.classList.remove("active"));
    document.querySelectorAll(".view-panel").forEach((panel) => {
      panel.hidden = panel.id !== button.dataset.view;
      panel.classList.toggle("active-view", panel.id === button.dataset.view);
    });
    button.classList.add("active");
    if (button.dataset.view === "speedView") calculateSpeedProjection();
  });
});

els.calibrateButton.addEventListener("click", () => {
  calibrationArmed = !calibrationArmed;
  calibrationPoints = [];
  els.calibrationState.textContent = calibrationArmed ? "Point 1" : "Manual";
});

els.clearCalibrationButton.addEventListener("click", () => {
  calibrationPoints = [];
  els.referencePixels.value = "0";
  els.autoCalibration.checked = true;
  calibrationArmed = false;
  syncCalibrationControls();
});

els.sourceCanvas.addEventListener("click", (event) => {
  if (!calibrationArmed || els.autoCalibration.checked) return;
  calibrationPoints.push(getCanvasPoint(event));
  if (calibrationPoints.length === 1) {
    els.calibrationState.textContent = "Point 2";
  }
  if (calibrationPoints.length === 2) {
    updateReferencePixelsFromPoints();
    calibrationArmed = false;
    els.calibrationState.textContent = "Manual";
    if (activeSource === "image") {
      analyzeStillFrame();
    }
  }
});

window.addEventListener("beforeunload", stopCamera);

syncCalibrationControls();
calculateSpeedProjection();
refreshStatus();
