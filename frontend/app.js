import { state, DEMO_PRESETS } from "./js/state.js";
import {
  fetchAvailableModels,
  fetchViolations,
  createSession,
  fetchNextFrame,
  stopSessionApi,
} from "./js/api.js";
import { drawDensityChart, InteractiveOverlay } from "./js/canvas.js";
import {
  renderMetrics,
  renderSessionSummary,
  renderViolationRows,
  setControlsEnabled,
  setStatus,
  updateLightBadge,
} from "./js/ui.js";

const LOG_REFRESH_INTERVAL_MS = 1500;

const elements = {
  form: document.getElementById("controlForm"),
  videoInput: document.getElementById("videoInput"),
  demoPreset: document.getElementById("demoPreset"),
  modelPath: document.getElementById("modelPath"),
  customModel: document.getElementById("customModel"),
  trafficLight: document.getElementById("trafficLight"),
  maxCapacity: document.getElementById("maxCapacity"),
  confidenceThreshold: document.getElementById("confidenceThreshold"),
  frameSkip: document.getElementById("frameSkip"),
  normalThreshold: document.getElementById("normalThreshold"),
  crowdedThreshold: document.getElementById("crowdedThreshold"),
  showBoxes: document.getElementById("showBoxes"),
  showRoi: document.getElementById("showRoi"),
  showLine: document.getElementById("showLine"),
  showLanes: document.getElementById("showLanes"),
  laneScenario: document.getElementById("laneScenario"),
  targetFilter: document.getElementById("targetFilter"),
  saveEvidence: document.getElementById("saveEvidence"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  statusText: document.getElementById("statusText"),
  lightBadge: document.getElementById("lightBadge"),
  videoFrame: document.getElementById("videoFrame"),
  emptyState: document.getElementById("emptyState"),
  densityChart: document.getElementById("densityChart"),
  totalVehicles: document.getElementById("totalVehicles"),
  carCount: document.getElementById("carCount"),
  motorcycleCount: document.getElementById("motorcycleCount"),
  busCount: document.getElementById("busCount"),
  truckCount: document.getElementById("truckCount"),
  densityValue: document.getElementById("densityValue"),
  processedFrames: document.getElementById("processedFrames"),
  totalViolations: document.getElementById("totalViolations"),
  averageDensity: document.getElementById("averageDensity"),
  maxDensity: document.getElementById("maxDensity"),
  averageFps: document.getElementById("averageFps"),
  averageVehicles: document.getElementById("averageVehicles"),
  fpsValue: document.getElementById("fpsValue"),
  roiCount: document.getElementById("roiCount"),
  pcuDensity: document.getElementById("pcuDensity"),
  motorcycleRatio: document.getElementById("motorcycleRatio"),
  trafficStatus: document.getElementById("trafficStatus"),
  recommendation: document.getElementById("recommendation"),
  violationRows: document.getElementById("violationRows"),
  refreshLogsBtn: document.getElementById("refreshLogsBtn"),
};

const overlayManager = new InteractiveOverlay(elements.videoFrame?.parentElement, (data) => {
  state.customRoiPoints = data.roiPoints;
  state.customLinePoints = data.linePoints;
});

elements.form?.addEventListener("submit", startProcessing);
elements.stopBtn?.addEventListener("click", () => stopProcessing(true));
elements.refreshLogsBtn?.addEventListener("click", () => loadViolationLogs(true));
elements.videoFrame?.addEventListener("load", updateVideoFrameAspectRatio);
elements.demoPreset?.addEventListener("change", applyDemoPreset);

loadViolationLogs(true);
loadModelOptions();
applyDemoPreset();
drawDensityChart(elements.densityChart, []);

async function startProcessing(event) {
  event.preventDefault();
  if (!elements.videoInput.files.length) {
    setStatus(elements.statusText, "Hay chon video truoc khi bat dau.");
    return;
  }
  const modelError = validateSelectedModel();
  if (modelError) {
    setStatus(elements.statusText, modelError);
    return;
  }

  await stopProcessing(false);
  setControlsEnabled(elements, false);
  resetSessionSummary();
  setStatus(elements.statusText, "Dang tai video va nap YOLO...");

  try {
    const payload = await createSession(buildSessionFormData());
    state.sessionId = payload.session_id;
    state.isRunning = true;
    state.lastLogFetchAt = 0;
    updateLightBadge(elements.lightBadge, elements.trafficLight.value);
    setStatus(elements.statusText, "Dang xu ly video...");
    requestAnimationFrame(processLoop);
  } catch (error) {
    setStatus(elements.statusText, `Khong the bat dau: ${error.message}`);
    setControlsEnabled(elements, true);
  }
}

async function stopProcessing(shouldUpdateStatus) {
  state.isRunning = false;
  state.isBusy = false;

  if (state.sessionId) {
    await stopSessionApi(state.sessionId);
  }
  state.sessionId = null;
  setControlsEnabled(elements, true);

  if (shouldUpdateStatus) {
    setStatus(elements.statusText, "Da dung xu ly.");
  }
}

async function processLoop() {
  if (!state.isRunning || !state.sessionId || state.isBusy) {
    return;
  }

  state.isBusy = true;
  try {
    const payload = await fetchNextFrame(state.sessionId);

    if (payload.done) {
      await finishProcessing("Da xu ly xong video.");
      return;
    }

    renderFrame(payload.frame);
    renderMetrics(elements, payload.metadata);
    renderSessionSummary(elements, payload.summary);
    drawDensityChart(elements.densityChart, payload.density_history || []);
    await loadViolationLogs(false);
  } catch (error) {
    const message = `Da dung xu ly: ${error.message}`;
    await stopProcessing(false);
    setStatus(elements.statusText, message);
    return;
  } finally {
    state.isBusy = false;
  }

  if (state.isRunning) {
    requestAnimationFrame(processLoop);
  }
}

async function finishProcessing(message) {
  state.isRunning = false;
  state.sessionId = null;
  setControlsEnabled(elements, true);
  setStatus(elements.statusText, message);
  await loadViolationLogs(true);
}

function buildSessionFormData() {
  const customModel = elements.customModel.value.trim();
  const data = new FormData();
  data.append("video", elements.videoInput.files[0]);
  data.append("model_path", customModel || elements.modelPath.value);
  data.append("traffic_light", elements.trafficLight.value);
  data.append("max_capacity", elements.maxCapacity.value);
  data.append("confidence_threshold", elements.confidenceThreshold.value);
  data.append("normal_threshold", elements.normalThreshold.value);
  data.append("crowded_threshold", elements.crowdedThreshold.value);
  data.append("frame_skip", elements.frameSkip.value);
  data.append("show_boxes", elements.showBoxes.checked);
  data.append("show_roi", elements.showRoi.checked);
  data.append("show_line", elements.showLine.checked);
  data.append("show_lanes", elements.showLanes ? elements.showLanes.checked : false);
  data.append("lane_scenario", elements.laneScenario ? elements.laneScenario.value : "none");
  data.append("target_classes", elements.targetFilter ? elements.targetFilter.value : "all");
  data.append("save_evidence", elements.saveEvidence.checked);

  if (state.customRoiPoints.length === 4) {
    data.append("custom_roi_json", JSON.stringify(state.customRoiPoints));
  }
  if (state.customLinePoints.length === 2) {
    data.append("custom_line_json", JSON.stringify(state.customLinePoints));
  }
  return data;
}

function applyDemoPreset() {
  const preset = DEMO_PRESETS[elements.demoPreset.value] || DEMO_PRESETS.balanced;
  elements.modelPath.value = preset.model;
  elements.confidenceThreshold.value = preset.confidence;
  elements.frameSkip.value = preset.frameSkip;
}

async function loadModelOptions() {
  const models = await fetchAvailableModels();
  if (!models.length) return;
  elements.modelPath.innerHTML = models
    .map((model) => `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`)
    .join("");
}

function validateSelectedModel() {
  const customModel = elements.customModel.value.trim();
  if (!customModel) return "";
  if (!customModel.endsWith(".pt")) {
    return "Model tuy chinh phai la file .pt trong thu muc models/.";
  }
  if (!customModel.startsWith("models/") || customModel.includes("..") || /^[a-zA-Z]:/.test(customModel)) {
    return "Duong dan model tuy chinh phai co dang models/my_model.pt.";
  }
  return "";
}

function renderFrame(frameDataUrl) {
  elements.videoFrame.src = frameDataUrl;
  elements.videoFrame.style.display = "block";
  elements.emptyState.style.display = "none";
}

function updateVideoFrameAspectRatio() {
  const width = elements.videoFrame.naturalWidth;
  const height = elements.videoFrame.naturalHeight;
  if (!width || !height) return;
  elements.videoFrame.parentElement.style.setProperty("--video-aspect-ratio", `${width} / ${height}`);
}

function resetSessionSummary() {
  renderSessionSummary(elements, {
    processed_frames: 0,
    total_violations: 0,
    average_density: 0,
    max_density: 0,
    average_fps: 0,
    average_class_counts: {},
  });
}

async function loadViolationLogs(force) {
  const now = Date.now();
  if (!force && now - state.lastLogFetchAt < LOG_REFRESH_INTERVAL_MS) return;
  state.lastLogFetchAt = now;
  const rows = await fetchViolations();
  renderViolationRows(elements.violationRows, rows);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
