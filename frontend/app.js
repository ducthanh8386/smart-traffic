const LOG_REFRESH_INTERVAL_MS = 1500;

const state = {
  sessionId: null,
  isRunning: false,
  isBusy: false,
  lastLogFetchAt: 0,
};

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

const DEMO_PRESETS = {
  fast: { model: "yolov8n.pt", confidence: "0.40", frameSkip: "3" },
  balanced: { model: "yolov8n.pt", confidence: "0.35", frameSkip: "2" },
  accurate: { model: "yolov8s.pt", confidence: "0.30", frameSkip: "1" },
};

elements.form.addEventListener("submit", startProcessing);
elements.stopBtn.addEventListener("click", () => stopProcessing(true));
elements.refreshLogsBtn.addEventListener("click", () => loadViolationLogs(true));
elements.videoFrame.addEventListener("load", updateVideoFrameAspectRatio);
elements.demoPreset.addEventListener("change", applyDemoPreset);

loadViolationLogs(true);
loadModelOptions();
applyDemoPreset();
drawDensityChart([]);

async function startProcessing(event) {
  event.preventDefault();
  if (!elements.videoInput.files.length) {
    setStatus("Hay chon video truoc khi bat dau.");
    return;
  }
  const modelError = validateSelectedModel();
  if (modelError) {
    setStatus(modelError);
    return;
  }

  await stopProcessing(false);
  setControlsEnabled(false);
  resetSessionSummary();
  setStatus("Dang tai video va nap YOLO...");

  try {
    const response = await fetch("/api/sessions", {
      method: "POST",
      body: buildSessionFormData(),
    });
    const payload = await readJsonResponse(response);

    state.sessionId = payload.session_id;
    state.isRunning = true;
    state.lastLogFetchAt = 0;
    updateLightBadge(elements.trafficLight.value);
    setStatus("Dang xu ly video...");
    requestAnimationFrame(processLoop);
  } catch (error) {
    setStatus(`Khong the bat dau: ${error.message}`);
    setControlsEnabled(true);
  }
}

async function stopProcessing(updateStatus) {
  state.isRunning = false;
  state.isBusy = false;

  if (state.sessionId) {
    await fetch(`/api/sessions/${state.sessionId}`, { method: "DELETE" }).catch((error) => {
      if (updateStatus) {
        setStatus(`Khong the dung xu ly: ${error.message}`);
      }
    });
  }
  state.sessionId = null;
  setControlsEnabled(true);

  if (updateStatus) {
    setStatus("Da dung xu ly.");
  }
}

async function processLoop() {
  if (!state.isRunning || !state.sessionId || state.isBusy) {
    return;
  }

  state.isBusy = true;
  try {
    const response = await fetch(`/api/sessions/${state.sessionId}/next-frame`, { method: "POST" });
    const payload = await readJsonResponse(response);

    if (payload.done) {
      await finishProcessing("Da xu ly xong video.");
      return;
    }

    renderFrame(payload.frame);
    renderMetrics(payload.metadata);
    renderSessionSummary(payload.summary);
    drawDensityChart(payload.density_history || []);
    await loadViolationLogs(false);
  } catch (error) {
    const message = `Da dung xu ly: ${error.message}`;
    await stopProcessing(false);
    setStatus(message);
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
  setControlsEnabled(true);
  setStatus(message);
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
  return data;
}

function applyDemoPreset() {
  const preset = DEMO_PRESETS[elements.demoPreset.value] || DEMO_PRESETS.balanced;
  elements.modelPath.value = preset.model;
  elements.confidenceThreshold.value = preset.confidence;
  elements.frameSkip.value = preset.frameSkip;
}

async function loadModelOptions() {
  try {
    const response = await fetch("/api/models");
    const payload = await readJsonResponse(response);
    const models = Array.isArray(payload.models) ? payload.models : [];
    if (!models.length) {
      return;
    }
    elements.modelPath.innerHTML = models
      .map((model) => `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`)
      .join("");
  } catch {
    // Keep the static defaults if the metadata endpoint is unavailable.
  }
}

function validateSelectedModel() {
  const customModel = elements.customModel.value.trim();
  if (!customModel) {
    return "";
  }
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
  if (!width || !height) {
    return;
  }

  elements.videoFrame.parentElement.style.setProperty("--video-aspect-ratio", `${width} / ${height}`);
}

function renderMetrics(metadata) {
  elements.totalVehicles.textContent = metadata.total_current_vehicles ?? 0;
  elements.carCount.textContent = metadata.car ?? 0;
  elements.motorcycleCount.textContent = metadata.motorcycle ?? 0;
  elements.busCount.textContent = metadata.bus ?? 0;
  elements.truckCount.textContent = metadata.truck ?? 0;
  elements.roiCount.textContent = metadata.vehicle_count_roi ?? 0;
  elements.densityValue.textContent = `${formatNumber(metadata.density_percent)}%`;
  if (elements.pcuDensity) {
    const pcuTotal = formatNumber(metadata.pcu_total ?? 0);
    const pcuPercent = formatNumber(metadata.pcu_density_percent ?? 0);
    elements.pcuDensity.textContent = `${pcuTotal} PCU (${pcuPercent}%)`;
  }
  if (elements.motorcycleRatio) {
    elements.motorcycleRatio.textContent = `${formatNumber(metadata.motorcycle_ratio_percent ?? 0)}%`;
  }
  elements.fpsValue.textContent = `FPS ${formatNumber(metadata.fps)}`;
  elements.trafficStatus.textContent = metadata.traffic_status || "Binh thuong";
  elements.recommendation.textContent = metadata.recommendation || "Luu luong on dinh.";
  updateLightBadge(metadata.traffic_light || elements.trafficLight.value);
}

function renderSessionSummary(summary) {
  if (!summary) {
    return;
  }
  const averageCounts = summary.average_class_counts || {};
  const averageVehicles =
    Number(averageCounts.car || 0) +
    Number(averageCounts.motorcycle || 0) +
    Number(averageCounts.bus || 0) +
    Number(averageCounts.truck || 0);

  elements.processedFrames.textContent = summary.processed_frames ?? 0;
  elements.totalViolations.textContent = summary.total_violations ?? 0;
  elements.averageDensity.textContent = `${formatNumber(summary.average_density)}%`;
  elements.maxDensity.textContent = `${formatNumber(summary.max_density)}%`;
  elements.averageFps.textContent = formatNumber(summary.average_fps);
  elements.averageVehicles.textContent = formatNumber(averageVehicles);
}

function resetSessionSummary() {
  renderSessionSummary({
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
  if (!force && now - state.lastLogFetchAt < LOG_REFRESH_INTERVAL_MS) {
    return;
  }
  state.lastLogFetchAt = now;

  try {
    const response = await fetch("/api/violations");
    renderViolationRows(await readJsonResponse(response));
  } catch {
    renderViolationRows([]);
  }
}

function renderViolationRows(rows) {
  if (!rows.length) {
    elements.violationRows.innerHTML = `<tr><td colspan="6">Chua co vi pham.</td></tr>`;
    return;
  }

  elements.violationRows.innerHTML = rows
    .slice()
    .reverse()
    .map((row) => {
      const evidence = renderEvidenceCell(row.evidence_path);
      return `
        <tr>
          <td>${escapeHtml(row.timestamp)}</td>
          <td>${escapeHtml(row.track_id)}</td>
          <td>${escapeHtml(row.class_name)}</td>
          <td>${escapeHtml(row.violation_type)}</td>
          <td>${escapeHtml(row.confidence)}</td>
          <td>${evidence}</td>
        </tr>
      `;
    })
    .join("");
}

function drawDensityChart(values) {
  const canvas = elements.densityChart;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const padding = 28;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f7f9fc";
  ctx.fillRect(0, 0, width, height);
  drawGrid(ctx, width, height, padding);

  if (!values.length) {
    drawNoData(ctx, width, height);
    return;
  }

  const points = values.map((value, index) => {
    const x = padding + (index / Math.max(values.length - 1, 1)) * (width - padding * 2);
    const y = height - padding - (Math.min(Number(value), 100) / 100) * (height - padding * 2);
    return { x, y };
  });

  ctx.strokeStyle = "#0f8f7a";
  ctx.lineWidth = 3;
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point.x, point.y);
    else ctx.lineTo(point.x, point.y);
  });
  ctx.stroke();

  const last = points[points.length - 1];
  ctx.fillStyle = "#0f8f7a";
  ctx.beginPath();
  ctx.arc(last.x, last.y, 4, 0, Math.PI * 2);
  ctx.fill();
}

function drawGrid(ctx, width, height, padding) {
  ctx.strokeStyle = "#d9e1ea";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#687789";
  ctx.font = "12px Segoe UI";

  [0, 25, 50, 75, 100].forEach((value) => {
    const y = height - padding - (value / 100) * (height - padding * 2);
    ctx.beginPath();
    ctx.moveTo(padding, y);
    ctx.lineTo(width - padding, y);
    ctx.stroke();
    ctx.fillText(`${value}%`, 4, y + 4);
  });
}

function drawNoData(ctx, width, height) {
  ctx.fillStyle = "#687789";
  ctx.font = "14px Segoe UI";
  ctx.textAlign = "center";
  ctx.fillText("Chua co du lieu mat do", width / 2, height / 2);
  ctx.textAlign = "left";
}

function setControlsEnabled(enabled) {
  elements.startBtn.disabled = !enabled;
  elements.stopBtn.disabled = enabled;
}

function setStatus(message) {
  elements.statusText.textContent = message;
}

function updateLightBadge(value) {
  elements.lightBadge.textContent = value;
  elements.lightBadge.className = "badge";
  if (value === "GREEN") elements.lightBadge.classList.add("badge-green");
  else if (value === "YELLOW") elements.lightBadge.classList.add("badge-yellow");
  else if (value === "NONE") elements.lightBadge.classList.add("badge-none");
  else elements.lightBadge.classList.add("badge-red");
}

async function readJsonResponse(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(formatApiError(payload.detail) || `API request failed (${response.status}).`);
  }
  return payload;
}

function formatApiError(detail) {
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg || item.message || String(item)).join("; ");
  }
  if (detail && typeof detail === "object") {
    return detail.message || JSON.stringify(detail);
  }
  return String(detail || "");
}

function formatNumber(value) {
  return Number(value || 0).toFixed(1);
}

function shortPath(path) {
  return String(path).split(/[\\/]/).slice(-3).join("/");
}

function renderEvidenceCell(path) {
  if (!path) {
    return "";
  }
  const value = String(path);
  const label = escapeHtml(shortPath(value));
  const title = escapeHtml(value);
  if (value.startsWith("/api/evidence/")) {
    return `<a href="${escapeHtml(value)}" target="_blank" rel="noopener" title="${title}">${label}</a>`;
  }
  return `<span title="${title}">${label}</span>`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
