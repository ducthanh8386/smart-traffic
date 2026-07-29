export function renderMetrics(elements, metadata) {
  if (!metadata) return;
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
  updateLightBadge(elements.lightBadge, metadata.traffic_light || elements.trafficLight.value);
}

export function renderSessionSummary(elements, summary) {
  if (!summary) return;
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

export function updateLightBadge(badgeElement, value) {
  if (!badgeElement) return;
  badgeElement.textContent = value;
  badgeElement.className = "badge";
  if (value === "GREEN") badgeElement.classList.add("badge-green");
  else if (value === "YELLOW") badgeElement.classList.add("badge-yellow");
  else if (value === "NONE") badgeElement.classList.add("badge-none");
  else if (value === "AUTO") badgeElement.classList.add("badge-auto");
  else badgeElement.classList.add("badge-red");
}

export function renderViolationRows(tableBodyElement, rows) {
  if (!rows || !rows.length) {
    tableBodyElement.innerHTML = `<tr><td colspan="6">Chua co vi pham.</td></tr>`;
    return;
  }

  tableBodyElement.innerHTML = rows
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

export function setControlsEnabled(elements, enabled) {
  elements.startBtn.disabled = !enabled;
  elements.stopBtn.disabled = enabled;
}

export function setStatus(statusElement, message) {
  if (statusElement) statusElement.textContent = message;
}

function renderEvidenceCell(path) {
  if (!path) return "";
  const value = String(path);
  const label = escapeHtml(shortPath(value));
  const title = escapeHtml(value);
  if (value.startsWith("/api/evidence/")) {
    return `<a href="${escapeHtml(value)}" target="_blank" rel="noopener" title="${title}">${label}</a>`;
  }
  return `<span title="${title}">${label}</span>`;
}

function shortPath(path) {
  return String(path).split(/[\\/]/).slice(-3).join("/");
}

function formatNumber(value) {
  return Number(value || 0).toFixed(1);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
