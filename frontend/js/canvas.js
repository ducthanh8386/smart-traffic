export function drawDensityChart(canvas, values) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const padding = 28;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f7f9fc";
  ctx.fillRect(0, 0, width, height);
  drawGrid(ctx, width, height, padding);

  if (!values || !values.length) {
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

export class InteractiveOverlay {
  constructor(containerElement, onPointsUpdated) {
    this.container = containerElement;
    this.onPointsUpdated = onPointsUpdated;
    this.roiPoints = []; // [{x: ratio, y: ratio}]
    this.linePoints = []; // [{x: ratio, y: ratio}]
    this.activeMode = "none"; // "roi", "line", "none"
  }

  setMode(mode) {
    this.activeMode = mode;
    if (mode === "roi") this.roiPoints = [];
    if (mode === "line") this.linePoints = [];
  }

  clear() {
    this.roiPoints = [];
    this.linePoints = [];
    this.activeMode = "none";
  }

  addNormalizedPoint(xRatio, yRatio) {
    if (this.activeMode === "roi") {
      if (this.roiPoints.length < 4) {
        this.roiPoints.push([roundRatio(xRatio), roundRatio(yRatio)]);
      }
      if (this.roiPoints.length === 4) {
        this.activeMode = "none";
      }
    } else if (this.activeMode === "line") {
      if (this.linePoints.length < 2) {
        this.linePoints.push([roundRatio(xRatio), roundRatio(yRatio)]);
      }
      if (this.linePoints.length === 2) {
        this.activeMode = "none";
      }
    }
    if (this.onPointsUpdated) {
      this.onPointsUpdated({
        roiPoints: this.roiPoints,
        linePoints: this.linePoints,
      });
    }
  }
}

function roundRatio(val) {
  return Math.round(Math.min(Math.max(val, 0.0), 1.0) * 1000) / 1000;
}
