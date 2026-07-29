export const state = {
  sessionId: null,
  isRunning: false,
  isBusy: false,
  lastLogFetchAt: 0,
  drawingMode: "none", // "none", "roi", "line"
  customRoiPoints: [], // [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
  customLinePoints: [], // [(x1,y1), (x2,y2)]
};

export const DEMO_PRESETS = {
  fast: { model: "yolov8n.pt", confidence: "0.40", frameSkip: "3" },
  balanced: { model: "yolov8n.pt", confidence: "0.35", frameSkip: "2" },
  accurate: { model: "yolov8s.pt", confidence: "0.30", frameSkip: "1" },
};
