from __future__ import annotations

import base64
import asyncio
from contextlib import asynccontextmanager
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.runtime import configure_runtime

configure_runtime()

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from core.model_registry import list_available_models, resolve_model_path, to_project_model_path
from core.storage import get_violation_storage
from core.utils import ensure_dirs, load_config
from core.video_processor import VideoProcessor


CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"
FRONTEND_DIR = ROOT_DIR / "frontend"
UPLOAD_DIR = ROOT_DIR / "uploads"
EVIDENCE_DIR = ROOT_DIR / "evidence"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_ACTIVE_SESSIONS = 3
SESSION_TIMEOUT_SECONDS = 20 * 60
CLEANUP_INTERVAL_SECONDS = 60
VALID_LIGHTS = {"RED", "GREEN", "YELLOW", "NONE"}
VALID_LANE_SCENARIOS = {"none", "city_standard", "highway"}
VALID_UPLOAD_EXTENSIONS = {".mp4", ".avi", ".mov"}
VALID_CONTENT_TYPES = {
    "application/octet-stream",
}


class ProcessingSession:
    """State for one uploaded video."""

    def __init__(self, session_id: str, video_path: Path, processor: VideoProcessor, frame_skip: int = 1):
        self.session_id = session_id
        self.video_path = video_path
        self.processor = processor
        self.frame_skip = max(int(frame_skip), 1)
        self.frame_index = 0
        self.density_history: list[float] = []
        self.processed_frames = 0
        self.total_violations = 0
        self.fps_history: list[float] = []
        self.class_totals = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        self.lock = Lock()
        self.capture = cv2.VideoCapture(str(video_path))
        self.last_access = time.time()
        self.closed = False

        if not self.capture.isOpened():
            self.close(delete_file=True)
            raise ValueError("Cannot read this video file.")

    def next_frame(self) -> dict[str, Any]:
        """Process the next frame. Calls are serialized by a per-session lock."""
        with self.lock:
            self.last_access = time.time()
            frame = self._read_next_selected_frame()
            if frame is None:
                return {"done": True}

            frame = resize_frame(frame, max_width=960)
            processed_frame, metadata = self.processor.process_frame(
                frame,
                session_id=self.session_id,
                frame_index=self.frame_index,
            )
            self.density_history.append(round(float(metadata["density_percent"]), 2))
            self.density_history = self.density_history[-200:]
            self._record_summary(metadata)

            return {
                "done": False,
                "frame": encode_frame_to_base64(processed_frame),
                "metadata": metadata,
                "density_history": self.density_history,
                "frame_index": self.frame_index,
                "summary": self._summary_unlocked(),
            }

    def summary(self) -> dict[str, Any]:
        with self.lock:
            return self._summary_unlocked()

    def close(self, delete_file: bool = False) -> None:
        if self.closed:
            return
        self.closed = True
        if self.capture is not None:
            self.capture.release()
        if delete_file:
            self.video_path.unlink(missing_ok=True)

    def _read_next_selected_frame(self):
        while True:
            ok, frame = self.capture.read()
            if not ok:
                return None

            self.frame_index += 1
            if self.frame_index % self.frame_skip == 0:
                return frame

    def _record_summary(self, metadata: dict[str, Any]) -> None:
        self.processed_frames += 1
        self.total_violations += len(metadata.get("violations", []))
        self.fps_history.append(float(metadata.get("fps", 0) or 0))
        self.fps_history = self.fps_history[-200:]
        for class_name in self.class_totals:
            self.class_totals[class_name] += int(metadata.get(class_name, 0) or 0)

    def _summary_unlocked(self) -> dict[str, Any]:
        densities = self.density_history
        fps_values = self.fps_history
        processed = max(self.processed_frames, 1)
        average_counts = {
            class_name: round(total / processed, 2)
            for class_name, total in self.class_totals.items()
        }
        return {
            "session_id": self.session_id,
            "source_file": self.video_path.name,
            "processed_frames": self.processed_frames,
            "total_violations": self.total_violations,
            "average_density": round(sum(densities) / len(densities), 2) if densities else 0.0,
            "max_density": round(max(densities), 2) if densities else 0.0,
            "average_fps": round(sum(fps_values) / len(fps_values), 2) if fps_values else 0.0,
            "class_totals": dict(self.class_totals),
            "average_class_counts": average_counts,
        }


sessions: dict[str, ProcessingSession] = {}
sessions_lock = Lock()
cleanup_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cleanup_task
    ensure_dirs(ROOT_DIR)
    get_violation_storage(load_runtime_config()["violation_db_path"])
    cleanup_task = asyncio.create_task(cleanup_inactive_sessions_loop())
    yield
    if cleanup_task is not None:
        cleanup_task.cancel()
    cleanup_all_sessions()


app = FastAPI(title="SMARTTRAFFIC - AI API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        FRONTEND_DIR / "index.html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
def get_models() -> dict[str, list[str]]:
    return {"models": list_available_models()}


TARGET_CLASS_MAP = {
    "all": ["car", "motorcycle", "bus", "truck", "person"],
    "car_motorcycle": ["car", "motorcycle"],
    "vehicles_only": ["car", "motorcycle", "bus", "truck"],
    "car": ["car"],
    "motorcycle": ["motorcycle"],
}


@app.post("/api/sessions")
async def create_session(
    video: UploadFile = File(...),
    model_path: str = Form("yolov8n.pt"),
    traffic_light: str = Form("RED"),
    max_capacity: int = Form(30),
    confidence_threshold: float = Form(0.35),
    normal_threshold: int = Form(40),
    crowded_threshold: int = Form(70),
    show_boxes: bool = Form(True),
    show_roi: bool = Form(True),
    show_line: bool = Form(True),
    show_lanes: bool = Form(False),
    lane_scenario: str = Form("none"),
    target_classes: str = Form("all"),
    save_evidence: bool = Form(True),
    frame_skip: int = Form(1),
) -> dict[str, str]:
    """Create a processing session from an uploaded traffic video."""
    suffix = validate_upload(
        video,
        traffic_light,
        max_capacity,
        confidence_threshold,
        normal_threshold,
        crowded_threshold,
        frame_skip,
    )
    resolved_model_path = validate_model_path(model_path)
    enforce_session_limit()

    session_id = uuid4().hex
    video_path = UPLOAD_DIR / f"{session_id}{suffix}"
    try:
        await save_upload(video, video_path)
        config = build_runtime_config(normal_threshold, crowded_threshold, confidence_threshold)

        if lane_scenario == "highway":
            config["lanes"] = [
                {"name": "Lane 1 (Cao toc)", "allowed_classes": ["car", "bus", "truck", "motorcycle"], "roi_ratio": {"x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 1.0}},
                {"name": "Lane 2 (Cao toc)", "allowed_classes": ["car", "bus", "truck", "motorcycle"], "roi_ratio": {"x1": 0.5, "y1": 0.0, "x2": 1.0, "y2": 1.0}},
            ]
        elif lane_scenario == "city_standard":
            config["lanes"] = [
                {"name": "Lane Oto", "allowed_classes": ["car", "bus", "truck"], "roi_ratio": {"x1": 0.0, "y1": 0.0, "x2": 0.5, "y2": 1.0}},
                {"name": "Lane Xe May", "allowed_classes": ["motorcycle"], "roi_ratio": {"x1": 0.5, "y1": 0.0, "x2": 1.0, "y2": 1.0}},
            ]
        elif lane_scenario == "none":
            config["lanes"] = []

        parsed_target_classes = TARGET_CLASS_MAP.get(target_classes, TARGET_CLASS_MAP["all"])

        processor = VideoProcessor(
            config=config,
            model_path=to_project_model_path(resolved_model_path),
            traffic_light=traffic_light,
            max_capacity=max_capacity,
            show_boxes=show_boxes,
            show_roi=show_roi,
            show_line=show_line,
            show_lanes=show_lanes,
            save_evidence=save_evidence,
            target_classes=parsed_target_classes,
        )
        session = ProcessingSession(session_id, video_path, processor, frame_skip)
        with sessions_lock:
            sessions[session_id] = session
    except HTTPException:
        video_path.unlink(missing_ok=True)
        raise
    except ValueError as exc:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=safe_error(str(exc))) from exc
    except Exception as exc:
        video_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Cannot create processing session.") from exc
    finally:
        await video.close()

    return {"session_id": session_id}


@app.post("/api/sessions/{session_id}/next-frame")
def process_next_frame(session_id: str) -> dict[str, Any]:
    session = get_session(session_id)
    try:
        payload = session.next_frame()
    except Exception as exc:
        cleanup_session(session_id, delete_file=True)
        raise HTTPException(status_code=500, detail="Frame processing failed.") from exc

    if payload.get("done"):
        cleanup_session(session_id, delete_file=True)
    return payload


@app.get("/api/sessions/{session_id}/summary")
def get_session_summary(session_id: str) -> dict[str, Any]:
    return get_session(session_id).summary()


@app.delete("/api/sessions/{session_id}")
def stop_session(session_id: str) -> dict[str, str]:
    cleanup_session(session_id, delete_file=True)
    return {"status": "stopped"}


@app.get("/api/violations")
def get_violations() -> list[dict[str, Any]]:
    return get_violation_storage(load_runtime_config()["violation_db_path"]).list_recent()


@app.get("/api/evidence/{relative_path:path}")
def get_evidence(relative_path: str) -> FileResponse:
    evidence_path = resolve_evidence_path(relative_path)
    if not evidence_path.exists() or not evidence_path.is_file():
        raise HTTPException(status_code=404, detail="Evidence not found.")
    return FileResponse(evidence_path)


def validate_upload(
    video: UploadFile,
    traffic_light: str,
    max_capacity: int,
    confidence_threshold: float,
    normal_threshold: int,
    crowded_threshold: int,
    frame_skip: int,
) -> str:
    if not video.filename:
        raise HTTPException(status_code=400, detail="Missing video file.")

    suffix = Path(video.filename).suffix.lower()
    if suffix not in VALID_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .mp4, .avi, and .mov videos are supported.")
    if video.content_type and not (video.content_type.startswith("video/") or video.content_type in VALID_CONTENT_TYPES):
        raise HTTPException(status_code=400, detail="Unsupported video content type.")
    if traffic_light not in VALID_LIGHTS:
        raise HTTPException(status_code=400, detail="traffic_light must be RED, GREEN, or YELLOW.")
    if max_capacity < 1:
        raise HTTPException(status_code=400, detail="max_capacity must be greater than 0.")
    if not (0.05 <= confidence_threshold <= 0.90):
        raise HTTPException(status_code=400, detail="confidence_threshold must be between 0.05 and 0.90.")
    if frame_skip < 1 or frame_skip > 30:
        raise HTTPException(status_code=400, detail="frame_skip must be between 1 and 30.")
    if not (0 <= normal_threshold < crowded_threshold <= 100):
        raise HTTPException(status_code=400, detail="Density thresholds must satisfy 0 <= normal < crowded <= 100.")

    return suffix


def validate_model_path(model_path: str) -> Path:
    try:
        return resolve_model_path(model_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=safe_error(str(exc))) from exc


async def save_upload(video: UploadFile, destination: Path) -> None:
    """Save an upload in chunks and enforce a size limit."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    with destination.open("wb") as file:
        while chunk := await video.read(1024 * 1024):
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                destination.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Video is larger than the 500 MB limit.")
            file.write(chunk)


def get_session(session_id: str) -> ProcessingSession:
    with sessions_lock:
        session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or already finished.")
    return session


def cleanup_session(session_id: str, delete_file: bool = False) -> None:
    with sessions_lock:
        session = sessions.pop(session_id, None)
    if session is not None:
        session.close(delete_file=delete_file)


def cleanup_all_sessions() -> None:
    with sessions_lock:
        session_ids = list(sessions)
    for session_id in session_ids:
        cleanup_session(session_id, delete_file=True)


async def cleanup_inactive_sessions_loop() -> None:
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
        cleanup_inactive_sessions()


def cleanup_inactive_sessions() -> None:
    now = time.time()
    with sessions_lock:
        expired_ids = [
            session_id
            for session_id, session in sessions.items()
            if now - session.last_access > SESSION_TIMEOUT_SECONDS
        ]
    for session_id in expired_ids:
        cleanup_session(session_id, delete_file=True)


def enforce_session_limit() -> None:
    cleanup_inactive_sessions()
    with sessions_lock:
        active_count = len(sessions)
    if active_count >= MAX_ACTIVE_SESSIONS:
        raise HTTPException(status_code=429, detail="Too many active sessions. Stop an existing session first.")


def build_runtime_config(normal_threshold: int, crowded_threshold: int, confidence_threshold: float) -> dict[str, Any]:
    config = load_runtime_config()
    config["confidence_threshold"] = float(confidence_threshold)
    config["density_threshold"] = {"normal": normal_threshold, "crowded": crowded_threshold}
    return config


def load_runtime_config() -> dict[str, Any]:
    config = load_config(CONFIG_PATH)
    config["violation_db_path"] = str(ROOT_DIR / config.get("violation_db_path", "logs/violations.sqlite3"))
    config["evidence_dir"] = str(ROOT_DIR / config.get("evidence_dir", "evidence"))
    return config


def resolve_evidence_path(relative_path: str) -> Path:
    if not relative_path or "\x00" in relative_path:
        raise HTTPException(status_code=400, detail="Invalid evidence path.")
    base = EVIDENCE_DIR.resolve()
    candidate = (base / relative_path).resolve()
    if base != candidate and base not in candidate.parents:
        raise HTTPException(status_code=400, detail="Invalid evidence path.")
    return candidate


def safe_error(message: str, max_length: int = 180) -> str:
    clean = " ".join(str(message).split())
    return clean[:max_length] or "Request failed."


def resize_frame(frame, max_width: int = 960):
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame

    scale = max_width / width
    return cv2.resize(frame, (max_width, int(height * scale)))


def encode_frame_to_base64(frame) -> str:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise RuntimeError("Could not encode frame.")
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)

