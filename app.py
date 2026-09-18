"""
app.py
=======
The backend server for the weather-severity classifier - the "fog-
density model" node in the sibling `fog_mine_yolov8` repo's architecture
diagram. Mirrors that repo's app.py: same shape of endpoints, same
FastAPI-startup-loads-the-model-once pattern, so the two services are
easy to run side by side (or eventually merge) on the same central
workstation.

WHAT THIS SERVER DOES
------------------------
1. On startup, loads the configured checkpoint (config.DEFAULT_CHECKPOINT,
   config.DEFAULT_USE_TTA, config.DEFAULT_USE_CLAHE) ONCE via
   `WeatherClassifier`, and creates a `WeatherStreamManager` that can
   watch several camera feeds at once.
2. Exposes a small HTTP API:
     GET    /health                            - is the server alive? which model is loaded?
     POST   /predict/video                      - upload a short MP4 clip, get a classification back
     POST   /cameras/{camera_id}                - start watching a new camera feed
     DELETE /cameras/{camera_id}                - stop watching a camera feed
     GET    /cameras                            - list currently-watched cameras + status
     GET    /cameras/{camera_id}/classification  - latest classification for one camera (JSON)

HOW TO RUN
-----------
    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 8001

(Port 8001, not 8000 - so it can run alongside the YOLO repo's app.py on
the same workstation without a port clash; change with --port as needed.)

Then open http://localhost:8001/docs for interactive API docs.

NOTE ON THE WEBCAM QUICK-DEMO ENDPOINT
------------------------------------------
For a demo without real CCTV hardware, register your laptop's webcam:
    curl -X POST http://localhost:8001/cameras/camera_0 \\
         -H "Content-Type: application/json" -d '{"source": "0"}'
"""

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

import config
from src.inference import WeatherClassifier
from src.infer_stream import WeatherStreamManager

app = FastAPI(
    title="VARG Weather-Severity Classification Service",
    description="Central inference service: 5-class weather/fog-severity classification over CCTV feeds and uploaded clips.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Global state, created once when the server starts (same pattern as the
# sibling YOLO repo's app.py - a couple of module-level globals is a
# reasonable, simple choice at this scale).
# ---------------------------------------------------------------------------
classifier: Optional[WeatherClassifier] = None
stream_manager: Optional[WeatherStreamManager] = None


@app.on_event("startup")
def on_startup() -> None:
    global classifier, stream_manager
    config.ensure_directories()

    print("Loading weather classifier...")
    print(f"  checkpoint = {config.DEFAULT_CHECKPOINT}")
    print(f"  use_clahe  = {config.DEFAULT_USE_CLAHE}")
    print(f"  use_tta    = {config.DEFAULT_USE_TTA}")

    classifier = WeatherClassifier(
        checkpoint_path=config.DEFAULT_CHECKPOINT,
        use_clahe=config.DEFAULT_USE_CLAHE,
        use_tta=config.DEFAULT_USE_TTA,
    )

    print("Starting multi-camera stream manager...")
    stream_manager = WeatherStreamManager(
        checkpoint_path=config.DEFAULT_CHECKPOINT,
        use_clahe=config.DEFAULT_USE_CLAHE,
        use_tta=config.DEFAULT_USE_TTA,
    )

    print("Server ready.")


@app.on_event("shutdown")
def on_shutdown() -> None:
    if stream_manager:
        stream_manager.shutdown()


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------
class AddCameraRequest(BaseModel):
    source: str
    classify_interval_seconds: Optional[float] = None


class ClassificationResponse(BaseModel):
    predicted_class: str
    predicted_class_id: int
    confidence: float
    probabilities: dict
    used_tta: bool
    used_clahe: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "classes": config.CLASS_NAMES,
        "checkpoint": str(config.DEFAULT_CHECKPOINT),
        "use_clahe": config.DEFAULT_USE_CLAHE,
        "use_tta": config.DEFAULT_USE_TTA,
    }


@app.post("/predict/video", response_model=ClassificationResponse)
async def predict_video(file: UploadFile = File(...)):
    """
    Upload a short MP4 clip, get a weather-severity classification back.
    Useful for testing the model from a browser/Postman, or for a
    teammate's module that has recorded clips rather than a live feed.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Server is still starting up.")

    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    contents = await file.read()

    # Ultralytics can take an in-memory array; OpenCV's VideoCapture can't
    # easily read from bytes, so we spool the upload to a temp file and
    # point read_video_frames() at that - deleted automatically after.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(contents)
        tmp.flush()
        try:
            result = classifier.predict_video(tmp.name)
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return ClassificationResponse(**result.to_dict())


@app.post("/cameras/{camera_id}")
def add_camera(camera_id: str, request: AddCameraRequest):
    """
    Start watching a new camera feed. Example:
        curl -X POST http://localhost:8001/cameras/entrance_cam \\
             -H "Content-Type: application/json" \\
             -d '{"source": "rtsp://192.168.1.10:554/stream1"}'
    """
    if stream_manager is None:
        raise HTTPException(status_code=503, detail="Server is still starting up.")

    resolved_source = int(request.source) if request.source.isdigit() else request.source
    interval = request.classify_interval_seconds or config.CLASSIFY_INTERVAL_SECONDS

    try:
        stream_manager.add_camera(camera_id, resolved_source, classify_interval=interval)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return {"camera_id": camera_id, "source": request.source, "status": "starting"}


@app.delete("/cameras/{camera_id}")
def remove_camera(camera_id: str):
    if stream_manager is None:
        raise HTTPException(status_code=503, detail="Server is still starting up.")

    stream_manager.remove_camera(camera_id)
    return {"camera_id": camera_id, "status": "removed"}


@app.get("/cameras")
def list_cameras():
    if stream_manager is None:
        raise HTTPException(status_code=503, detail="Server is still starting up.")

    return {
        camera_id: {
            "connected": state.connected,
            "fps": round(state.fps, 1),
            "frames_buffered": state.frames_buffered,
            "last_update_time": state.last_update_time,
            "has_classification": state.last_classification is not None,
            "error": state.error,
        }
        for camera_id, state in stream_manager.states.items()
    }


@app.get("/cameras/{camera_id}/classification", response_model=ClassificationResponse)
def get_camera_classification(camera_id: str):
    """
    The main endpoint a DOWNSTREAM MODULE (e.g. the alert-combining logic
    that also reads the YOLO detector's output) would poll: the latest
    weather-severity classification for one camera.
    """
    if stream_manager is None:
        raise HTTPException(status_code=503, detail="Server is still starting up.")

    state = stream_manager.get_state(camera_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Unknown camera_id: {camera_id}")
    if state.last_classification is None:
        raise HTTPException(status_code=404, detail="No classification yet - buffer still filling.")

    return ClassificationResponse(**state.last_classification.to_dict())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
