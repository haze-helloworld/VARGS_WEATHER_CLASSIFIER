"""
src/infer_stream.py
=====================
Live, multi-camera weather-severity classification for the same central-
workstation architecture as the sibling `fog_mine_yolov8` repo's
`src/infer_stream.py`: every camera feed is read on the central
workstation, not on per-camera edge hardware.

WHY THIS LOOKS DIFFERENT FROM THE YOLO VERSION
------------------------------------------------
The object detector classifies ONE frame at a time. This model needs
`config.NUM_FRAMES` frames of TEMPORAL context per prediction, and
weather severity changes slowly (over seconds, not milliseconds) - so
instead of running inference on every incoming frame, each camera worker:
  1. Keeps a rolling buffer of the most recent frames (already resized +
     CLAHE'd if configured - see `WeatherClassifier.prepare_frame`).
  2. Re-runs classification every `config.CLASSIFY_INTERVAL_SECONDS`
     once the buffer has enough frames, instead of every frame.

HOW TO RUN A QUICK STANDALONE TEST
--------------------------------------
    python -m src.infer_stream --sources 0
        (source "0" = your laptop's webcam)

In production, `app.py` imports `WeatherStreamManager` and starts it as
part of the backend service, same as the YOLO repo's `app.py` does with
its own `StreamManager`.
"""

import argparse
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2

import config
from src.inference import WeatherClassifier, ClassificationResult


@dataclass
class CameraState:
    camera_id: str
    connected: bool = False
    last_classification: Optional[ClassificationResult] = None
    last_update_time: float = 0.0
    frames_buffered: int = 0
    fps: float = 0.0
    error: Optional[str] = None


class WeatherCameraWorker(threading.Thread):
    """One background thread per camera: reads frames, buffers them, classifies periodically."""

    def __init__(
        self,
        camera_id: str,
        source,
        classifier: WeatherClassifier,
        state: CameraState,
        classify_interval: float = config.CLASSIFY_INTERVAL_SECONDS,
        sample_every_n_frames: int = config.STREAM_SAMPLE_EVERY_N_FRAMES,
    ):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.source = source
        self.classifier = classifier
        self.state = state
        self.classify_interval = classify_interval
        self.sample_every_n_frames = max(1, sample_every_n_frames)

        # Rolling buffer of preprocessed (resized/CLAHE'd) frames. maxlen
        # keeps memory bounded regardless of how long the camera has run.
        buffer_len = classifier.num_frames * self.sample_every_n_frames
        self._buffer: deque = deque(maxlen=buffer_len)
        self._frame_counter = 0
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.state.connected = False
            self.state.error = f"Could not open video source: {self.source}"
            print(f"[{self.camera_id}] ERROR: {self.state.error}")
            return

        self.state.connected = True
        print(f"[{self.camera_id}] Connected to {self.source}")

        last_classify_time = 0.0
        last_loop_time = time.time()

        while not self._stop_event.is_set():
            ret, frame_bgr = cap.read()
            if not ret:
                self.state.connected = False
                self.state.error = "Stream ended or dropped (no more frames)."
                print(f"[{self.camera_id}] {self.state.error}")
                break

            self._frame_counter += 1
            if self._frame_counter % self.sample_every_n_frames == 0:
                self._buffer.append(self.classifier.prepare_frame(frame_bgr))
                self.state.frames_buffered = len(self._buffer)

            now = time.time()
            if now - last_classify_time >= self.classify_interval and len(self._buffer) >= self.classifier.num_frames:
                try:
                    self.state.last_classification = self.classifier.predict_frames(list(self._buffer))
                    self.state.last_update_time = now
                except Exception as e:  # keep the worker alive on a transient bad prediction
                    self.state.error = f"Classification error: {e}"
                    print(f"[{self.camera_id}] {self.state.error}")
                last_classify_time = now

            loop_duration = now - last_loop_time
            self.state.fps = 1.0 / loop_duration if loop_duration > 0 else 0.0
            last_loop_time = now

        cap.release()
        print(f"[{self.camera_id}] Worker stopped.")


class WeatherStreamManager:
    """Owns and coordinates one `WeatherCameraWorker` per camera feed - the object app.py talks to."""

    def __init__(
        self,
        checkpoint_path=config.DEFAULT_CHECKPOINT,
        device=None,
        use_clahe: bool = config.DEFAULT_USE_CLAHE,
        use_tta: bool = config.DEFAULT_USE_TTA,
    ):
        # One shared classifier/model for every camera worker - loading a
        # checkpoint per camera would waste memory and startup time.
        self.classifier = WeatherClassifier(checkpoint_path, device=device, use_clahe=use_clahe, use_tta=use_tta)

        self.workers: Dict[str, WeatherCameraWorker] = {}
        self.states: Dict[str, CameraState] = {}

    def add_camera(self, camera_id: str, source, classify_interval: float = config.CLASSIFY_INTERVAL_SECONDS) -> None:
        if camera_id in self.workers:
            raise ValueError(f"Camera '{camera_id}' is already registered.")

        state = CameraState(camera_id=camera_id)
        worker = WeatherCameraWorker(camera_id, source, self.classifier, state, classify_interval=classify_interval)

        self.states[camera_id] = state
        self.workers[camera_id] = worker
        worker.start()

    def remove_camera(self, camera_id: str) -> None:
        worker = self.workers.pop(camera_id, None)
        if worker:
            worker.stop()
            worker.join(timeout=5)
        self.states.pop(camera_id, None)

    def get_state(self, camera_id: str) -> Optional[CameraState]:
        return self.states.get(camera_id)

    def shutdown(self) -> None:
        for camera_id in list(self.workers.keys()):
            self.remove_camera(camera_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick standalone test of the multi-camera weather classification engine.")
    parser.add_argument("--sources", nargs="+", default=["0"])
    parser.add_argument("--checkpoint", type=str, default=str(config.DEFAULT_CHECKPOINT))
    parser.add_argument("--seconds", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    manager = WeatherStreamManager(checkpoint_path=args.checkpoint)

    for i, source in enumerate(args.sources):
        resolved_source = int(source) if source.isdigit() else source
        manager.add_camera(camera_id=f"camera_{i}", source=resolved_source)

    try:
        end_time = time.time() + args.seconds
        while time.time() < end_time:
            for camera_id, state in manager.states.items():
                if state.last_classification:
                    r = state.last_classification
                    print(f"[{camera_id}] fps={state.fps:.1f} -> {r.predicted_class} ({r.confidence:.2f})")
            time.sleep(1.0)
    finally:
        manager.shutdown()
