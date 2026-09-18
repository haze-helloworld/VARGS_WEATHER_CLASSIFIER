"""
src/inference.py
=================
The inference wrapper app.py (and src/infer_stream.py) actually calls.
Loads a checkpoint once and exposes simple `predict_video()` /
`predict_frames()` functions that return plain dicts - no Ultralytics-
style result objects, no notebook globals, just JSON-serializable output.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import torch

import config
from src.dataset import IMAGENET_MEAN, IMAGENET_STD, apply_clahe, read_video_frames, sample_frame_indices
from src.model import load_checkpoint


@dataclass
class ClassificationResult:
    predicted_class: str
    predicted_class_id: int
    confidence: float
    probabilities: dict  # {class_name: probability}
    used_tta: bool
    used_clahe: bool

    def to_dict(self) -> dict:
        return asdict(self)


class WeatherClassifier:
    """
    Wraps a loaded TSMResNet50 for repeated inference calls. Create ONE
    instance at server startup (loading the model is relatively slow) and
    reuse it for every request/frame - same pattern as the YOLO repo's
    `StreamManager` / `single_image_model`.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = config.DEFAULT_CHECKPOINT,
        device: Union[str, torch.device] = None,
        use_clahe: bool = config.DEFAULT_USE_CLAHE,
        use_tta: bool = config.DEFAULT_USE_TTA,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model, self.checkpoint = load_checkpoint(checkpoint_path, self.device)
        self.use_clahe = use_clahe
        self.use_tta = use_tta
        self.num_frames = self.checkpoint.get("num_segments", config.NUM_FRAMES)

    # -- frames already in memory (e.g. from a live camera buffer) ----------
    @torch.no_grad()
    def predict_frames(self, frames: List[np.ndarray]) -> ClassificationResult:
        """
        `frames`: a list of RGB uint8 HxWxC frames, already the size the
        model expects (config.IMAGE_SIZE) and already CLAHE'd if
        `self.use_clahe` is True - use `predict_video` instead if you're
        starting from a raw video file, since it handles all of that.
        """
        if self.use_tta:
            probs = self._tta_probs_from_frames(frames)
        else:
            indices = sample_frame_indices(len(frames), self.num_frames, train=False)
            clip = [frames[i] for i in indices]
            probs = self._forward_probs(clip)

        return self._to_result(probs)

    # -- a video file on disk -------------------------------------------------
    def predict_video(self, video_path: Union[str, Path]) -> ClassificationResult:
        frames = read_video_frames(Path(video_path), size=config.IMAGE_SIZE, use_clahe=self.use_clahe)
        return self.predict_frames(frames)

    # -- a raw frame captured live from a camera (BGR, as OpenCV gives it) ------
    def prepare_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Resize + convert to RGB + (if configured) apply CLAHE, matching how
        every frame is preprocessed during training/evaluation. Call this
        on every frame BEFORE appending it to a rolling buffer for
        `predict_frames` (see src/infer_stream.py) - `predict_frames`
        assumes frames are already in this format.
        """
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        if self.use_clahe:
            frame = apply_clahe(frame)
        return frame

    # -- internals --------------------------------------------------------------
    @torch.no_grad()
    def _forward_probs(self, clip: List[np.ndarray]) -> torch.Tensor:
        x = torch.from_numpy(np.stack(clip)).permute(3, 0, 1, 2).float() / 255.0
        x = ((x - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(self.device)
        logits = self.model(x)
        return torch.softmax(logits, dim=1)[0].cpu()

    @torch.no_grad()
    def _tta_probs_from_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Mirrors evaluate_with_tta() from src/evaluate.py, operating on in-memory frames."""
        n = len(frames)
        base = np.linspace(0, max(n - 1, 0), self.num_frames).astype(int)
        step = max(1, n // (self.num_frames * 2))
        offsets = [0, step, -step][:max(1, config.TTA_N_OFFSETS)]

        probs_accum = torch.zeros(config.NUM_CLASSES)
        views = 0

        for off in offsets:
            idxs = np.clip(base + off, 0, n - 1)
            clip = [frames[i] for i in idxs]

            variants = [clip]
            if config.TTA_USE_FLIP:
                variants.append([np.ascontiguousarray(f[:, ::-1]) for f in clip])

            for v in variants:
                probs_accum += self._forward_probs(v)
                views += 1

        return probs_accum / views

    def _to_result(self, probs: torch.Tensor) -> ClassificationResult:
        probs_np = probs.numpy()
        pred_id = int(probs_np.argmax())
        return ClassificationResult(
            predicted_class=config.CLASS_NAMES[pred_id],
            predicted_class_id=pred_id,
            confidence=float(probs_np[pred_id]),
            probabilities={name: float(p) for name, p in zip(config.CLASS_NAMES, probs_np)},
            used_tta=self.use_tta,
            used_clahe=self.use_clahe,
        )
