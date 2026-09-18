"""
src/dataset.py
===============
The MP4 video dataset (reads clips directly, no pre-extracted JPG frame
dump), CLAHE preprocessing, and the frame-sampling helpers shared by
training, evaluation/TTA, and live inference - so all three read frames
and normalize them in exactly the same way.
"""

import random
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import config

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """CLAHE-normalize one RGB uint8 frame (contrast-limited adaptive histogram equalization on L channel)."""
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


def read_video_frames(video_path: Path, size: int = config.IMAGE_SIZE, use_clahe: bool = False) -> List[np.ndarray]:
    """Decode every frame of an MP4, resized to (size, size), RGB order. Raises if none are readable."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        if use_clahe:
            frame = apply_clahe(frame)
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No readable frames in: {video_path}")
    return frames


def sample_frame_indices(n: int, num_frames: int = config.NUM_FRAMES, train: bool = False) -> np.ndarray:
    """
    Pick `num_frames` indices out of `n` available frames.
    - train=True: a random (but time-ordered) subset - light temporal augmentation.
    - train=False: evenly spaced across the whole clip - deterministic for eval.
    - n < num_frames: repeat the last frame to pad out to num_frames.
    """
    if n <= 0:
        raise RuntimeError("Video has no readable frames")

    if n >= num_frames:
        if train:
            return np.sort(np.random.choice(n, num_frames, replace=False))
        return np.linspace(0, n - 1, num_frames).astype(int)

    return np.array(list(range(n)) + [n - 1] * (num_frames - n))


def frames_to_tensor(frames: List[np.ndarray]) -> torch.Tensor:
    """List of HxWx3 uint8 RGB frames -> normalized (C, T, H, W) float tensor, ImageNet-normalized."""
    x = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float() / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


class VARGVideoDataset(Dataset):
    """Reads MP4s directly. Set `use_clahe=True` to reproduce the Variant 1 + CLAHE experiment."""

    def __init__(self, df, video_dir, num_frames: int = config.NUM_FRAMES, train: bool = False,
                 size: int = config.IMAGE_SIZE, use_clahe: bool = False):
        self.df = df.reset_index(drop=True)
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        self.train = train
        self.size = size
        self.use_clahe = use_clahe

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        video_path = self.video_dir / f"{row.filename}.mp4"

        all_frames = read_video_frames(video_path, size=self.size, use_clahe=self.use_clahe)
        indices = sample_frame_indices(len(all_frames), self.num_frames, train=self.train)
        frames = [all_frames[i] for i in indices]

        # Whole-clip horizontal flip (same flip applied to every frame, training only)
        if self.train and random.random() < 0.5:
            frames = [np.ascontiguousarray(f[:, ::-1]) for f in frames]

        x = frames_to_tensor(frames)
        y = torch.tensor(int(row.label), dtype=torch.long)
        return x, y


def build_datasets(train_part, val_part, test_df, use_clahe: bool = False):
    """Convenience factory used by both src/train.py and src/evaluate.py."""
    train_ds = VARGVideoDataset(train_part, config.VIDEO_DIR, train=True, use_clahe=use_clahe)
    val_ds = VARGVideoDataset(val_part, config.VIDEO_DIR, train=False, use_clahe=use_clahe)
    test_ds = VARGVideoDataset(test_df, config.VIDEO_DIR, train=False, use_clahe=use_clahe)
    return train_ds, val_ds, test_ds
