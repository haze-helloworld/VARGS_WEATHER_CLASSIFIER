"""
src/model.py
============
The model architecture (Temporal Shift Module + ResNet-50) and checkpoint
loading helpers, in ONE place. Every other module (training, evaluation,
inference, the API server) imports `TSMResNet50` and `load_checkpoint`
from here instead of redefining the class - this is the single most
important thing to keep centralized, since a checkpoint's `state_dict`
will only load into an architecture that matches exactly.
"""

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

import config


class TemporalShift(nn.Module):
    """
    Shifts a fraction of channels forward/backward one timestep, letting a
    2D CNN mix a little information across time "for free" (no extra
    parameters, negligible extra compute). See the TSM paper (Lin et al.,
    2019, https://arxiv.org/abs/1811.08383).
    """

    def __init__(self, channels: int, num_segments: int = 16, shift_div: int = 8):
        super().__init__()
        self.num_segments = num_segments
        self.shift_div = shift_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        BT, C, H, W = x.shape
        if BT % self.num_segments != 0:
            raise ValueError(f"{BT} samples cannot be grouped into {self.num_segments} temporal segments")

        B = BT // self.num_segments
        x = x.view(B, self.num_segments, C, H, W)
        fold = C // self.shift_div

        out = torch.zeros_like(x)
        out[:, 1:, :fold] = x[:, :-1, :fold]                  # earlier -> later
        out[:, :-1, fold:2 * fold] = x[:, 1:, fold:2 * fold]  # later -> earlier
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]             # no shift

        return out.reshape(BT, C, H, W)


class TSMResNet50(nn.Module):
    """ImageNet-pretrained ResNet-50 with a TemporalShift inserted before every block's conv1."""

    def __init__(self, num_classes: int = 5, num_segments: int = 16, pretrained_backbone: bool = False):
        super().__init__()
        self.num_segments = num_segments

        # pretrained_backbone=False by default: for INFERENCE/loading a
        # trained checkpoint you don't need (or want) to download
        # ImageNet weights first - the checkpoint's state_dict overwrites
        # every weight anyway. Set pretrained_backbone=True only when
        # training a fresh model from scratch (see src/train.py).
        weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None
        self.backbone = resnet50(weights=weights)

        for layer in [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]:
            for block in layer:
                block.conv1 = nn.Sequential(
                    TemporalShift(block.conv1.in_channels, num_segments=num_segments),
                    block.conv1,
                )

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        if T != self.num_segments:
            raise ValueError(f"Expected {self.num_segments} frames, got {T}")

        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        features = self.backbone(x).view(B, T, -1).mean(dim=1)
        return self.classifier(features)


def build_model(device: Union[str, torch.device], pretrained_backbone: bool = False) -> TSMResNet50:
    """Convenience constructor using config.py's class count / frame count."""
    model = TSMResNet50(
        num_classes=config.NUM_CLASSES,
        num_segments=config.NUM_FRAMES,
        pretrained_backbone=pretrained_backbone,
    )
    return model.to(device)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device],
    strict: bool = True,
) -> tuple[TSMResNet50, dict]:
    """
    Build a fresh model and load trained weights into it. Returns
    `(model, checkpoint_dict)` so callers can also inspect metadata like
    `checkpoint["val_f1"]`.

    Checkpoints are expected to be the dict format saved by src/train.py
    (and the original notebooks):
        {"model_state_dict": ..., "class_names": [...], "num_segments": 16, "val_f1": 0.xx}
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Copy your trained .pth file here, or point config.py at the right one."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ckpt_classes = checkpoint.get("class_names")
    if ckpt_classes is not None and list(ckpt_classes) != list(config.CLASS_NAMES):
        print(
            f"⚠️  Checkpoint class_names {ckpt_classes} do not match config.CLASS_NAMES "
            f"{config.CLASS_NAMES} - predictions will be mislabelled if you proceed."
        )

    num_segments = checkpoint.get("num_segments", config.NUM_FRAMES)
    model = TSMResNet50(num_classes=config.NUM_CLASSES, num_segments=num_segments, pretrained_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model.to(device)
    model.eval()

    print(f"✅ Loaded checkpoint (val_f1={checkpoint.get('val_f1', 'n/a')}) <- {checkpoint_path}")
    return model, checkpoint


def save_checkpoint(model: TSMResNet50, path: Union[str, Path], val_f1: float, num_segments: Optional[int] = None) -> None:
    """Save in the same dict format every script/notebook in this project expects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": config.CLASS_NAMES,
            "num_segments": num_segments or config.NUM_FRAMES,
            "val_f1": val_f1,
        },
        path,
    )
