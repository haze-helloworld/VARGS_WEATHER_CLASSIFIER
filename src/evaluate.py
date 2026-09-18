"""
src/evaluate.py
================
Evaluate a trained checkpoint on the untouched VARG test set. Supports
the same test-time augmentation (TTA) used to get the current best
result (Variant 1 + TTA).

Usage:
    python -m src.evaluate --variant variant1 --tta            # current best result
    python -m src.evaluate --variant variant1_clahe             # once that checkpoint exists
    python -m src.evaluate --checkpoint checkpoints/custom.pth --clahe
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import config
from src.dataset import VARGVideoDataset, IMAGENET_MEAN, IMAGENET_STD, read_video_frames, sample_frame_indices
from src.dataset_utils import load_clean_labels, make_split
from src.model import load_checkpoint
from src.train import get_device, VARIANT_CHECKPOINTS


def evaluate_standard(model, test_ds, device, batch_size: int = config.BATCH_SIZE):
    """Plain test-set pass, no TTA - one forward pass per clip."""
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS)
    preds, targets = [], []

    model.eval()
    with torch.no_grad():
        for videos, labels in tqdm(loader, desc="Evaluating"):
            videos = videos.to(device, non_blocking=True)
            logits = model(videos)
            preds.extend(logits.argmax(1).cpu().numpy())
            targets.extend(labels.numpy())

    return preds, targets


@torch.no_grad()
def evaluate_with_tta(test_df, model, device, use_clahe: bool = False,
                       n_offsets: int = config.TTA_N_OFFSETS, use_flip: bool = config.TTA_USE_FLIP):
    """
    Re-samples each clip at a few different frame offsets (+ optional flip),
    averages softmax probabilities, then argmaxes. This is the recipe
    behind the current best result (Variant 1 + TTA).
    """
    model.eval()
    preds, targets = [], []
    num_frames = config.NUM_FRAMES

    for idx in tqdm(range(len(test_df)), desc="TTA eval"):
        row = test_df.iloc[idx]
        video_path = config.VIDEO_DIR / f"{row.filename}.mp4"
        frames = read_video_frames(video_path, size=config.IMAGE_SIZE, use_clahe=use_clahe)

        n = len(frames)
        base = np.linspace(0, max(n - 1, 0), num_frames).astype(int)
        step = max(1, n // (num_frames * 2))
        offsets = [0, step, -step][:max(1, n_offsets)]

        probs_accum = torch.zeros(config.NUM_CLASSES)
        views = 0

        for off in offsets:
            idxs = np.clip(base + off, 0, n - 1)
            clip = [frames[i] for i in idxs]

            variants = [clip]
            if use_flip:
                variants.append([np.ascontiguousarray(f[:, ::-1]) for f in clip])

            for v in variants:
                x = torch.from_numpy(np.stack(v)).permute(3, 0, 1, 2).float() / 255.0
                x = ((x - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(device)
                logits = model(x)
                probs_accum += torch.softmax(logits, dim=1)[0].cpu()
                views += 1

        preds.append(int((probs_accum / views).argmax()))
        targets.append(int(row.label))

    return preds, targets


def report(preds, targets, title: str, save_confusion_matrix_to: Path = None):
    acc = accuracy_score(targets, preds)
    bal_acc = balanced_accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, labels=list(range(config.NUM_CLASSES)), average="macro", zero_division=0
    )

    print(f"\n========== {title} — TEST SET ==========")
    print(f"Accuracy          : {acc:.4f}")
    print(f"Balanced Accuracy : {bal_acc:.4f}")
    print(f"Macro Precision   : {precision:.4f}")
    print(f"Macro Recall      : {recall:.4f}")
    print(f"Macro F1          : {f1:.4f}")
    print("\nClassification report:\n")
    print(classification_report(targets, preds, target_names=config.CLASS_NAMES, digits=4, zero_division=0))

    cm = confusion_matrix(targets, preds, labels=list(range(config.NUM_CLASSES)))
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES, cbar=False)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"{title} — Test Confusion Matrix")
    plt.tight_layout()

    if save_confusion_matrix_to:
        save_confusion_matrix_to.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_confusion_matrix_to, dpi=150)
        print(f"Confusion matrix saved -> {save_confusion_matrix_to}")
    plt.show()

    return {"accuracy": acc, "balanced_accuracy": bal_acc, "precision": precision, "recall": recall, "f1": f1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the untouched test set.")
    parser.add_argument("--variant", choices=list(VARIANT_CHECKPOINTS), default=None,
                         help="Shortcut for --checkpoint pointing at that variant's config.py path.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path (overrides --variant).")
    parser.add_argument("--clahe", action="store_true", help="Read frames with CLAHE applied (must match how the checkpoint was trained).")
    parser.add_argument("--tta", action="store_true", help="Use test-time augmentation instead of a single forward pass.")
    parser.add_argument("--n_offsets", type=int, default=config.TTA_N_OFFSETS)
    parser.add_argument("--no_flip", action="store_true", help="Disable the horizontal-flip view in TTA.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    elif args.variant:
        checkpoint_path = VARIANT_CHECKPOINTS[args.variant]
    else:
        checkpoint_path = config.DEFAULT_CHECKPOINT

    device = get_device()
    print("Device:", device)

    model, ckpt = load_checkpoint(checkpoint_path, device)

    train_df, test_df = load_clean_labels()
    _, _ = make_split(train_df)  # not needed here, but keeps the split call identical everywhere

    title = checkpoint_path.stem
    if args.tta:
        preds, targets = evaluate_with_tta(test_df, model, device, use_clahe=args.clahe,
                                            n_offsets=args.n_offsets, use_flip=not args.no_flip)
        title += " + TTA"
    else:
        test_ds = VARGVideoDataset(test_df, config.VIDEO_DIR, train=False, use_clahe=args.clahe)
        preds, targets = evaluate_standard(model, test_ds, device)

    report(preds, targets, title, save_confusion_matrix_to=config.PROJECT_ROOT / "docs" / "images" / f"confusion_matrix_{checkpoint_path.stem}.png")
