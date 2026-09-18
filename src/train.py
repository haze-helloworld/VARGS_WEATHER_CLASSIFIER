"""
src/train.py
=============
CLI training script. Reproduces the three recipes from the original
notebooks as `--variant` choices:

    baseline        class-weighted CrossEntropy, no sampler
    variant1        balanced WeightedRandomSampler + unweighted CE + grad clip   (best result so far, with TTA)
    variant1_clahe  identical recipe to variant1, but every frame is CLAHE-normalized

Usage:
    python -m src.train --variant variant1_clahe --epochs 5
    python -m src.train --variant variant1 --epochs 5 --lr 5e-5

Run `python -m src.dataset_utils` first to make sure paths are correct.
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm.auto import tqdm

import config
from src.dataset_utils import load_clean_labels, make_split
from src.dataset import build_datasets
from src.model import build_model, save_checkpoint

VARIANT_CHECKPOINTS = {
    "baseline": config.BASELINE_CHECKPOINT,
    "variant1": config.VARIANT1_CHECKPOINT,
    "variant1_clahe": config.VARIANT1_CLAHE_CHECKPOINT,
}

# Matches the learning rate / weight decay each recipe actually used in the original notebooks.
VARIANT_DEFAULT_LR = {
    "baseline": config.LR,
    "variant1": config.VARIANT1_LR,
    "variant1_clahe": config.LR,
}
VARIANT_DEFAULT_WD = {
    "baseline": config.WEIGHT_DECAY,
    "variant1": config.WEIGHT_DECAY,
    "variant1_clahe": 0.0,
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_balanced_sampler(train_part: pd.DataFrame) -> WeightedRandomSampler:
    class_counts = train_part["label"].value_counts().reindex(range(config.NUM_CLASSES), fill_value=0)
    class_weights = 1.0 / class_counts.values
    sample_weights = torch.DoubleTensor(train_part["label"].map(dict(enumerate(class_weights))).values)
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def evaluate(model, loader, criterion, device):
    model.eval()
    losses, preds, targets = [], [], []

    with torch.no_grad():
        for videos, labels in loader:
            videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(videos)
            loss = criterion(logits, labels)

            losses.append(loss.item())
            preds.extend(logits.argmax(1).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, preds, labels=list(range(config.NUM_CLASSES)), average="macro", zero_division=0
    )
    return np.mean(losses), accuracy_score(targets, preds), precision, recall, f1


def train_one_variant(variant: str, epochs: int, lr: float, weight_decay: float, batch_size: int) -> None:
    device = get_device()
    print("Device:", device)
    config.ensure_directories()

    train_df, test_df = load_clean_labels()
    train_part, val_part = make_split(train_df)
    print(f"Train: {len(train_part)}  Val: {len(val_part)}  Test: {len(test_df)}")

    use_clahe = variant == "variant1_clahe"
    train_ds, val_ds, _ = build_datasets(train_part, val_part, test_df, use_clahe=use_clahe)

    loader_kwargs = dict(num_workers=config.NUM_WORKERS, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    model = build_model(device, pretrained_backbone=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    if variant == "baseline":
        counts = train_part["label"].value_counts().reindex(range(config.NUM_CLASSES), fill_value=0).values
        class_weights = torch.tensor(
            len(train_part) / (config.NUM_CLASSES * np.maximum(counts, 1)), dtype=torch.float32, device=device
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
        grad_clip = None
    else:  # variant1 / variant1_clahe: balanced sampler + unweighted CE + grad clip
        criterion = nn.CrossEntropyLoss()
        sampler = make_balanced_sampler(train_part)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, **loader_kwargs)
        grad_clip = config.GRAD_CLIP

    checkpoint_path = VARIANT_CHECKPOINTS[variant]
    best_f1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n_samples = 0.0, 0
        start = time.time()

        for videos, labels in tqdm(train_loader, desc=f"[{variant}] Epoch {epoch}/{epochs}"):
            videos, labels = videos.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(videos)
            loss = criterion(logits, labels)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)

        train_loss = running_loss / n_samples
        val_loss, val_acc, val_p, val_r, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_f1)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "val_accuracy": val_acc, "val_macro_f1": val_f1})

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(model, checkpoint_path, val_f1=best_f1)
            print(f"💾 New best checkpoint (val_f1={best_f1:.4f}) -> {checkpoint_path}")

        elapsed = time.time() - start
        print(f"[{variant}] Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_acc={val_acc:.4f} | macro_F1={val_f1:.4f} | time={elapsed/60:.1f} min")

    print("\nTraining history:")
    print(pd.DataFrame(history))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the VARG weather-severity classifier.")
    parser.add_argument("--variant", choices=list(VARIANT_CHECKPOINTS), default="variant1_clahe")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=None, help="Defaults to the LR the original notebook used for --variant (see VARIANT_DEFAULT_LR).")
    parser.add_argument("--weight_decay", type=float, default=None, help="Defaults to the value the original notebook used for --variant.")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    lr = args.lr if args.lr is not None else VARIANT_DEFAULT_LR[args.variant]
    weight_decay = args.weight_decay if args.weight_decay is not None else VARIANT_DEFAULT_WD[args.variant]
    train_one_variant(args.variant, args.epochs, lr, weight_decay, args.batch_size)
