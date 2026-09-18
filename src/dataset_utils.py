"""
src/dataset_utils.py
=====================
Label loading/cleaning, dataset verification, and the train/val split -
run this once before training to catch a missing CSV, a missing video
folder, or a class-count typo early, instead of failing 20 minutes into
an epoch.

    python -m src.dataset_utils
"""

import pandas as pd
from sklearn.model_selection import train_test_split

import config


def prepare_labels(csv_path):
    """
    Load one of the VARG multi-label CSVs and keep only single-label rows
    (exactly one active weather column). Returns (raw_df, clean_df, num_removed).
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    active = df[config.CLASS_NAMES].fillna(0).astype(int).sum(axis=1)
    clean = df[active == 1].copy()
    clean["class_name"] = clean[config.CLASS_NAMES].idxmax(axis=1)
    clean["label"] = clean["class_name"].map(config.CLASS_TO_ID).astype(int)
    clean = clean[["filename", "num_frames", "class_name", "label"]].reset_index(drop=True)

    removed = int((active != 1).sum())
    return df, clean, removed


def load_clean_labels():
    """Load + clean both train and test CSVs. Returns (train_df, test_df)."""
    _, train_df, _ = prepare_labels(config.TRAIN_CSV)
    _, test_df, _ = prepare_labels(config.TEST_CSV)
    return train_df, test_df


def verify_videos_exist(train_df, test_df) -> None:
    video_stems = {p.stem for p in config.VIDEO_DIR.glob("*.mp4")}
    train_missing = int((~train_df["filename"].isin(video_stems)).sum())
    test_missing = int((~test_df["filename"].isin(video_stems)).sum())

    print("MP4 files found:", len(video_stems))
    print("Train missing  :", train_missing)
    print("Test missing   :", test_missing)

    assert train_missing == 0, "Some train clips are missing their .mp4 file"
    assert test_missing == 0, "Some test clips are missing their .mp4 file"
    print("✅ Every cleaned train/test sample has an MP4")


def make_split(train_df):
    """Reproducible train/val split - same random_state everywhere so results stay comparable."""
    train_part, val_part = train_test_split(
        train_df, test_size=config.VAL_SPLIT, random_state=config.SEED, stratify=train_df["label"]
    )
    return train_part.reset_index(drop=True), val_part.reset_index(drop=True)


def _main() -> None:
    for name, path in {
        "TRAIN_CSV": config.TRAIN_CSV,
        "TEST_CSV": config.TEST_CSV,
        "VIDEO_DIR": config.VIDEO_DIR,
    }.items():
        print(f"{name:12s}: {'OK' if path.exists() else 'MISSING'} -> {path}")

    train_df, test_df = load_clean_labels()

    print("\nTRAIN distribution:")
    print(train_df["class_name"].value_counts().reindex(config.CLASS_NAMES))
    print("\nTEST distribution:")
    print(test_df["class_name"].value_counts().reindex(config.CLASS_NAMES))

    print()
    verify_videos_exist(train_df, test_df)

    train_part, val_part = make_split(train_df)
    print(f"\nTrain: {len(train_part)}  Val: {len(val_part)}  Test: {len(test_df)}")


if __name__ == "__main__":
    _main()
