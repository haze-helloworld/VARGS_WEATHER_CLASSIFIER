"""
config.py
=========
Central configuration for the whole project.

Same idea as the sibling `fog_mine_yolov8` repo's config.py: every path,
class name, and hyperparameter lives in ONE place, so every other module
just does `import config` instead of hard-coding paths. Edit the values
below to match your machine; nothing else in this project should need
changing.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. PROJECT ROOT
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 2. DATASET PATHS
# ---------------------------------------------------------------------------
# Expected layout (matches the original training notebook):
#
#   VARG_Dataset/
#   ├── data_split/multi_label/multi_label_train.csv
#   ├── data_split/multi_label/multi_label_test.csv
#   └── videos/*.mp4
#
# Only used by src/dataset_utils.py and src/train.py (training/evaluation
# on the labelled dataset) - the app.py backend does NOT need this, it
# only needs a trained checkpoint.
DATASET_DIR = PROJECT_ROOT / "VARG_Dataset"
TRAIN_CSV = DATASET_DIR / "data_split" / "multi_label" / "multi_label_train.csv"
TEST_CSV = DATASET_DIR / "data_split" / "multi_label" / "multi_label_test.csv"
VIDEO_DIR = DATASET_DIR / "videos"

# ---------------------------------------------------------------------------
# 3. CLASS DEFINITIONS
# ---------------------------------------------------------------------------
# IMPORTANT: order defines the class index used everywhere (checkpoints,
# the API response, the confusion matrix). Keep this in sync with any
# checkpoint's own stored "class_names" list - src/model.py checks this
# for you on load and warns if they don't match.
CLASS_NAMES = [
    "Clear",
    "Rain Moderate",
    "Rain Heavy",
    "Fog Moderate",
    "Fog Heavy",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# 4. MODEL / DATA SHAPE
# ---------------------------------------------------------------------------
NUM_FRAMES = 16     # frames sampled per clip (TSM "num_segments")
IMAGE_SIZE = 224    # frames are resized to IMAGE_SIZE x IMAGE_SIZE
BATCH_SIZE = 4
NUM_WORKERS = 0     # bump to 2-4 on Linux/macOS; keep 0 on Windows

# ---------------------------------------------------------------------------
# 5. CHECKPOINTS
# ---------------------------------------------------------------------------
# Copy your trained .pth files here (see README "Sharing your trained
# model" section - they're gitignored, same policy as the YOLO repo).
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

BASELINE_CHECKPOINT = CHECKPOINT_DIR / "best_tsm_resnet50_varg.pth"
VARIANT1_CHECKPOINT = CHECKPOINT_DIR / "TSM_ResNet50_balanced_best.pth"
VARIANT1_CLAHE_CHECKPOINT = CHECKPOINT_DIR / "variant1_clahe_best.pth"
COMBINED_CHECKPOINT = CHECKPOINT_DIR / "combined_sampler_focal_best.pth"
COMBINED_V2_CHECKPOINT = CHECKPOINT_DIR / "combined_v2_sampler_focal_jitter_best.pth"

# ---------------------------------------------------------------------------
# 6. WHICH MODEL DOES app.py SERVE?
# ---------------------------------------------------------------------------
# Per the current experiment results: Variant 1 + TTA is the best result
# so far, Variant 1 + CLAHE has not finished evaluation yet. Once the
# CLAHE run beats Variant 1 + TTA, flip these two lines and nothing else
# in the codebase needs to change.
DEFAULT_CHECKPOINT = VARIANT1_CHECKPOINT
DEFAULT_USE_CLAHE = False     # must match how DEFAULT_CHECKPOINT was trained
DEFAULT_USE_TTA = True        # apply test-time augmentation at inference

# ---------------------------------------------------------------------------
# 7. TEST-TIME AUGMENTATION (TTA) SETTINGS
# ---------------------------------------------------------------------------
# Mirrors evaluate_with_tta() from the training notebook: re-sample each
# clip at a few frame offsets (+ optional horizontal flip), average the
# softmax probabilities, then argmax.
TTA_N_OFFSETS = 2
TTA_USE_FLIP = True

# ---------------------------------------------------------------------------
# 8. LIVE-STREAM CLASSIFICATION SETTINGS
# ---------------------------------------------------------------------------
# The weather classifier needs NUM_FRAMES of temporal context, unlike the
# YOLO detector which classifies one frame at a time. src/infer_stream.py
# keeps a rolling buffer of the most recent frames per camera and re-runs
# classification every CLASSIFY_INTERVAL_SECONDS instead of every frame -
# weather severity changes slowly, so there's no need to classify at full
# camera frame rate.
CLASSIFY_INTERVAL_SECONDS = 2.0
STREAM_SAMPLE_EVERY_N_FRAMES = 2  # thin out the rolling buffer before sampling NUM_FRAMES from it

# ---------------------------------------------------------------------------
# 9. TRAINING HYPERPARAMETERS
# ---------------------------------------------------------------------------
# Defaults for src/train.py; override from the command line for one-off
# experiments instead of editing this file each time (see `--help`).
EPOCHS = 5
LR = 1e-4
VARIANT1_LR = 5e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 5.0
VAL_SPLIT = 0.15
SEED = 42


def ensure_directories() -> None:
    """Create every output directory this project writes to. Safe to call repeatedly."""
    for directory in [CHECKPOINT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
