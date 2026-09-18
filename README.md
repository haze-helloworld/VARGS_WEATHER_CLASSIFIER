# VARG Weather / Fog Severity Classification (TSM + ResNet-50)

A CCTV-based video classifier that grades weather severity into 5 classes — `Clear`, `Rain Moderate`,
`Rain Heavy`, `Fog Moderate`, `Fog Heavy` — from a short clip. Built on ImageNet-pretrained ResNet-50 with
Temporal Shift Modules (TSM) for cheap temporal reasoning across 16 sampled frames.

This is the **fog-density / weather-severity model** referenced in the sibling
[`fog_mine_yolov8`](../fog_mine_yolov8) repo's architecture diagram: that repo finds *what and where*
(human / vehicle / obstacle); this repo answers *how bad are conditions right now*, so a downstream
alert/dashboard can combine both — e.g. tightening a distance/TTC safety margin when this model reports
`Fog Heavy`.

## Model Architecture
<img src="/assets/flowchart.png" alt="Model architecture" width="700"/>

The final weather classification model uses a **Temporal Shift Module (TSM)** integrated with an **ImageNet-pretrained ResNet-50** backbone.

| Component | Configuration |
|---|---|
| Architecture | TSM + ResNet-50 |
| Backbone | ResNet-50 (ImageNet-pretrained) |
| Temporal Segments | 16 |
| Input Resolution | 224 × 224 |
| Input Tensor | `(batch, 3, 16, 224, 224)` |
| Classifier | Dropout (0.5) → Linear (2048 → 5) |
| Total Parameters | 23,518,277 |
| Trainable Parameters | 23,518,277 |
| Number of Classes | 5 |

### Classes

The model classifies each video clip into one of five weather-severity categories:

1. **Clear**
2. **Rain Moderate**
3. **Rain Heavy**
4. **Fog Moderate**
5. **Fog Heavy**



## Architecture

<img src="/assets/archit.png" alt="Architecture" width="700"/>

Same central-workstation model as the YOLO repo: no per-camera edge hardware, every feed streams to one
place and both models run there. See `src/infer_stream.py` for the live multi-camera engine.

## Repository structure

```text
varg_weather_classifier/
├── app.py                    # FastAPI backend - the central inference service (port 8001)
├── config.py                 # every path/constant/hyperparameter, in one place
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── VARG_TSM_ResNet50_Local_CLAHE_TTA.ipynb   # original training notebook
├── src/
│   ├── dataset.py             # MP4 video dataset, CLAHE preprocessing, frame-sampling helpers
│   ├── dataset_utils.py       # label cleaning, dataset verification, train/val split (python -m src.dataset_utils)
│   ├── model.py                # TemporalShift + TSMResNet50, checkpoint load/save
│   ├── train.py                 # CLI training script: baseline / variant1 / variant1_clahe (python -m src.train)
│   ├── evaluate.py             # test-set metrics + optional TTA + confusion matrix (python -m src.evaluate)
│   ├── inference.py             # WeatherClassifier - the wrapper app.py actually calls
│   └── infer_stream.py         # multi-camera live-video classification engine
├── checkpoints/                # (gitignored) trained .pth files go here
├── VARG_Dataset/                # (gitignored) the training dataset
└── docs/images/                 # confusion matrices etc. land here for the README
```

### Where do my files go?

| File | Goes here |
|---|---|
| `TSM_ResNet50_balanced_best.pth` (Variant 1, current best w/ TTA) | `checkpoints/TSM_ResNet50_balanced_best.pth` |
| `variant1_clahe_best.pth` (once trained) | `checkpoints/variant1_clahe_best.pth` |
| Original `.ipynb` notebook(s) | `notebooks/` |
| Training dataset | `VARG_Dataset/` (same layout as before: `data_split/multi_label/multi_label_{train,test}.csv`, `videos/`) |

Everything above matches `config.py` exactly, so no code changes are needed once your files are in place.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If your workstation has an NVIDIA GPU, install a CUDA-enabled PyTorch build first (see the note at the
bottom of `requirements.txt`) — this is a video model, CPU training/TTA is slow.

## Usage

**1. Verify your dataset** (do this before training anything):
```bash
python -m src.dataset_utils
```

**2. Train** — pick a recipe. `variant1_clahe` is the one currently pending (not yet trained locally):
```bash
python -m src.train --variant variant1_clahe --epochs 5
python -m src.train --variant variant1 --epochs 5       # already trained on Colab; only rerun if needed
```

**3. Evaluate** — reproduce the current best result (Variant 1 + TTA), or check the CLAHE run once trained:
```bash
python -m src.evaluate --variant variant1 --tta                  # current best result
python -m src.evaluate --variant variant1_clahe                  # plain eval
python -m src.evaluate --variant variant1_clahe --tta --clahe    # CLAHE + TTA together
```
> `--clahe` tells the evaluator to read frames with CLAHE applied — pass it whenever you're evaluating a
> checkpoint that was *trained* with CLAHE, or you'll get a train/eval mismatch.

**4. Run the live central inference service:**
```bash
uvicorn app:app --host 0.0.0.0 --port 8001
```
Then open `http://localhost:8001/docs` for interactive API docs.

Test with an uploaded clip (no camera needed):
```bash
curl -X POST http://localhost:8001/predict/video -F "file=@some_clip.mp4"
```

Register a camera feed for continuous classification:
```bash
curl -X POST http://localhost:8001/cameras/entrance_cam \
     -H "Content-Type: application/json" \
     -d '{"source": "rtsp://<camera-ip>/stream1"}'
```

For a **quick demo without real CCTV hardware**, register your own laptop webcam:
```bash
curl -X POST http://localhost:8001/cameras/camera_0 \
     -H "Content-Type: application/json" \
     -d '{"source": "0"}'
```
Then poll `http://localhost:8001/cameras/camera_0/classification` for the latest reading (it fills in once
the rolling buffer has enough frames — a few seconds after registering).

## Which model does `app.py` serve?

Controlled entirely by `config.py`:
```python
DEFAULT_CHECKPOINT = VARIANT1_CHECKPOINT   # Variant 1 (balanced sampler) is the current best-performing checkpoint
DEFAULT_USE_CLAHE = False                  # must match how DEFAULT_CHECKPOINT was trained
DEFAULT_USE_TTA = True                     # TTA is what got the current best result
```
Once `variant1_clahe` is trained and evaluated, and **if** it beats Variant 1 + TTA, flip:
```python
DEFAULT_CHECKPOINT = VARIANT1_CLAHE_CHECKPOINT
DEFAULT_USE_CLAHE = True
```
Nothing else in the codebase needs to change — `src/inference.py` and `src/infer_stream.py` both read these
three config values.

> **Cost note on `DEFAULT_USE_TTA = True` for the live stream:** TTA runs several forward passes per
> classification (offsets × flip). Fine for the `/predict/video` endpoint and for the multi-second
> `CLASSIFY_INTERVAL_SECONDS` cadence used by `infer_stream.py`, but don't lower that interval too far on
> CPU without checking it still keeps up.

## Results

> Fill in after running `python -m src.evaluate`. Screenshots referenced below are auto-saved by
> `src/evaluate.py` into `docs/images/confusion_matrix_<checkpoint_name>.png`.


### DATASET 
The dataset was obtained from a paper 
"Video WeAther RecoGnition (VARG): An Intensity-Labeled Video Weather Recognition Dataset" by Himanshu Gupta et al : https://doi.org/10.3390/jimaging10110281

The dataset originally had 6742 annotated clips from 1079 videos, with the training set containing 5159 clips and the test set containing 1583 clips. With 7 classes categorized into three major weather categories, rain, fog, and snow, with three intensity classes: absent/no, moderate, and high. 

Snow is not a relevant weather condition for our target NDCC mine deployment, so all snow-labeled clips were excluded from training.

For our task, VARG's multi-label annotation file was used as the label source, but only clips with exactly one active weather condition were retained, converting the problem into single-label 5-class classification. Clips with zero or multiple simultaneously active labels were excluded.

5 classes we trained upon : clear , high fog, moderate fog , high rain , moderate rain

The final split was :
Train: 3778 | Val: 667 | Test: 1408


### ARCHITECTURE

### Architecture Choice

**Problem framing.** Our target problem is not generic fog/rain detection — that is
already well-solved by established single-image methods (e.g., dark-channel-prior and
atmospheric-scattering-model-based dehazing). Our actual problem is fog and rain
**severity** classification for HEMM driver alerting in mine terrain, where the practical
decision (whether to trigger a speed-limit warning) hinges specifically on the
Moderate-vs-Heavy boundary. This boundary was consistently the hardest part of every
experiment we ran (see Results), which shaped our architecture choice more than the
general "is there fog" question would have.

**Why a temporal architecture (TSM).** TSM (Lin et al., 2019) was chosen based on two
distinct, weather-type-specific justifications rather than one blanket "temporal is
better" claim:

- *Rain:* a single still frame cannot reliably distinguish rain droplets on a lens from
  static condensation/dew — both appear as blur or spots in one frame. What
  distinguishes them is behavior across frames: raindrops fall and visibly displace
  frame to frame, while dew remains static. This is not just our own intuition — Garg
  and Nayar (2004) established that rain's visual signature is inseparable from its
  temporal dynamics, showing that a single frame's intensity fluctuation from a
  raindrop is ambiguous, while its behavior across a short window of consecutive
  frames reliably distinguishes real rain from other effects. TSM gives a 2D CNN
  cheap access to this kind of inter-frame information, without the computational
  cost of full 3D convolutions or optical flow.
- *Fog:* the case for motion is weaker here, and we want to be upfront about that.
  Established fog/haze detection literature (dark-channel-prior, atmospheric-scattering
  approaches) generally works from a single static frame, since fog's core effect —
  scattering that reduces contrast and washes out color — is a global property of one
  image, not a discrete moving signal. Our temporal justification for fog is therefore
  narrower: aggregating information across multiple frames (via temporal mean-pooling
  in our architecture) provides a more *stable* severity estimate than any single frame
  would, buffering against transient per-frame noise specific to mine haul-road
  conditions — headlight/floodlight glare off suspended particles, dust from nearby
  HEMM traffic, and patchy, unevenly-distributed fog density.

**Why ResNet-50 as the backbone.** ResNet-50's residual (skip) connections allow
effective training at 50 layers of depth without the vanishing-gradient degradation
that affects plain deep CNNs, and its bottleneck block design (1×1 → 3×3 → 1×1
convolutions) keeps per-layer computation relatively low for that depth (He et al.,
2015). Combined with freely available ImageNet-pretrained weights, it offered the best
accuracy-to-setup-effort tradeoff among the backbones evaluated in the VARG paper
itself (TSM, I3D, X3D, VideoSWIN, MViT-v2) for our free-tier Colab/CPU compute
constraints — the alternatives require heavier 3D convolutions or transformer-scale
attention and less plug-and-play library support.

**Deployment-specific considerations.** Two gaps between VARG's training data and our
actual deployment context are worth stating explicitly:

- Our cameras are **fixed environmental perceptors** mounted on mine infrastructure,
  not vehicle-mounted, whereas VARG's clips are drawn from a mix of static, handheld,
  dashcam, and drone footage with no per-clip camera-type label — meaning we cannot
  verify what fraction of our training data matches our actual camera setup.
- Night-time artificial lighting (mine floodlights, vehicle headlights) is a realistic
  source of localized glare that could be mistaken for a change in weather intensity by
  a model relying on frame brightness as a cue — a risk directly suggested by the
  brightness/contrast distribution shift we measured between our train and test splits
  (see Limitations).

**What remains unverified.** We have not yet run a single-frame (non-temporal) baseline
on this same task and data. Doing so is the single most important next experiment to
confirm that TSM's temporal modeling is actually contributing to performance here,
rather than being a well-motivated assumption we have not directly tested against a
simpler spatial-only model.

**References**
- Lin, J., Gan, C., Han, S. (2019). *TSM: Temporal Shift Module for Efficient Video
  Understanding.* ICCV.
- He, K., Zhang, X., Ren, S., Sun, J. (2015). *Deep Residual Learning for Image
  Recognition.*
- Garg, K., Nayar, S.K. (2004). *Detection and Removal of Rain from Videos.* CVPR.
- Gupta, H., Kotlyar, O., Andreasson, H., Lilienthal, A.J. (2024). *Video WeAther
  RecoGnition (VARG): An Intensity-Labeled Video Weather Recognition Dataset.*
  J. Imaging, 10, 281.



**TLDR :** TSM + ResNet-50
Identical architecture across all five checkpoints (16 temporal segments, 224×224 frames)


### Brighness and Contrast Diagnostics : 

| Class | Train Brightness | Test Brightness | Train Contrast | Test Contrast |
|---|---:|---:|---:|---:|
| Clear | 82.17 | 77.00 | 3.25 | 0.74 |
| Rain Moderate | 85.34 | 117.30 | 1.49 | 1.21 |
| Rain Heavy | 116.21 | 91.86 | 2.62 | 1.39 |
| Fog Moderate | 127.74 | 117.65 | 1.43 | 0.90 |
| Fog Heavy | 129.67 | 139.73 | 1.69 | 1.51 |

### Results

The balanced-sampler variant achieved the highest test performance among the evaluated experiments, with **67.97% test accuracy**, **64.34% balanced accuracy**, and **65.08% macro F1**. The other sampler + focal-loss variants did not improve test performance over the balanced-sampler model. CLAHE also resulted in lower test metrics, suggesting that the additional preprocessing did not improve generalization on the test split.

The gap between the stored validation Macro F1 and the corresponding test metrics also indicates a noticeable **validation-to-test distribution shift**, which is consistent with the brightness and contrast differences observed between the train and test sets.


### Experiment Comparison

| Experiment | Checkpoint | Val Macro F1 (Stored) | Test Accuracy | Test Balanced Acc | Test Macro Precision | Test Macro Recall | Test Macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | `TSM_ResNet50_baseline_64.99.pth` | — | 0.6499 | 0.6342 | 0.6214 | 0.6342 | 0.6210 |
| Variant 1 (Balanced Sampler) | `TSM_ResNet50_balanced_best.pth` | 0.9268 | 0.6797 | 0.6434 | 0.6664 | 0.6434 | 0.6508 |
| Combined v1 (Sampler + Focal + Alpha) | `combined_sampler_focal_best.pth` | 0.9258 | 0.6385 | 0.6050 | 0.6238 | 0.6050 | 0.5997 |
| Combined v2 (Sampler + Focal, No Alpha + Jitter) | `combined_v2_sampler_focal_jitter_best.pth` | 0.9194 | 0.6548 | 0.6169 | 0.6568 | 0.6169 | 0.6200 |
| Variant 1 + CLAHE | `variant1_clahe_best.pth` | 0.9093 | 0.6349 | 0.6058 | 0.6080 | 0.6058 | 0.5962 |
| **Variant 1 + TTA** | Same checkpoint + TTA | — | **0.6825** | **0.6450** | **0.6701** | **0.6450** | **0.6532** |

### Test-Time Augmentation (TTA)

Test-Time Augmentation (TTA) was applied to the selected Variant 1 checkpoint at inference time.

| Metric | Plain Inference | TTA | Change |
|---|---:|---:|---:|
| Test Accuracy | 0.6797 | **0.6825** | +0.0028 |
| Balanced Accuracy | 0.6434 | **0.6450** | +0.0016 |
| Macro Precision | 0.6664 | **0.6701** | +0.0037 |
| Macro Recall | 0.6434 | **0.6450** | +0.0016 |
| Macro F1 | 0.6508 | **0.6532** | +0.0024 |

## Experiments

We ran five training configurations to see what actually moves the needle on weather severity classification. Short version: a balanced sampler helped a lot, and everything we tried on top of it made things worse.

**Baseline.** Plain cross entropy training with no class balancing, as a reference point. VARG's classes aren't evenly represented, so we expected this to lean toward the more common weather classes and undersell the rarer ones, and that's what happened (Acc 0.6499, macro F1 0.6210). The gap between accuracy and macro F1 is basically a symptom of imbalance: the model does fine on average but worse once every class is weighted equally.

**Variant 1, balanced sampler.** We addressed the imbalance directly by resampling batches so every class shows up roughly equally often during training, keeping the loss function itself unweighted. This was our strongest result by a clear margin (macro F1 0.6508, and 0.6532 with test time augmentation), confirming that class imbalance really was the main thing holding the baseline back, and that fixing it at the sampling stage was enough on its own.

**Combined v1, sampler plus focal loss plus class weighting.** Here we tried stacking a second balancing mechanism (focal loss with per class weights) on top of the sampler from Variant 1, expecting the two to reinforce each other. Instead, performance dropped (macro F1 0.5997), worse than Variant 1, and even worse than the baseline. Correcting for the same imbalance twice, once in the data and once in the loss, seems to have overcorrected and made training harder rather than better.

**Combined v2, sampler plus focal loss plus color jitter.** A follow up that removed the redundant class weighting (since the sampler already balances classes) and added color jitter augmentation instead. This recovered most of what Combined v1 lost (macro F1 0.6200), which confirms the class weighting was the culprit, but it still didn't beat the simpler Variant 1 recipe. Conclusion: neither focal loss nor jitter earned their added complexity here.

**Variant 1 plus CLAHE.** We'd noticed a brightness and contrast difference between our training and test videos and worried the model might be leaning on that as a shortcut instead of learning real weather cues, so we applied CLAHE (contrast normalization) to every clip to remove it. This backfired. It was our worst result (macro F1 0.5962), below even the baseline. Our best guess: some of that contrast difference wasn't a shortcut at all, it was real signal. Fog and heavy rain genuinely do reduce visual contrast, so flattening it out removed information the model actually needed, not just noise.

Bottom line: the balanced sampler alone, without extra loss tricks or preprocessing, gave us the best model, and adding test time augmentation improved it slightly further.

### Confusion Matrix

<img src="/assets/confusion matrix.png" alt="Confusion Matrix" width="700"/>

### Per-class results *(best checkpoint)*

### Per-Class Performance — Variant 1 (Balanced Sampler)

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Clear | 0.803 | 0.775 | 0.789 | 436 |
| Rain Moderate | 0.669 | 0.690 | 0.680 | 355 |
| Rain Heavy | 0.590 | 0.510 | 0.547 | 206 |
| Fog Moderate | 0.584 | 0.716 | 0.643 | 278 |
| Fog Heavy | 0.686 | 0.526 | 0.596 | 133 |
| **Accuracy** | — | — | **0.680** | 1408 |
| **Macro Avg** | **0.666** | **0.643** | **0.651** | 1408 |
| **Weighted Avg** | **0.684** | **0.680** | **0.679** | 1408 |

## Sharing our trained model
 
-  https://drive.google.com/drive/folders/1J-MWLi08SlvgPX20Rakkdm-Iznb24IKV?usp=sharing


## Limitations & honest caveats


* **The model has never seen a mine:** VARG contains general outdoor footage from roads, cities, forests, etc., but no open-pit mine or HEMM footage. Mine conditions such as dust, floodlights, unpaved roads, and heavy machinery may look different. **Real mine-site footage should be used for validation before deployment.**

* **Fog Heavy is the weakest class:** Recall for Fog Heavy was only **40–53% (best: 52.6%)**, so the model misses roughly half of the actual Fog Heavy clips, usually confusing them with Fog Moderate. This is especially important because Fog Heavy is the safety-critical condition.

* **Train/test differences need more investigation:** We found differences in brightness and contrast between the train and test sets. CLAHE was tested to reduce this effect, but performance actually decreased. We therefore cannot yet say whether these differences are harmful bias or genuine weather-related visual information.

* **TSM has not been properly ablated:** We have not compared the model against a **single-frame ResNet-50 baseline**. So we cannot confirm how much of the performance comes specifically from temporal modeling.

* **TSM may limit performance:** The original VARG study found TSM weaker than several other video architectures. Our model also uses ImageNet-pretrained ResNet-50 rather than video-pretrained weights, so a stronger temporal architecture may perform better.

* **Real weather can overlap:** We converted VARG into five single-label classes, removing clips with multiple weather labels. This means situations such as **rain + fog** are not represented, even though they can occur in real mines.

* **Camera mismatch:** VARG contains dashcam, handheld, drone, and static footage, while our deployment uses fixed CCTV cameras. The extent of this mismatch has not been measured.

* **Night lighting is still an untested factor:** Mine floodlights and headlights could influence brightness-based predictions, but we have not yet verified whether this affects classification.

* **Limited training experiments:** Due to CPU/Colab constraints, experiments were limited to **5 epochs with fixed training settings**. More tuning and training could change the results.

* **Annotation uncertainty:** VARG's Moderate/Heavy labels involve some manual judgment, so borderline predictions may partly reflect ambiguity in the original labels.

* **TTA adds latency:** Test-Time Augmentation slightly improved metrics but requires multiple predictions per clip. Its effect on real-time alert latency has not yet been benchmarked.

* Future work includes loading Kinetics-400-pretrained TSM weights (e.g. from the original TSM release or MMAction2's model zoo) instead of ImageNet-pretrained weights, which would require remapping checkpoint keys to match our architecture's module structure, but should better align with the reference paper's setup and likely improve the model's ability to exploit temporal information from the first epoch of fine-tuning.

