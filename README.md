# PCGKolect_model

**Heart-sound recording-quality models for PCGKolect.** Training, evaluation and deployment code
for the models that run on the PCGKolect device's Rockchip RV1103.

> **PCGKolect**: *System and Method for Intelligent Phonocardiogram Dataset Acquisition Using
> Real-Time Recording Quality Assessment.* A DIY digital stethoscope and phonocardiogram (PCG)
> data-acquisition platform. It judges, while recording, whether the heart sounds being captured
> are usable for a research dataset. It is a research instrument, **not a diagnostic device**.

| Repository | Role |
|---|---|
| [PCGKolect](https://github.com/cosmosapien200/PCGKolect) | ESP32-S3 firmware: PCM1803A ADC capture (I²S/DMA), USB audio, 8 kHz framed PCM over UART to the RV1103 |
| [PCGKolect_rk](https://github.com/cosmosapien200/PCGKolect_rk) | RV1103 (Luckfox Pico) firmware: UART receiver, quality pipeline in C++, RKNN NPU inference, SSD1306 OLED, session recording |
| **PCGKolect_model** (this repo) | Datasets, training, evaluation, ONNX → RKNN export, and Python reference implementations of everything the firmware runs |

The full story of how the models were built, and why each choice was made, is in **[DEVLOG.md](DEVLOG.md)**.

---

## What the models do

Two models, both deployed on the RV1103:

| | **v2: beat-level segmentation quality** (primary) | **v1: SQI features + logistic regression** |
|---|---|---|
| Runs on | NPU (INT8 RKNN, 1.65 ms per inference) + CPU decoder | CPU (≈ 0.4 ms per second of audio) |
| Output | per **20 ms frame**: P(S1, systole, S2, diastole, noise); per **beat**: usable / rejected + confidence | per **1 s**: quality score, GOOD / MARGINAL / BAD + reason |
| Role on device | decides which heartbeats are usable; drives the OLED | instant deterministic gates (clipping, no contact, too loud), and the fallback when no model file is present |

v2 pipeline on the device (identical in Python and C++):

```
UART int16 @ 8 kHz
 → 63-tap FIR low-pass 800 Hz + ÷4 → 2 kHz → 20 Hz high-pass (Butterworth, 2nd order)
 → every 1 s, last 5.164 s: 128-pt Hann STFT, hop 40 (20 ms) → 32 mel bands 25–900 Hz → log → −mean, ÷4
 → NPU: 1D U-Net [1,32,1,256] → [1,5,1,256] logits                       (364k params, INT8)
 → CPU: softmax → frames with ≥ 1.28 s future context → 20 s history
 → duration-constrained Viterbi (S1 → systole → S2 → diastole, noise may interrupt)
 → beats: usable if confidence ≥ 0.70, 0.3–2.0 s long, within ±30 % of neighbouring beats
 → v1 gates veto beats → usable beats, longest run, coverage, HR → OLED / CSV
```

## Results (5-fold cross-validation on held-out PhysioNet recordings; v1 retrained on the same splits)

| Metric | v1 | **v2** |
|---|---|---|
| 2 s window usable / unusable vs expert noise marks: AUC | 0.923 | **0.964** |
| same: best balanced accuracy | 0.852 | **0.898** |
| Recording accept / reject vs expert signal-quality label: AUC (mean of folds) | 0.932 | **0.943** |
| same: best balanced accuracy | 0.879 | **0.894** |
| Per beat usable / unusable, unseen noise at −10…+20 dB: AUC | n/a | **0.949** |
| same: balanced accuracy of the device's usable flag | n/a | **0.919** |
| S1 / S2 onset detection F1 (±60 ms) | n/a | **0.973 / 0.963** |
| Frame state accuracy (S1 / systole / S2 / diastole) | n/a | 0.946 |
| Real-noise frame detection AUC | n/a | 0.955 |

On the board: the INT8 model agrees with PyTorch float on **99.9 %** of frames (logits cosine 0.9999).
A replay of 15 held-out recordings (clean + noisy) through the full C++ firmware pipeline produced
**507 / 507 identical beats and decisions** compared with the Python float reference.
Raw logs and per-fold JSON are in [`results/`](results/).

**Caveats.**
- The per-beat numbers use controlled corruption, because real noise stretches carry no beat
  annotations.
- Splits are by recording; only subset b has subject IDs.
- All training data come from electronic stethoscopes. Accuracy on the PCGKolect piezo sensor has
  **not yet been measured**: calibrate or fine-tune on device recordings first (see DEVLOG §9).

---

## Repository layout

```
README.md, DEVLOG.md
requirements.txt          training / evaluation environment (Python 3.12)
requirements-rknn.txt     separate environment for RKNN conversion (Python 3.11, rknn-toolkit2 2.3.2)

seg/                      ── v2 (segmentation-based beat quality) ──
  prep_seg.py             build the labelled corpus (PhysioNet 2016 + CirCor) → data/seg_corpus.npz
  prep_noise.py           build the noise bank (ESC-50, Mini LibriSpeech, PhysioNet noise) → data/noise_bank.npz
  model.py                front-end (STFT + filterbank) and the 1D U-Net
  train_seg.py            corruption / augmentation, training, 5-fold CV, window-level comparison with v1
  decode.py               duration-constrained Viterbi + beat scoring (mirrored in C++)
  eval_beats.py           S1/S2 F1, recording-level vs expert SQI, per-beat accuracy under controlled noise
  stream.py               Python reference of the device's streaming scheduler
  export_onnx.py          checkpoint → ONNX + INT8 calibration set + reference I/O
  convert_rknn.py         ONNX → RKNN INT8 for rv1103 + PC-simulator accuracy check
  gen_c_tables.py         generates PCGKolect_rk/include/seg_params.h
  check_board.py          on-board replay of recordings vs the float reference (needs adb)
  review_session.py       run v2 on any recording, write an Audacity label track

sqi_features.py           ── v1 ── 19 signal-quality features (mirrored in C++)
sqi_frontend.py           v1 device front-end + streaming scorer (Python reference)
build_dataset.py          v1 window dataset from PhysioNet noise annotations
train.py                  v1 training / CV / thresholds → models/sqi_lr_v1.json + PCGKolect_rk/include/sqi_model_params.h
score_wav.py              v1 per-second timeline for any WAV
models/sqi_lr_v1.json     v1 model

tests/                    C++ ↔ Python parity (host g++): test_parity.py (v1), test_seg_parity.py (v2)
release/                  the deployed v2 model: seg_final.pt, segnet.onnx, segnet_w8a8_normal.rknn
results/                  cross-validation logs and per-fold beat/recording metrics

data/ runs/ export/       working directories (git-ignored, regenerated by the scripts)
```

---

## Datasets

| Dataset | Used for | Size used | Source | Licence |
|---|---|---|---|---|
| PhysioNet/CinC Challenge 2016, training set a–f | v1 + v2 training and **all evaluation** | 3,153 recordings, 19.6 h, 2 kHz | `training.zip` and `annotations.zip` from [physionet.org/content/challenge-2016/1.0.0](https://physionet.org/content/challenge-2016/1.0.0/) | open access (see PhysioNet page) |
| PhysioNet 2016 hand-corrected annotations (Liu et al.) | S1 / systole / S2 / diastole labels and `(N … N)` noise brackets | 3,153 files | `annotations/hand_corrected/` in `annotations.zip` | as above |
| PhysioNet 2016 signal-quality labels | recording-level quality (SQI 0/1) | all | `annotations/updated/training-*/REFERENCE_withSQI.csv` | as above |
| CirCor DigiScope 1.0.3 | v2 training (segmentation only) | 3,163 recordings, 20.1 h, 4 kHz | [physionet.org/content/circor-heart-sound/1.0.3](https://physionet.org/content/circor-heart-sound/1.0.3/) | ODC-By 1.0 |
| ESC-50 | non-cardiac noise (knocks, footsteps, breathing, coughs, …) | 1,999 clips | [github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50) | **CC BY-NC 3.0, non-commercial** |
| Mini LibriSpeech dev-clean-2 | speech noise | 1,089 utterances | [openslr.org/31](https://www.openslr.org/31/) | CC BY 4.0 |

> **Licence note for a product or patent:** ESC-50 is non-commercial. For commercial use, replace it
> with your own noise recordings, which `prep_noise.py` accepts as an extra source.

PhysioNet's own server can be very slow; the official AWS mirror is much faster:
```bash
mkdir -p ~/Datasets/pcg && cd ~/Datasets/pcg
curl -LO https://physionet-open.s3.amazonaws.com/challenge-2016/1.0.0/training.zip    && unzip -q training.zip -d pn2016
curl -LO https://physionet-open.s3.amazonaws.com/challenge-2016/1.0.0/annotations.zip && unzip -q annotations.zip -d pn2016
aws s3 sync --no-sign-request s3://physionet-open/circor-heart-sound/1.0.3/ circor    # or the ZIP from the CirCor page
curl -L -o esc50.zip https://github.com/karoldvl/ESC-50/archive/master.zip && unzip -q esc50.zip -d esc50
curl -LO https://www.openslr.org/resources/31/dev-clean-2.tar.gz && mkdir -p speech && tar xzf dev-clean-2.tar.gz -C speech
```
This gives `pn2016/training-a … training-f/` and `pn2016/annotations/{hand_corrected,updated}/`.
`prep_seg.py` expects `hand_corrected` and `updated` as siblings, which is how the zip unpacks.

---

## Setup

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt     # training (GPU recommended)
python3.11 -m venv .venv-rknn && .venv-rknn/bin/pip install -r requirements-rknn.txt   # conversion only
```
Tested on an RTX 3060 (12 GB) with 32 GB RAM. One CV fold trains in about 7 min; peak RAM is about 3 GB.

## Train and evaluate v2

```bash
# 1. data (≈ 1 GB in data/)
python seg/prep_seg.py --pn ~/Datasets/pcg/pn2016 --pn-ann ~/Datasets/pcg/pn2016/annotations/hand_corrected \
       --circor ~/Datasets/pcg/circor/training_data --out data/seg_corpus.npz
python seg/prep_noise.py --esc50 ~/Datasets/pcg/esc50 --speech ~/Datasets/pcg/speech \
       --corpus data/seg_corpus.npz --out data/noise_bank.npz

# 2. 5-fold cross-validation (+ v1 retrained on the same folds), then beat/recording-level metrics
python seg/train_seg.py --fold all                       # ≈ 45 min
for k in 0 1 2 3 4; do python seg/eval_beats.py --fold $k; done

# 3. final model on all data
python seg/train_seg.py --final                          # → runs/seg_final.pt
```

## Deploy v2 to the RV1103

```bash
python seg/export_onnx.py --ckpt runs/seg_final.pt --width 32,48,64,96 --out export   # ONNX + calibration
.venv-rknn/bin/python seg/convert_rknn.py --export export      # → export/segnet_w8a8_normal.rknn
python seg/gen_c_tables.py                                     # → PCGKolect_rk/include/seg_params.h
python tests/test_seg_parity.py                                # must print SEG PARITY PASS
cp runs/seg_final.pt export/segnet.onnx export/segnet_w8a8_normal.rknn release/
cd ../PCGKolect_rk && make deploy                              # firmware + release/segnet_w8a8_normal.rknn
python ../PCGKolect_model/seg/check_board.py --export ../PCGKolect_model/export --bin build --fold 0   # optional
```
To include device audio in the INT8 calibration set (recommended once you have it), add
`--device-wavs rec1.wav rec2.wav ...` to `export_onnx.py`.

## v1 (CPU model)

```bash
python build_dataset.py --physionet ~/Datasets/pcg/pn2016 --out data/windows.npz
python train.py                # CV report, thresholds → models/sqi_lr_v1.json + PCGKolect_rk/include/sqi_model_params.h
python tests/test_parity.py    # must print PARITY PASS
python score_wav.py some.wav   # per-second v1 timeline
```

## Reviewing device recordings

The firmware records every session to `/root/recordings/session_NNN.wav`, plus `_beats.csv` and
`_quality.csv`. From `PCGKolect_rk`:
```bash
make pull-recordings
python tools/view_in_audacity.py          # menu → opens Audacity with the audio + Beats + Quality label tracks
python tools/view_in_audacity.py -s 002   # a specific session
```
`seg/review_session.py` here does the same analysis on the PC for any WAV (used for pre-labelling
fine-tuning data).

## Invariants

- Every constant shared with the firmware is generated: `train.py` → `sqi_model_params.h`,
  `seg/gen_c_tables.py` → `seg_params.h`. Never edit those headers by hand.
- Any change to features, front-end or decoder must be made in Python **and** C++, and both parity
  tests must pass. Do not build the firmware with `-ffast-math`.
