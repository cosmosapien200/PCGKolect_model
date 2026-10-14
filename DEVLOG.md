# PCGKolect_model — Development Log

How the PCGKolect recording-quality models were designed, trained, evaluated and deployed, and
**why** each decision was made. Dates are 2026. Numbers are from the scripts in this repo; raw
logs are in `results/`.

Related repos: [PCGKolect](https://github.com/cosmosapien200/PCGKolect) (ESP32-S3 acquisition) ·
[PCGKolect_rk](https://github.com/cosmosapien200/PCGKolect_rk) (RV1103 firmware).

---

## 0. The problem and the constraints

**Goal.** While a heart sound is being recorded, tell the operator whether the audio is usable for
a PCG research dataset, continuously (every 1–2 s or per heartbeat). Aggregate the decisions into
session metrics: usable time, usable beats, longest clean run. Quality only; no diagnosis.

**Constraints that shaped everything:**

| Constraint | Consequence |
|---|---|
| ESP32 → RV1103 link is **8 kHz, int16, mono** over UART (100 ms frames) | Heart sounds (20–400 Hz) and the murmur band (≤ 800 Hz) fit easily. All models analyse at 2 kHz, an exact ÷4 decimation |
| Sensor is a **piezo contact pickup** with fixed analog gain | Absolute level is meaningful (contact / clipping gates), but the frequency response differs from the electronic stethoscopes in public data, so models must be loudness-invariant and robust to spectral tilt |
| RV1103: Cortex-A7, 64 MB RAM, **INT8-only NPU** (0.5 TOPS), `librknnmrt` zero-copy API, no Python on board | Small models, NPU-supported operators only, C++ inference, every Python step re-implemented in C++ and parity-tested |
| No labelled data from the device yet | Train on public corpora; design for later calibration / fine-tuning on device audio |

---

## 1. Groundwork (2026-10-13)

- **Prior work found on the machine:**
  - `PGT-1`: a heavy, never-trained segmentation + quality design.
  - `Model`: age/sex classifiers on CirCor.
  - Reused: the PhysioNet 2016 and CirCor downloads, and PGT-1's insight that **2 kHz** is reachable
    from every corpus by integer decimation (PhysioNet 2 kHz ÷1, CirCor 4 kHz ÷2, device 8 kHz ÷4),
    so no fractional resampler sits between training and deployment.
  - Not reused: PGT-1's assumptions (16 kHz codec, non-piezo sensor), which don't match the hardware.
- **Literature** (what quality assessment for PCG actually uses):
  - Springer et al. 2016, mobile-phone PCG quality: kurtosis, degree of periodicity and similar
    features with logistic regression; experts agreed poorly (kappa 0.24–0.54).
  - Tang et al. 2021: feature-based; *degree of periodicity* is the single best feature.
  - Grooby et al. 2021: real-time multi-level neonatal quality.
  - A 2026 CinC bone-conducted PCG quality paper: LightGBM on handcrafted features beats deep nets
    on small data.
  - Takeaway: with scarce labels, handcrafted SQIs are hard to beat. Segmentation U-Nets reach
    ≈ 0.96 F1 on PhysioNet 2016, and self-supervision helps out-of-distribution robustness.
- **The label source.** PhysioNet/CinC 2016's hand-corrected segmentation (Liu et al.) brackets
  unusable stretches with `(N … N)`. That turned out to be the only real *segment-level* quality
  label in public data, and it became the evaluation ground truth for everything.

---

## 2. v1: SQI features + logistic regression (CPU)

### Labels
- **Windows:** 2 s scored windows, hop 0.5 s.
- **BAD:** ≥ 25 % of the window inside a noise bracket, or the recording has SQI = 0 and < 50 % of
  the window is segmented.
- **GOOD:** ≥ 90 % inside annotated S1/systole/S2/diastole, no noise, and the recording has SQI = 1.
- **Everything else is skipped.** The thresholds leave a margin so ambiguous windows don't
  contaminate either class.

### Experiments
Patient-grouped 5-fold CV on the PhysioNet subsets available then (a, c, d, f):

| Variant | AUC | Decision |
|---|---|---|
| 17 SQI features + logistic regression | 0.875 | baseline |
| same + LightGBM | 0.866 | no gain, less transparent |
| + CirCor weak labels (annotated = good, unannotated = bad) | ≤ 0.875, BA dropped to 0.67 | **rejected**: CirCor's unannotated regions are not reliably noisy |
| tiny depthwise CNN on log-mel (15–31k params), with/without CirCor | 0.73–0.78 | **rejected**: 2.4k bad windows are too few for a CNN |
| SQI features with **4 s context** (1 s each side of the 2 s window) | **0.905** | **kept**: periodicity needs more than 2 s to see 2+ beats |

**Strongest single feature:** envelope-autocorrelation periodicity (AUC 0.80), consistent with Tang 2021.

### C-friendly redesign
Features were re-specified so C++ could reproduce them bit-for-bit (`sqi_features.py`):
- causal 20 Hz high-pass instead of `filtfilt`;
- 256-point Welch PSD, so power-of-2 FFTs;
- a 20 ms block-mean envelope instead of a homomorphic envelope;
- an explicit unbiased autocorrelation.

AUC stayed at 0.902, so nothing was lost.

### Model, thresholds and gates
- **Model:** standardised logistic regression (C = 0.1), 19 features.
- **Why LR:** it was the best scorer, it's 20 multiply-adds on device, it's explainable, and it can be
  re-fitted on a few hundred device windows.
- **Thresholds:** picked from out-of-fold scores.
  - GOOD when ≤ 2 % of GOOD windows are noisy. Admitting a bad segment into a dataset costs more than
    rejecting a good one.
  - BAD when ≥ 50 % of BAD windows are noisy.
- **Deterministic gates on the raw int16:**
  - clipping > 0.1 %;
  - level < −60 dBFS (no contact) or > −3 dBFS (too loud), both provisional.
  - These are deterministic facts; putting them in a model would make them harder to verify and no
    more accurate.

### Results and transfer
- **Cross-validation:** AUC 0.902, balanced accuracy 0.82.
- **Leave-one-database-out:** AUC 0.79–0.96, but fixed-threshold accuracy collapsed to 0.40 on one
  subset. **The score ranks well across devices; thresholds do not transfer.**
- **Recalibrating** the threshold with 30 recordings of the new database gives balanced accuracy
  0.72–0.89.
- **Conclusion:** device calibration is mandatory, and about 82 % is the ceiling of this approach.
  That motivated v2.

### Deployment
- `PCGKolect_rk/src/signal_quality.cpp`, with constants generated into `sqi_model_params.h` by `train.py`.
- C++/Python parity: max difference 2.75e-13, identical verdicts.
- Ran on the board first time; about 0.36 ms per second of audio on x86.

---

## 3. Why v2 changes the question

v1 learns "does this window look like PhysioNet's good windows?" from about 2.4k bad examples
labelled by people who disagree with each other. A bigger network on the same labels just overfits.
That was observed directly: the CNN scored worse.

v2 asks instead: **"can S1 and S2 be located reliably in this beat?"** That question:
- **has abundant labels:** about 3,150 + 3,160 recordings are hand-segmented into
  S1/systole/S2/diastole;
- **matches the goal:** usable heartbeats for a dataset;
- **should be more device-independent** than "looks like the training windows";
- **is checkable by physiology:** S1 → systole → S2 → diastole with plausible durations.

To stop a segmenter from calling every knock "S1", the model gets an explicit **noise** class (§4).
Its confidence is then judged along a decoded, physiologically constrained path (§6).

---

## 4. v2 data

### Corpora (`seg/prep_seg.py`)

**PhysioNet 2016**
- All subsets a–f: 3,153 recordings, 19.6 h. Full set from the AWS mirror; PhysioNet's own server
  ran at 6 KB/s.
- Labels per sample: S1 / systole / S2 / diastole from the hand-corrected states, and **noise**
  inside `(N … N)`.
- Before the first and after the last annotation → *ignore*.
- 8 % of labelled time is noise, about 1.6 h of real expert-marked noise. v1 had 2.4k windows.
- Recording-level SQI from `annotations/updated/*/REFERENCE_withSQI.csv`.

**CirCor 1.0.3**
- 3,163 recordings, 20.1 h.
- States from the `.tsv` files; **unannotated → ignore, not noise.** The v1 experiment showed
  "unannotated" is a poor noise proxy.
- Contributes paediatric cardiac variety; never used for evaluation.

**Preprocessing**
- Every recording is resampled to 2 kHz, causally high-passed at 20 Hz, and peak-normalised.
- Peak normalisation is irrelevant to the model (the front-end is gain-invariant) but keeps float16
  storage well-conditioned.

### Noise bank (`seg/prep_noise.py`)

| Source | Clips | Minutes | Why |
|---|---|---|---|
| ESC-50 | 1,999 | 167 | knocks, footsteps, clicks, breathing, coughing, crackling, the impulsive sounds most easily mistaken for S1 |
| Mini LibriSpeech dev-clean-2 | 1,089 | 114 | talking during auscultation |
| PhysioNet `(N … N)` stretches ≥ 1 s | 423 | 103 | real stethoscope handling and rub noise |
| synthetic (generated on the fly) | n/a | n/a | white (sensor floor), pink, brown (pressure / rub drift), mains hum 50/60 Hz + harmonics, bursty friction |

**Leakage controls**
- Every PhysioNet noise clip is tagged with its source recording. A fold never trains on noise cut
  from its own test recordings; caught and fixed before the first real run.
- 20 % of clips (hash-selected) are held out as *test noise*, used only in the per-beat evaluation.

---

## 5. v2 model

### Front-end (`seg/model.py: Frontend`), with the reason for each choice

| Step | Choice | Why |
|---|---|---|
| Rate | 2 kHz | integer decimation from every source; Nyquist 1 kHz covers S1/S2 and the murmur band |
| Frames | 128-pt (64 ms) periodic Hann, hop 40 (20 ms, 50 fps) | 50 fps is the standard PCG segmentation rate (Springer); 64 ms resolves S1/S2 (≈ 100 ms) |
| Bands | 32 mel-spaced triangular bands, 25–900 Hz | compact (32 = an NPU-friendly channel count); low-frequency resolution where S1/S2 energy is |
| Compression | `log(power + 1e-10)` | heart sounds span a large dynamic range |
| Normalisation | subtract the window's global mean, ÷4 | **gain invariance**: piezo pressure changes the level, which must not change the decision. Absolute level is handled by the v1 gates. The global mean (not per-band) keeps the spectral shape. ÷4 brings values to about ±6, a good INT8 range |
| Window | 256 frames = 5.12 s (10,328 samples) | several beats of context to tell S1 from S2 (systole is shorter than diastole); width 256 is 16-aligned, as RV1103 requires |

Why not raw waveform? A 1D conv front-end on 10k samples costs far more NPU work, and the STFT plus
filterbank is about 40 lines of C++, trivially parity-tested.

### Network (`SegNet`): 1D U-Net, 364,325 parameters

```
[B,32,1,256] → e0: 2×(conv 1×5) 32ch → e1: stride-2, 48ch → e2: stride-2, 64ch
            → e3: stride-2, 96ch, dilations 2/4/8           (bottleneck at 32 frames ≈ 3 s receptive field)
            → up×2 + concat e2 → 64ch → up×2 + concat e1 → 48ch → up×2 + concat e0 → 32ch → 1×1 → 5 logits
```

- **Written as Conv2d with kernel 1×k on [B, C, 1, T]**, so the ONNX graph contains only Conv, ReLU,
  Resize (nearest ×2) and Concat. All are INT8-native on RV1103; Resize is supported only at integer
  ×2–×8 scales, which this is.
- **No GlobalAveragePool** (limited to ≤ 7×7 on RV1103) and **no softmax in the graph** (a CPU-fallback
  op). Softmax runs on the CPU after dequantisation. BatchNorm folds into the convolutions at conversion.
- **Why a U-Net:** frame-wise labels need both the local onset detail (skip connections) and multi-beat
  context (dilated bottleneck). Published PCG segmenters with this shape reach about 0.96 F1.
- **Size check:** a wider variant (48/64/96/128, 706k parameters, 2× steps) scored *worse* on fold 0
  (window AUC 0.958 vs 0.967), so the smaller model was kept.

### Corruption: teaching "noise" vs "S1 under light noise" (`train_seg.py: corrupt`)

**Each training window gets one of four treatments:**

| Mode | Probability | What |
|---|---|---|
| clean | 0.35 | original |
| partial overlay | 0.35 | noise over a random 0.3–4 s span (20 ms ramps) |
| full overlay | 0.15 | noise over the whole window |
| pure noise | 0.15 | window replaced by noise → all frames labelled noise |

- Noise is 70 % from the bank and 30 % synthetic. SNR is drawn uniformly from −20 to +25 dB,
  relative to the window's beat-average heart power.
- **Relabelling rule:** per frame, local noise power is smoothed over 200 ms and compared with the
  beat-average heart power.
  - Below 0 dB (`TAU_DB`), a cardiac frame becomes **noise**.
  - An *ignore* frame becomes noise only below −10 dB.
  - So light noise keeps the S1/S2 labels (robustness) and heavy noise flips them (detection).
- **Why 0 dB:** S1/S2 peaks sit roughly 10 dB above the beat-average power, so at 0 dB they are
  still visibly above the noise; below that, timing becomes unreliable. A conservative choice for
  dataset quality; it's a single constant.

**Other augmentation**
- Random polarity flip.
- Feature-domain "sensor EQ": a random spectral tilt of ±1.5 log units across the bands plus a
  smooth random curve. This simulates different transducer frequency responses, preparing for the
  piezo.

### Training
- **Optimiser:** AdamW, lr 2e-3, weight decay 1e-3, OneCycle schedule (10 % warm-up), **12,000
  steps × batch 64**.
- **Loss:** cross-entropy with ignore index and label smoothing 0.05, which helps calibration.
- **Sampling:** windows by recording length; PhysioNet (adult) weighted ×1.5; re-drawn until
  ≥ 30 % of frames are labelled.
- **Cost:** about 7 min per fold on an RTX 3060. Data generation on the CPU is the bottleneck; peak
  RAM about 3 GB.

---

## 6. Beat decoder (`seg/decode.py`, mirrored in `seg_quality.cpp`)

**Hybrid HMM / Viterbi over the network posteriors**
- Cardiac states must cycle S1 → systole → S2 → diastole.
- Each state has a **minimum duration** (sub-state chain: 3/4/3/4 frames) and a geometric tail
  (mean 6/14/5/25 frames).
- A **noise** state (minimum 5, mean 50 frames) can be entered from the end of any state
  (p = 0.01) and resumes into any state.

**Why a decoder:** the order of heart sounds is a physiological fact, so enforcing it beats learning
it. It turns frames into discrete beats, and random impulsive sounds rarely form a consistent cycle.

**Beats:** S1-onset to next S1-onset with exactly one S1/systole/S2/diastole and no noise.
**Usable** when all of:
- mean posterior of the decoded state ≥ 0.70;
- 0.3–2.0 s long (30–200 bpm);
- within ±30 % of the median of the neighbouring beats.

These thresholds are physiological priors, **not tuned** on the test data. Re-tune them on device data.

**Behaviour:** noisy stretches produce *no beats at all* rather than rejected beats. So the session
metrics are usable beats, the longest consecutive run (reset on gaps), and **coverage** (usable beat
time / scored time), not usable/total beats.

---

## 7. Evaluation

**Protocol**
- 5 folds by `crc32(recording id) % 5` over PhysioNet. CirCor is always training data.
- **v1 is retrained on exactly the same folds** (PhysioNet windows only, as in its own pipeline),
  so the comparison is like-for-like.
- **Inference for evaluation mimics the device:** 256-frame windows, hop 128, central frames kept.

**Results** (pooled or averaged over 5 folds; see `results/cv5.log`, `results/beats_fold*.json`)

| Metric | v1 | v2 |
|---|---|---|
| 2 s window usable / unusable vs expert noise marks: AUC | 0.923 | **0.964** |
| same: best balanced accuracy | 0.852 | **0.898** |
| Recording accept / reject vs expert SQI: AUC | 0.932 | **0.943** |
| same: best balanced accuracy | 0.879 | **0.894** |
| Per beat usable, held-out noise at −10…+20 dB: AUC / BA of usable flag | n/a | **0.949 / 0.919** |
| S1 / S2 onset F1 (±60 ms) | n/a | **0.973 / 0.963** |
| Frame state accuracy / real-noise frame AUC | n/a | 0.946 / 0.955 |

**Per fold:** window AUC 0.95–0.97, beat BA 0.916–0.923. Weakest subsets: b (AUC 0.82; short,
noisy recordings) and f (0.81).

**Ablations (fold 0)**
- Wider model: worse (§5).
- Rank-fusion of v1 and v2: +0.008 balanced accuracy at best, so not worth the complexity.
- Window-score variants (mean(1 − P_noise), mean(u · conf), 20th percentile): within 0.002 AUC of
  each other.

**Caveats to state in a paper**
- Per-beat numbers use controlled corruption, because real noise stretches carry no beat annotations.
- Splits are by recording; only subset b has subject IDs, so some subjects may straddle folds.
- Recording-level SQI labels are themselves noisy.
- All data are electronic stethoscopes, not the piezo device.
- ESC-50 is a non-commercial licence.

---

## 8. Deployment to the RV1103 NPU

**Conversion**
- PyTorch → ONNX (opset 13) → rknn-toolkit2 **2.3.2**, matching the board's `librknnmrt 2.3.2`.
- `target_platform='rv1103'`, `w8a8`, `normal` quantisation, `optimization_level=3`.
- **Calibration:** 300 public windows mixing clean, noisy and pure noise, so activation ranges cover
  both regimes.
- PC simulator vs float: 99.95 % frame agreement, cosine 0.9999. The fallbacks (`mmse`, `w16a16i`)
  were not needed.
- **Toolchain gotchas:**
  - `onnx` must be 1.16.1 (newer versions removed `onnx.mapping`).
  - `.npy` calibration files must be NCHW.

**C++ runtime** (`PCGKolect_rk/src/seg_model.cpp`)
- The zero-copy API (the only one on RV1103).
- **The native input is NC1HWC2 `[1,2,1,256,16]`.** The runtime can't convert int8 NHWC to it, so
  the CPU quantises the features (scale 0.0498, zp −2) and writes the native layout with
  `pass_through = 1`.
- **The NHWC output is dense `[256 × 5]`** (scale 0.0282, zp −16). `size_with_stride` (4096) is
  allocation padding, not a channel stride. Misreading it gave cosine ≈ 0 until diagnosed by dumping
  raw bytes.
- **Speed:** 1.65 ms per inference including quantise, dequantise and softmax. Model file 425 KB.

**Streaming** (`seg_quality.cpp`, reference `seg/stream.py`)
- Inference every 1 s on the last 5.164 s.
- Only frames with ≥ 1.28 s of future context are accepted (frames 142–191 of each window), so
  every frame is judged with context on both sides.
- 20 s posterior history; Viterbi re-run each second; beats emitted 0.5 s after they end.
- **Latency:** about 2–3 s from heartbeat to OLED. CPU about 6 %.

**Verification chain**
- Host parity (`tests/test_seg_parity.py`): features within 2e-4 of Python; Viterbi, beats and
  the whole streaming path identical, with an NPU replay stub.
- Board numerics (`tools/npu_selftest`): 99.9 % frame agreement with PyTorch.
- Board end-to-end (`tools/pcg_replay`, `seg/check_board.py`): 15 held-out recordings, clean +
  noisy → **507 / 507 identical beats and decisions**.
- Firmware replay mode (`PCG_REPLAY=`) on a scripted demo: the OLED went GOOD → NOISY →
  NO_CONTACT → GOOD.
  - This caught a tracker bug: the longest run didn't reset across noise gaps. Fixed.
  - It also caught a v1 `log(0)` on digitally silent windows. Fixed with an epsilon on both sides.

---

## 9. First device recordings and next steps (2026-10-14)

**First piezo session** (`session_001`, 74.5 s)
- The model locked onto a steady heartbeat (≈ 62 bpm, beat confidence 0.8–0.93) without any device
  fine-tuning.
- But 25 of 33 beats were vetoed by the **clipping gate**: S1 peaks reach full scale (0.1–0.6 %
  of samples during clean stretches, up to 10 % during handling).
- **Recommendation:** lower the analog gain about 6 dB. Optionally make clipping a warning rather
  than a veto unless it's heavy.

**Next**
1. Rebuild the INT8 calibration with device audio (`export_onnx.py --device-wavs`); set the level
   gates from measured off-chest / on-chest levels.
2. Feature-statistics alignment (per-band mean/std of device vs training), which needs no labels.
3. Device noise bank: noise-only recordings (no heart in range) need no subjects and mix into clean
   device beats with exact labels.
4. Light fine-tune: last layers + BatchNorm, device data mixed 50/50 with public data. Labels are
   high-confidence pseudo-labels plus human review in Audacity (`PCGKolect_rk/tools/view_in_audacity.py`,
   `seg/review_session.py`). An ECG module would give objective beat timing.
5. Evaluate leave-one-person-out on scripted sessions (quiet → talk → rub → lift → quiet). With
   2 people, report it as a pilot study.
