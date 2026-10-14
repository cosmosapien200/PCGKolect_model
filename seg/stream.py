"""Python reference of the device's streaming SegQuality (PCGKolect_rk/src/seg_quality.cpp).

  beats = stream_beats(pcm16_8k, infer)   # infer(feat [32, 256]) -> probs [256, 5]

Same decimator/HP as v1, inference every 1 s on the last 5.164 s, frames accepted once they
have 1.28 s of future context, 20 s posterior history, beats emitted 0.5 s after they end.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from sqi_frontend import device_to_2k  # noqa: E402
from model import FRAMES, SAMPLES, HOP, NFFT, NBANDS, HANN, FB, FEAT_SCALE, NOISE  # noqa: E402
from decode import viterbi, beats, FPS  # noqa: E402

STEP_SAMPLES = 2000
STEP_FRAMES = STEP_SAMPLES // HOP
RIGHT_CTX = 64
HIST_FRAMES = 1000
STABLE_MARGIN = 25


def features(x):
    """x [SAMPLES] -> [32, 256] float64 (numpy twin of model.Frontend)."""
    idx = np.arange(FRAMES)[:, None] * HOP + np.arange(NFFT)[None, :]
    S = np.fft.rfft(x[idx].astype(np.float64) * HANN, axis=1)
    M = np.log(FB.astype(np.float64) @ (np.abs(S) ** 2).T + 1e-10)
    return (M - M.mean()) / FEAT_SCALE


def stream_beats(pcm16, infer):
    x = device_to_2k(pcm16).astype(np.float32)
    hist = np.zeros((0, 5), np.float32)
    hist_start = next_frame = 0
    last_end = -1
    out = []
    for total in range(SAMPLES, len(x) + 1, STEP_SAMPLES):
        P = infer(features(x[total - SAMPLES:total]).astype(np.float32))
        g0 = (total - SAMPLES) // HOP
        a = 0 if next_frame == 0 else FRAMES - RIGHT_CTX - STEP_FRAMES
        new = [i for i in range(a, FRAMES - RIGHT_CTX) if g0 + i >= next_frame]
        hist = np.concatenate([hist, P[new]])
        next_frame += len(new)
        if len(hist) > HIST_FRAMES:
            hist_start += len(hist) - HIST_FRAMES
            hist = hist[-HIST_FRAMES:]
        if len(hist) < 2 * FPS:
            continue
        for b in beats(viterbi(hist), hist):
            s, e = hist_start + b['start'], hist_start + b['end']
            if e > next_frame - STABLE_MARGIN or e <= last_end + 5 or s < last_end - 5:
                continue
            out.append(dict(b, start=s, end=e))
            last_end = e
    return out
