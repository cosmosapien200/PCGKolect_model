"""Front-end + streaming scorer — the Python reference for src/signal_quality.cpp.

Device path:  int16 @ 8 kHz  ->  63-tap FIR low-pass (800 Hz) + keep every 4th  ->  2 kHz
              ->  2nd-order Butterworth high-pass 20 Hz (causal)  ->  4 s ring
Every 1 s (2000 new samples) the 4 s buffer is scored; the result describes the
centre 2 s, i.e. audio from 3 s ago to 1 s ago.
"""
import json, os
import numpy as np
from scipy.signal import butter, firwin, lfilter, sosfilt

from sqi_features import features, MODEL_FEATURES, NAMES, CTX

FS = 2000
DEVICE_FS = 8000
DECIM = DEVICE_FS // FS
HP_SOS = butter(2, 20, 'highpass', fs=FS, output='sos')[0]   # b0 b1 b2 a0 a1 a2
DECIM_TAPS = firwin(63, 800, fs=DEVICE_FS)
STEP = FS                    # 1 s between results
CLIP_LEVEL = 32000           # |int16| at or above this counts as clipped

VERDICTS = ['GOOD', 'MARGINAL', 'BAD']
REASONS = ['OK', 'NOISY', 'NO_CONTACT', 'TOO_LOUD', 'CLIPPING']
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'sqi_lr_v1.json')


def highpass(x):
    return sosfilt(HP_SOS[None, :], np.asarray(x, np.float64))


def device_to_2k(pcm16):
    """int16 8 kHz -> float 2 kHz (full scale = 1.0), high-passed. Identical to the C++ path."""
    x = np.asarray(pcm16, np.float64) / 32768.0
    y = lfilter(DECIM_TAPS, [1.0], x)[::DECIM]
    return highpass(y)


def load_model(path=MODEL_PATH):
    with open(path) as f:
        return json.load(f)


def score_features(F, m):
    z = (F[:, :len(MODEL_FEATURES)] - np.array(m['mean'])) / np.array(m['scale'])
    return 1.0 / (1.0 + np.exp(-(z @ np.array(m['coef']) + m['intercept'])))


def decide(q, level_dbfs, clip_frac, m):
    g = m['gates']
    if clip_frac > g['clip_frac_max']:
        return 'BAD', 'CLIPPING'
    if level_dbfs < g['level_dbfs_min']:
        return 'BAD', 'NO_CONTACT'
    if level_dbfs > g['level_dbfs_max']:
        return 'BAD', 'TOO_LOUD'
    if q >= m['t_good']:
        return 'GOOD', 'OK'
    if q >= m['t_bad']:
        return 'MARGINAL', 'NOISY'
    return 'BAD', 'NOISY'


def score_stream(pcm16, m=None):
    """Score a whole device recording (int16 @ 8 kHz) exactly as the device would.

    Returns a list of dicts, one per second once 4 s of audio is available.
    """
    m = m or load_model()
    pcm16 = np.asarray(pcm16)
    x = device_to_2k(pcm16)
    clipped = (np.abs(pcm16.astype(np.int32)) >= CLIP_LEVEL)
    out = []
    for end in range(CTX, len(x) + 1, STEP):
        buf = x[end - CTX:end]
        F = features(buf[None, :])[0]
        q = float(score_features(F[None, :], m)[0])
        c0 = (end - CTX + FS) * DECIM                # centre 2 s, in 8 kHz samples
        clip_frac = float(clipped[c0:c0 + 2 * FS * DECIM].mean())
        level = 20 * np.log10(np.exp(F[NAMES.index('log_rms')]) + 1e-12)
        verdict, reason = decide(q, level, clip_frac, m)
        per_peak, per_lag = F[NAMES.index('per_peak')], F[NAMES.index('per_lag')]
        hr = 60.0 / per_lag if verdict == 'GOOD' and per_peak >= m['hr_min_periodicity'] else 0.0
        out.append(dict(t_start=(end - CTX + FS) / FS, t_end=(end - FS) / FS, q=q,
                        verdict=verdict, reason=reason, level_dbfs=float(level), clip_frac=clip_frac,
                        hr_bpm=float(hr)))
    return out


def summarize(results, step_s=1.0):
    """Aggregate per-second results the same way the device's QualityTracker does."""
    n = len(results)
    good = [r for r in results if r['verdict'] == 'GOOD']
    run = best = 0
    for r in results:
        run = run + 1 if r['verdict'] == 'GOOD' else 0
        best = max(best, run)
    return dict(scored_s=n * step_s,
                good_s=len(good) * step_s,
                marginal_s=sum(r['verdict'] == 'MARGINAL' for r in results) * step_s,
                bad_s=sum(r['verdict'] == 'BAD' for r in results) * step_s,
                usable_ratio=len(good) / n if n else 0.0,
                usable_beats=sum(r['hr_bpm'] / 60.0 * step_s for r in good),
                longest_good_run_s=best * step_s,
                mean_q=float(np.mean([r['q'] for r in results])) if n else 0.0)
