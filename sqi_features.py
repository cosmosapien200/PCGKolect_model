"""PCG signal-quality features, spec v1 (mirrored 1:1 in C++ on the RV1103).

Input: X [B, 8000] float, 4 s @ 2 kHz, already high-passed (20 Hz, causal).
The scored region is the centre 2 s (samples 2000..5999); the full 4 s is
context for rhythm/periodicity features.
"""
import numpy as np

FS = 2000
CTX = 8000
C0, C1 = 2000, 6000
NFFT, WHOP = 256, 128
BANDS = [(20, 60), (60, 150), (150, 300), (300, 600), (600, 1001)]
ENV_BLK = 40           # 20 ms -> 50 Hz envelope
LAG_LO, LAG_HI = 15, 75  # 0.3 .. 1.5 s

MODEL_FEATURES = ['lkurt', 'crest', 'zcr', 'blk_cv', 'blk_maxmed', 'burst', 'lflat', 'centroid',
                  'band0', 'band1', 'band2', 'band3', 'band4',
                  'blk_cv4', 'blk_maxmed4', 'burst4', 'per_peak', 'per_lag', 'env_entropy']
EXTRA = ['log_rms']  # used by deterministic gates only, not the model
NAMES = MODEL_FEATURES + EXTRA
HANN = np.hanning(NFFT).astype(np.float64)


def _blocks(z, n):
    e = (z.reshape(len(z), n, -1) ** 2).mean(2) + 1e-12
    med = np.median(e, 1)
    return np.log(e).std(1), np.log(e.max(1) / med), (e > 10 * med[:, None]).mean(1)


def features(X):
    x = np.asarray(X, np.float64)
    x = x - x.mean(1, keepdims=True)
    z = x / (np.sqrt((x ** 2).mean(1, keepdims=True)) + 1e-8)
    c = x[:, C0:C1]
    c = c - c.mean(1, keepdims=True)
    rmsc = np.sqrt((c ** 2).mean(1)) + 1e-8
    zc = c / rmsc[:, None]
    lkurt = np.log((zc ** 4).mean(1) + 1e-12)
    crest = np.abs(zc).max(1)
    pos = zc >= 0
    zcr = (pos[:, 1:] != pos[:, :-1]).mean(1)
    blk_cv, blk_maxmed, burst = _blocks(zc, 20)
    # Welch PSD, 256-pt Hann, hop 128
    starts = range(0, zc.shape[1] - NFFT + 1, WHOP)
    P = np.mean([np.abs(np.fft.rfft(zc[:, s:s + NFFT] * HANN, axis=1)) ** 2 for s in starts], 0) + 1e-12
    f = np.arange(NFFT // 2 + 1) * FS / NFFT
    sel = f >= 20
    Ps, fs_ = P[:, sel], f[sel]
    tot = Ps.sum(1)
    lflat = np.log(Ps).mean(1) - np.log(Ps.mean(1))
    centroid = (Ps * fs_).sum(1) / tot / 1000.0
    bands = [np.log(Ps[:, (fs_ >= a) & (fs_ < b)].sum(1) / tot) for a, b in BANDS]
    # context (4 s)
    blk_cv4, blk_maxmed4, burst4 = _blocks(z, 40)
    env = np.abs(z).reshape(len(z), -1, ENV_BLK).mean(2)
    env = env - env.mean(1, keepdims=True)
    env = env / (np.sqrt((env ** 2).sum(1, keepdims=True)) + 1e-8)
    m = env.shape[1]
    ac = np.stack([(env[:, :m - L] * env[:, L:]).sum(1) / (1 - L / m) for L in range(LAG_LO, LAG_HI)], 1)
    per_peak = ac.max(1)
    per_lag = (ac.argmax(1) + LAG_LO) / (FS / ENV_BLK)
    p = env ** 2 + 1e-12
    p = p / p.sum(1, keepdims=True)
    env_entropy = -(p * np.log(p)).sum(1) / np.log(m)
    log_rms = np.log(rmsc)
    return np.stack([lkurt, crest, zcr, blk_cv, blk_maxmed, burst, lflat, centroid] + bands +
                    [blk_cv4, blk_maxmed4, burst4, per_peak, per_lag, env_entropy, log_rms], 1)
