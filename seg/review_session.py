"""Review a device session recording with the v2 pipeline and export an Audacity label track.

  python seg/review_session.py ../PCGKolect_rk/recordings/session_001.wav [--onnx export/segnet.onnx]

Writes <wav>_labels.txt (Audacity: File > Import > Labels):
  usable beats as 'B' (or 'b?' when rejected), stretches with P(noise) > 0.5 as 'NOISE'.
Use it as a pre-label: listen, fix the boundaries, and the corrected track becomes training data.
"""
import argparse, os, sys
import numpy as np, soundfile as sf, onnxruntime as ort
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, HERE)
from stream import stream_beats, features  # noqa: E402
from sqi_frontend import device_to_2k  # noqa: E402
from model import SAMPLES, HOP  # noqa: E402
from decode import FPS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('wav')
    ap.add_argument('--onnx', default=os.path.join(HERE, '..', 'release', 'segnet.onnx'))
    a = ap.parse_args()
    pcm, fs = sf.read(a.wav, dtype='int16')
    if pcm.ndim > 1:
        pcm = pcm[:, 0]
    if fs != 8000:
        pcm = np.clip(np.round(resample_poly(pcm.astype(np.float64), 8000, fs)), -32768, 32767).astype(np.int16)
    sess = ort.InferenceSession(a.onnx)
    pnoise = {}

    def infer(feat):
        z = sess.run(None, {'feat': feat[None, :, None, :].astype(np.float32)})[0][0, :, 0].T
        e = np.exp(z - z.max(1, keepdims=True))
        return (e / e.sum(1, keepdims=True)).astype(np.float32)

    beats = stream_beats(pcm, infer)
    # frame-wise noise probability over the whole file (non-overlapping 256-frame windows, hop 128)
    x = device_to_2k(pcm).astype(np.float32)
    T = (len(x) - 128) // HOP + 1
    pn = np.zeros(max(T, 0))
    cnt = np.zeros(max(T, 0))
    for s in range(0, max(len(x) - SAMPLES + 1, 0), 128 * HOP):
        p = infer(features(x[s:s + SAMPLES]).astype(np.float32))
        t0 = s // HOP
        pn[t0:t0 + 256] += p[:, 4]; cnt[t0:t0 + 256] += 1
    pn = pn / np.maximum(cnt, 1)

    lines = [f"{b['start'] / FPS:.3f}\t{b['end'] / FPS:.3f}\t{'B' if b['usable'] else 'b?'} conf={b['conf']:.2f}"
             for b in beats]
    noisy = np.concatenate([[0], (pn > 0.5).astype(int), [0]])
    ed = np.flatnonzero(np.diff(noisy))
    for s, e in zip(ed[::2], ed[1::2]):
        if e - s >= 10:
            lines.append(f'{s / FPS:.3f}\t{e / FPS:.3f}\tNOISE p={pn[s:e].mean():.2f}')
    out = os.path.splitext(a.wav)[0] + '_labels.txt'
    with open(out, 'w') as f:
        f.write('\n'.join(sorted(lines, key=lambda l: float(l.split()[0]))) + '\n')
    dur = len(pcm) / 8000
    u = [b for b in beats if b['usable']]
    hr = 60 / np.median([b['end'] / FPS - b['start'] / FPS for b in u]) if u else 0
    print(f'{os.path.basename(a.wav)}: {dur:.1f} s, {len(beats)} beats ({len(u)} usable, HR ~{hr:.0f}), '
          f'noise > 0.5 on {100 * (pn > 0.5).mean():.0f}% of frames, '
          f'usable coverage {100 * sum(b["end"] - b["start"] for b in u) / FPS / max(dur, 1e-9):.0f}%')
    print('labels ->', out)


if __name__ == '__main__':
    main()
