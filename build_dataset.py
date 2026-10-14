"""Build the labelled window dataset for the PCG signal-quality model.

Each sample is a 4 s buffer @ 2 kHz (1 s context | 2 s scored centre | 1 s context),
hop 0.5 s. Labels come from the centre 2 s:

  PhysioNet/CinC 2016 hand-corrected segmentation (Liu et al.):
    - the annotators bracket unusable stretches with '(N' ... 'N)'
    - BAD  (0): >= 25 % of the centre lies inside a noise bracket,
                or the recording is SQI=0 and < 50 % of the centre is segmented
    - GOOD (1): >= 90 % of the centre is inside S1/systole/S2/diastole, no noise,
                recording SQI=1
    - anything else is ambiguous and skipped
  CirCor (optional, --circor): annotated = GOOD, unannotated = BAD.  These are weak
    labels; they reduced accuracy on PhysioNet in our tests and are off by default.

Usage:
  python build_dataset.py --physionet ~/Datasets/pcg/pn2016 --out data/windows.npz
"""
import argparse, glob, os
import numpy as np, scipy.io as sio, soundfile as sf
from scipy.signal import resample_poly

from sqi_frontend import FS, highpass

WIN = 2 * FS
CTX = FS
HOP = FS // 2
STATES = ('S1', 'systole', 'S2', 'diastole')


def load(path):
    x, fs = sf.read(path, dtype='float32')
    if x.ndim > 1:
        x = x[:, 0]
    if fs != FS:
        x = resample_poly(x, FS, fs)
    return highpass(x).astype(np.float32)


def windows(x, good, bad, rec_bad):
    W, Y = [], []
    for s in range(CTX, len(x) - WIN - CTX + 1, HOP):
        g = good[s:s + WIN].mean()
        b = bad[s:s + WIN].mean()
        if b >= 0.25 or (rec_bad and g < 0.5):
            y = 0
        elif g >= 0.9 and b == 0 and not rec_bad:
            y = 1
        else:
            continue
        W.append(x[s - CTX:s + WIN + CTX]); Y.append(y)
    return W, Y


def physionet(root):
    for sub_dir in sorted(glob.glob(os.path.join(root, 'training-*'))):
        sub = os.path.basename(sub_dir)
        # SQI labels: per-subset REFERENCE-SQI.csv (individual subset downloads) or
        # annotations/updated/training-*/REFERENCE_withSQI.csv (annotations.zip)
        sqi_csv = os.path.join(sub_dir, 'REFERENCE-SQI.csv')
        if not os.path.exists(sqi_csv):
            sqi_csv = os.path.join(root, 'annotations', 'updated', sub, 'REFERENCE_withSQI.csv')
        sqi = {r[0]: r[2] for r in (l.strip().split(',') for l in open(sqi_csv)) if len(r) >= 3}
        for wav in sorted(glob.glob(os.path.join(sub_dir, '*.wav'))):
            rid = os.path.basename(wav)[:-4]
            ann = os.path.join(root, 'annotations', 'hand_corrected', f'{sub}_StateAns', f'{rid}_StateAns.mat')
            if not os.path.exists(ann):
                continue
            x = load(wav)
            a = sio.loadmat(ann)
            a = a[[k for k in a if not k.startswith('__')][0]]
            starts = [int(np.squeeze(r[0])) - 1 for r in a]
            labs = [str(r[1].flat[0].flat[0]) for r in a]
            good = np.zeros(len(x), bool)
            bad = np.zeros(len(x), bool)
            in_noise = False
            for i, (st, lb) in enumerate(zip(starts, labs)):
                en = starts[i + 1] if i + 1 < len(starts) else len(x)
                if '(N' in lb:
                    in_noise = True
                if in_noise:
                    bad[st:en] = True
                elif lb in STATES and i + 1 < len(starts):
                    good[st:en] = True
                if 'N)' in lb:
                    in_noise = False
            yield 'pn_' + sub[-1], 'pn' + rid, x, good, bad, sqi.get(rid, '1') == '0'


def circor(root):
    for wav in sorted(glob.glob(os.path.join(root, '*.wav'))):
        tsv = wav[:-4] + '.tsv'
        if not os.path.exists(tsv):
            continue
        x = load(wav)
        good = np.zeros(len(x), bool)
        bad = np.zeros(len(x), bool)
        for st, en, lb in np.loadtxt(tsv, ndmin=2):
            (good if lb > 0 else bad)[int(st * FS):int(en * FS)] = True
        yield 'circor', 'cc' + os.path.basename(wav).split('_')[0], x, good, bad, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--physionet', required=True, help='dir with training-*/ and annotations/hand_corrected/')
    ap.add_argument('--circor', help='CirCor training_data/ dir (weak labels, optional)')
    ap.add_argument('--out', default='data/windows.npz')
    a = ap.parse_args()
    X, Y, G, S = [], [], [], []
    gens = [physionet(a.physionet)] + ([circor(a.circor)] if a.circor else [])
    for gen in gens:
        for src, group, x, good, bad, rec_bad in gen:
            W, Yw = windows(x, good, bad, rec_bad)
            X += W; Y += Yw; G += [group] * len(W); S += [src] * len(W)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    np.savez_compressed(a.out, X=np.stack(X).astype(np.float16), y=np.array(Y, np.int8),
                        group=np.array(G), src=np.array(S))
    S, Y = np.array(S), np.array(Y)
    for s in sorted(set(S)):
        m = S == s
        print(f'{s:8s} windows={m.sum():6d} bad={(Y[m] == 0).sum():5d}')


if __name__ == '__main__':
    main()
