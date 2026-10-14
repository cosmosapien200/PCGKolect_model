"""Build the per-sample labelled corpus for the segmentation-based quality model (v2).

Output (one .npz): concatenated 2 kHz float16 audio, per-sample int8 labels and a
recording index.  Labels: 0=S1 1=systole 2=S2 3=diastole 4=noise 255=ignore.

  PhysioNet/CinC 2016 (all subsets a–f) with Liu et al. hand-corrected state annotations;
      '(N' .. 'N)' brackets -> noise.  Before the first / after the last annotation -> ignore.
  CirCor 1.0.3 .tsv segmentation: states 1..4 -> S1..diastole, 0 (unannotated) -> ignore.
      (unannotated is NOT necessarily noisy, so it is not used as a noise label)

  python seg/prep_seg.py --pn ~/Datasets/pcg/pn2016 --pn-ann ~/Downloads/PGT-1/data/annotations/hand_corrected \
         --circor ~/Downloads/Model/data/circor/training_data --out data/seg_corpus.npz
"""
import argparse, glob, os, sys
import numpy as np, scipy.io as sio, soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from sqi_frontend import FS, highpass  # noqa: E402

STATE = {'S1': 0, 'systole': 1, 'S2': 2, 'diastole': 3}
IGN, NOISE = 255, 4


def load(path):
    x, fs = sf.read(path, dtype='float32')
    if x.ndim > 1:
        x = x[:, 0]
    if fs != FS:
        x = resample_poly(x, FS, fs)
    x = highpass(x)
    return (x / (np.abs(x).max() + 1e-9)).astype(np.float32)   # peak-normalise; model input is gain-invariant


def physionet(root, ann_root, rows):
    # signal-quality labels live in annotations/updated/training-*/REFERENCE_withSQI.csv
    sqi_root = os.path.join(os.path.dirname(os.path.normpath(ann_root)), 'updated')
    for d in sorted(glob.glob(os.path.join(root, 'training-*'))):
        sub = os.path.basename(d)
        sqi_csv = os.path.join(sqi_root, sub, 'REFERENCE_withSQI.csv')
        sqi = {r[0]: int(r[2]) for r in (l.strip().split(',') for l in open(sqi_csv)) if len(r) >= 3}
        for wav in sorted(glob.glob(os.path.join(d, '*.wav'))):
            rid = os.path.basename(wav)[:-4]
            ann = os.path.join(ann_root, f'{sub}_StateAns', f'{rid}_StateAns.mat')
            if not os.path.exists(ann):
                continue
            x = load(wav)
            a = sio.loadmat(ann)
            a = a[[k for k in a if not k.startswith('__')][0]]
            starts = [int(np.squeeze(r[0])) - 1 for r in a]
            labs = [str(r[1].flat[0].flat[0]) for r in a]
            y = np.full(len(x), IGN, np.uint8)
            in_noise = False
            for i, (st, lb) in enumerate(zip(starts, labs)):
                if i + 1 >= len(starts):
                    break                                   # last marker: open-ended -> ignore after it
                en = min(starts[i + 1], len(x))
                if '(N' in lb:
                    in_noise = True
                if in_noise:
                    y[st:en] = NOISE
                elif lb in STATE:
                    y[st:en] = STATE[lb]
                if 'N)' in lb:
                    in_noise = False
            rows.append(('pn_' + sub[-1], 'pn' + rid, x, y, sqi.get(rid, 1)))


def circor(root, rows):
    for wav in sorted(glob.glob(os.path.join(root, '*.wav'))):
        tsv = wav[:-4] + '.tsv'
        if not os.path.exists(tsv):
            continue
        x = load(wav)
        y = np.full(len(x), IGN, np.uint8)
        for st, en, lb in np.loadtxt(tsv, ndmin=2):
            if lb > 0:
                y[int(st * FS):int(en * FS)] = int(lb) - 1
        rows.append(('circor', 'cc' + os.path.basename(wav).split('_')[0], x, y, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pn', required=True)
    ap.add_argument('--pn-ann', required=True)
    ap.add_argument('--circor')
    ap.add_argument('--out', default='data/seg_corpus.npz')
    a = ap.parse_args()
    rows = []
    physionet(a.pn, a.pn_ann, rows)
    if a.circor:
        circor(a.circor, rows)
    lens = np.array([len(r[2]) for r in rows])
    off = np.concatenate([[0], np.cumsum(lens)])
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    np.savez(a.out, audio=np.concatenate([r[2] for r in rows]).astype(np.float16),
             label=np.concatenate([r[3] for r in rows]), offset=off,
             src=np.array([r[0] for r in rows]), group=np.array([r[1] for r in rows]),
             rid=np.array([r[1] for r in rows]), sqi=np.array([r[4] for r in rows]))
    src = np.array([r[0] for r in rows])
    lab = np.concatenate([r[3] for r in rows])
    for s in sorted(set(src)):
        m = np.repeat(src == s, lens)
        l = lab[m]
        print(f'{s:7s} recs={int((src == s).sum()):5d} hours={m.sum() / FS / 3600:5.1f} '
              f'labelled={np.mean(l != IGN):.2f} noise={np.mean(l == NOISE):.3f}')


if __name__ == '__main__':
    main()
