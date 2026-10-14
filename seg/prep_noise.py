"""Build the non-cardiac noise bank used to corrupt training windows (2 kHz, high-passed).

Sources (each clip is tagged so train/test noise can be split by clip):
  esc50   ESC-50 environmental sounds (knocks, footsteps, breathing, coughing, clicks, ...)
  speech  Mini LibriSpeech dev-clean-2 (talking during auscultation)
  pnnoise the '(N'..'N)' stretches of PhysioNet 2016 (real stethoscope handling noise)

  python seg/prep_noise.py --esc50 ~/Datasets/pcg/esc50 --speech ~/Datasets/pcg/speech \
         --corpus data/seg_corpus.npz --out data/noise_bank.npz
"""
import argparse, glob, os, sys
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from sqi_frontend import FS, highpass  # noqa: E402

MIN_LEN = FS  # 1 s


def load(path, max_s=None):
    x, fs = sf.read(path, dtype='float32')
    if x.ndim > 1:
        x = x.mean(1)
    if max_s:
        x = x[:int(max_s * fs)]
    x = highpass(resample_poly(x, FS, fs))
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--esc50', required=True)
    ap.add_argument('--speech', required=True)
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--out', default='data/noise_bank.npz')
    a = ap.parse_args()
    clips, tags, rids = [], [], []
    for p in sorted(glob.glob(os.path.join(a.esc50, '**', 'audio', '*.wav'), recursive=True)):
        x = load(p)
        if np.sqrt((x ** 2).mean()) > 1e-5:         # some ESC-50 clips are mostly digital silence
            clips.append(x); tags.append('esc50'); rids.append('')
    for p in sorted(glob.glob(os.path.join(a.speech, '**', '*.flac'), recursive=True)):
        clips.append(load(p, max_s=12)); tags.append('speech'); rids.append('')
    c = np.load(a.corpus)
    lab, aud, off = c['label'], c['audio'], c['offset']
    is_n = (lab == 4).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], is_n, [0]])))
    for s, e in zip(edges[::2], edges[1::2]):
        if e - s >= MIN_LEN:
            r = np.searchsorted(off, s, side='right') - 1
            clips.append(aud[s:e].astype(np.float32)); tags.append('pnnoise'); rids.append(str(c['rid'][r]))
    keep = [i for i, x in enumerate(clips) if len(x) >= MIN_LEN]
    clips = [clips[i] / (np.sqrt((clips[i] ** 2).mean()) + 1e-9) for i in keep]   # unit RMS
    tags = [tags[i] for i in keep]
    rids = [rids[i] for i in keep]
    lens = np.array([len(x) for x in clips])
    np.savez(a.out, audio=np.concatenate(clips).astype(np.float16),
             offset=np.concatenate([[0], np.cumsum(lens)]), tag=np.array(tags), rid=np.array(rids))
    for t in sorted(set(tags)):
        m = np.array(tags) == t
        print(f'{t:8s} clips={m.sum():5d} minutes={lens[m].sum() / FS / 60:6.1f}')


if __name__ == '__main__':
    main()
