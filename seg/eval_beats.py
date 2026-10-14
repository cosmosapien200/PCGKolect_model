"""Beat- and recording-level evaluation of a trained fold model.

  python seg/eval_beats.py --fold 0 --tag _w --width 48,64,96,128

1. S1 / S2 onset detection F1 (±60 ms) on held-out PhysioNet recordings (expert segmentation).
2. Recording-level acceptance vs the PhysioNet expert signal-quality label (SQI 0/1):
   v2 = fraction of time covered by usable beats, v1 = mean q over 1 s steps (LR retrained
   on the same training fold). Threshold-free AUC + best balanced accuracy.
3. Per-beat usability under controlled corruption: held-out recordings + held-out noise
   clips at fixed SNRs. A true beat is "unusable" when its beat-average SNR < 0 dB
   (same definition the model was trained with). Each true beat gets the score of the
   decoded beat that overlaps it most (0 if none was decoded).
"""
import argparse, json, os, sys, zlib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from model import Frontend, SegNet, FS, HOP, NFFT, IGNORE, NOISE, frame_labels  # noqa: E402
from train_seg import Data, infer, fold_of, any_noise, v1_cache, RUNS, dev  # noqa: E402
from decode import viterbi, beats, recording_summary, FPS  # noqa: E402
from sqi_features import features as v1_features, MODEL_FEATURES  # noqa: E402
from train import fit as v1_fit, predict as v1_predict  # noqa: E402

TOL = 3  # frames = 60 ms


def onsets(cls, c):
    m = cls == c
    return np.flatnonzero(m & ~np.concatenate([[False], m[:-1]]))


def match_f1(pred, ref, tol=TOL):
    ref = list(ref)
    tp = 0
    used = set()
    for p in pred:
        j = next((j for j, r in enumerate(ref) if j not in used and abs(r - p) <= tol), None)
        if j is not None:
            used.add(j); tp += 1
    prec = tp / max(len(pred), 1)
    rec = tp / max(len(ref), 1)
    return tp, len(pred), len(ref)


def true_beats(yf):
    """Beats from frame labels: S1 onset -> next S1 onset with no ignore/noise inside."""
    s1 = onsets(yf, 0)
    out = []
    for a, b in zip(s1[:-1], s1[1:]):
        seg = yf[a:b]
        if np.all(seg < 4) and set(np.unique(seg)) == {0, 1, 2, 3}:
            out.append((a, b))
    return out


def best_ba(y, s):
    ts = np.unique(np.quantile(s, np.linspace(0.005, 0.995, 199)))
    return max(balanced_accuracy_score(y, s >= t) for t in ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--width', default='32,48,64,96')
    ap.add_argument('--max-recs', type=int, default=100000)
    a = ap.parse_args()
    d = Data()
    d.set_fold(a.fold)
    width = tuple(int(v) for v in a.width.split(','))
    fe, net = Frontend().to(dev), SegNet(width).to(dev)
    net.load_state_dict(torch.load(os.path.join(RUNS, f'seg{a.tag}_fold{a.fold}.pt')))
    net.eval()
    folds = np.array([fold_of(r) if p else -1 for r, p in zip(d.rid, d.is_pn)])
    te = np.flatnonzero(folds == a.fold)[:a.max_recs]
    tr = np.flatnonzero((folds != a.fold) & d.is_pn)

    # v1 retrained on the training fold
    F, Y, R, _ = v1_cache(d)
    m = np.isin(R, tr)
    sc, lr = v1_fit(F[m], Y[m])

    rng = np.random.default_rng(1000 + a.fold)
    cnt = {'S1': [0, 0, 0], 'S2': [0, 0, 0]}
    rec_y, rec_v1, rec_v2 = [], [], []
    beat_rows = []      # (snr, truth_usable, score, usable_flag)
    for r in te:
        x, ys = d.rec(r)
        if len(x) < 6 * FS:
            continue
        P = infer(net, fe, x)
        yf = frame_labels(ys, 0, len(P))
        cls = viterbi(P)
        bts = beats(cls, P)
        # 1. onset F1 on frames that are labelled clean cardiac
        lab = (yf < 4)
        for name, c in (('S1', 0), ('S2', 2)):
            ref = [t for t in onsets(yf, c) if lab[max(t - 5, 0):t + 5].all()]
            pred = [t for t in onsets(cls, c) if lab[max(t - 5, 0):t + 5].all()]
            tp, npred, nref = match_f1(pred, ref)
            cnt[name][0] += tp; cnt[name][1] += npred; cnt[name][2] += nref
        # 2. recording level
        summ = recording_summary(bts, len(P))
        X4 = np.stack([x[s:s + 4 * FS] for s in range(0, len(x) - 4 * FS + 1, FS)])
        q1 = v1_predict(sc, lr, v1_features(X4)[:, :len(MODEL_FEATURES)])
        rec_y.append(int(d.sqi[r])); rec_v2.append(summ['usable_frac_time']); rec_v1.append(float(q1.mean()))
        # 3. controlled corruption (only recordings that have clean annotated beats)
        tb = true_beats(yf)
        if len(tb) < 4 or d.sqi[r] == 0 or (yf == NOISE).any():
            continue
        for snr in (20, 10, 5, 0, -5, -10):
            L = len(x)
            span = int(min(L, rng.uniform(3, 8) * FS))
            s0 = int(rng.integers(0, max(1, L - span)))
            n = np.zeros(L, np.float32)
            n[s0:s0 + span] = any_noise(rng, d, span, test=True)
            lab_s = ys != IGNORE
            pc = (x[lab_s] ** 2).mean()
            n *= np.sqrt(pc / 10 ** (snr / 10))
            Pn = infer(net, fe, x + n)
            bn = beats(viterbi(Pn), Pn)
            pn_frame = np.convolve(n ** 2, np.ones(400) / 400, 'same')[HOP * np.arange(len(Pn)) + NFFT // 2]
            for (ba, bb) in tb:
                snr_b = 10 * np.log10(pc / (pn_frame[ba:bb].mean() + 1e-12))
                truth = snr_b >= 0
                best, ov = None, 0
                for b in bn:
                    o = min(bb, b['end']) - max(ba, b['start'])
                    if o > ov:
                        best, ov = b, o
                hit = best is not None and ov >= 0.5 * (bb - ba)
                beat_rows.append((snr, truth, best['score'] if hit else 0.0, bool(hit and best['usable'])))

    res = {}
    for name, (tp, npred, nref) in cnt.items():
        p, rc = tp / max(npred, 1), tp / max(nref, 1)
        res[f'{name}_onset_F1'] = 2 * p * rc / max(p + rc, 1e-9)
    ry = np.array(rec_y)
    res['recordings'] = len(ry)
    res['recordings_sqi0'] = int((ry == 0).sum())
    for k, s in (('v1', rec_v1), ('v2', rec_v2)):
        res[f'rec_AUC_{k}'] = roc_auc_score(ry, s)
        res[f'rec_bestBA_{k}'] = best_ba(ry, np.array(s))
    B = np.array(beat_rows, dtype=float)
    res['beats'] = len(B)
    res['beat_AUC'] = roc_auc_score(B[:, 1], B[:, 2])
    res['beat_BA_usable_flag'] = balanced_accuracy_score(B[:, 1], B[:, 3])
    res['beat_by_snr'] = {int(s): dict(true_usable=float(B[B[:, 0] == s, 1].mean()),
                                       flagged_usable=float(B[B[:, 0] == s, 3].mean()))
                          for s in np.unique(B[:, 0])}
    print(json.dumps(res, indent=1))
    with open(os.path.join(RUNS, f'beats{a.tag}_fold{a.fold}.json'), 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
