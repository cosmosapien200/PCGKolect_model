"""Train / cross-validate the segmentation-quality network (v2) and compare with v1.

  python seg/train_seg.py --fold 0            # one CV fold (PhysioNet recordings, hash-assigned)
  python seg/train_seg.py --fold all          # 5 folds, then pooled report
  python seg/train_seg.py --final             # train on everything -> runs/seg_final.pt

CirCor is always training data.  Evaluation uses only held-out PhysioNet recordings
(real expert labels): frame state accuracy, real-noise frame AUC, and 2 s window
usability on exactly the v1 label rule, with v1 retrained on the same split.
"""
import argparse, json, os, sys, time, zlib
import numpy as np
import torch
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from model import (Frontend, SegNet, FS, NFFT, HOP, FRAMES, SAMPLES, NBANDS, FEAT_SCALE,  # noqa: E402
                   NOISE, IGNORE, frame_labels)
from sqi_features import features as v1_features, MODEL_FEATURES  # noqa: E402
from train import fit as v1_fit, predict as v1_predict  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

DATA = os.path.join(HERE, '..', 'data')
RUNS = os.path.join(HERE, '..', 'runs')
TAU_DB = 0.0          # a frame is "noise" when local noise power exceeds the beat-average heart power
NFOLD = 5
dev = 'cuda'


def fold_of(rid):
    return zlib.crc32(rid.encode()) % NFOLD


class Data:
    def __init__(self):
        c = np.load(os.path.join(DATA, 'seg_corpus.npz'))
        self.audio, self.lab, self.off = c['audio'], c['label'], c['offset']
        self.src, self.rid, self.sqi = c['src'], c['rid'], c['sqi']
        self.len = np.diff(self.off)
        nb = np.load(os.path.join(DATA, 'noise_bank.npz'))
        self.naud, self.noff, self.ntag, self.nrid = nb['audio'], nb['offset'], nb['tag'], nb['rid']
        n = len(self.ntag)
        # held-out noise clips for the synthetic test; real PhysioNet noise follows its recording's fold
        self.ntest = np.array([zlib.crc32(str(i).encode()) % 5 == 0 for i in range(n)])
        self.nfold = np.array([fold_of(r) if r else -1 for r in self.nrid])
        self.exclude_fold = -2
        self.is_pn = np.char.startswith(self.src, 'pn')

    def set_fold(self, k):
        self.exclude_fold = k

    def rec(self, i):
        return self.audio[self.off[i]:self.off[i + 1]].astype(np.float32), self.lab[self.off[i]:self.off[i + 1]]


# ----------------------------------------------------------------------------------------
# Corruption
# ----------------------------------------------------------------------------------------

def synth_noise(rng, L):
    kind = rng.integers(5)
    w = rng.standard_normal(L)
    if kind == 0:                                   # white / sensor floor
        x = w
    elif kind == 1:                                 # pink-ish
        X = np.fft.rfft(w); f = np.arange(len(X)) + 1
        x = np.fft.irfft(X / np.sqrt(f), L)
    elif kind == 2:                                 # brown (rub / pressure drift)
        X = np.fft.rfft(w); f = np.arange(len(X)) + 1
        x = np.fft.irfft(X / f, L)
    elif kind == 3:                                 # mains hum + harmonics
        t = np.arange(L) / FS
        f0 = rng.choice([50.0, 60.0])
        x = sum(rng.random() / h * np.sin(2 * np.pi * f0 * h * t + rng.random() * 6.28) for h in range(1, 8))
        x = x + 0.05 * w
    else:                                           # bursty friction
        env = np.repeat(rng.random(L // 200 + 1) ** 4, 200)[:L]
        x = w * env
    return x / (np.sqrt((x ** 2).mean()) + 1e-9)


def bank_noise(rng, d, L, test):
    pool = np.flatnonzero((d.ntest == test) & (d.nfold != d.exclude_fold))
    out = np.empty(0, np.float32)
    while len(out) < L:
        k = rng.choice(pool)
        seg = d.naud[d.noff[k]:d.noff[k + 1]].astype(np.float32)
        s = rng.integers(0, max(1, len(seg) - (L - len(out))))
        out = np.concatenate([out, seg[s:s + L - len(out)]])
    return out / (np.sqrt((out ** 2).mean()) + 1e-9)


def any_noise(rng, d, L, test=False):
    return bank_noise(rng, d, L, test) if rng.random() < 0.7 else synth_noise(rng, L)


def corrupt(rng, d, x, y_s, y_f, test=False, mode=None, snr=None):
    """x audio [SAMPLES], y_s per-sample labels, y_f per-frame labels -> corrupted x, new y_f."""
    L = len(x)
    if mode is None:
        mode = rng.choice(4, p=[0.35, 0.35, 0.15, 0.15])   # clean, partial, full overlay, pure noise
    if mode == 0:
        return x, y_f
    if mode == 3:
        return any_noise(rng, d, L, test).astype(np.float32), np.full_like(y_f, NOISE)
    lab = y_s != IGNORE
    pc = (x[lab] ** 2).mean() if lab.sum() > FS else (x ** 2).mean()
    snr = rng.uniform(-20, 25) if snr is None else snr
    n = any_noise(rng, d, L, test) * np.sqrt(pc / 10 ** (snr / 10))
    if mode == 1:
        span = int(rng.uniform(0.3, 4.0) * FS)
        s = rng.integers(0, max(1, L - span))
        m = np.zeros(L)
        m[s:s + span] = 1
        ramp = np.convolve(m, np.ones(40) / 40, 'same')
        n = n * ramp
    p_loc = np.convolve(n ** 2, np.ones(400) / 400, 'same')[HOP * np.arange(FRAMES) + NFFT // 2]
    snr_t = 10 * np.log10(pc / (p_loc + 1e-12) + 1e-12)
    y = y_f.copy()
    y[(y != IGNORE) & (snr_t < TAU_DB)] = NOISE
    y[(y == IGNORE) & (snr_t < TAU_DB - 10)] = NOISE
    return (x + n).astype(np.float32), y


def make_batch(rng, d, recs, w, B):
    X = np.empty((B, SAMPLES), np.float32)
    Y = np.empty((B, FRAMES), np.int64)
    for j, r in enumerate(rng.choice(recs, B, p=w)):
        for _ in range(8):
            s = d.off[r] + rng.integers(0, d.len[r] - SAMPLES + 1)
            ys = d.lab[s:s + SAMPLES]
            yf = frame_labels(ys, 0, FRAMES)
            if (yf != IGNORE).mean() > 0.3:
                break
        x = d.audio[s:s + SAMPLES].astype(np.float32) * rng.choice([-1, 1])
        X[j], yf = corrupt(rng, d, x, ys, yf)
        Y[j] = yf
    return torch.from_numpy(X), torch.from_numpy(Y)


def spec_augment(f, rng):
    B = f.shape[0]
    band = torch.linspace(-1, 1, NBANDS, device=f.device)
    tilt = torch.tensor(rng.uniform(-1.5, 1.5, B), device=f.device, dtype=f.dtype)
    curve = sum(torch.tensor(rng.uniform(-0.3, 0.3, B), device=f.device, dtype=f.dtype)[:, None] *
                torch.cos(np.pi * k * (band[None] + 1) / 2 + float(rng.random() * 6.28)) for k in (1, 2, 3))
    return f + ((tilt[:, None] * band[None] + curve) / FEAT_SCALE)[:, :, None, None]


# ----------------------------------------------------------------------------------------
# Inference over a whole recording, as the device would (256-frame windows, hop 128)
# ----------------------------------------------------------------------------------------

@torch.no_grad()
def infer(net, fe, x):
    T = (len(x) - NFFT) // HOP + 1
    if T < FRAMES:
        x = np.resize(x, SAMPLES)                   # tile short recordings
    Tx = (len(x) - NFFT) // HOP + 1
    starts = list(range(0, Tx - FRAMES + 1, FRAMES // 2))
    if starts[-1] != Tx - FRAMES:
        starts.append(Tx - FRAMES)
    W = torch.from_numpy(np.stack([x[s * HOP:s * HOP + SAMPLES] for s in starts])).to(dev)
    P = torch.softmax(net(fe(W)), 1)[:, :, 0, :].cpu().numpy()   # [n, 5, 256]
    out = np.zeros((Tx, 5), np.float32)
    q = FRAMES // 4
    for i, s in enumerate(starts):
        a = 0 if i == 0 else q
        b = FRAMES if i == len(starts) - 1 else FRAMES - q
        out[s + a:s + b] = P[i, :, a:b].T
    return out[:T]


# ----------------------------------------------------------------------------------------
# Window-level comparison with v1 (same label rule as build_dataset.py)
# ----------------------------------------------------------------------------------------

def windows(x, ys, rec_bad):
    WIN, CTX = 2 * FS, FS
    good = (ys != IGNORE) & (ys != NOISE)
    bad = ys == NOISE
    out = []
    for s in range(CTX, len(x) - WIN - CTX + 1, FS // 2):
        g, b = good[s:s + WIN].mean(), bad[s:s + WIN].mean()
        if b >= 0.25 or (rec_bad and g < 0.5):
            out.append((s, 0))
        elif g >= 0.9 and b == 0 and not rec_bad:
            out.append((s, 1))
    return out


def v2_window_scores(P, s):
    t0, t1 = int(np.ceil((s - NFFT // 2) / HOP)), int((s + 2 * FS - NFFT // 2) / HOP)
    p = P[max(t0, 0):t1]
    u = 1 - p[:, NOISE]
    conf = p[:, :4].max(1) / np.maximum(u, 1e-6)
    return [u.mean(), (u * conf).mean(), np.quantile(u, 0.2)]


def v1_cache(d):
    """v1 features for every labelled window of every PhysioNet recording (fold-independent; cached)."""
    path = os.path.join(RUNS, 'v1_windows_cache.npz')
    if os.path.exists(path):
        c = np.load(path)
        return c['F'], c['y'], c['rec'], c['s']
    F, Y, R, S = [], [], [], []
    for r in np.flatnonzero(d.is_pn):
        x, ys = d.rec(r)
        w = windows(x, ys, d.sqi[r] == 0)
        if not w:
            continue
        X = np.stack([x[s - FS:s + 3 * FS] for s, _ in w])
        F.append(v1_features(X)[:, :len(MODEL_FEATURES)])
        Y += [y for _, y in w]; R += [r] * len(w); S += [s for s, _ in w]
    F, Y, R, S = np.concatenate(F), np.array(Y), np.array(R), np.array(S)
    np.savez(path, F=F, y=Y, rec=R, s=S)
    return F, Y, R, S


def v1_window_feats(d, recs):
    F, Y, R, _ = v1_cache(d)
    m = np.isin(R, recs)
    return F[m], Y[m], R[m]


# ----------------------------------------------------------------------------------------

def train_net(d, recs, steps, seed, width=(32, 48, 64, 96), log_every=1000):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    recs = np.array([r for r in recs if d.len[r] >= SAMPLES])
    w = (d.len[recs] - SAMPLES + 1).astype(np.float64)
    w[d.is_pn[recs]] *= 1.5                       # adult recordings slightly up-weighted
    w /= w.sum()
    fe, net = Frontend().to(dev), SegNet(width).to(dev)
    print(f'  params {sum(p.numel() for p in net.parameters())}', flush=True)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.1)
    cw = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], device=dev)
    t0 = time.time()
    run = 0.0
    for st in range(steps):
        X, Y = make_batch(rng, d, recs, w, 64)
        with torch.no_grad():
            f = spec_augment(fe(X.to(dev)), rng)
        logits = net(f)[:, :, 0, :]
        loss = Fn.cross_entropy(logits, Y.to(dev), weight=cw, ignore_index=IGNORE, label_smoothing=0.05)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
        run = 0.98 * run + 0.02 * loss.item()
        if (st + 1) % log_every == 0:
            print(f'  step {st + 1}/{steps} loss {run:.3f}  {time.time() - t0:.0f}s', flush=True)
    net.eval()
    return fe, net


def evaluate(d, fe, net, test_recs, train_recs):
    st_ok = st_n = 0
    nz_s, nz_y = [], []
    W2, WY, WR = [], [], []
    for r in test_recs:
        x, ys = d.rec(r)
        if len(x) < NFFT * 2:
            continue
        P = infer(net, fe, x)
        yf = frame_labels(ys, 0, len(P))
        m = yf < 4
        st_ok += (P[m, :4].argmax(1) == yf[m]).sum(); st_n += m.sum()
        m = yf != IGNORE
        nz_s.append(P[m, NOISE]); nz_y.append(yf[m] == NOISE)
        for s, y in windows(x, ys, d.sqi[r] == 0):
            W2.append(v2_window_scores(P, s)); WY.append(y); WR.append(r)
    nz_s, nz_y = np.concatenate(nz_s), np.concatenate(nz_y)
    # v1 retrained on the same training recordings (PhysioNet only, as in train.py)
    Ftr, Ytr, _ = v1_window_feats(d, [r for r in train_recs if d.is_pn[r]])
    Fte, Yte, Gte = v1_window_feats(d, test_recs)
    sc, lr = v1_fit(Ftr, Ytr)
    v1s = v1_predict(sc, lr, Fte)
    assert np.array_equal(Yte, np.array(WY)) and np.array_equal(Gte, np.array(WR))
    return dict(state_acc=float(st_ok / max(st_n, 1)),
                noise_frame_auc=float(roc_auc_score(nz_y, nz_s)) if nz_y.any() else None,
                n_windows=len(WY), n_bad=int((np.array(WY) == 0).sum())), \
        dict(y=np.array(WY), rec=np.array(WR), v1=v1s, v2=np.array(W2))


def report(win, d, tag=''):
    from sklearn.metrics import balanced_accuracy_score as BA
    y = win['y']
    src = d.src[win['rec']]
    lines = []
    for name, s in [('v1 (features+LR)', win['v1'])] + [(f'v2 {n}', win['v2'][:, i]) for i, n in
                                                       enumerate(['mean(1-Pnoise)', 'mean(u*conf)', 'q20(1-Pnoise)'])]:
        auc = roc_auc_score(y, s)
        ts = np.quantile(s, np.linspace(0.01, 0.99, 197))
        bas = [BA(y, s >= t) for t in ts]
        per = ' '.join(f'{k[-1]}:{roc_auc_score(y[src == k], s[src == k]):.3f}'
                       for k in sorted(set(src)) if len(set(y[src == k])) == 2)
        lines.append(f'{tag}{name:22s} AUC={auc:.3f}  best-BA={max(bas):.3f}  per-subset AUC {per}')
    print('\n'.join(lines), flush=True)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', default='0')
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--final', action='store_true')
    ap.add_argument('--width', default='32,48,64,96')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    width = tuple(int(v) for v in a.width.split(','))
    d = Data()
    print(f'{len(d.rid)} recordings, {d.len.sum() / FS / 3600:.1f} h; noise bank {len(d.ntag)} clips', flush=True)
    folds = np.array([fold_of(r) if p else -1 for r, p in zip(d.rid, d.is_pn)])
    if a.final:
        fe, net = train_net(d, np.arange(len(d.rid)), a.steps, seed=123, width=width)
        torch.save(net.state_dict(), os.path.join(RUNS, f'seg{a.tag}_final.pt'))
        print('saved', os.path.join(RUNS, f'seg{a.tag}_final.pt'))
        return
    ks = range(NFOLD) if a.fold == 'all' else [int(a.fold)]
    for k in ks:
        tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        print(f'fold {k}: train {len(tr)} recs, test {len(te)} PhysioNet recs', flush=True)
        d.set_fold(k)
        fe, net = train_net(d, tr, a.steps, seed=k, width=width)
        torch.save(net.state_dict(), os.path.join(RUNS, f'seg{a.tag}_fold{k}.pt'))
        m, win = evaluate(d, fe, net, te, tr)
        print(json.dumps(m), flush=True)
        np.savez(os.path.join(RUNS, f'seg{a.tag}_fold{k}_windows.npz'), **win)
        report(win, d, f'[fold {k}] ')
    if a.fold == 'all':
        W = [np.load(os.path.join(RUNS, f'seg{a.tag}_fold{k}_windows.npz')) for k in range(NFOLD)]
        win = {k: np.concatenate([w[k] for w in W]) for k in W[0].files}
        report(win, d, '[pooled] ')


if __name__ == '__main__':
    main()
