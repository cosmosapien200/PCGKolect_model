"""C++ SegQuality (PCGKolect_rk/src/seg_quality.cpp) vs Python reference (seg/*.py).

  python tests/test_seg_parity.py [--ckpt runs/seg_w_final.pt --width 48,64,96,128]
1. front-end features vs seg/model.py Frontend (torch) and seg/stream.py features (numpy)
2. Viterbi + beat scoring on real network posteriors
3. full streaming path (decimation, scheduling, history, emission) with a stub NPU
"""
import argparse, glob, os, subprocess, sys, tempfile
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'seg'))
from model import Frontend, SegNet, SAMPLES, FRAMES  # noqa: E402
from decode import viterbi, beats  # noqa: E402
from stream import features as np_features, stream_beats  # noqa: E402

from sqi_frontend import device_to_2k  # noqa: E402

RK = os.path.normpath(os.path.join(ROOT, '..', 'PCGKolect_rk'))
BIN = os.path.join(tempfile.gettempdir(), 'seg_cli_parity')
TMP = tempfile.mkdtemp()


def build():
    subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-I', os.path.join(RK, 'include'),
                    os.path.join(HERE, 'seg_cli.cpp'), os.path.join(RK, 'src', 'seg_quality.cpp'), '-o', BIN], check=True)


def cli(*args):
    return subprocess.run([BIN, *map(str, args)], check=True, capture_output=True, text=True).stdout


def parse_beats(txt):
    return [tuple(float(v) for v in l.split()[1:]) for l in txt.splitlines() if l.startswith('beat')]


def stub(feat):
    l = 3.0 * feat[[0, 6, 12, 18, 24], :].T.astype(np.float64)
    e = np.exp(l - l.max(1, keepdims=True))
    return (e / e.sum(1, keepdims=True)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=os.path.join(ROOT, 'release', 'seg_final.pt'))
    ap.add_argument('--width', default='32,48,64,96')
    ap.add_argument('--physionet', default=os.path.expanduser('~/Datasets/pcg/pn2016'))
    a = ap.parse_args()
    build()
    ok = True
    wavs = sorted(glob.glob(os.path.join(a.physionet, 'training-*', '*.wav')))
    rng = np.random.default_rng(3)
    picks = [wavs[i] for i in rng.choice(len(wavs), 4, replace=False)]
    fe = Frontend()
    net = SegNet(tuple(int(v) for v in a.width.split(','))).eval()
    net.load_state_dict(torch.load(a.ckpt, map_location='cpu'))

    worst_f = 0.0
    for w in picks:
        x, fs = sf.read(w, dtype='float64')
        pcm = np.clip(np.round(resample_poly(x, 8000, fs) / np.abs(x).max() * 8000), -32768, 32767).astype(np.int16)
        x2 = device_to_2k(pcm).astype(np.float32)
        # 1. features
        win = x2[:SAMPLES]
        p = os.path.join(TMP, 'w.f32'); win.tofile(p)
        cli('feat', p, p + '.out')
        fc = np.fromfile(p + '.out', np.float32).reshape(32, FRAMES)
        ft = fe(torch.from_numpy(win)[None])[0, :, 0].numpy()
        fn = np_features(win)
        worst_f = max(worst_f, np.abs(fc - ft).max(), np.abs(fc - fn).max())
        # 2. decoder on real posteriors over the whole recording
        T = (len(x2) - 128) // 40 + 1
        with torch.no_grad():
            Tp = T // 8 * 8
            xx = torch.from_numpy(x2[:128 + 40 * (Tp - 1)])[None]
            P = torch.softmax(net(fe(xx)), 1)[0, :, 0].T.numpy().astype(np.float32)
        p = os.path.join(TMP, 'P.f32'); P.tofile(p)
        out = cli('decode', p, len(P))
        cls_c = np.array(out.splitlines()[0].split()[1:], int)
        cls_p = viterbi(P)
        bc = parse_beats(out)
        bp = [(b['start'], b['end'], b['conf'], b['score'], int(b['usable'])) for b in beats(cls_p, P)]
        dec_ok = np.array_equal(cls_c, cls_p) and len(bc) == len(bp) and all(
            c[0] == q[0] and c[1] == q[1] and c[4] == q[4] and abs(c[2] - q[2]) < 1e-6 for c, q in zip(bc, bp))
        # 3. streaming: NPU replaced by a replay of the network's per-window posteriors
        p = os.path.join(TMP, 'pcm.raw'); pcm.astype('<i2').tofile(p)
        replay = []

        def net_infer(feat):
            with torch.no_grad():
                pr = torch.softmax(net(torch.from_numpy(feat.astype(np.float32))[None, :, None, :]), 1)[0, :, 0].T.numpy()
            replay.append(pr.astype(np.float32))
            return pr.astype(np.float32)
        sp = [(b['start'], b['end'], b['conf'], b['score'], int(b['usable'])) for b in stream_beats(pcm, net_infer)]
        rp = os.path.join(TMP, 'replay.f32')
        (np.stack(replay) if replay else np.zeros((0, 256, 5), np.float32)).tofile(rp)
        sc = parse_beats(cli('stream', p, rp))
        st_ok = len(sc) == len(sp) and all(c[0] == q[0] and c[1] == q[1] and c[4] == q[4] and abs(c[2] - q[2]) < 1e-4
                                           for c, q in zip(sc, sp))
        print(f'{os.path.basename(w):10s} decode {"OK " if dec_ok else "FAIL"} ({len(bp)} beats, '
              f'{sum(b[4] for b in bp)} usable)   stream {"OK " if st_ok else "FAIL"} ({len(sp)} beats)')
        ok &= dec_ok and st_ok
    print(f'features max abs diff {worst_f:.2e}')
    ok &= worst_f < 1e-3
    print('SEG PARITY PASS' if ok else 'SEG PARITY FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
