"""On-board end-to-end check: held-out recordings (clean + corrupted with held-out noise) are
replayed through the firmware pipeline on the RV1103 (INT8 NPU) and through the Python
streaming reference with the float ONNX model. Reports beat agreement.

  python seg/check_board.py --export export/ --fold 0 --n 20
Requires adb and PCGKolect_rk build output (pcg_replay) in --bin.
"""
import argparse, glob, os, subprocess, sys, tempfile
import numpy as np, soundfile as sf, onnxruntime as ort
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, HERE)
from stream import stream_beats  # noqa: E402
from train_seg import fold_of, Data, any_noise  # noqa: E402


def run(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', default='export')
    ap.add_argument('--rknn', default=None)
    ap.add_argument('--bin', required=True, help='dir with cross-compiled pcg_replay')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--physionet', default=os.path.expanduser('~/Datasets/pcg/pn2016'))
    a = ap.parse_args()
    rknn = a.rknn or sorted(glob.glob(os.path.join(a.export, 'segnet_w8a8_*.rknn')))[0]
    sess = ort.InferenceSession(os.path.join(a.export, 'segnet.onnx'))

    def infer(feat):
        z = sess.run(None, {'feat': feat[None, :, None, :].astype(np.float32)})[0][0, :, 0].T
        e = np.exp(z - z.max(1, keepdims=True))
        return (e / e.sum(1, keepdims=True)).astype(np.float32)

    run(['adb', 'shell', 'mkdir -p /root/replay'])
    run(['adb', 'push', rknn, '/root/replay/segnet.rknn'])
    run(['adb', 'push', os.path.join(a.bin, 'pcg_replay'), '/root/replay/'])
    run(['adb', 'shell', 'chmod +x /root/replay/pcg_replay'])

    d = Data()
    rng = np.random.default_rng(11)
    wavs = [w for w in sorted(glob.glob(os.path.join(a.physionet, 'training-*', '*.wav')))
            if fold_of('pn' + os.path.basename(w)[:-4]) == a.fold]
    wavs = [wavs[i] for i in rng.choice(len(wavs), a.n, replace=False)]
    tot = dict(py=0, board=0, matched=0, usable_agree=0)
    tmp = tempfile.mkdtemp()
    for w in wavs:
        x, fs = sf.read(w, dtype='float64')
        x = resample_poly(x, 8000, fs)
        x = x / np.abs(x).max() * 8000
        for variant in ('clean', 'noisy'):
            y = x.copy()
            if variant == 'noisy':
                span = min(len(y), int(rng.uniform(3, 8) * 8000))
                s0 = int(rng.integers(0, max(1, len(y) - span)))
                nz = resample_poly(any_noise(rng, d, span // 4 + 1, test=True), 4, 1)[:span]
                y[s0:s0 + span] += nz * np.sqrt((x ** 2).mean() / 10 ** (rng.uniform(-10, 5) / 10))
            pcm = np.clip(np.round(y), -32768, 32767).astype(np.int16)
            p = os.path.join(tmp, 'r.raw'); pcm.astype('<i2').tofile(p)
            run(['adb', 'push', p, '/root/replay/r.raw'])
            out = run(['adb', 'shell', 'cd /root/replay && ./pcg_replay segnet.rknn r.raw'])
            bb = [tuple(int(float(v)) if i != 2 else float(v) for i, v in enumerate(l.split()[1:]))
                  for l in out.splitlines() if l.startswith('beat')]
            pb = stream_beats(pcm, infer)
            m = 0; ua = 0
            for b in pb:
                hit = [q for q in bb if abs(q[0] - b['start']) <= 2 and abs(q[1] - b['end']) <= 2]
                if hit:
                    m += 1; ua += int(hit[0][3] == int(b['usable']))
            tot['py'] += len(pb); tot['board'] += len(bb); tot['matched'] += m; tot['usable_agree'] += ua
            print(f'{os.path.basename(w):10s} {variant:5s} float beats={len(pb):3d} '
                  f'(usable {sum(b["usable"] for b in pb):3d})  board beats={len(bb):3d} '
                  f'(usable {sum(q[3] for q in bb):3d})  matched={m:3d}', flush=True)
    print(f"\nTOTAL float beats {tot['py']}, board beats {tot['board']}, matched {tot['matched']} "
          f"({tot['matched'] / max(tot['py'], 1):.3f}); usable decision agreement on matched beats "
          f"{tot['usable_agree'] / max(tot['matched'], 1):.3f}")


if __name__ == '__main__':
    main()
