"""C++ (PCGKolect_rk/src/signal_quality.cpp) vs Python reference parity test.

  python tests/test_parity.py --physionet ~/Downloads/PGT-1/data
Builds tests/sqi_cli with the host g++, streams several signals through both and
requires identical verdicts and features within 1e-6.
"""
import argparse, glob, os, subprocess, sys, tempfile
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from sqi_frontend import score_stream, summarize, load_model  # noqa: E402
from sqi_features import NAMES  # noqa: E402

RK = os.path.normpath(os.path.join(HERE, '..', '..', 'PCGKolect_rk'))
BIN = os.path.join(tempfile.gettempdir(), 'sqi_cli_parity')


def build():
    subprocess.run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-I', os.path.join(RK, 'include'),
                    os.path.join(HERE, 'sqi_cli.cpp'), os.path.join(RK, 'src', 'signal_quality.cpp'),
                    '-o', BIN], check=True)


def run_cpp(pcm):
    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as f:
        pcm.astype('<i2').tofile(f.name)
    out = subprocess.run([BIN, f.name], check=True, capture_output=True, text=True).stdout
    os.unlink(f.name)
    rows = [l.split(',') for l in out.strip().splitlines() if not l.startswith('#')]
    summ = [l for l in out.splitlines() if l.startswith('#summary')][0].split(',')[1:]
    return rows, [float(v) for v in summ]


def signals(physionet):
    rng = np.random.default_rng(0)
    wavs = sorted(glob.glob(os.path.join(physionet, 'training-*', '*.wav')))
    for w in [wavs[i] for i in rng.choice(len(wavs), 6, replace=False)]:
        x, fs = sf.read(w, dtype='float64')
        x = resample_poly(x, 8000, fs)
        yield os.path.basename(w), np.clip(np.round(x / np.abs(x).max() * 8000), -32768, 32767).astype(np.int16)
    yield 'silence', np.round(rng.normal(0, 2, 8000 * 8)).astype(np.int16)
    yield 'white_noise', np.round(rng.normal(0, 3000, 8000 * 8)).astype(np.int16)
    x, fs = sf.read(wavs[0], dtype='float64')
    x = resample_poly(x, 8000, fs)
    yield 'clipped', np.clip(np.round(x / np.abs(x).max() * 60000), -32768, 32767).astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--physionet', default=os.path.expanduser('~/Downloads/PGT-1/data'))
    a = ap.parse_args()
    build()
    m = load_model()
    worst, ok = 0.0, True
    for name, pcm in signals(a.physionet):
        py = score_stream(pcm, m)
        cpp, csum = run_cpp(pcm)
        if len(py) != len(cpp):
            print(f'FAIL {name}: {len(py)} python results vs {len(cpp)} C++'); ok = False; continue
        from sqi_features import features
        from sqi_frontend import device_to_2k, CTX, STEP
        x = device_to_2k(pcm)
        for k, (p, c) in enumerate(zip(py, cpp)):
            Fp = features(x[k * STEP:k * STEP + CTX][None, :])[0]
            Fc = np.array([float(v) for v in c[8:]])
            d = np.abs(Fp - Fc).max()
            worst = max(worst, d, abs(p['q'] - float(c[2])))
            if c[3] != p['verdict'] or c[4] != p['reason'] or d > 1e-6 or abs(p['q'] - float(c[2])) > 1e-6:
                ok = False
                print(f'FAIL {name} #{k}: py={p["verdict"]}/{p["reason"]} q={p["q"]:.6f}  '
                      f'cpp={c[3]}/{c[4]} q={float(c[2]):.6f}  max feat diff {d:.2e} '
                      f'({NAMES[int(np.abs(Fp - Fc).argmax())]})')
        s = summarize(py)
        verdicts = ''.join(r['verdict'][0] for r in py)
        print(f'{name:14s} n={len(py):3d} usable={s["usable_ratio"]:.2f} beats={s["usable_beats"]:.1f} '
              f'cpp_usable={csum[4]:.2f} cpp_beats={csum[5]:.1f}  {verdicts}')
    print(f'max abs diff {worst:.2e}')
    print('PARITY PASS' if ok else 'PARITY FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
