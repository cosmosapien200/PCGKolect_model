"""Step 1 of deployment (main venv): checkpoint -> ONNX + calibration set + reference I/O.

  python seg/export_onnx.py --ckpt runs/seg_w_final.pt --width 48,64,96,128 --out export/

Writes
  export/segnet.onnx                 input  'feat'   [1, 32, 1, 256] float (CPU front-end output)
                                     output 'logits' [1, 5, 1, 256]
  export/calib/*.npy + calib.txt     calibration inputs, NCHW [1, 32, 1, 256]
  export/ref_in.npy / ref_out.npy    reference batch for the RKNN / board numerics check
  export/frontend_ref.npz            audio -> features pairs for the C++ front-end parity test
"""
import argparse, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from model import Frontend, SegNet, SAMPLES, FRAMES, NBANDS  # noqa: E402
from train_seg import Data, corrupt, frame_labels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--width', default='48,64,96,128')
    ap.add_argument('--out', default='export')
    ap.add_argument('--n-calib', type=int, default=300)
    ap.add_argument('--device-wavs', nargs='*', default=[],
                    help='device recordings (8 kHz WAV) to include in calibration')
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'calib'), exist_ok=True)
    width = tuple(int(v) for v in a.width.split(','))
    net = SegNet(width).eval()
    net.load_state_dict(torch.load(a.ckpt, map_location='cpu'))
    fe = Frontend().eval()

    dummy = torch.zeros(1, NBANDS, 1, FRAMES)
    onnx_path = os.path.join(a.out, 'segnet.onnx')
    torch.onnx.export(net, dummy, onnx_path, input_names=['feat'], output_names=['logits'],
                      opset_version=13, do_constant_folding=True, dynamo=False)
    print('wrote', onnx_path)

    # calibration: clean / partially noisy / pure-noise public windows (+ device audio if given)
    d = Data()
    rng = np.random.default_rng(7)
    X = []
    while len(X) < a.n_calib:
        r = rng.integers(len(d.rid))
        if d.len[r] < SAMPLES:
            continue
        s = d.off[r] + rng.integers(0, d.len[r] - SAMPLES + 1)
        x = d.audio[s:s + SAMPLES].astype(np.float32)
        ys = d.lab[s:s + SAMPLES]
        x, _ = corrupt(rng, d, x, ys, frame_labels(ys, 0, FRAMES))
        X.append(x)
    for w in a.device_wavs:
        import soundfile as sf
        from sqi_frontend import device_to_2k
        pcm, fs = sf.read(w, dtype='int16')
        x2 = device_to_2k(pcm).astype(np.float32)
        x2 = x2 / (np.abs(x2).max() + 1e-9)
        for s in range(0, len(x2) - SAMPLES + 1, SAMPLES // 2):
            X.append(x2[s:s + SAMPLES])
    X = torch.from_numpy(np.stack(X))
    with torch.no_grad():
        Fe = fe(X).numpy()                                    # [N, 32, 1, 256]
    lines = []
    for i, f in enumerate(Fe):
        p = os.path.abspath(os.path.join(a.out, 'calib', f'{i:04d}.npy'))
        np.save(p, f[None].astype(np.float32))   # NCHW [1, 32, 1, 256] (what rknn-toolkit2 2.3.2 expects for .npy)
        lines.append(p)
    with open(os.path.join(a.out, 'calib.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with torch.no_grad():
        ref_in = torch.from_numpy(Fe[:16])
        ref_out = net(ref_in).numpy()
    np.save(os.path.join(a.out, 'ref_in.npy'), Fe[:16])
    np.save(os.path.join(a.out, 'ref_out.npy'), ref_out)
    np.savez(os.path.join(a.out, 'frontend_ref.npz'), audio=X[:4].numpy(), feat=Fe[:4])
    print(f'calibration set: {len(lines)} windows; feature range [{Fe.min():.2f}, {Fe.max():.2f}]')


if __name__ == '__main__':
    main()
