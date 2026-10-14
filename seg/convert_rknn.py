"""Step 2 of deployment (rknn-toolkit2 2.3.2 venv): ONNX -> RKNN INT8 for RV1103, then
compare the quantised model (PC simulator) against the float reference.

  <rknn-venv>/bin/python seg/convert_rknn.py --export export/ [--dtype w8a8|w16a16i] [--algo normal|mmse|kl_divergence]
"""
import argparse, os
import numpy as np
from rknn.api import RKNN


def softmax(z, ax):
    z = z - z.max(ax, keepdims=True)
    e = np.exp(z)
    return e / e.sum(ax, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', default='export')
    ap.add_argument('--dtype', default='w8a8')
    ap.add_argument('--algo', default='normal')
    a = ap.parse_args()
    out = os.path.join(a.export, f'segnet_{a.dtype}_{a.algo}.rknn')
    rknn = RKNN(verbose=False)
    rknn.config(target_platform='rv1103', quantized_dtype=a.dtype, quantized_algorithm=a.algo,
                optimization_level=3)
    assert rknn.load_onnx(model=os.path.join(a.export, 'segnet.onnx')) == 0
    assert rknn.build(do_quantization=True, dataset=os.path.join(a.export, 'calib.txt')) == 0
    assert rknn.export_rknn(out) == 0
    print('wrote', out, os.path.getsize(out), 'bytes')

    assert rknn.init_runtime(target=None) == 0            # PC simulator
    ref_in, ref_out = np.load(os.path.join(a.export, 'ref_in.npy')), np.load(os.path.join(a.export, 'ref_out.npy'))
    agree, cos = [], []
    for i in range(len(ref_in)):
        o = rknn.inference(inputs=[ref_in[i:i + 1]], data_format='nchw')[0]
        o = np.asarray(o).reshape(ref_out[i:i + 1].shape)
        pq, pf = softmax(o[0, :, 0], 0), softmax(ref_out[i, :, 0], 0)
        agree.append((pq.argmax(0) == pf.argmax(0)).mean())
        cos.append(float((o.ravel() @ ref_out[i].ravel()) / (np.linalg.norm(o) * np.linalg.norm(ref_out[i]) + 1e-9)))
    print(f'{a.dtype}/{a.algo}: frame argmax agreement {np.mean(agree):.4f}, logits cosine {np.mean(cos):.4f}')
    rknn.release()


if __name__ == '__main__':
    main()
