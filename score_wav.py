"""Score a recording with the exact device algorithm and print the quality timeline.

  python score_wav.py ../PCGKolect_rk/recordings/recording.wav [--json out.json]

Any sample rate is accepted (resampled to the device's 8 kHz int16 first).
"""
import argparse, json
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

from sqi_frontend import score_stream, summarize, DEVICE_FS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('wav')
    ap.add_argument('--json', help='write per-second results + summary (sidecar format)')
    a = ap.parse_args()
    x, fs = sf.read(a.wav, dtype='int16')
    if x.ndim > 1:
        x = x[:, 0]
    if fs != DEVICE_FS:
        x = np.clip(np.round(resample_poly(x.astype(np.float64), DEVICE_FS, fs)), -32768, 32767).astype(np.int16)
    res = score_stream(x)
    print(f'{"span (s)":>12s}  {"q":>5s}  {"verdict":8s} {"reason":10s} {"level dBFS":>10s} {"HR":>5s}')
    for r in res:
        print(f'{r["t_start"]:5.1f}-{r["t_end"]:<5.1f}  {r["q"]:5.2f}  {r["verdict"]:8s} {r["reason"]:10s} '
              f'{r["level_dbfs"]:10.1f} {r["hr_bpm"]:5.0f}')
    s = summarize(res)
    print(json.dumps(s, indent=1))
    if a.json:
        with open(a.json, 'w') as f:
            json.dump(dict(file=a.wav, results=res, summary=s), f, indent=1)


if __name__ == '__main__':
    main()
