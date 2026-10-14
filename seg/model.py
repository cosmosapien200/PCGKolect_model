"""PCG segmentation-quality network (v2) and its front-end.

Front-end (CPU on the device): 2 kHz audio -> 128-pt Hann STFT, hop 40 (20 ms, 50 fps)
  -> 32-band triangular filterbank 25..900 Hz -> log -> minus window mean, / 4.
Network (NPU): 1D U-Net written with Conv2d(kernel 1xk) on [B, 32, 1, T] so every op is
  plain Conv / ReLU / Resize(x2 nearest) / Concat — all INT8-native on RV1103.
Output: [B, 5, 1, T] logits per 20 ms frame: S1, systole, S2, diastole, noise.
  (softmax is done on the CPU: it is a CPU-fallback op on RV1103)
"""
import numpy as np
import torch
import torch.nn as nn

FS = 2000
NFFT = 128
HOP = 40
NBANDS = 32
FRAMES = 256                          # model window: 5.12 s, 16-aligned width
SAMPLES = NFFT + HOP * (FRAMES - 1)   # 10328 audio samples per window
FEAT_SCALE = 4.0
CLASSES = ['S1', 'systole', 'S2', 'diastole', 'noise']
NOISE = 4
IGNORE = 255


def filterbank(n=NBANDS, fmin=25.0, fmax=900.0):
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = imel(np.linspace(mel(fmin), mel(fmax), n + 2))
    f = np.arange(NFFT // 2 + 1) * FS / NFFT
    fb = np.zeros((n, len(f)), np.float32)
    for i in range(n):
        l, c, r = pts[i:i + 3]
        fb[i] = np.clip(np.minimum((f - l) / (c - l), (r - f) / (r - c)), 0, None)
        if fb[i].sum() == 0:                       # narrow low bands: nearest bin
            fb[i, np.argmin(np.abs(f - c))] = 1.0
    return fb


HANN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / NFFT)).astype(np.float32)   # periodic Hann
FB = filterbank()


class Frontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('win', torch.tensor(HANN))
        self.register_buffer('fb', torch.tensor(FB))

    def forward(self, x):                          # x [B, N] -> [B, 32, 1, T]
        S = torch.stft(x, NFFT, HOP, window=self.win, center=False, return_complex=True)
        P = S.real ** 2 + S.imag ** 2
        M = torch.log(self.fb @ P + 1e-10)
        M = (M - M.mean((1, 2), keepdim=True)) / FEAT_SCALE
        return M.unsqueeze(2)


def cbr(cin, cout, k=5, s=1, d=1):
    return nn.Sequential(nn.Conv2d(cin, cout, (1, k), (1, s), (0, d * (k // 2)), (1, d), bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SegNet(nn.Module):
    def __init__(self, c=(32, 48, 64, 96)):
        super().__init__()
        c0, c1, c2, c3 = c
        self.e0 = nn.Sequential(cbr(NBANDS, c0), cbr(c0, c0))
        self.e1 = nn.Sequential(cbr(c0, c1, s=2), cbr(c1, c1))
        self.e2 = nn.Sequential(cbr(c1, c2, s=2), cbr(c2, c2))
        self.e3 = nn.Sequential(cbr(c2, c3, s=2), cbr(c3, c3, d=2), cbr(c3, c3, d=4), cbr(c3, c3, d=8))
        self.up = nn.Upsample(scale_factor=(1, 2), mode='nearest')
        self.d2 = nn.Sequential(cbr(c3 + c2, c2), cbr(c2, c2))
        self.d1 = nn.Sequential(cbr(c2 + c1, c1), cbr(c1, c1))
        self.d0 = nn.Sequential(cbr(c1 + c0, c0), cbr(c0, c0))
        self.head = nn.Conv2d(c0, len(CLASSES), 1)

    def forward(self, f):                          # f [B, 32, 1, T], T % 8 == 0
        x0 = self.e0(f)
        x1 = self.e1(x0)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        y = self.d2(torch.cat([self.up(x3), x2], 1))
        y = self.d1(torch.cat([self.up(y), x1], 1))
        y = self.d0(torch.cat([self.up(y), x0], 1))
        return self.head(y)


def frame_labels(lab, start, T):
    """Label of each frame = label at the frame's centre sample."""
    return lab[start + HOP * np.arange(T) + NFFT // 2]
