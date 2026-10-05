"""Operational modal analysis on the Z24 progressive damage test (ambient
vibration). For each time window we form the cross-power spectral density matrix
of the five permanently installed reference accelerometers, take its first
singular value (frequency domain decomposition) and peak-pick tracked modes in
fixed bands with parabolic interpolation.
"""
import glob, os, re, json
import numpy as np
import pandas as pd
from scipy import signal

ROOT = '/path/to/Z24-master'   # KU Leuven Z24 parquet files (not distributed)
FS = 100.0
WIN = 8192            # 81.9 s per monitoring epoch
NPERSEG = 2048
BANDS = [(3.60, 4.10), (9.60, 10.15), (12.05, 12.95)]   # tracked modes 1, 3, 5
REF = ['R1V ', 'R2L ', 'R2T ', 'R2V ', 'R3V ']


def fdd_spectrum(x, fs=FS, nperseg=NPERSEG):
    """First singular value of the CSD matrix, per frequency line."""
    m = x.shape[1]
    f = None
    G = None
    for i in range(m):
        for j in range(i, m):
            f, c = signal.csd(x[:, i], x[:, j], fs=fs, nperseg=nperseg,
                              noverlap=nperseg // 2)
            if G is None:
                G = np.zeros((len(f), m, m), dtype=complex)
            G[:, i, j] = c
            if i != j:
                G[:, j, i] = np.conj(c)
    s = np.linalg.svd(G, compute_uv=False)[:, 0]
    return f, s


def pick(f, s, band):
    lo, hi = band
    m = (f >= lo) & (f <= hi)
    idx = np.where(m)[0]
    k = idx[np.argmax(s[idx])]
    if k <= 0 or k >= len(f) - 1:
        return f[k], s[k]
    y0, y1, y2 = np.log(s[k - 1]), np.log(s[k]), np.log(s[k + 1])
    d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-30)
    d = float(np.clip(d, -1, 1))
    df = f[1] - f[0]
    return f[k] + d * df, s[k]


def process_file(path):
    df = pd.read_parquet(path)
    cols = [c for c in df.columns if c.strip().startswith('R')]
    x = df[cols].to_numpy(dtype=float)
    x = x - x.mean(0)
    out = []
    n = x.shape[0] // WIN
    for w in range(n):
        seg = x[w * WIN:(w + 1) * WIN]
        f, s = fdd_spectrum(seg)
        out.append([pick(f, s, b)[0] for b in BANDS])
    return np.array(out)


def build(scenarios=range(1, 11), cache='z24_modal.npz', mod='avt'):
    cache = cache if mod == 'avt' else cache.replace('.npz', '_%s.npz' % mod)
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return {int(k): d[k] for k in d.files}
    res = {}
    for sc in scenarios:
        rows, meta = [], []
        for su in range(1, 10):
            p = os.path.join(ROOT, '%02dsetup%02d_%s.parquet' % (sc, su, mod))
            if not os.path.exists(p):
                continue
            a = process_file(p)
            rows.append(a)
            meta += [(su, w) for w in range(len(a))]
        res[sc] = np.concatenate(rows) if rows else np.zeros((0, len(BANDS)))
        print('scenario %02d: %d epochs' % (sc, len(res[sc])), flush=True)
    np.savez(cache, **{str(k): v for k, v in res.items()})
    return res


if __name__ == '__main__':
    r = build()
    for k, v in sorted(r.items()):
        print('%02d  n=%3d  mean %s  cv%% %s' % (
            k, len(v), np.round(v.mean(0), 3),
            np.round(100 * v.std(0) / v.mean(0), 2)))
