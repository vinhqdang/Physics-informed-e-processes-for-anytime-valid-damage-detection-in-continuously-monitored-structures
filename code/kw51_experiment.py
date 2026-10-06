"""PHASE (reduced form) and comparators on the KW51 railway-bridge monitoring data
(Maes and Lombaert; Zenodo record 3745914, file trackedmodes.zip, CC BY-NC-SA 4.0).

The data hold hourly identified natural frequencies of 14 tracked modes, ambient
temperature and humidity, for 15 months that include a retrofit.  No finite element
model is available here, so the physics-structured surrogate and the physics-derived
cone are replaced by (i) a per-mode regression on temperature (and its square) with
sequential plug-in and forgetting, and (ii) the plain non-positive orthant as damage
cone.  Everything else (bounded score, bet grid, allowance, discounting, filtered
adaptation, alarm at 1/alpha) is as in monitors.py.

The raw file is NOT redistributed; fetch it with

    python3 kw51_experiment.py --download          (writes kw51/trackedmodes.mat)

and run   python3 kw51_experiment.py   to reproduce kw51_results.json.
"""
import datetime as dt
import json
import os
import sys
import urllib.request
import warnings

import numpy as np
import scipy.io as sio
from scipy.stats import chi2

warnings.filterwarnings('ignore')
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kw51', 'trackedmodes.mat')
URL = 'https://zenodo.org/api/records/3745914/files/trackedmodes.zip/content'

MODES = [2, 4, 5, 8, 12]               # tracked modes with the fewest identification gaps
ALPHA = 0.01
C_SCALE = 3.0
HUBER = 3.0
FORGET = 0.9995
N_LAM = 12
EPS_CAL = ALPHA / 2
WORKS_START = dt.datetime(2019, 5, 13)  # first visible change of the tracked frequencies
WORKS_END = dt.datetime(2019, 9, 16)    # frequencies settle at their post-retrofit levels


def download():
    import io
    import zipfile
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    raw = urllib.request.urlopen(URL, timeout=300).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with z.open('trackedmodes/trackedmodes.mat') as f, open(DATA, 'wb') as g:
            g.write(f.read())
    print('saved', DATA)


def load(agg='hourly'):
    """Valid epochs (all five modes and all covariates present), optionally averaged
    per calendar day (days with at least 12 valid hours).  Covariate columns:
    air temperature, air relative humidity, wind speed."""
    d = sio.loadmat(DATA, squeeze_me=True, struct_as_record=False)['modes']
    t = np.array([dt.datetime.fromordinal(int(x)) + dt.timedelta(days=float(x % 1))
                  - dt.timedelta(days=366) for x in d.sdn])
    lab = list(d.labels_env)
    E = d.env[:, [lab.index('tVL'), lab.index('rhVL'), lab.index('wsVL')]].astype(float)
    f = d.f[:, MODES]
    ok = np.isfinite(f).all(1) & np.isfinite(E).all(1)
    t, y, E = t[ok], np.log(f[ok]), E[ok]
    if agg == 'hourly':
        return t, y, E
    day = np.array([x.date() for x in t])
    td, yd, Ed = [], [], []
    for u in sorted(set(day)):
        m = day == u
        if m.sum() >= 12:
            td.append(dt.datetime(u.year, u.month, u.day, 12)); yd.append(y[m].mean(0)); Ed.append(E[m].mean(0))
    return np.array(td), np.array(yd), np.array(Ed)


# ------------------------------------------------------------------ generators
def orth_generators(m):
    E = np.eye(m)
    g = [-E[j] for j in range(m)]
    for j in range(m):
        for k in range(j + 1, m):
            for w in (0.25, 0.5, 0.75):
                g.append(-(w * E[j] + (1 - w) * E[k]))
    g.append(-np.ones(m) / np.sqrt(m))
    G = np.array(g)
    return G / np.linalg.norm(G, axis=1, keepdims=True)


def null_means(V, m, n=400_000, seed=3):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, m))
    c_om = C_SCALE * np.sqrt(m) / 1.2
    x = np.minimum(np.clip((e @ V.T).max(1), 0, None) / C_SCALE, 1.0)
    xo = np.minimum(np.linalg.norm(e, axis=1) / c_om, 1.0)
    return float(x.mean()), float(xo.mean()), c_om


class Surrogate:
    """Per-mode regression on features of temperature, recursive with forgetting."""

    def __init__(self, m, kind, forget=FORGET):
        self.kind = kind
        self.forget = forget
        self.p = {'lin': 2, 'quad': 3, 'env': 5, 'frz': 5}[kind]
        self.A = np.eye(self.p) * 1e-6
        self.b = np.zeros((self.p, m))
        self.coef = np.zeros((self.p, m))

    def feat(self, E):
        T, rh, ws = E
        if self.kind == 'lin':
            return np.array([1.0, T])
        if self.kind == 'quad':
            return np.array([1.0, T, T * T / 10.0])
        if self.kind == 'frz':   # post hoc: freezing hinge instead of the quadratic term
            return np.array([1.0, T, max(0.0, -T) / 5.0, rh / 100.0, ws])
        return np.array([1.0, T, T * T / 10.0, rh / 100.0, ws])

    def predict(self, T):
        return self.feat(T) @ self.coef

    def update(self, T, y):
        x = self.feat(T)
        self.A = self.forget * self.A + np.outer(x, x)
        self.b = self.forget * self.b + np.outer(x, y)
        self.coef = np.linalg.solve(self.A + 1e-9 * np.eye(self.p), self.b)


def robust_scale(res):
    s0 = np.maximum(1.4826 * np.median(np.abs(res - np.median(res, 0)), 0), 1e-7)
    keep = np.abs(res) <= 6 * s0
    return np.maximum(np.array([res[keep[:, i], i].std(ddof=1) for i in range(res.shape[1])]), 1e-7)


def mix(l):
    mx = l.max()
    return mx + np.log(np.exp(l - mx).mean())


def run(t, y, T, t_start, n_comm, kind='quad', keep_path=False, forget=FORGET):
    """Commission on the n_comm valid epochs from t_start; monitor everything after."""
    m = y.shape[1]
    i0 = int(np.searchsorted(t, t_start))
    i1 = i0 + n_comm
    half = n_comm // 2
    V = orth_generators(m)
    mu0, mu0o, c_om = null_means(V, m)
    lams = np.geomspace(0.05, 0.9 / mu0, N_LAM)
    lams_o = np.geomspace(0.05, 0.9 / mu0o, N_LAM)
    sg = Surrogate(m, kind, forget)

    def score(r):
        return min(max((r @ V.T).max(), 0.0) / C_SCALE, 1.0)

    def filt(yv, sig, r):
        pj = r @ V.T
        w = max(pj.max(), 0.0) * V[int(np.argmax(pj))]
        return yv - sig * (r - np.clip(r - w, -HUBER, HUBER))

    # pass A
    rA = []
    for k in range(i0, i0 + half):
        if k - i0 > max(5, min(30, half // 4)):
            rA.append(y[k] - sg.predict(T[k]))
        sg.update(T[k], y[k])
    sig = robust_scale(np.array(rA[len(rA) // 3:]))
    # pass B
    xb, xob, RB = [], [], []
    for k in range(i0 + half, i1):
        r = (y[k] - sg.predict(T[k])) / sig
        xb.append(score(r)); xob.append(min(np.linalg.norm(r) / c_om, 1.0)); RB.append(r)
        sg.update(T[k], filt(y[k], sig, r))
    xb, xob = np.array(xb), np.array(xob)
    dl = lambda x, m0: max(0.0, x.mean() - m0 + 2 * x.std(ddof=1) / np.sqrt(len(x)))
    delta, delta_o = dl(xb, mu0), dl(xob, mu0o)
    delta_c = max(0.0, xb.mean() - mu0 + np.sqrt(np.log(1 / EPS_CAL) / (2 * len(xb))))
    RA = np.array(rA[len(rA) // 3:]) / sig
    xa = np.minimum(np.clip((RA @ V.T).max(1), 0, None) / C_SCALE, 1.0)
    cal_f = np.sort(np.concatenate([xa, xb]))
    eps = np.linspace(0.05, 0.95, 12)
    rng = np.random.default_rng(20240518)
    chi_c = chi2.ppf(1 - ALPHA, m)
    thr = np.log(1 / ALPHA)

    lw = np.zeros(N_LAM); lwd = np.zeros(N_LAM); lwo = np.zeros(N_LAM); lwc = np.zeros(N_LAM)
    lctm = np.zeros(len(eps))
    names = ('phase', 'phase_nd', 'phase_c', 'phase_omni', 'ctm_f', 'chart3', 'chi2')
    al = {k: None for k in names}
    path = []
    n_used = 0
    for k in range(i1, len(t)):
        r = (y[k] - sg.predict(T[k])) / sig
        x = score(r)
        xo = min(np.linalg.norm(r) / c_om, 1.0)
        inc = np.log1p(lams * (x - mu0))
        lw += inc
        lwd += inc - np.log1p(lams * delta)
        lwc += inc - np.log1p(lams * delta_c)
        lwo += np.log1p(lams_o * (xo - mu0o)) - np.log1p(lams_o * delta_o)
        lo = np.searchsorted(cal_f, x, 'left'); hi = np.searchsorted(cal_f, x, 'right')
        p = lo + rng.random() * (1 + hi - lo)
        p = min(max((len(cal_f) - p + 1) / (len(cal_f) + 1), 1e-6), 1.0)
        lctm += np.log(eps) + (eps - 1.0) * np.log(p)
        Ld, Lw, Lc, Lo, Lt = mix(lwd), mix(lw), mix(lwc), mix(lwo), mix(lctm)
        for nm, cond in (('phase', Ld >= thr), ('phase_nd', Lw >= thr),
                         ('phase_c', Lc >= np.log(1 / (ALPHA - EPS_CAL))),
                         ('phase_omni', Lo >= thr), ('ctm_f', Lt >= thr),
                         ('chart3', np.abs(r).max() > 3.0), ('chi2', r @ r > chi_c)):
            if al[nm] is None and cond:
                al[nm] = t[k]
        if keep_path:
            path.append((t[k].isoformat(), Ld, Lo, x))
        sg.update(T[k], filt(y[k], sig, r))
        n_used += 1
    info = dict(delta=float(delta), delta_c=float(delta_c), delta_o=float(delta_o), mu0=mu0,
                sig=sig.tolist(), comm_end=t[i1 - 1].isoformat(), x_comm_mean=float(xb.mean()))
    return al, info, path


def classify(ts):
    if ts is None:
        return 'none'
    if ts < WORKS_START:
        return 'before works (false alarm)'
    if ts < WORKS_END:
        return 'during works'
    return 'after works'


def main():
    out = dict(modes=MODES, works_start=WORKS_START.isoformat(), works_end=WORKS_END.isoformat(), runs=[])
    for agg in ('hourly', 'daily'):
        t, y, E = load(agg)
        out['n_epochs_' + agg] = int(len(t))
        pre = t < WORKS_START
        out['n_pre_works_' + agg] = int(pre.sum())
        if agg == 'hourly':
            T = E[:, 0]
            X = np.stack([np.ones(pre.sum()), T[pre], T[pre] ** 2 / 10.0], 1)
            B = np.linalg.lstsq(X, y[pre], rcond=None)[0]
            res = y[pre] - X @ B
            out['pre_cv_pct'] = (100 * y[pre].std(0)).tolist()
            out['pre_res_sd_pct'] = (100 * res.std(0)).tolist()
            out['pre_slope_pct_per_C'] = (100 * B[1]).tolist()
        forget = FORGET if agg == 'hourly' else FORGET ** 24
        for kind in ('lin', 'quad', 'env', 'frz'):
            t0 = dt.datetime(t[0].year, t[0].month, t[0].day)
            for start in (t0, t0 + dt.timedelta(days=14), t0 + dt.timedelta(days=28)):
                for days in (45, 60, 90, 120, 150):
                    n_comm = int(((t >= start) & (t < start + dt.timedelta(days=days))).sum())
                    if n_comm < 24:
                        continue
                    al, info, _ = run(t, y, E, start, n_comm, kind, forget=forget)
                    out['runs'].append(dict(agg=agg, kind=kind, start=start.isoformat(), days=days,
                                            n_comm=n_comm,
                                            alarms={k: (v.isoformat() if v else None) for k, v in al.items()},
                                            cls={k: classify(v) for k, v in al.items()}, info=info))
                    print(agg, kind, start.date(), days, {k: (v.date().isoformat() if v else None) for k, v in al.items()},
                          round(info['delta'], 3), flush=True)
    out['paths'] = {}
    for agg, kind, forget in (('hourly', 'quad', FORGET), ('daily', 'env', FORGET ** 24), ('hourly', 'frz', FORGET)):
        t, y, E = load(agg)
        start = dt.datetime(t[0].year, t[0].month, t[0].day)
        n_comm = int(((t >= start) & (t < start + dt.timedelta(days=90))).sum())
        al, info, path = run(t, y, E, start, n_comm, kind, keep_path=True, forget=forget)
        out['paths']['%s_%s' % (agg, kind)] = dict(path=path, comm_end=info['comm_end'],
                                                  alarms={k: (v.isoformat() if v else None) for k, v in al.items()})
    json.dump(out, open('kw51_results.json', 'w'))
    print('saved kw51_results.json')


if __name__ == '__main__':
    if '--download' in sys.argv:
        download()
    else:
        main()
