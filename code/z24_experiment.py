"""PHASE and baselines on the Z24 progressive damage test (real data).

Features: three natural frequencies tracked hourly-equivalent (81.9 s epochs) by
band-limited frequency domain decomposition on the five permanent reference
accelerometers (see z24_modal.py). No environmental covariate was recorded in
the released ambient-vibration files, so the nuisance surrogate degenerates to a
predictable running intercept and the damage cone is instantiated at its coarsest
physically justified level: stiffness loss cannot raise a natural frequency, so
the admissible set is the non-positive orthant in standardised log-frequency
space.
"""
import json
import numpy as np
from scipy.stats import chi2
import z24_modal as Z

ALPHA = 0.01
C_SCALE = 2.5
HUBER = 3.0
FORGET = 0.995
N_LAM = 12
M = len(Z.BANDS)
SCEN_NAME = {1: 'reference 1', 2: 'reference 2', 3: 'settlement 20 mm',
             4: 'settlement 40 mm', 5: 'settlement 80 mm', 6: 'settlement 95 mm',
             7: 'foundation tilt', 8: 'reference 3 (settlement reversed)',
             9: 'spalling 12 m$^2$', 10: 'spalling 24 m$^2$'}


def cone_generators(sigma, m=M):
    """Generators of the non-positive orthant: singletons, pairs and the
    all-modes direction, standardised and normalised."""
    gens = []
    base = -np.eye(m) / sigma
    for j in range(m):
        gens.append(base[j])
    for j in range(m):
        for k in range(j + 1, m):
            for w in (0.25, 0.5, 0.75):
                gens.append(w * base[j] + (1 - w) * base[k])
    gens.append(base.mean(axis=0))
    V = np.array(gens)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def null_means(V, m=M, c=C_SCALE, n=400_000, seed=3):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, m))
    x = np.minimum(np.clip((e @ V.T).max(axis=1), 0, None) / c, 1.0)
    xo = np.minimum(np.linalg.norm(e, axis=1) / (c * np.sqrt(m) / 1.2), 1.0)
    return float(x.mean()), float(xo.mean())


def monitor(burn, stream, alpha=ALPHA, seed=0):
    """burn: (n0, m) log-frequency array; stream: (n, m). Returns alarm epochs."""
    mu = burn[:len(burn) // 2].mean(0)
    res0 = burn[len(burn) // 2:] - mu
    sig = np.maximum(1.4826 * np.median(np.abs(res0 - np.median(res0, 0)), 0), 1e-5)
    V = cone_generators(sig)
    mu0, mu0o = null_means(V)
    c_omni = C_SCALE * np.sqrt(M) / 1.2

    # predictable calibration of delta on the held-out half of the burn-in
    rr = res0 / sig
    xb = np.minimum(np.clip((rr @ V.T).max(1), 0, None) / C_SCALE, 1.0)
    xob = np.minimum(np.linalg.norm(rr, axis=1) / c_omni, 1.0)
    d = lambda x, m0: max(0.0, x.mean() - m0 + 2 * x.std(ddof=1) / np.sqrt(len(x)))
    delta, delta_o = d(xb, mu0), d(xob, mu0o)

    # additional comparators, all calibrated on the commissioning window only
    Yc = burn - burn.mean(0)
    _, _, Vt = np.linalg.svd(Yc, full_matrices=False)
    Ppca = np.eye(M) - Vt[:1].T @ Vt[:1]
    sp = np.linalg.norm(Yc @ Ppca.T, axis=1)
    sp_s = max(sp.std(), 1e-12)
    Cinv = np.linalg.pinv(np.cov(rr.T) + 1e-9 * np.eye(M))
    h_pca = float(np.quantile(np.linalg.norm((res0) @ Ppca.T, axis=1) / sp_s, 0.99)) * 1.5
    h_msd = float(np.quantile(np.sqrt(np.maximum((rr @ Cinv * rr).sum(1), 0)), 0.99)) * 1.5
    h_ewma = float(np.quantile(np.abs(xb - mu0), 0.99)) * 1.5
    h_sr = 6.0
    cal = np.sort(xb)
    n_cal = len(cal)
    eps_grid = np.linspace(0.05, 0.95, 12)
    lctm = np.zeros(len(eps_grid))
    rng_c = np.random.default_rng(1234 + seed)
    ew, ew_lam, srv = mu0, 0.1, 0.0

    lams = np.geomspace(0.05, 0.9 / mu0, N_LAM)
    lams_o = np.geomspace(0.05, 0.9 / mu0o, N_LAM)
    lw = np.zeros(N_LAM); lwo = np.zeros(N_LAM)
    disc = np.log1p(lams * delta); disc_o = np.log1p(lams_o * delta_o)
    logthr = np.log(1 / alpha)
    chi_c = chi2.ppf(1 - alpha, M)
    hcus = np.quantile(np.maximum.accumulate(np.cumsum(xb - mu0 - 0.05)), 0.99)
    hcus = max(hcus, 1.0)
    cus = 0.0
    mix = lambda l: l.max() + np.log(np.exp(l - l.max()).mean())
    al = {k: -1 for k in ('phase', 'phase_omni', 'chart3', 'chi2', 'cusum',
                          'pca', 'msd', 'ewma', 'sr', 'ctm')}
    path = []
    est = mu.copy()
    for t, y in enumerate(stream):
        r = (y - est) / sig
        proj = r @ V.T
        x = min(max(proj.max(), 0.0) / C_SCALE, 1.0)
        xo = min(np.linalg.norm(r) / c_omni, 1.0)
        lw += np.log1p(lams * (x - mu0)) - disc
        lwo += np.log1p(lams_o * (xo - mu0o)) - disc_o
        Lp, Lo = mix(lw), mix(lwo)
        cus = max(0.0, cus + (x - mu0) - 0.05)
        s_pca = np.linalg.norm(Ppca @ (y - burn.mean(0))) / sp_s
        s_msd = float(np.sqrt(max(r @ Cinv @ r, 0.0)))
        ew = (1 - ew_lam) * ew + ew_lam * x
        s_ewma = abs(ew - mu0)
        srv = min((1.0 + srv) * np.exp(4.0 * (x - mu0 - 0.05)), 1e290)
        s_sr = np.log1p(srv)
        lo = np.searchsorted(cal, x, 'left'); hi = np.searchsorted(cal, x, 'right')
        pc = lo + rng_c.random() * (1 + hi - lo)
        pc = min(max((n_cal - pc + 1) / (n_cal + 1), 1e-6), 1.0)
        lctm += np.log(eps_grid) + (eps_grid - 1.0) * np.log(pc)
        L_ctm = mix(lctm)
        if al['phase'] < 0 and Lp >= logthr: al['phase'] = t
        if al['phase_omni'] < 0 and Lo >= logthr: al['phase_omni'] = t
        if al['chart3'] < 0 and np.abs(r).max() > 3: al['chart3'] = t
        if al['chi2'] < 0 and (r @ r) > chi_c: al['chi2'] = t
        if al['cusum'] < 0 and cus > hcus: al['cusum'] = t
        for nm, st, h_ in (('pca', s_pca, h_pca), ('msd', s_msd, h_msd),
                           ('ewma', s_ewma, h_ewma), ('sr', s_sr, h_sr)):
            if al[nm] < 0 and st > h_: al[nm] = t
        if al['ctm'] < 0 and L_ctm >= logthr: al['ctm'] = t
        path.append([Lp, Lo, x])
        # cone-filtered, Huber-clipped adaptation of the intercept
        k = int(np.argmax(proj))
        w = max(proj[k], 0.0) * V[k]
        keep = np.clip(r - w, -HUBER, HUBER)
        est = est + (1 - FORGET) * sig * keep
    return al, np.array(path), dict(delta=delta, delta_o=delta_o, mu0=mu0,
                                    sigma=sig.tolist(), hcus=float(hcus))


def experiment(burn_scen, mon_scen, reps=200, seed=0, data=None, mod='avt'):
    data = data if data is not None else Z.build(mod=mod)
    rng = np.random.default_rng(seed)
    keys = ('phase', 'phase_omni', 'ctm', 'cusum', 'sr', 'ewma',
            'chart3', 'chi2', 'pca', 'msd')
    fired = {k: [] for k in keys}
    first = {k: [] for k in keys}
    deltas = []
    for r in range(reps):
        burn = np.concatenate([np.log(data[s]) for s in burn_scen])
        burn = burn[rng.permutation(len(burn))]
        blocks, bounds = [], []
        n = 0
        for s in mon_scen:
            a = np.log(data[s])
            a = a[rng.permutation(len(a))]
            blocks.append(a); n += len(a); bounds.append((s, n))
        stream = np.concatenate(blocks)
        al, _, info = monitor(burn, stream, seed=r)
        deltas.append([info['delta'], info['delta_o']])
        for k in keys:
            fired[k].append(al[k] >= 0)
            first[k].append(al[k])
    stage = {}
    for k in keys:
        cnt = {}
        for v in first[k]:
            if v < 0:
                cnt['none'] = cnt.get('none', 0) + 1
                continue
            lab = 'none'
            prev = 0
            for s_, b in bounds:
                if v < b:
                    lab = s_; break
                prev = b
            cnt[lab] = cnt.get(lab, 0) + 1
        stage[k] = {str(a): b / reps for a, b in sorted(cnt.items(), key=str)}
    out = {k: dict(rate=float(np.mean(fired[k])),
                   se=float(np.std(fired[k]) / np.sqrt(reps)),
                   median_epoch=(float(np.median([v for v in first[k] if v >= 0]))
                                 if any(f >= 0 for f in first[k]) else None))
           for k in keys}
    out['_bounds'] = bounds
    out['_stage'] = stage
    out['_delta'] = np.mean(deltas, 0).tolist()
    out['_reps'] = reps
    return out


if __name__ == '__main__':
    data = Z.build()
    res = {}
    print('R1a burn=ref1, monitor=ref2 (undamaged, different test day)')
    res['R1a'] = experiment([1], [2], data=data)
    print('   ', {k: (v['rate'], v['median_epoch']) for k, v in res['R1a'].items()
                  if not k.startswith('_')}, 'delta', np.round(res['R1a']['_delta'], 3))

    print('R1b burn=ref1+ref2 pooled, monitor=held-out reference epochs')
    d2 = dict(data)
    pool = np.concatenate([data[1], data[2]])
    rng = np.random.default_rng(11)
    idx = rng.permutation(len(pool))
    d2[101], d2[102] = pool[idx[:84]], pool[idx[84:]]
    res['R1b'] = experiment([101], [102], data=d2)
    print('   ', {k: (v['rate'], v['median_epoch']) for k, v in res['R1b'].items()
                  if not k.startswith('_')}, 'delta', np.round(res['R1b']['_delta'], 3))

    print('R2 burn=ref1+ref2, monitor=settlement 20->40->80->95 mm')
    res['R2'] = experiment([1, 2], [3, 4, 5, 6], data=data)
    print('   ', {k: (v['rate'], v['median_epoch']) for k, v in res['R2'].items()
                  if not k.startswith('_')}, 'bounds', res['R2']['_bounds'])

    print('R3 burn=settlement 95 mm, monitor=reference 3 (settlement reversed)')
    res['R3'] = experiment([6], [8], data=data)
    print('   ', {k: (v['rate'], v['median_epoch']) for k, v in res['R3'].items()
                  if not k.startswith('_')})

    print('R4 burn=ref1+ref2, monitor=spalling 12 then 24 m2')
    res['R4'] = experiment([1, 2], [9, 10], data=data)
    print('   ', {k: (v['rate'], v['median_epoch']) for k, v in res['R4'].items()
                  if not k.startswith('_')}, 'bounds', res['R4']['_bounds'])

    json.dump(res, open('z24_results.json', 'w'), indent=1)
