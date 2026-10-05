"""Chronological and dependence-preserving sensitivity checks on the Z24 data
(Section 5.3 of the manuscript), exchangeability diagnostics, and the sign pattern
of the tracked-frequency shifts.  Output: z24_sensitivity.json.

Run from this directory:  python3 z24_sensitivity.py     (REPS=200 by default)
"""
import json, os
import numpy as np
from scipy import stats
import z24_modal as Z
import z24_experiment as E

REPS = int(os.environ.get('REPS', 200))
KEYS = ['phase', 'phase_omni', 'ctm', 'cusum', 'sr', 'ewma', 'chart3', 'chi2', 'pca', 'msd']
ORDERS = os.environ.get('ORDERS', 'perm,chron,rev,rot,blockperm,poolblock').split(',')
OUTF = os.environ.get('OUT', 'z24_sensitivity.json')
EXPS = [('R1a', [1], [2]), ('R2', [1, 2], [3, 4, 5, 6]), ('R3', [6], [8]),
        ('R4', [1, 2], [9, 10]), ('R5', [1, 2], [8])]
out = dict(reps=REPS, runs={}, diag={}, shifts={})

for mod in ('avt', 'fvt'):
    d = Z.build(mod=mod)
    # ---- exchangeability diagnostics, natural order within each state ----------
    rows = []
    for s in range(1, 11):
        x = np.log(d[s])
        n = len(x)
        for j in range(x.shape[1]):
            z = x[:, j] - x[:, j].mean()
            r1 = float((z[:-1] @ z[1:]) / (z @ z))
            rho, p = stats.spearmanr(np.arange(n), x[:, j])
            # share of variance between the nine measurement set-ups (one-way ANOVA ICC)
            blocks = np.array_split(x[:, j], 9)
            F, pF = stats.f_oneway(*blocks)
            rows.append(dict(state=s, mode=j + 1, n=n, lag1=r1, bound=1.96 / np.sqrt(n),
                             trend_rho=float(rho), trend_p=float(p), setup_F=float(F),
                             setup_p=float(pF)))
    out['diag'][mod] = dict(
        rows=rows,
        frac_lag1_sig=float(np.mean([abs(r['lag1']) > r['bound'] for r in rows])),
        median_lag1=float(np.median([r['lag1'] for r in rows])),
        frac_trend_sig=float(np.mean([r['trend_p'] < 0.05 for r in rows])),
        frac_setup_sig=float(np.mean([r['setup_p'] < 0.05 for r in rows])))
    # ---- sign pattern of state means relative to reference 1+2 ----------------
    ref = np.log(np.concatenate([d[1], d[2]])).mean(0)
    out['shifts'][mod] = {str(s): (100 * (np.log(d[s]).mean(0) - ref)).tolist()
                          for s in range(1, 11)}
    # ---- alarm rates under the five orderings -------------------------------
    for nm, burn, mon in EXPS:
        for order in ORDERS:
            reps = 1 if False else REPS
            r = E.experiment(burn, mon, reps=reps, data=d, order=order)
            key = '%s_%s_%s' % (mod, nm, order)
            out['runs'][key] = {k: (round(r[k]['rate'], 3), r[k]['median_epoch'])
                                for k in KEYS}
            print(key, {k: out['runs'][key][k] for k in ('phase', 'phase_omni', 'ctm', 'cusum')},
                  flush=True)
json.dump(out, open(OUTF, 'w'), indent=1)
