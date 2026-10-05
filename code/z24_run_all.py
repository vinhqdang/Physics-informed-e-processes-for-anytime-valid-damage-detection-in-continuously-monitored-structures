"""Reproduces Table 'Z24 bridge' (all six experiments, both measurement
modalities, ten monitors) from the extracted modal features.

Run from this directory:   python3 z24_run_all.py
Uses z24_modal.npz (ambient) and z24_modal_fvt.npz (forced), so the raw
acceleration records are not required. Output: z24_full.json

Environment variables:  REPS (default 200 orderings per experiment),
                        OUT  (default z24_full.json)
"""
import json, os
import numpy as np
import z24_modal as Z
import z24_experiment as E

KEYS = ['phase', 'phase_omni', 'ctm', 'cusum', 'sr', 'ewma',
        'chart3', 'chi2', 'pca', 'msd']
REPS = int(os.environ.get('REPS', 200))
OUT = os.environ.get('OUT', 'z24_full.json')

out = {}
for mod in ['avt', 'fvt']:
    d = Z.build(mod=mod)
    # split the pooled reference days into two halves for R1b
    pool = np.concatenate([d[1], d[2]])
    rng = np.random.default_rng(11)
    idx = rng.permutation(len(pool))
    d2 = dict(d)
    d2[101], d2[102] = pool[idx[:84]], pool[idx[84:]]
    exps = [('R1a', [1], [2], d), ('R1b', [101], [102], d2),
            ('R2', [1, 2], [3, 4, 5, 6], d), ('R3', [6], [8], d),
            ('R4', [1, 2], [9, 10], d), ('R5', [1, 2], [8], d)]
    for nm, burn, mon, dd in exps:
        r = E.experiment(burn, mon, reps=REPS, data=dd)
        key = '%s_%s' % (mod, nm)
        out[key] = {k: (round(r[k]['rate'], 3), r[k]['median_epoch']) for k in KEYS}
        out[key]['_delta'] = [round(v, 3) for v in r['_delta']]
        print(key, out[key], flush=True)
json.dump(out, open(OUT, 'w'), indent=1)
