"""Generate the LaTeX bodies of the result tables from summary.json,
identifiability.json and z24_sensitivity.json, so that every number in the
manuscript tables is produced by code.   Run:  python3 make_tables.py
Writes tables/*.tex (fragments that main.tex \\input's are NOT used; the
manuscript contains the pasted text, regenerate with tables_to_tex.py)."""
import json, os
import numpy as np

S = json.load(open('summary.json'))
os.makedirs('tables', exist_ok=True)
NAME = {'phase_d': 'PHASE', 'phase_c': 'PHASE-C', 'phase_r': 'PHASE-R', 'phase_e': 'PHASE-E',
        'phase_omni': 'PHASE-$\\Omega$', 'phase_bb': 'PHASE-BB', 'phase': 'PHASE-ND',
        'ctm': 'CTM', 'ctm_f': 'CTM-F', 'phase_na': 'PHASE-NA', 'phase_orth': 'PHASE-S', 'cusum': 'CUSUM', 'sr': 'Shiryaev--Roberts',
        'ewma': 'EWMA', 'chart_cal': 'Calibrated chart (oracle)',
        'msd': 'Mahalanobis novelty index', 'pca': 'PCA-EOV chart',
        'chart3': '$3\\sigma$ chart', 'hotelling': 'Repeated $T^2$ test'}


def f3(x):
    return '%.3f' % x


def cell_rate(d):
    return '%s [%s, %s]' % (f3(d['p']), f3(d['lo']), f3(d['hi']))


def fwer_table():
    groups = [('anytime-valid by construction (conditionally)',
               ['phase_d', 'phase_c', 'phase_r', 'phase_e', 'phase_omni', 'phase_bb', 'phase',
                'ctm', 'ctm_f']),
              ('sequential detection, horizon-calibrated oracles', ['cusum', 'sr', 'ewma']),
              ('current SHM practice', ['chart_cal', 'msd', 'pca', 'chart3', 'hotelling'])]
    L = []
    for gname, ms in groups:
        L.append('\\multicolumn{3}{l}{\\textit{%s}}\\\\' % gname)
        for m in ms:
            a, b = S['fwer_2y'][m], S['fwer_4y'][m]
            L.append('%s & %s & %s \\\\' % (NAME[m], cell_rate(a), cell_rate(b)))
        L.append('\\midrule')
    return '\n'.join(L[:-1])


def delay_detail():
    L = []
    for sc, lab in [('storey3_2', '2\\% storey 3'), ('storey3_5', '5\\% storey 3'),
                    ('storey3_10', '10\\% storey 3'), ('gradual_6', 'Gradual 6\\% storey 3'),
                    ('storey1_5', '5\\% storey 1'), ('storey6_5', '5\\% storey 6')]:
        r = S['scenarios'][sc]; p = r['phase_d']
        loc = '%d/%d' % (round(r['loc_det_alarm'] * r['n_loc_det']), r['n_loc_det'])
        L.append('%s & %d/%d & %s [%s, %s] & %s (%s--%s) & %d & %d & %s \\\\' % (
            lab, p['n_det'], p['n'], f3(p['det_rate'])[:4], f3(p['det_lo'])[:4], f3(p['det_hi'])[:4],
            '%.0f' % (p['med'] / 4), '%.0f' % (p['q25'] / 4), '%.0f' % (p['q75'] / 4),
            p['n_pre'], p['n_undet'], loc))
    r = S['scenarios']['stiffening']; p = r['phase_d']
    L.append('Benign 3\\%% stiffening & %d/%d & %s [%s, %s] & --- & %d & %d & --- \\\\' % (
        p['n_det'], p['n'], f3(p['det_rate'])[:4], f3(p['det_lo'])[:4], f3(p['det_hi'])[:4],
        p['n_pre'], p['n_undet']))
    return '\n'.join(L)


def delay_compare():
    ms = ['phase_d', 'phase_r', 'phase_c', 'phase_omni', 'phase_bb', 'ctm', 'ctm_f', 'cusum']
    L = []
    for sc, lab in [('storey3_2', '2\\% storey 3'), ('storey3_5', '5\\% storey 3'),
                    ('storey3_10', '10\\% storey 3'), ('gradual_6', 'Gradual 6\\%'),
                    ('storey1_5', '5\\% storey 1'), ('storey6_5', '5\\% storey 6'),
                    ('stiffening', 'Benign stiffening')]:
        cells = []
        for m in ms:
            p = S['scenarios'][sc][m]
            if p['med'] is None:
                cells.append('%.2f / ---' % p['det_rate'])
            else:
                cells.append('%.2f / %.0f' % (p['det_rate'], p['med'] / 4))
        L.append('%s & %s \\\\' % (lab, ' & '.join(cells)))
    return '\n'.join(L)


def burn_table():
    L = []
    for tb in sorted(S['commissioning'], key=int):
        r = S['commissioning'][tb]
        def c(m, key):
            d = r[m]
            if key == 'fwer':
                return '%.2f' % d['fwer']
            return '%.2f' % d['det']
        L.append('%d & %.2f & %.3f & %s & %s & %s & %s & %s & %s & %s \\\\' % (
            round(r['days']), r['coverage'], r['delta'][0], c('phase_d', 'fwer'),
            c('phase_d', 'det'), c('phase_c', 'fwer'), c('phase_c', 'det'),
            c('phase_bb', 'fwer'), c('phase_bb', 'det'), c('phase_e', 'fwer')))
    return '\n'.join(L)


def late_table():
    L = []
    for sc, lab in [('storey3_5', '5\\% storey 3'), ('storey3_2', '2\\% storey 3'),
                    ('storey3_10', '10\\% storey 3'), ('stiffening', 'Benign stiffening')]:
        r = S['late'][sc]
        for m in ('phase_d', 'phase_r', 'phase_c', 'ctm_f', 'cusum'):
            p = r[m]
            med = '---' if p['med'] is None else '%.0f (%.0f--%.0f)' % (p['med'] / 4, p['q25'] / 4, p['q75'] / 4)
            L.append('%s & %s & %d/%d [%s, %s] & %s & %d \\\\' % (
                lab if m == 'phase_d' else '', NAME[m], p['n_det'], p['n'],
                f3(p['det_lo'])[:4], f3(p['det_hi'])[:4], med, p['n_pre']))
    return '\n'.join(L)


def nuisance_table():
    I = json.load(open('identifiability.json'))
    L = []
    for nm, v in I['nuisance_cases'].items():
        if 'roof' in nm:
            continue
        nm = nm.replace('+ support (storey-1) condition', '+ support condition (or roof mass)')
        L.append('%s & %d & %d & %s & %.3f \\\\' % (
            nm, v['dim'], v['modes_left'], ' & '.join('%.2f' % x for x in v['retention']),
            min(v['retention'])))
    return '\n'.join(L)


def ablation_table():
    ms = ['phase_d', 'phase_orth', 'phase_na', 'phase_omni']
    L = []
    for sc, lab in [('storey3_2', '2\\%% storey 3'), ('storey3_5', '5\\%% storey 3'),
                    ('storey3_10', '10\\%% storey 3'), ('gradual_6', 'Gradual 6\\%%'),
                    ('storey1_5', '5\\%% storey 1'), ('storey6_5', '5\\%% storey 6'),
                    ('stiffening', 'Benign stiffening')]:
        cells = []
        for m in ms:
            p = S['scenarios'][sc][m]
            cells.append('%.2f / ---' % p['det_rate'] if p['med'] is None
                         else '%.2f / %.0f' % (p['det_rate'], p['med'] / 4))
        L.append('%s & %s \\\\' % (lab, ' & '.join(cells)))
    r = S['late']['storey3_5']
    cells = []
    for m in ms:
        p = r[m]
        cells.append('%.2f / ---' % p['det_rate'] if p['med'] is None
                     else '%.2f / %.0f' % (p['det_rate'], p['med'] / 4))
    L.append('Late onset, 5\\%% storey 3 & %s \\\\' % ' & '.join(cells))
    L.append('\\midrule')
    cells = []
    for m in ms:
        cells.append('%d/300 ; %d/300' % (S['fwer_2y'][m]['k'], S['fwer_4y'][m]['k']))
    L.append('False alarms, 1-year ; 3-year & %s \\\\' % ' & '.join(cells))
    return '\n'.join(L)


def twin_table():
    ms = ['phase_d', 'phase_bb', 'phase_omni', 'phase_na', 'ctm_f', 'cusum']
    L = []
    for lvl, lab in (('10', '10\\%% stiffness, 5\\%% mass'), ('20', '20\\%% stiffness, 10\\%% mass')):
        h = S['twin'][lvl + '_healthy']; d = S['twin'][lvl + '_storey3_5']
        cells = []
        for m in ms:
            pf = h[m]; pd = d[m]
            med = '---' if pd['med'] is None else '%.0f' % (pd['med'] / 4)
            cells.append('%d / %d / %s' % (pf['k'], pd['k'], med))
        L.append('%s & %s \\\\' % (lab, ' & '.join(cells)))
    return '\n'.join(L)


def z24_table():
    d = json.load(open('z24_sensitivity.json'))
    pool = json.load(open('z24_pool.json'))['runs']
    runs = dict(d['runs']); runs.update(pool)
    L = []
    for mod, mlab in (('avt', 'ambient'), ('fvt', 'forced')):
        L.append('\\multicolumn{5}{l}{\\textit{%s vibration}}\\\\' % mlab)
        for e in ('R1a', 'R2', 'R3', 'R4', 'R5'):
            cells = []
            for o in ('perm', 'poolblock', 'chron', 'rot'):
                r = runs['%s_%s_%s' % (mod, e, o)]
                def c(k):
                    rate, med = r[k]
                    return '%.2f' % rate + ('' if med is None else ' (%d)' % med)
                cells.append('%s / %s' % (c('phase'), c('phase_omni')))
            L.append('%s & %s \\\\' % (e, ' & '.join(cells)))
    return '\n'.join(L)


if __name__ == '__main__':
    out = dict(fwer=fwer_table(), delay_detail=delay_detail(), delay_compare=delay_compare(),
               burn=burn_table(), late=late_table(), nuisance=nuisance_table(),
               z24=z24_table() if os.path.exists('z24_pool.json') else '',
               ablation=ablation_table(), twin=twin_table())
    for k, v in out.items():
        open('tables/%s.tex' % k, 'w').write(v + '\n')
        print('=== ', k); print(v)
