import json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'axes.grid': True,
                     'grid.alpha': 0.3, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'axes.spines.top': False, 'axes.spines.right': False})

T_STAR, T_BURN, T_REC, T_LONG = 1825, 1460, 2920, 5840
T_LATE = 4745
EP_DAY = 4.0
SCEN = ['storey3_2', 'storey3_5', 'storey3_10', 'gradual_6',
        'storey1_5', 'storey6_5', 'stiffening']
LBL = {'phase_d': 'PHASE', 'phase_e': 'PHASE-E', 'phase': 'PHASE-ND',
       'phase_omni': 'PHASE-$\\Omega$', 'phase_bb': 'PHASE-BB', 'phase_c': 'PHASE-C',
       'phase_r': 'PHASE-R', 'phase_na': 'PHASE-NA', 'phase_orth': 'PHASE-S',
       'chart3': '$3\\sigma$ chart', 'chart_cal': 'calibrated chart',
       'cusum': 'CUSUM', 'hotelling': 'repeated $T^2$',
       'pca': 'PCA-EOV chart', 'msd': 'Mahalanobis index', 'ewma': 'EWMA',
       'sr': 'Shiryaev--Roberts', 'ctm': 'CTM', 'ctm_f': 'CTM-F'}
ORDER = ['phase_d', 'phase_r', 'phase_c', 'phase_na', 'phase_orth', 'phase', 'phase_omni', 'phase_bb', 'phase_e', 'ctm', 'ctm_f',
         'cusum', 'sr', 'ewma', 'chart_cal', 'msd', 'pca', 'chart3', 'hotelling']


def L(n):
    return json.load(open('part_%s.json' % n))


geo, cal, hea = L('geometry'), L('calib'), L('healthy')
lng = [L('long')] + [L('long%d' % i) for i in (1, 2)]
scen = {s: L('scen%d' % k) for k, s in enumerate(SCEN)}
TBS = (120, 240, 480, 720, 960, 1200, 1460)
burn = {tb: L('burn%d' % tb) for tb in TBS}
late = L('late')
twin = L('twin')
r3s = L('rate3s')
from scipy.stats import beta as _beta
res = {}


def se(p, n):
    return float(np.sqrt(max(p * (1 - p), 1e-12) / n))


def cp(k, n, conf=0.95):
    """Clopper-Pearson two-sided interval for k events in n trials."""
    a = 1 - conf
    lo = 0.0 if k == 0 else float(_beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(_beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


# ------------------------------------------------------------------ geometry
res['geometry'] = dict(
    f0=geo['f0'], retention=geo['retention'], signature=geo['signature_norm'],
    identifiable=geo['identifiable_norm'],
    max_rel_err=max(r['rel_err'] for r in geo['validation']),
    validation=geo['validation'], mu0=geo['mu0'], mu0_omni=geo['mu0_omni'],
    n_gen=geo['n_gen'], lams=geo['lams'])


# ------------------------------------------------------------------- FWER tables
def fwer(d, methods=None):
    n = d['reps']
    out = {}
    for m, v in d['alarms'].items():
        k = int(np.sum(np.array(v) >= 0))
        lo, hi = cp(k, n)
        out[m] = dict(k=k, n=n, p=k / n, se=se(k / n, n), lo=lo, hi=hi)
    return out


res['fwer_2y'] = fwer(hea)
alL = {m: np.concatenate([np.array(b['alarms'][m]) for b in lng]) for m in ORDER}
nL = sum(b['reps'] for b in lng)
res['fwer_4y'] = {}
for m, a in alL.items():
    k = int((a >= 0).sum()); lo, hi = cp(k, nL)
    res['fwer_4y'][m] = dict(k=k, n=nL, p=k / nL, se=se(k / nL, nL), lo=lo, hi=hi)
res['n_long'] = nL
# pooled healthy records for the PHASE family: 600 records
res['fwer_pooled'] = {}
for m in ('phase_d', 'phase_c', 'phase_bb', 'phase_omni', 'phase_e'):
    k = int((np.array(hea['alarms'][m]) >= 0).sum() + (alL[m] >= 0).sum())
    lo, hi = cp(k, hea['reps'] + nL)
    res['fwer_pooled'][m] = dict(k=k, n=hea['reps'] + nL, lo=lo, hi=hi)
res['delta_calib'] = np.array(cal['delta']).mean(0).tolist()
res['delta_calib_sd'] = np.array(cal['delta']).std(0).tolist()
res['delta_c_mean'] = float(np.mean([d['delta_c'] for d in hea['diag']]))
res['h_chart'], res['h_cusum'] = cal['h_chart'], cal['h_cusum']
res['h_extra'] = cal['h_extra']

# assumption diagnostics on healthy records (two- and four-year)
dg = hea['diag'] + [x for b in lng for x in b['diag']]
dl = [d[0] for d in hea['delta']] + [d[0] for b in lng for d in b['delta']]
res['assumption'] = dict(
    n=len(dg),
    mon_mean_excess_med=float(np.median([d['x_mon_mean'] - geo['mu0'] for d in dg])),
    burn_mean_excess_med=float(np.median([d['x_burn_mean'] - geo['mu0'] for d in dg])),
    delta_med=float(np.median(dl)),
    frac_mean_excess_above_delta=float(np.mean(
        [(d['x_mon_mean'] - geo['mu0']) > dd for d, dd in zip(dg, dl)])),
    frac_blockmax_above_delta=float(np.mean(
        [(d['excess_blockmax'] is not None) and d['excess_blockmax'] > dd
         for d, dd in zip(dg, dl)])),
    frac_blocks_above_delta=float(np.mean(
        [b > dd for d, dd in zip(dg, dl) for b in d['excess_blocks']])),
    n_blocks=int(sum(len(d['excess_blocks']) for d in dg)),
    acf1_med=float(np.median([d['acf1'] for d in dg])),
    acf1_q=[float(np.quantile([d['acf1'] for d in dg], q)) for q in (.1, .9)])

# --------------------------------------------------------- damage scenario table
def scen_row(d, t_star, n_after):
    n = d['reps']
    row = {}
    for m in ORDER:
        a = np.array(d['alarms'][m])
        pre = (a >= 0) & (a < t_star)
        det = a >= t_star
        dl_ = a[det] - t_star
        lo, hi = cp(int(det.sum()), n)
        row[m] = dict(n=n, n_pre=int(pre.sum()), n_det=int(det.sum()),
                      n_undet=int(((a < 0)).sum()),
                      det_rate=float(det.mean()), det_lo=lo, det_hi=hi,
                      pre_onset_fa=float(pre.mean()),
                      med=float(np.median(dl_)) if det.any() else None,
                      q25=float(np.quantile(dl_, .25)) if det.any() else None,
                      q75=float(np.quantile(dl_, .75)) if det.any() else None)
    return row


res['scenarios'] = {}
for s in SCEN:
    d = scen[s]
    row = scen_row(d, T_STAR, T_REC - T_STAR)
    tgt = 0 if s.startswith('storey1') else (5 if s.startswith('storey6') else 2)
    loc = np.array(d['loc']); a = np.array(d['alarms']['phase_d'])
    det = a >= T_STAR
    laa = np.array([x['loc_at_alarm'] for x in d['diag']])
    if s != 'stiffening':
        row['loc_all_end'] = float((loc == tgt).mean())            # all records, end of record
        row['loc_det_end'] = float((loc[det] == tgt).mean()) if det.any() else None
        row['loc_det_alarm'] = float((laa[det] == tgt).mean()) if det.any() else None
        row['loc_det_alarm_counts'] = np.bincount(laa[det], minlength=8).tolist() if det.any() else None
        row['loc_all_counts'] = np.bincount(loc, minlength=8).tolist()
        row['n_loc_det'] = int(det.sum())
    row['reps'] = d['reps']
    res['scenarios'][s] = row

# ---------------------------------------------------- delay-bound check (plug-in)
lam = np.array(geo['lams']); mu0 = geo['mu0']
b_const = np.log(1 / 0.01) + np.log(len(lam))
res['delay_bound'] = {}
for s in ('storey3_5', 'storey3_10', 'storey6_5', 'storey1_5'):
    d = scen[s]
    a = np.array(d['alarms']['phase_d'])
    ratios, holds, bnd, obs = [], [], [], []
    for i, dg_ in enumerate(d['diag']):
        if a[i] < T_STAR or dg_['onset'] is None:
            continue
        delta = d['delta'][i][0]
        g = np.array(dg_['onset']['g']) - np.log1p(lam * delta)
        Lm = np.maximum(-np.array(dg_['onset']['lw']), 0.0)
        Bc = np.log1p(lam * (1 - mu0)) - np.log1p(lam * delta)
        with np.errstate(divide='ignore', invalid='ignore'):
            bd = np.where(g > 0, (b_const + Bc + Lm) / g, np.inf)
        k = bd.min()
        if np.isfinite(k):
            bnd.append(k); obs.append(a[i] - T_STAR)
    bnd, obs = np.array(bnd), np.array(obs)
    res['delay_bound'][s] = dict(n=int(len(bnd)), bound_med=float(np.median(bnd)),
                                 obs_med=float(np.median(obs)),
                                 frac_holds=float(np.mean(bnd >= obs)),
                                 ratio_med=float(np.median(bnd / obs)))

# ---------------------------------------------------------- late-onset experiment
res['late'] = {}
for s, d in late.items():
    row = scen_row(d, T_LATE, T_LONG - T_LATE)
    row['onset_lw_mix_med'] = float(np.median([x['onset']['mix'] for x in d['diag']]))
    row['onset_lw_comp_med'] = np.median([x['onset']['lw'] for x in d['diag']], axis=0).tolist()
    row['reps'] = d['reps']
    res['late'][s] = row
res['T_LATE'] = T_LATE

# ---------------------------------------------------- commissioning windows
res['commissioning'] = {}
for tb, d in burn.items():
    h, g = d['healthy'], d['damage']
    row = dict(days=tb / EP_DAY, coverage=float(np.mean(h['coverage'])),
               delta=np.array(h['delta']).mean(0).tolist(),
               delta_c=float(np.mean([x['delta_c'] for x in h['diag']])), reps=h['reps'])
    for m in ['phase_d', 'phase_c', 'phase_e', 'phase_bb', 'ctm']:
        a = np.array(h['alarms'][m]); b = np.array(g['alarms'][m])
        det = b >= T_STAR
        kf = int((a >= 0).sum()); kd = int(det.sum())
        row[m] = dict(k=kf, fwer=kf / len(a), fwer_ci=cp(kf, len(a)),
                      kd=kd, det=float(det.mean()), det_ci=cp(kd, len(b)),
                      med=float(np.median(b[det] - T_STAR)) if det.any() else None)
    res['commissioning'][tb] = row

# ---------------------------------------------------- twin-error experiment
res['twin'] = {}
for key, d in twin.items():
    n = d['reps']; row = {}
    healthy = key.endswith('healthy')
    for m in ORDER:
        a = np.array(d['alarms'][m])
        if healthy:
            k = int((a >= 0).sum()); row[m] = dict(k=k, n=n, ci=cp(k, n))
        else:
            kd = int((a >= T_STAR).sum()); kp = int(((a >= 0) & (a < T_STAR)).sum())
            dl_ = a[a >= T_STAR] - T_STAR
            row[m] = dict(k=kd, n=n, ci=cp(kd, n), n_pre=kp,
                          med=float(np.median(dl_)) if len(dl_) else None)
    row['delta_mean'] = float(np.mean([x for x in d['delta']]))
    res['twin'][key] = row

# ---------------------------------------------------- 3-sigma trigger rates
rows = r3s['rows']
res['rate3s'] = dict(
    reps=r3s['reps'], exceed_frac=float(np.mean([r['exceed'] for r in rows])),
    per_year={lk: float(np.mean([r['n_lock%d' % lk] / r['years'] for r in rows]))
              for lk in r3s['lockouts']})

# ---------------------------------------------------- simulation accounting
res['accounting'] = dict(
    calibration=cal['reps'], healthy_2y=hea['reps'], healthy_4y=nL,
    damage=sum(scen[s]['reps'] for s in SCEN), sweep=sum(2 * burn[t]['healthy']['reps'] for t in (120, 240, 480, 1460)),
    sweep_extra=sum(2 * burn[t]['healthy']['reps'] for t in (720, 960, 1200)),
    late=sum(late[s]['reps'] for s in late), illustrative=3)

json.dump(res, open('summary.json', 'w'), indent=1)

# =============================================================== FIGURES
days = lambda e: e / EP_DAY

# --- Fig 2: one healthy and one damaged record
p = L('paths')
fig, ax = plt.subplots(3, 2, figsize=(7.2, 6.0), sharex=True)
for c, sc, ttl in [(0, 'healthy', 'Undamaged'),
                   (1, 'storey3_5', '5% stiffness loss, storey 3')]:
    d = p[sc]
    y = np.array(d['y']); dT = np.array(d['dT'])
    t = days(np.arange(len(y)))
    ax[0, c].plot(t, 100 * (np.exp(y[:, 0] - y[:, 0].mean()) - 1), lw=.4, color='C0')
    a2 = ax[0, c].twinx(); a2.plot(t, dT + 24, lw=.4, color='C3', alpha=.55)
    a2.set_ylabel('ambient $T$ ($^\\circ$C)', color='C3', fontsize=7)
    a2.tick_params(labelsize=7, colors='C3'); a2.grid(False)
    ax[0, c].set_title(ttl, fontsize=9)
    ax[0, c].set_ylabel('$f_1$ dev. (%)' if c == 0 else '')
    tm = days(np.arange(T_BURN, T_BURN + len(d['x'])))
    xs_ = np.array(d['x']); k = 28
    sm_ = np.convolve(xs_, np.ones(k) / k, mode='valid')
    ax[1, c].plot(tm, xs_, lw=.2, color='0.75')
    ax[1, c].plot(tm[k - 1:], sm_, lw=1.0, color='C2')
    ax[1, c].axhline(geo['mu0'], color='k', lw=.8, ls='--')
    ax[1, c].set_ylabel('cone score $X_t$' if c == 0 else '')
    for k, st in [('lwd', '-'), ('lwo', '--'), ('lwb', ':')]:
        ax[2, c].plot(tm, d[k], st, lw=.9,
                      label={'lwd': 'PHASE', 'lwo': 'PHASE-$\\Omega$',
                             'lwb': 'PHASE-BB'}[k])
    ax[2, c].axhline(np.log(100), color='k', lw=.8)
    ax[2, c].set_ylabel('$\\log W_t$' if c == 0 else '')
    ax[2, c].set_xlabel('day')
    for r in range(3):
        if sc != 'healthy':
            ax[r, c].axvline(days(T_STAR), color='C3', lw=.8, ls='-.')
        ax[r, c].axvspan(0, days(T_BURN), color='0.85', alpha=.5, lw=0)
ax[2, 0].legend(fontsize=7, loc='upper left')
ax[2, 0].set_ylim(-12, 14); ax[2, 1].set_ylim(-12, 14)
fig.tight_layout(); fig.savefig('fig_records.png'); plt.close(fig)

# --- Fig 3: FWER vs monitoring horizon
fig, ax = plt.subplots(figsize=(5.4, 3.1))
grid = np.arange(1, T_LONG - T_BURN, 20)
for m, st in [('phase_d', '-'), ('phase_r', '-'), ('ctm', '-.'), ('ctm_f', ':'),
              ('chart_cal', '--'), ('pca', ':'), ('cusum', '-'), ('chart3', '-'),
              ('hotelling', '--')]:
    a = alL[m]
    cur = np.array([np.mean((a >= 0) & (a - T_BURN <= g)) for g in grid])
    ax.plot(days(grid), np.maximum(cur, 4.5e-4), st, lw=1.2, label=LBL[m])
ax.axhline(0.01, color='k', lw=.8)
ax.text(30, 0.013, r'$\alpha=0.01$', fontsize=7)
ax.set_yscale('log'); ax.set_ylim(3.5e-4, 1.9)
ax.axvline(365, color='0.5', lw=.7, ls=':')
ax.text(372, 3.0e-3, 'calibration horizon', fontsize=6, color='0.4', rotation=90)

ax.set_xlabel('monitoring horizon (days)')
ax.set_ylabel('family-wise false-alarm probability')
ax.legend(fontsize=6.5, ncol=1, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)
fig.tight_layout(); fig.savefig('fig_fwer.png', bbox_inches='tight'); plt.close(fig)

# --- Fig 4: detection delay
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
lv = ['storey3_2', 'storey3_5', 'storey3_10']
xs = [2, 5, 10]
for m, mk in [('phase_d', 'o-'), ('ctm', 's--'), ('phase_bb', '^-.'),
              ('phase_omni', 'v:'), ('cusum', 'd-'), ('sr', 'P-'), ('ewma', 'x--')]:
    med = [res['scenarios'][s][m]['med'] for s in lv]
    ax[0].plot(xs, [np.nan if v is None else days(v) for v in med], mk, ms=4,
               lw=1, label=LBL[m])
ax[0].set_xlabel('stiffness loss at storey 3 (%)')
ax[0].set_ylabel('median detection delay (days)')
ax[0].set_xticks(xs); ax[0].legend(fontsize=5.8, ncol=2)
for m, mk in [('phase_d', 'o-'), ('ctm', 's--'), ('phase_bb', '^-.'),
              ('phase_omni', 'v:'), ('cusum', 'd-'), ('sr', 'P-'), ('ewma', 'x--')]:
    dr = [res['scenarios'][s][m]['det_rate'] for s in lv]
    ax[1].plot(xs, dr, mk, ms=4, lw=1)
ax[1].set_xlabel('stiffness loss at storey 3 (%)')
ax[1].set_ylabel('detection rate within 9 months')
ax[1].set_xticks(xs); ax[1].set_ylim(-0.05, 1.05)
fig.tight_layout(); fig.savefig('fig_delay.png'); plt.close(fig)

# --- Fig 5: detectability map
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
st = np.arange(1, 9)
w = 0.38
ax[0].bar(st - w / 2, geo['signature_norm'], w, label='full signature $\\|d_j\\|$')
ax[0].bar(st + w / 2, geo['identifiable_norm'], w,
          label='identifiable $\\|P_\\perp d_j\\|$')
ax[0].set_xlabel('storey $j$'); ax[0].set_ylabel('norm (sd units per unit loss)')
ax[0].legend(fontsize=7)
ax[1].bar(st, 100 * np.array(geo['retention']), 0.6, color='C2')
ax[1].set_xlabel('storey $j$'); ax[1].set_ylabel('identifiable fraction (%)')
for s, c in [('storey1_5', 'C3'), ('storey3_5', 'C0'), ('storey6_5', 'C1')]:
    j = int(s[6]); ax[1].plot(j, 100 * geo['retention'][j - 1], 'k*', ms=9)
fig.tight_layout(); fig.savefig('fig_detectability.png'); plt.close(fig)

# --- Fig 6: commissioning sweep
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
tbs = sorted(burn)
dd = [res['commissioning'][t]['days'] for t in tbs]
for m, mk in [('phase_d', 'o-'), ('phase_c', 'v-'), ('phase_e', 's--'), ('phase_bb', '^-.'),
              ('ctm', 'd:')]:
    ax[0].plot(dd, [res['commissioning'][t][m]['fwer'] for t in tbs], mk, ms=4, lw=1,
               label=LBL[m])
    ax[1].plot(dd, [res['commissioning'][t][m]['det'] for t in tbs], mk, ms=4, lw=1)
ax[0].axhline(0.01, color='k', lw=.8)
ax[0].set_xscale('log'); ax[1].set_xscale('log')
ax[0].set_xlabel('commissioning window (days)')
ax[0].set_ylabel('false-alarm probability'); ax[0].legend(fontsize=7)
ax[1].set_xlabel('commissioning window (days)')
ax[1].set_ylabel('detection rate, 5% loss')
from matplotlib.ticker import NullFormatter
for a_ in ax:
    a_.set_xticks([30, 60, 120, 240, 365]); a_.set_xticklabels([30, 60, 120, 240, 365])
    a_.xaxis.set_minor_formatter(NullFormatter())
fig.tight_layout(); fig.savefig('fig_commissioning.png'); plt.close(fig)

print(json.dumps({k: res[k] for k in ['fwer_pooled', 'delta_calib', 'delta_c_mean',
                                      'h_chart', 'h_cusum', 'n_long', 'assumption',
                                      'accounting', 'rate3s', 'delay_bound']}, indent=1))
print({m: (v['k'], v['n'], round(v['lo'], 4), round(v['hi'], 4)) for m, v in res['fwer_2y'].items()})
print({m: (v['k'], v['n'], round(v['lo'], 4), round(v['hi'], 4)) for m, v in res['fwer_4y'].items()})
for s in SCEN:
    r = res['scenarios'][s]
    print(s, {k: r.get(k) for k in ('loc_all_end', 'loc_det_end', 'loc_det_alarm', 'loc_det_alarm_counts')},
          {m: (r[m]['det_rate'], r[m]['med'], r[m]['q25'], r[m]['q75'], r[m]['n_pre'], r[m]['n_undet'])
           for m in ['phase_d', 'phase_c', 'phase_omni', 'phase_bb', 'cusum', 'chart_cal']})
for s, r in res['late'].items():
    print('late', s, {m: (r[m]['det_rate'], r[m]['med'], r[m]['q25'], r[m]['q75'], r[m]['n_pre'])
                      for m in ['phase_d', 'phase_c', 'phase_omni', 'phase_bb', 'cusum', 'ctm']},
          r['onset_lw_mix_med'])
for t in tbs:
    print('burn', t, res['commissioning'][t])
