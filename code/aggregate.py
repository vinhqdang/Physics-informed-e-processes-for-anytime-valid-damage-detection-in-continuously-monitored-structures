import json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'axes.grid': True,
                     'grid.alpha': 0.3, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'axes.spines.top': False, 'axes.spines.right': False})

T_STAR, T_BURN, T_REC, T_LONG = 1825, 1460, 2920, 5840
EP_DAY = 4.0
SCEN = ['storey3_2', 'storey3_5', 'storey3_10', 'gradual_6',
        'storey1_5', 'storey6_5', 'stiffening']
LBL = {'phase_d': 'PHASE', 'phase_e': 'PHASE-E', 'phase': 'PHASE (no discount)',
       'phase_omni': 'PHASE-$\\Omega$ (no cone)', 'phase_bb': 'PHASE-BB (black box)',
       'chart3': '3$\\sigma$ chart', 'chart_cal': 'calibrated chart',
       'cusum': 'CUSUM (oracle)', 'hotelling': 'repeated $T^2$',
       'pca': 'PCA-EOV chart', 'msd': 'Mahalanobis index', 'ewma': 'EWMA (oracle)',
       'sr': 'Shiryaev--Roberts', 'ctm': 'conformal martingale'}
ORDER = ['phase_d', 'phase', 'phase_omni', 'phase_bb', 'phase_e', 'ctm',
         'cusum', 'sr', 'ewma', 'chart_cal', 'msd', 'pca', 'chart3', 'hotelling']


def L(n):
    return json.load(open('part_%s.json' % n))


geo, cal, hea = L('geometry'), L('calib'), L('healthy')
lng = [L('long')] + [L('long%d' % i) for i in (1, 2)]
scen = {s: L('scen%d' % k) for k, s in enumerate(SCEN)}
burn = {tb: L('burn%d' % tb) for tb in (120, 240, 480, 1460)}
res = {}


def se(p, n):
    return float(np.sqrt(max(p * (1 - p), 1e-12) / n))


# ------------------------------------------------------------------ Table: geometry
res['geometry'] = dict(
    f0=geo['f0'], retention=geo['retention'], signature=geo['signature_norm'],
    identifiable=geo['identifiable_norm'],
    max_rel_err=max(r['rel_err'] for r in geo['validation']),
    validation=geo['validation'], mu0=geo['mu0'], mu0_omni=geo['mu0_omni'],
    n_gen=geo['n_gen'])

# ------------------------------------------------------------------- Table: FWER
def fwer(d):
    n = d['reps']
    return {m: (float(np.mean(np.array(v) >= 0)), se(float(np.mean(np.array(v) >= 0)), n))
            for m, v in d['alarms'].items()}


res['fwer_2y'] = fwer(hea)
alL = {m: np.concatenate([np.array(b['alarms'][m]) for b in lng]) for m in ORDER}
nL = sum(b['reps'] for b in lng)
res['fwer_4y'] = {m: (float((a >= 0).mean()), se(float((a >= 0).mean()), nL))
                  for m, a in alL.items()}
res['n_long'] = nL
res['delta_calib'] = np.array(cal['delta']).mean(0).tolist()
res['delta_calib_sd'] = np.array(cal['delta']).std(0).tolist()
res['h_chart'], res['h_cusum'] = cal['h_chart'], cal['h_cusum']

# --------------------------------------------------------- Table: damage scenarios
res['scenarios'] = {}
for s in SCEN:
    d = scen[s]
    n = d['reps']
    row = {}
    for m in ORDER:
        a = np.array(d['alarms'][m])
        pre = float(((a >= 0) & (a < T_STAR)).mean())
        det = a >= T_STAR
        dl = a[det] - T_STAR
        row[m] = dict(pre_onset_fa=pre, det_rate=float(det.mean()),
                      det_se=se(float(det.mean()), n),
                      med=float(np.median(dl)) if det.any() else None,
                      q25=float(np.quantile(dl, .25)) if det.any() else None,
                      q75=float(np.quantile(dl, .75)) if det.any() else None)
    tgt = 0 if s.startswith('storey1') else (5 if s.startswith('storey6') else 2)
    row['loc_acc'] = None if s == 'stiffening' else float(
        (np.array(d['loc']) == tgt).mean())
    row['reps'] = n
    res['scenarios'][s] = row

# ---------------------------------------------------- Table: commissioning windows
res['commissioning'] = {}
for tb, d in burn.items():
    h, g = d['healthy'], d['damage']
    row = dict(days=tb / EP_DAY, coverage=float(np.mean(h['coverage'])),
               delta=np.array(h['delta']).mean(0).tolist(), reps=h['reps'])
    for m in ['phase_d', 'phase_e', 'phase_bb', 'ctm']:
        a = np.array(h['alarms'][m]); b = np.array(g['alarms'][m])
        det = b >= T_STAR
        row[m] = dict(fwer=float((a >= 0).mean()),
                      det=float(det.mean()),
                      med=float(np.median(b[det] - T_STAR)) if det.any() else None)
    res['commissioning'][tb] = row

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
fig, ax = plt.subplots(figsize=(4.4, 3.1))
grid = np.arange(1, T_LONG - T_BURN, 20)
for m, st in [('phase_d', '-'), ('ctm', '-.'), ('chart_cal', '--'),
              ('pca', ':'), ('cusum', '-'), ('chart3', '-'), ('hotelling', '--')]:
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
ax.legend(fontsize=6.5, ncol=2, loc='center right')
fig.tight_layout(); fig.savefig('fig_fwer.png'); plt.close(fig)

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
for m, mk in [('phase_d', 'o-'), ('phase_e', 's--'), ('phase_bb', '^-.'),
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
ax[0].set_xticks(dd); ax[0].set_xticklabels([int(x) for x in dd])
ax[1].set_xticks(dd); ax[1].set_xticklabels([int(x) for x in dd])
fig.tight_layout(); fig.savefig('fig_commissioning.png'); plt.close(fig)

print(json.dumps({k: res[k] for k in ['fwer_2y', 'fwer_4y', 'delta_calib',
                                      'h_chart', 'h_cusum', 'n_long']}, indent=1))
for s in SCEN:
    r = res['scenarios'][s]
    print(s, 'loc=%s' % r['loc_acc'],
          {m: (r[m]['det_rate'], r[m]['med'], r[m]['pre_onset_fa'])
           for m in ['phase_d', 'phase_omni', 'phase_bb', 'cusum', 'chart_cal']})
for t in tbs:
    print('burn', t, res['commissioning'][t])
