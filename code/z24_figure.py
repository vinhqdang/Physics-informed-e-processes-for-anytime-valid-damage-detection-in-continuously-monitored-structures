import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import z24_modal as Z
import z24_experiment as E

plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'axes.grid': True,
                     'grid.alpha': 0.3, 'figure.dpi': 300, 'savefig.dpi': 300,
                     'axes.spines.top': False, 'axes.spines.right': False})

d = Z.build()
lab = ['ref 1', 'ref 2', '20 mm', '40 mm', '80 mm', '95 mm', 'tilt',
       'ref 3', 'spall 12', 'spall 24']

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9),
                       gridspec_kw={'width_ratios': [1.15, 1]})

# --- tracked frequencies by structural state, normalised to reference 2
base = d[2].mean(0)
for j, nm in enumerate(['mode 1 ($\\approx$3.87 Hz)', 'mode 2 ($\\approx$9.82 Hz)',
                        'mode 3 ($\\approx$12.7 Hz)']):
    m = [100 * (d[s][:, j].mean() / base[j] - 1) for s in range(1, 11)]
    e = [100 * d[s][:, j].std() / base[j] / np.sqrt(len(d[s])) for s in range(1, 11)]
    ax[0].errorbar(np.arange(1, 11), m, yerr=e, marker='o', ms=3.5, lw=1, capsize=2,
                   label=nm)
ax[0].axhline(0, color='k', lw=.7)
ax[0].axvspan(2.5, 6.5, color='C3', alpha=.08, lw=0)
ax[0].axvspan(7.5, 8.5, color='C2', alpha=.10, lw=0)
ax[0].set_xticks(np.arange(1, 11)); ax[0].set_xticklabels(lab, rotation=45, ha='right')
ax[0].set_ylabel('frequency shift vs reference 2 (%)')
ax[0].legend(fontsize=6.5, loc='lower left')

# --- evidence paths
rng = np.random.default_rng(5)
burn = np.log(np.concatenate([d[1], d[2]]))
burn = burn[rng.permutation(len(burn))]
stream = np.concatenate([np.log(d[s])[rng.permutation(len(d[s]))]
                         for s in [3, 4, 5, 6]])
_, p2, _ = E.monitor(burn, stream)
burn3 = np.log(d[6])[rng.permutation(len(d[6]))]
stream3 = np.log(d[8])[rng.permutation(len(d[8]))]
_, p3, _ = E.monitor(burn3, stream3)

ax[1].plot(p2[:, 0], '-', lw=1.1, color='C0', label='PHASE, settlement')
ax[1].plot(p2[:, 1], '--', lw=1.1, color='C1', label='PHASE-$\\Omega$, settlement')
ax[1].plot(p3[:, 0], '-', lw=1.1, color='C2', label='PHASE, settlement reversed')
ax[1].plot(p3[:, 1], ':', lw=1.3, color='C3', label='PHASE-$\\Omega$, reversed')
ax[1].axhline(np.log(100), color='k', lw=.8)
for b in [71, 143, 215]:
    ax[1].axvline(b, color='0.7', lw=.6, ls='-.')
for x, t in [(30, '20 mm'), (100, '40 mm'), (172, '80 mm'), (245, '95 mm')]:
    ax[1].text(x, -10.6, t, fontsize=6, color='0.35')
ax[1].set_xlabel('monitoring epoch'); ax[1].set_ylabel('$\\log W_t$')
ax[1].set_ylim(-12, 11); ax[1].legend(fontsize=6, loc='upper left')

fig.tight_layout()
fig.savefig('fig_z24.png')
print('saved fig_z24.png')
