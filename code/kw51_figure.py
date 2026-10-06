"""Draws fig_kw51.png from kw51_results.json and the KW51 tracked-mode file."""
import json
import datetime as dt
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio
import kw51_experiment as K

plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.grid': True, 'grid.alpha': 0.3,
                     'figure.dpi': 300, 'savefig.dpi': 300, 'axes.spines.top': False,
                     'axes.spines.right': False})
R = json.load(open('kw51_results.json'))
t, y, E = K.load('daily')
fig, ax = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True, gridspec_kw={'height_ratios': [1.1, 0.8, 1.3]})
names = ['mode at 1.89 Hz', '2.57 Hz', '2.93 Hz', '4.10 Hz', '6.34 Hz']
base = (t >= dt.datetime(2018, 11, 1)) & (t < dt.datetime(2019, 2, 1))
for j in range(y.shape[1]):
    ax[0].plot(t, 100 * (np.exp(y[:, j] - np.median(y[base, j])) - 1), lw=0.7, label=names[j])
ax[0].set_ylabel('frequency deviation (%)'); ax[0].legend(fontsize=6, ncol=5, loc='lower left')
ax[1].plot(t, E[:, 0], lw=0.7, color='C3'); ax[1].set_ylabel('air temperature ($^\\circ$C)')
for k, (key, lab, c) in enumerate([('hourly_quad', 'hourly epochs, temperature surrogate', 'C0'),
                                   ('daily_env', 'daily epochs, temperature, humidity, wind', 'C2')]):
    p = R['paths'][key]['path']
    tt = [dt.datetime.fromisoformat(a[0]) for a in p]
    ax[2].plot(tt, [a[1] for a in p], lw=0.9, color=c, label='PHASE, ' + lab)
    ax[2].plot(tt, [a[2] for a in p], lw=0.7, color=c, ls=':', label='PHASE-$\\Omega$, ' + lab)
ax[2].axhline(np.log(100), color='k', lw=0.8)
ax[2].set_ylabel('$\\log W_t$'); ax[2].set_ylim(-12, 25); ax[2].legend(fontsize=6, loc='upper left')
for a in ax:
    a.axvspan(K.WORKS_START, K.WORKS_END, color='C1', alpha=0.12, lw=0)
ax[2].axvspan(dt.datetime(2018, 11, 1), dt.datetime.fromisoformat(R['paths']['daily_env']['comm_end']), color='0.85', alpha=0.5, lw=0)
fig.tight_layout(); fig.savefig('fig_kw51.png'); print('saved fig_kw51.png')
