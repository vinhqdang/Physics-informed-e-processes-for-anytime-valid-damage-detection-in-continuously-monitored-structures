"""Run one stage of the Monte Carlo study and append to partial JSON files."""
import json, os, sys, time
import numpy as np
import model as M
import monitors as MO

T_REC, T_STAR, T_LONG = 2920, 1825, 5840
T_LATE = 4745            # late onset: year 3.25 of a four-year record
R_CAL, R_H, R_D, R_SW, R_LONG = 150, 300, 150, 100, 100
METHODS = ['phase_d', 'phase_e', 'phase', 'phase_omni', 'phase_bb',
           'chart3', 'chart_cal', 'cusum', 'hotelling',
           'pca', 'msd', 'ewma', 'sr', 'ctm', 'phase_c', 'ctm_f']
EXTRA = ['pca', 'msd', 'ewma', 'sr']
SCEN = ['storey3_2', 'storey3_5', 'storey3_10', 'gradual_6',
        'storey1_5', 'storey6_5', 'stiffening']


def save(name, obj):
    with open('part_%s.json' % name, 'w') as f:
        json.dump(obj, f)


def load(name):
    with open('part_%s.json' % name) as f:
        return json.load(f)


def sweep(scenario, reps, seed0, T=T_REC, t_burn=MO.T_BURN, h=None, hx=None,
          t_star=T_STAR):
    rows = {m: [] for m in METHODS}
    loc, dlt, cov, smax = [], [], [], []
    diag = []
    for r in range(reps):
        rng = np.random.default_rng(seed0 + r)
        rec = M.simulate(rng, T, scenario, t_star=t_star)
        o = MO.run_record(rec, t_burn=t_burn, h_extra=hx, t_onset=t_star,
                          h_chart=None if h is None else h[0],
                          h_cusum=None if h is None else h[1])
        diag.append({k: o[k] for k in ('delta_c', 'loc_at_alarm', 'x_burn_mean',
                                       'x_mon_mean', 'excess_blockmax', 'acf1',
                                       'onset')})
        for m in METHODS:
            rows[m].append(o['alarms'][m])
        loc.append(o['loc']); cov.append(o['coverage'])
        dlt.append([o['delta'], o['delta_o'], o['delta_b']])
        smax.append([o['stat_max'][k] for k in
                     ['chart', 'cusum'] + EXTRA])
    return dict(alarms={m: rows[m] for m in METHODS}, loc=loc, delta=dlt,
                coverage=cov, stat_max=smax, reps=reps, T=T, t_burn=t_burn,
                scenario=scenario, t_star=t_star, diag=diag)


def main(stage):
    t0 = time.time()
    if stage == 'calib':
        d = sweep('healthy', R_CAL, 100000)
        sm = np.array(d['stat_max'])
        d['h_chart'] = float(np.quantile(sm[:, 0], 1 - MO.ALPHA))
        d['h_cusum'] = float(np.quantile(sm[:, 1], 1 - MO.ALPHA))
        d['h_extra'] = {k: float(np.quantile(sm[:, 2 + i], 1 - MO.ALPHA))
                        for i, k in enumerate(EXTRA)}
        save('calib', d)
        print('h_chart=%.3f h_cusum=%.3f delta=%s'
              % (d['h_chart'], d['h_cusum'], np.round(np.array(d['delta']).mean(0), 4)))
    elif stage == 'geometry':
        Sn = np.linalg.norm(-M.REF['S'] / MO.SIGMA_D[:, None], axis=0)
        # first-order vs exact eigen-shift validation
        rows = []
        for j in [0, 2, 5]:
            for lvl in [0.02, 0.05, 0.10]:
                th = np.ones((2, M.N_STOREY)); th[1, j] = 1 - lvl
                fr = M.eig_freqs(th, np.tile(M.M0, (2, 1)))[:, :M.N_MODES]
                exact = np.log(fr[1] / fr[0])
                first = -0.5 * M.REF['S'][:, j] * lvl
                rows.append(dict(storey=j + 1, level=lvl,
                                 exact=exact.tolist(), first_order=first.tolist(),
                                 max_abs_err=float(np.abs(exact - first).max()),
                                 rel_err=float(np.abs(exact - first).max()
                                               / np.abs(exact).max())))
        save('geometry', dict(f0=M.REF['f0'].tolist(), S=M.REF['S'].tolist(),
                              kappa=M.REF['kappa'].tolist(), eta=M.REF['eta'].tolist(),
                              signature_norm=Sn.tolist(),
                              retention=MO.DET_RETENTION.tolist(),
                              identifiable_norm=(MO.DET_RETENTION * Sn).tolist(),
                              validation=rows,
                              mu0=MO.MU0, mu0_se=MO.MU0_SE, mu0_omni=MO.MU0_OMNI,
                              n_gen=int(MO.V.shape[0]), lams=MO.LAMS.tolist(),
                              sig_f=M.SIG_F.tolist(), alpha=MO.ALPHA,
                              c_scale=MO.C_SCALE, c_omni=MO.C_OMNI,
                              t_burn=MO.T_BURN, epoch_hours=M.EPOCH_H,
                              n_storey=M.N_STOREY, n_modes=M.N_MODES,
                              beta_T=M.BETA_T, nu_t=M.NU_T, forget=MO.FORGET))
        print('geometry saved; max first-order rel. error %.4f'
              % max(r['rel_err'] for r in rows))
    elif stage == 'healthy':
        c = load('calib')
        save('healthy', sweep('healthy', R_H, 200000, h=(c['h_chart'], c['h_cusum']), hx=c['h_extra']))
        d = load('healthy')
        print({m: float(np.mean(np.array(v) >= 0)) for m, v in d['alarms'].items()})
    elif stage.startswith('long'):
        c = load('calib')
        blk = int(stage[4:] or 0)
        save('long%d' % blk if blk else 'long',
             sweep('healthy', R_LONG, 400000 + 7000 * blk, T=T_LONG,
                   h=(c['h_chart'], c['h_cusum']), hx=c['h_extra']))
        d = load('long%d' % blk if blk else 'long')
        print({m: float(np.mean(np.array(v) >= 0)) for m, v in d['alarms'].items()})
    elif stage.startswith('scen'):
        k = int(stage[4:])
        c = load('calib')
        sc = SCEN[k]
        d = sweep(sc, R_D, 300000 + 1000 * k, h=(c['h_chart'], c['h_cusum']), hx=c['h_extra'])
        save('scen%d' % k, d)
        a = np.array(d['alarms']['phase_d'])
        det = a >= T_STAR
        print('%s det=%.3f delay_med=%s' % (sc, det.mean(),
              np.median(a[det] - T_STAR) if det.any() else None))
    elif stage.startswith('burn'):
        tb = int(stage[4:])
        c = load('calib')
        h = (c['h_chart'], c['h_cusum']); hx = c['h_extra']
        d = dict(healthy=sweep('healthy', R_SW, 500000 + tb, t_burn=tb, h=h, hx=hx),
                 damage=sweep('storey3_5', R_SW, 600000 + tb, t_burn=tb, h=h, hx=hx))
        save('burn%d' % tb, d)
        ah = np.array(d['healthy']['alarms']['phase_d'])
        ab = np.array(d['healthy']['alarms']['phase_bb'])
        print('tb=%d FA phase_d=%.3f phase_bb=%.3f cov=%.2f'
              % (tb, (ah >= 0).mean(), (ab >= 0).mean(),
                 np.mean(d['healthy']['coverage'])))
    elif stage == 'late':
        c = load('calib')
        out = {}
        for k, sc in enumerate(['storey3_5', 'storey3_2', 'storey3_10', 'stiffening']):
            out[sc] = sweep(sc, R_LONG, 700000 + 1000 * k, T=T_LONG, t_star=T_LATE,
                            h=(c['h_chart'], c['h_cusum']), hx=c['h_extra'])
            a = np.array(out[sc]['alarms']['phase_d'])
            det = a >= T_LATE
            print('%s late onset det=%.3f delay_med=%s pre=%.3f' % (
                sc, det.mean(), np.median(a[det] - T_LATE) if det.any() else None,
                ((a >= 0) & (a < T_LATE)).mean()))
        save('late', out)
    elif stage == 'rate3s':
        # spurious-trigger rate of the 3-sigma chart under an inspect-and-resume rule
        out = dict(lockouts=[1, 4, 28], reps=60, rows=[])
        for r in range(out['reps']):
            rng = np.random.default_rng(800000 + r)
            rec = M.simulate(rng, T_REC, 'healthy', t_star=T_STAR)
            o = MO.run_record(rec, collect_paths=True)
            ch = np.abs(np.array(o['paths']['r'])).max(axis=1) > 3.0
            row = dict(exceed=float(ch.mean()))
            for lk in out['lockouts']:
                n, t = 0, 0
                while t < len(ch):
                    if ch[t]:
                        n += 1; t += lk
                    else:
                        t += 1
                row['n_lock%d' % lk] = n
            row['years'] = len(ch) / (4 * 365)
            out['rows'].append(row)
        save('rate3s', out)
        print({k: float(np.mean([r[k] for r in out['rows']]))
               for k in out['rows'][0]})
    elif stage == 'paths':
        out = {}
        for sc in ['healthy', 'storey3_5', 'stiffening']:
            rng = np.random.default_rng(777)
            rec = M.simulate(rng, T_REC, sc, t_star=T_STAR)
            o = MO.run_record(rec, collect_paths=True)
            out[sc] = dict(y=rec['y'].tolist(), dT=rec['dT'].tolist(),
                           q=rec['q'].tolist(),
                           lwd=o['paths']['lwd'].tolist(),
                           lw=o['paths']['lw'].tolist(),
                           lwo=o['paths']['lwo'].tolist(),
                           lwb=o['paths']['lwb'].tolist(),
                           x=o['paths']['x'].tolist(),
                           cus=o['paths']['cus'].tolist(),
                           r=o['paths']['r'].tolist(),
                           alarms=o['alarms'], delta=o['delta'])
        save('paths', out)
        print('paths saved')
    else:
        raise SystemExit('unknown stage ' + stage)
    print('stage %s done in %.0fs' % (stage, time.time() - t0))


if __name__ == '__main__':
    main(sys.argv[1])
