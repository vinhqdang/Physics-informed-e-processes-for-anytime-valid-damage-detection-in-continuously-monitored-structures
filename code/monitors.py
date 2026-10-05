"""
Online monitors for a single simulated SHM record.

PHASE = Physics-informed Hedged Anytime-valid Sequential Evidence
  (i)   physics-structured EOV surrogate with sequential (predictable) plug-in
  (ii)  polyhedral inner approximation of the physics-derived damage cone
  (iii) mixture ("hedged") betting e-process, alarm by Ville's inequality at 1/alpha
  (iv)  identifiability-aware adaptation: residual components lying in the damage
        cone *and* outside the nuisance sensitivity subspace are withheld from the
        nuisance model, so drift is tracked but identifiable damage is not absorbed
  (v)   surrogate-bias discount delta (Proposition 2)

Ablations (PHASE-Omega: no cone; PHASE-BB: black-box surrogate) and conventional
baselines (fixed 3-sigma chart, horizon-calibrated chart, CUSUM, repeated
Hotelling test) share the same residual stream wherever possible.
"""
import numpy as np
from scipy.stats import chi2
import model as M

ALPHA = 0.01
T_BURN = 1460                    # commissioning window: one full seasonal cycle
C_SCALE = 3.0                    # saturation scale of the cone score
C_OMNI = 8.0                     # saturation scale of the omnibus score
HUBER = 3.0                      # Huber clip on adaptation residuals
FORGET = 0.9995
N_LAM = 12


# ------------------------------------------------------------- geometry (design time)
def cone_generators(S, sigma, pair_weights=(0.25, 0.5, 0.75)):
    """Unit generators of a polyhedral inner approximation of the standardised
    damage cone C = { -diag(1/sigma) S rho : rho >= 0 }, rho = stiffness loss."""
    A = -(S / sigma[:, None])
    gens, support = [], []
    n = A.shape[1]
    for j in range(n):
        gens.append(A[:, j]); support.append({j: 1.0})
    for j in range(n):
        for k in range(j + 1, n):
            for w in pair_weights:
                gens.append(w * A[:, j] + (1 - w) * A[:, k])
                support.append({j: w, k: 1 - w})
    Vm = np.array(gens)
    return Vm / np.linalg.norm(Vm, axis=1, keepdims=True), support


def nuisance_projector(sigma):
    """Projector onto the complement of the nuisance sensitivity subspace
    N = span{kappa/sigma, eta/sigma}. Temperature reaches the modal set only
    through the uniform-modulus direction kappa and live load only through eta,
    so dim N = 2 whatever the (possibly nonlinear) EOV law."""
    U = np.stack([M.REF['kappa'] / sigma, M.REF['eta'] / sigma], axis=1)
    Q, _ = np.linalg.qr(U)
    return np.eye(len(sigma)) - Q @ Q.T


def score_null_mean(Vm, c, n_mc=400_000, seed=7):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n_mc, Vm.shape[1]))
    x = np.minimum(np.clip((e @ Vm.T).max(axis=1), 0, None) / c, 1.0)
    return float(x.mean()), float(x.std() / np.sqrt(n_mc))


def omni_null_mean(m, c, n_mc=400_000, seed=8):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n_mc, m))
    return float(np.minimum(np.linalg.norm(e, axis=1) / c, 1.0).mean())


SIGMA_D = M.SIG_F.copy()
V, SUPPORT = cone_generators(M.REF['S'], SIGMA_D)
P_PERP = nuisance_projector(SIGMA_D)
_VT = (P_PERP @ V.T).T
_n = np.linalg.norm(_VT, axis=1)
VT = _VT[_n > 1e-8] / _n[_n > 1e-8, None]        # identifiable cone directions
DET_RETENTION = (np.linalg.norm(P_PERP @ (-M.REF['S'] / SIGMA_D[:, None]), axis=0)
                 / np.linalg.norm(-M.REF['S'] / SIGMA_D[:, None], axis=0))
MU0, MU0_SE = score_null_mean(V, C_SCALE)
MU0_OMNI = omni_null_mean(M.N_MODES, C_OMNI)
LAMS = np.geomspace(0.05, 0.9 / MU0, N_LAM)
LAMS_OMNI = np.geomspace(0.05, 0.9 / MU0_OMNI, N_LAM)
CHI2_C = chi2.ppf(1 - ALPHA, M.N_MODES)
SUP_MAT = np.zeros((V.shape[0], M.N_STOREY))
for _i, _s in enumerate(SUPPORT):
    for _j, _w in _s.items():
        SUP_MAT[_i, _j] = _w


# ---------------------------------------------------------------------- surrogates
class PooledPI:
    """Physics-informed surrogate
        log f_i(t) = a_i + b * kappa_i * dT(t) + c * eta_i * q(t),
    with kappa_i, eta_i eigen-sensitivities supplied by the digital twin:
    m + 2 free scalars, temperature response tied across modes."""
    p = M.N_MODES + 2

    def __init__(self):
        self.A = np.eye(self.p) * 1e-6
        self.b = np.zeros(self.p)
        self.coef = np.zeros(self.p)

    def design(self, dT, q):
        X = np.zeros((M.N_MODES, self.p))
        X[:, :M.N_MODES] = np.eye(M.N_MODES)
        X[:, M.N_MODES] = M.REF['kappa'] * dT
        X[:, M.N_MODES + 1] = M.REF['eta'] * q
        return X

    def predict(self, dT, q):
        return self.design(dT, q) @ self.coef

    def update(self, dT, q, y):
        X = self.design(dT, q)
        self.A = FORGET * self.A + X.T @ X
        self.b = FORGET * self.b + X.T @ y
        self.coef = np.linalg.solve(self.A + 1e-9 * np.eye(self.p), self.b)


class PerModeBB:
    """Black-box surrogate: independent quadratic regression per mode on
    [1, dT, dT^2, q, dT*q] -- 5m free parameters, no physics tying."""
    p = 5

    def __init__(self):
        self.A = np.repeat(np.eye(self.p)[None] * 1e-6, M.N_MODES, axis=0)
        self.b = np.zeros((M.N_MODES, self.p))
        self.coef = np.zeros((M.N_MODES, self.p))

    @staticmethod
    def feat(dT, q):
        return np.array([1.0, dT, dT * dT, q, dT * q])

    def predict(self, dT, q):
        return self.coef @ self.feat(dT, q)

    def update(self, dT, q, y):
        x = self.feat(dT, q)
        for i in range(M.N_MODES):
            self.A[i] = FORGET * self.A[i] + np.outer(x, x)
            self.b[i] = FORGET * self.b[i] + x * y[i]
            self.coef[i] = np.linalg.solve(self.A[i] + 1e-9 * np.eye(self.p), self.b[i])


def _filtered_target(y, sig, r):
    """Adaptation target: withhold the identifiable damage-consistent component of
    the residual, then Huber-clip what remains."""
    pj = r @ VT.T
    w = max(pj.max(), 0.0) * VT[int(np.argmax(pj))]
    return y - sig * (r - np.clip(r - w, -HUBER, HUBER))


def _robust_scale(res):
    s0 = np.maximum(1.4826 * np.median(np.abs(res - np.median(res, axis=0)), axis=0), 1e-5)
    keep = np.abs(res) <= 6 * s0
    return np.maximum(np.array([res[keep[:, i], i].std(ddof=1)
                                for i in range(res.shape[1])]), 1e-5)


# ------------------------------------------------------------------------- monitor
EPS_CAL = ALPHA / 2.0            # calibration-failure budget of PHASE-C


def run_record(rec, h_chart=None, h_cusum=None, cusum_k=0.05, alpha=ALPHA,
               collect_paths=False, t_burn=T_BURN, h_extra=None, t_onset=None):
    y, dT, q = rec['y'], rec['dT'], rec['q']
    T = len(y)
    pi, bb = PooledPI(), PerModeBB()
    half = t_burn // 2

    # ---- commissioning pass A: unfiltered fitting, scale estimation
    rA_pi, rA_bb = [], []
    for t in range(half):
        if t > M.N_MODES + 4:
            rA_pi.append(y[t] - pi.predict(dT[t], q[t]))
            rA_bb.append(y[t] - bb.predict(dT[t], q[t]))
        pi.update(dT[t], q[t], y[t])
        bb.update(dT[t], q[t], y[t])
    sig_pi = _robust_scale(np.array(rA_pi[len(rA_pi) // 3:]))
    sig_bb = _robust_scale(np.array(rA_bb[len(rA_bb) // 3:]))

    # ---- commissioning pass B: monitoring update rule; calibrate delta
    xb, xob, xbb = [], [], []
    for t in range(half, t_burn):
        r = (y[t] - pi.predict(dT[t], q[t])) / sig_pi
        rb = (y[t] - bb.predict(dT[t], q[t])) / sig_bb
        xb.append(min(max((r @ V.T).max(), 0.0) / C_SCALE, 1.0))
        xob.append(min(np.linalg.norm(r) / C_OMNI, 1.0))
        xbb.append(min(max((rb @ V.T).max(), 0.0) / C_SCALE, 1.0))
        pi.update(dT[t], q[t], _filtered_target(y[t], sig_pi, r))
        bb.update(dT[t], q[t], _filtered_target(y[t], sig_bb, rb))

    env_lo, env_hi = np.quantile(dT[:t_burn], [0.005, 0.995])
    q_hi = np.quantile(q[:t_burn], 0.995)

    def _delta(x, m0):
        x = np.asarray(x)
        return max(0.0, x.mean() - m0 + 2.0 * x.std(ddof=1) / np.sqrt(len(x)))

    delta = _delta(xb, MU0)
    # PHASE-C: finite-sample certified allowance.  Azuma-Hoeffding upper limit on
    # the commissioning average of the conditional-mean excess (scores lie in
    # [0,1] and are sequential plug-in scores), failure probability EPS_CAL; the
    # alarm threshold is spent on alpha - EPS_CAL so that the total budget is alpha.
    xb_arr = np.asarray(xb)
    delta_c = max(0.0, xb_arr.mean() - MU0
                  + np.sqrt(np.log(1.0 / EPS_CAL) / (2.0 * len(xb_arr))))
    delta_o = _delta(xob, MU0_OMNI)
    delta_b = _delta(xbb, MU0)

    # ---- monitoring
    logthr = np.log(1.0 / alpha)
    lw = np.zeros(N_LAM); lwd = np.zeros(N_LAM); lwe = np.zeros(N_LAM)
    lwo = np.zeros(N_LAM); lwb = np.zeros(N_LAM); lwc = np.zeros(N_LAM)
    n_in = 0
    disc_c = np.log1p(LAMS * delta_c)
    xs_mon = []
    gacc = np.zeros(N_LAM); gn = 0
    onset_state = None
    loc_at_alarm = -1
    disc = np.log1p(LAMS * delta)
    disc_o = np.log1p(LAMS_OMNI * delta_o)
    disc_b = np.log1p(LAMS * delta_b)
    cus = 0.0
    loc = np.zeros(M.N_STOREY)
    keys = ('phase_d', 'phase_e', 'phase', 'phase_omni', 'phase_bb',
            'chart3', 'chart_cal', 'cusum', 'hotelling',
            'pca', 'msd', 'ewma', 'sr', 'ctm', 'phase_c', 'ctm_f')
    alarms = {k: -1 for k in keys}
    stat_max = {'chart': 0.0, 'cusum': 0.0, 'pca': 0.0, 'msd': 0.0,
                'ewma': 0.0, 'sr': 0.0}
    paths = ({'lwd': [], 'lw': [], 'lwo': [], 'lwb': [], 'x': [], 'r': [], 'cus': []}
             if collect_paths else None)

    def mix(l):
        mx = l.max()
        return mx + np.log(np.exp(l - mx).mean())

    # ---- extra comparators calibrated on the commissioning window ------------
    # (i) PCA-based EOV removal: drop the leading components of the raw
    #     log-frequency matrix, which are dominated by environmental response,
    #     and monitor the norm in the retained subspace.
    Yb = y[:t_burn]
    Yc = Yb - Yb.mean(0)
    _, _, Vt = np.linalg.svd(Yc, full_matrices=False)
    n_pc = 2
    Ppca = np.eye(M.N_MODES) - Vt[:n_pc].T @ Vt[:n_pc]
    ybar = Yb.mean(0)
    spca = np.linalg.norm((Yc @ Ppca.T), axis=1)
    spca_s = max(spca.std(), 1e-9)
    # (ii) Mahalanobis novelty index on the surrogate residuals
    Rb = np.array(rA_pi[len(rA_pi) // 3:]) / sig_pi
    Cinv = np.linalg.pinv(np.cov(Rb.T) + 1e-9 * np.eye(M.N_MODES))
    # (iii)-(iv) EWMA and Shiryaev-Roberts run on the cone score
    ew = MU0
    ew_lam = 0.05
    sr = 0.0
    # (v) conformal test martingale on the cone score (distribution-free)
    cal = np.sort(xb)
    n_cal = len(cal)
    eps_grid = np.linspace(0.05, 0.95, 12)
    lctm = np.zeros(len(eps_grid))
    rng_ctm = np.random.default_rng(20240517)
    # (v-b) the same martingale with a calibration set drawn from the WHOLE
    #       commissioning window (plug-in residuals of pass A plus pass B scores)
    RA = np.array(rA_pi[len(rA_pi) // 3:]) / sig_pi
    xa = np.minimum(np.clip((RA @ V.T).max(axis=1), 0, None) / C_SCALE, 1.0)
    cal_f = np.sort(np.concatenate([xa, xb_arr]))
    n_cal_f = len(cal_f)
    lctm_f = np.zeros(len(eps_grid))
    rng_ctm_f = np.random.default_rng(20240518)

    for t in range(t_burn, T):
        r = (y[t] - pi.predict(dT[t], q[t])) / sig_pi
        rb = (y[t] - bb.predict(dT[t], q[t])) / sig_bb
        proj = r @ V.T
        k_arg = int(np.argmax(proj))
        u = max(proj[k_arg], 0.0)
        x = min(u / C_SCALE, 1.0)
        xo = min(np.linalg.norm(r) / C_OMNI, 1.0)
        xbn = min(max((rb @ V.T).max(), 0.0) / C_SCALE, 1.0)

        in_env = (env_lo <= dT[t] <= env_hi) and (q[t] <= q_hi)
        if in_env:
            n_in += 1
            lwe += np.log1p(LAMS * (x - MU0)) - disc
        lw += np.log1p(LAMS * (x - MU0))
        lwd += np.log1p(LAMS * (x - MU0)) - disc
        lwo += np.log1p(LAMS_OMNI * (xo - MU0_OMNI)) - disc_o
        lwb += np.log1p(LAMS * (xbn - MU0)) - disc_b
        lwc += np.log1p(LAMS * (x - MU0)) - disc_c
        if t_onset is not None:
            if t == t_onset:
                onset_state = (lwd.copy(), mix(lwd))
            if t >= t_onset:
                gacc += np.log1p(LAMS * (x - MU0)); gn += 1
        xs_mon.append(x)
        Lpc = mix(lwc)
        Lp, Lpd, Lpo, Lpb, Lpe = mix(lw), mix(lwd), mix(lwo), mix(lwb), mix(lwe)

        cus = max(0.0, cus + (x - MU0) - cusum_k)
        ch = np.abs(r).max()
        s_pca = np.linalg.norm(Ppca @ (y[t] - ybar)) / spca_s
        s_msd = float(np.sqrt(max(r @ Cinv @ r, 0.0)))
        ew = (1 - ew_lam) * ew + ew_lam * x
        s_ewma = (ew - MU0) / np.sqrt(ew_lam / (2 - ew_lam))
        sr = min((1.0 + sr) * np.exp(4.0 * (x - MU0 - cusum_k)), 1e290)
        s_sr = np.log1p(sr)
        p_c = (np.searchsorted(cal, x, side='left') +
               rng_ctm.random() * (1 + np.searchsorted(cal, x, side='right')
                                   - np.searchsorted(cal, x, side='left')))
        p_c = min(max((n_cal - p_c + 1) / (n_cal + 1), 1e-6), 1.0)
        lctm += np.log(eps_grid) + (eps_grid - 1.0) * np.log(p_c)
        L_ctm = mix(lctm)
        lo_f = np.searchsorted(cal_f, x, side='left'); hi_f = np.searchsorted(cal_f, x, side='right')
        p_f = lo_f + rng_ctm_f.random() * (1 + hi_f - lo_f)
        p_f = min(max((n_cal_f - p_f + 1) / (n_cal_f + 1), 1e-6), 1.0)
        lctm_f += np.log(eps_grid) + (eps_grid - 1.0) * np.log(p_f)
        L_ctm_f = mix(lctm_f)
        loc += SUP_MAT[k_arg] * max(x - MU0, 0.0)
        stat_max['chart'] = max(stat_max['chart'], ch)
        stat_max['cusum'] = max(stat_max['cusum'], cus)
        stat_max['pca'] = max(stat_max['pca'], s_pca)
        stat_max['msd'] = max(stat_max['msd'], s_msd)
        stat_max['ewma'] = max(stat_max['ewma'], s_ewma)
        stat_max['sr'] = max(stat_max['sr'], s_sr)

        if alarms['phase'] < 0 and Lp >= logthr: alarms['phase'] = t
        if alarms['phase_d'] < 0 and Lpd >= logthr:
            alarms['phase_d'] = t
            loc_at_alarm = int(np.argmax(loc))
        if alarms['phase_c'] < 0 and Lpc >= np.log(1.0 / (alpha - EPS_CAL)):
            alarms['phase_c'] = t
        if alarms['phase_e'] < 0 and Lpe >= logthr: alarms['phase_e'] = t
        if alarms['phase_omni'] < 0 and Lpo >= logthr: alarms['phase_omni'] = t
        if alarms['phase_bb'] < 0 and Lpb >= logthr: alarms['phase_bb'] = t
        if alarms['chart3'] < 0 and ch > 3.0: alarms['chart3'] = t
        if h_chart is not None and alarms['chart_cal'] < 0 and ch > h_chart:
            alarms['chart_cal'] = t
        if h_cusum is not None and alarms['cusum'] < 0 and cus > h_cusum:
            alarms['cusum'] = t
        if alarms['hotelling'] < 0 and (r @ r) > CHI2_C: alarms['hotelling'] = t
        if h_extra is not None:
            for nm, st in (('pca', s_pca), ('msd', s_msd), ('ewma', s_ewma),
                           ('sr', s_sr)):
                if alarms[nm] < 0 and st > h_extra[nm]:
                    alarms[nm] = t
        if alarms['ctm'] < 0 and L_ctm >= logthr: alarms['ctm'] = t
        if alarms['ctm_f'] < 0 and L_ctm_f >= logthr: alarms['ctm_f'] = t

        if collect_paths:
            paths['lwd'].append(Lpd); paths['lw'].append(Lp)
            paths['lwo'].append(Lpo); paths['lwb'].append(Lpb)
            paths['x'].append(x); paths['r'].append(r.copy()); paths['cus'].append(cus)

        pi.update(dT[t], q[t], _filtered_target(y[t], sig_pi, r))
        bb.update(dT[t], q[t], _filtered_target(y[t], sig_bb, rb))

    xm = np.asarray(xs_mon)
    blk = 360                                    # 90-day blocks
    nb = len(xm) // blk
    blockmax = (float(max(xm[i * blk:(i + 1) * blk].mean() for i in range(nb)) - MU0)
                if nb else None)
    acf1 = float(np.corrcoef(xm[:-1], xm[1:])[0, 1])
    out = dict(alarms=alarms, stat_max=stat_max, delta=delta, delta_o=delta_o,
               delta_c=float(delta_c), loc_at_alarm=loc_at_alarm,
               x_burn_mean=float(xb_arr.mean()), x_mon_mean=float(xm.mean()),
               excess_blockmax=blockmax, acf1=acf1,
               onset=(None if onset_state is None else dict(
                   lw=onset_state[0].tolist(), mix=float(onset_state[1]),
                   g=(gacc / max(gn, 1)).tolist(), n=gn)),
               mu0=MU0,
               delta_b=delta_b, loc=int(np.argmax(loc)), sig_pi=sig_pi, sig_bb=sig_bb,
               coverage=float(n_in / max(T - t_burn, 1)),
               envelope=(float(env_lo), float(env_hi)))
    if collect_paths:
        out['paths'] = {k: np.array(v) for k, v in paths.items()}
    return out
