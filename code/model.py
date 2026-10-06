"""
Physics model + synthetic SHM data generator for the PHASE experiments.

Structure: n-storey shear frame (lumped mass, storey stiffnesses).
Nuisance (EOV) effects:
  * seasonal + diurnal ambient temperature, storey-wise thermal LAG (first-order),
    storey-wise gradient  -> unmodelled by a single-sensor surrogate
  * occupancy / live-load mass on intermediate floors, measured with error
Damage: multiplicative stiffness loss on one or more storeys.
Observation: first m identified natural frequencies with heavy-tailed (Student-t)
    multiplicative identification noise.
"""
import numpy as np

# ----------------------------------------------------------------------------- model
N_STOREY = 8
M_STOREY = 2.5e5          # kg
K_STOREY = 6.0e8          # N/m
N_MODES = 6               # identified modes
BETA_T = 2.5e-3           # modulus temperature coefficient, 1/degC
T_REF = 24.0              # degC reference
LOADED_FLOORS = np.arange(1, 7)   # floors carrying live load (0-indexed storeys 1..6)
SIG_F = np.array([0.0060, 0.0072, 0.0084, 0.0096, 0.0108, 0.0120])  # rel. id. noise (std)
NU_T = 5.0                # Student-t dof of identification noise


def storey_stiffness_matrices(n=N_STOREY):
    """K = sum_j theta_j * K_j  for a shear frame."""
    Ks = np.zeros((n, n, n))
    for j in range(n):
        Kj = np.zeros((n, n))
        Kj[j, j] += 1.0
        if j > 0:
            Kj[j - 1, j - 1] += 1.0
            Kj[j - 1, j] -= 1.0
            Kj[j, j - 1] -= 1.0
        Ks[j] = Kj
    return Ks * K_STOREY


KS = storey_stiffness_matrices()
M0 = np.full(N_STOREY, M_STOREY)


def eig_freqs(theta, mass):
    """Batched natural frequencies (Hz) for stiffness multipliers theta (B,n)
    and floor masses mass (B,n)."""
    theta = np.atleast_2d(theta)
    mass = np.atleast_2d(mass)
    K = np.einsum('bj,jkl->bkl', theta, KS)
    s = 1.0 / np.sqrt(mass)                      # (B,n)
    A = K * s[:, :, None] * s[:, None, :]        # M^{-1/2} K M^{-1/2}
    lam = np.linalg.eigvalsh(A)                  # ascending
    lam = np.clip(lam, 1e-12, None)
    return np.sqrt(lam) / (2.0 * np.pi)


def reference_state():
    """Reference (healthy, T=T_REF, no live load) modal properties and the
    physics-derived sensitivity matrices used by the surrogate and the cone."""
    theta = np.ones((1, N_STOREY))
    mass = M0[None, :].copy()
    K = np.einsum('bj,jkl->bkl', theta, KS)[0]
    s = 1.0 / np.sqrt(mass[0])
    A = K * s[:, None] * s[None, :]
    lam, Q = np.linalg.eigh(A)
    phi = Q * s[:, None]                     # mass-normalised-ish mode shapes
    lam = lam[:N_MODES]
    phi = phi[:, :N_MODES]
    f0 = np.sqrt(lam) / (2 * np.pi)

    # d log(omega_i) / d(stiffness loss rho_j)  =  -0.5 * S_ij ,  S_ij >= 0
    S = np.zeros((N_MODES, N_STOREY))
    for i in range(N_MODES):
        v = phi[:, i]
        denom = v @ (M0 * v) * lam[i]
        for j in range(N_STOREY):
            S[i, j] = (v @ KS[j] @ v) / denom

    # temperature sensitivity: uniform modulus scaling -> kappa_i = sum_j S_ij
    kappa = S.sum(axis=1)                    # d log omega_i / d(uniform stiffness loss)
    # added-mass sensitivity on LOADED_FLOORS (per unit relative added mass)
    eta = np.zeros(N_MODES)
    for i in range(N_MODES):
        v = phi[:, i]
        eta[i] = (v[LOADED_FLOORS] ** 2 * M0[LOADED_FLOORS]).sum() / (v @ (M0 * v))
    return dict(f0=f0, lam=lam, phi=phi, S=S, kappa=kappa, eta=eta)


REF = reference_state()


# ------------------------------------------------------------------ EOV / load driver
def lowpass(x, tau):
    """First-order lag with time constant tau (in epochs)."""
    a = np.exp(-1.0 / tau)
    y = np.empty_like(x)
    acc = x[0]
    for t in range(len(x)):
        acc = a * acc + (1 - a) * x[t]
        y[t] = acc
    return y


EPOCH_H = 6.0          # hours per interrogation epoch (modal id. every 6 h)
EP_DAY = int(24 / EPOCH_H)
EP_YEAR = EP_DAY * 365


def eov_series(rng, T):
    """Ambient temperature (degC), storey temperatures, live-load ratio and its
    imperfect measurement. 6-hourly epochs, northern-Vietnam monsoon calendar."""
    t = np.arange(T)
    seasonal = 24.0 - 7.0 * np.cos(2 * np.pi * (t / EP_YEAR - 0.05))
    diurnal = 4.0 * np.sin(2 * np.pi * (t % EP_DAY) / EP_DAY - 1.2)
    noise = np.zeros(T)
    e = rng.standard_normal(T) * 1.1
    acc = 0.0
    for k in range(T):
        acc = 0.6 * acc + e[k]
        noise[k] = acc
    T_amb = seasonal + diurnal + noise

    # storey temperatures: thermal lag + vertical gradient (unmodelled)
    taus = np.linspace(0.7, 2.0, N_STOREY)
    grad = np.linspace(0.86, 1.16, N_STOREY)
    T_st = np.empty((T, N_STOREY))
    for j in range(N_STOREY):
        T_st[:, j] = T_REF + grad[j] * (lowpass(T_amb, taus[j]) - T_REF)

    # live load: occupancy cycle + weekly pattern + noise
    hod = (t % EP_DAY) * EPOCH_H
    occ = np.exp(-0.5 * ((hod - 12.0) / 6.0) ** 2)
    week = np.where(((t // EP_DAY) % 7) < 5, 1.0, 0.25)
    q = 0.045 * occ * week + 0.004 * rng.standard_normal(T)
    q = np.clip(q, 0.0, None)
    q_meas = q * (1 + 0.10 * rng.standard_normal(T)) + 0.0015 * rng.standard_normal(T)
    return T_amb, T_st, q, q_meas


def damage_profile(T, scenario, t_star):
    """Stiffness-loss matrix rho (T, n)."""
    rho = np.zeros((T, N_STOREY))
    if scenario == 'healthy':
        pass
    elif scenario.startswith('abrupt'):
        lvl = float(scenario.split('_')[1]) / 100.0
        rho[t_star:, 2] = lvl                       # storey 3 (0-indexed 2)
    elif scenario.startswith('gradual'):
        lvl = float(scenario.split('_')[1]) / 100.0
        dur = 240        # 60-day degradation ramp
        ramp = np.clip((np.arange(T) - t_star) / dur, 0.0, 1.0)
        rho[:, 2] = lvl * ramp
    elif scenario.startswith('storey'):
        body, lvl = scenario.split('_')
        j = int(body[6:]) - 1
        rho[t_star:, j] = float(lvl) / 100.0
    elif scenario == 'stiffening':
        rho[t_star:, :] = -0.03                     # temporary propping: NOT damage
    elif scenario == 'stiffdam':
        # persistent 3% stiffening from t_star, then a 5% loss at storey 3 180 days later
        rho[t_star:, :] = -0.03
        rho[t_star + 720:, 2] = 1.0 - 1.03 * 0.95
    else:
        raise ValueError(scenario)
    return rho


def simulate(rng, T, scenario='healthy', t_star=None):
    """Return dict with observed log-frequencies and measured covariates."""
    if t_star is None:
        t_star = T // 2
    T_amb, T_st, q, q_meas = eov_series(rng, T)
    rho = damage_profile(T, scenario, t_star)

    theta = (1.0 - BETA_T * (T_st - T_REF)) * (1.0 - rho)
    mass = np.tile(M0, (T, 1))
    mass[:, LOADED_FLOORS] *= (1.0 + q[:, None])

    f = eig_freqs(theta, mass)[:, :N_MODES]
    # heavy-tailed multiplicative identification noise
    z = rng.standard_t(NU_T, size=(T, N_MODES)) / np.sqrt(NU_T / (NU_T - 2))
    f_obs = f * (1.0 + SIG_F[None, :] * z)
    return dict(y=np.log(f_obs), dT=T_amb - T_REF, q=q_meas,
                f_true=f, t_star=t_star, scenario=scenario)
