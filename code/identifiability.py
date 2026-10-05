"""Identifiability, localisability and nuisance-model sensitivity of the shear-frame
detectability map (Section 3.2 and 4.2.5 of the manuscript).

Outputs identifiability.json and prints the numbers quoted in the text.
Run:  python3 identifiability.py
"""
import json
import numpy as np
from scipy.optimize import linprog
import model as M

SIG = M.SIG_F.copy()


def frame_sens(n, m, loaded=None):
    """First-order sensitivities of an n-storey shear frame with the storey
    properties of model.py; returns S (m x n), kappa, eta."""
    Ks = np.zeros((n, n, n))
    for j in range(n):
        Kj = np.zeros((n, n))
        Kj[j, j] += 1.0
        if j > 0:
            Kj[j - 1, j - 1] += 1.0
            Kj[j - 1, j] -= 1.0
            Kj[j, j - 1] -= 1.0
        Ks[j] = Kj * M.K_STOREY
    mass = np.full(n, M.M_STOREY)
    s = 1 / np.sqrt(mass)
    A = Ks.sum(0) * s[:, None] * s[None, :]
    lam, Q = np.linalg.eigh(A)
    phi = Q * s[:, None]
    S = np.zeros((m, n)); eta = np.zeros(m)
    loaded = np.arange(1, n - 1) if loaded is None else loaded
    for i in range(m):
        v = phi[:, i]
        den = (v @ (mass * v)) * lam[i]
        for j in range(n):
            S[i, j] = (v @ Ks[j] @ v) / den
        eta[i] = (v[loaded] ** 2 * mass[loaded]).sum() / (v @ (mass * v))
    return S, S.sum(1), eta, phi, mass


def projector(U):
    Q, _ = np.linalg.qr(U)
    return np.eye(U.shape[0]) - Q @ Q.T


def retention(S, sig, U):
    A = S / sig[:, None]
    P = projector(U)
    return np.linalg.norm(P @ A, axis=0) / np.linalg.norm(A, axis=0), P, A


out = {}
S, kappa, eta = M.REF['S'], M.REF['kappa'], M.REF['eta']
n, m = S.shape[1], S.shape[0]
A = S / SIG[:, None]

# ---- rank structure of D^{-1}S -------------------------------------------------
sv = np.linalg.svd(A, compute_uv=False)
rank = int((sv > 1e-9 * sv[0]).sum())
out['rank_DinvS'] = rank
out['n_substructures'] = n
out['n_modes'] = m
out['null_dim_S'] = n - rank
U2 = np.stack([kappa / SIG, eta / SIG], axis=1)
ret2, P2, _ = retention(S, SIG, U2)
out['retention_k_eta'] = ret2.tolist()

# dimension of the confounded set {rho : D^{-1} S rho in N}
Pn = P2 @ A                                   # (m x n); rho confounded iff Pn rho = 0
sv2 = np.linalg.svd(Pn, compute_uv=False)
rank2 = int((sv2 > 1e-9 * sv2[0]).sum())
out['confounded_dim_k_eta'] = n - rank2
U1 = (kappa / SIG)[:, None]
P1 = projector(U1)
sv1 = np.linalg.svd(P1 @ A, compute_uv=False)
out['confounded_dim_k_only'] = n - int((sv1 > 1e-9 * sv1[0]).sum())

# does the confounded set contain non-uniform *physical* (rho >= 0) patterns?
def extreme_confounded(Pmat, j):
    """min and max of rho_j over {rho >= 0, sum(rho)=1, Pmat rho = 0}."""
    res = []
    for sgn in (1, -1):
        c = np.zeros(n); c[j] = sgn
        r = linprog(c, A_eq=np.vstack([Pmat, np.ones((1, n))]),
                    b_eq=np.concatenate([np.zeros(Pmat.shape[0]), [1.0]]),
                    bounds=[(0, None)] * n, method='highs')
        res.append(None if r.status != 0 else float(sgn * r.fun))
    return res


out['nonneg_confounded_range_k_eta'] = [extreme_confounded(P2 @ A, j) for j in range(n)]
out['nonneg_confounded_range_k_only'] = [extreme_confounded(P1 @ A, j) for j in range(n)]

# ---- the stronger claim holds when S is injective and only temperature acts -------
S6, k6, e6, _, _ = frame_sens(6, 6)
A6 = S6 / SIG[:, None]
P16 = projector((k6 / SIG)[:, None])
s6 = np.linalg.svd(P16 @ A6, compute_uv=False)
out['six_storey_six_modes'] = dict(
    rank=int((np.linalg.svd(A6, compute_uv=False) > 1e-9).sum()),
    confounded_dim_k_only=int(6 - (s6 > 1e-9 * s6[0]).sum()),
    retention_k_only=(np.linalg.norm(P16 @ A6, axis=0) / np.linalg.norm(A6, axis=0)).tolist(),
    retention_k_eta=retention(S6, SIG, np.stack([k6 / SIG, e6 / SIG], 1))[0].tolist())

# ---- localisability: coherence of identifiable signatures --------------------------
G = P2 @ A
Gn = G / np.linalg.norm(G, axis=0)
coh = Gn.T @ Gn
out['coherence_identifiable'] = coh.tolist()
An = A / np.linalg.norm(A, axis=0)
out['coherence_full'] = (An.T @ An).tolist()
offd = coh - np.eye(n)
out['max_offdiag_coherence'] = float(np.abs(offd).max())
out['storey1_vs_3_6'] = [float(coh[0, 2]), float(coh[0, 5])]
out['rank_identifiable_signatures'] = int((np.linalg.svd(G, compute_uv=False) > 1e-9 * np.linalg.svd(G, compute_uv=False)[0]).sum())

# ---- sensitivity of the detectability map to additional nuisance directions ---------
j_idx = np.arange(n)
grad = (j_idx - j_idx.mean()) / (n - 1) * 2          # linear-in-height modulus profile
mass_all = np.array([(M.REF['phi'][:, i] ** 2 * M.M0).sum() /
                     (M.REF['phi'][:, i] @ (M.M0 * M.REF['phi'][:, i])) for i in range(m)])  # = 1
phi = M.REF['phi']
eta_top = np.array([(phi[n - 1, i] ** 2 * M.M0[n - 1]) /
                    (phi[:, i] @ (M.M0 * phi[:, i])) for i in range(m)])   # roof mass change
cases = {
    'temperature only ($\\kappa$)': [kappa],
    'temperature + live load ($\\kappa,\\eta$)  [baseline]': [kappa, eta],
    '+ vertical gradient': [kappa, eta, S @ grad],
    '+ support (storey-1) condition': [kappa, eta, S[:, 0]],
    '+ roof/equipment mass': [kappa, eta, eta_top],
    '+ gradient + support': [kappa, eta, S @ grad, S[:, 0]],
}
out['nuisance_cases'] = {}
for nm, cols in cases.items():
    U = np.stack([c / SIG for c in cols], 1)
    r, _, _ = retention(S, SIG, U)
    out['nuisance_cases'][nm] = dict(dim=int(np.linalg.matrix_rank(U)), retention=r.tolist(),
                                     modes_left=int(m - np.linalg.matrix_rank(U)))
json.dump(out, open('identifiability.json', 'w'), indent=1)

np.set_printoptions(precision=3, suppress=True, linewidth=160)
print('rank D^-1 S = %d of n = %d substructures, m = %d modes' % (rank, n, m))
print('dim confounded set (kappa,eta) = %d ; (kappa only) = %d' % (
    out['confounded_dim_k_eta'], out['confounded_dim_k_only']))
print('nonneg confounded range of rho_j, (kappa,eta):')
for j, r in enumerate(out['nonneg_confounded_range_k_eta']):
    print('  storey', j + 1, r)
print('(kappa only):')
for j, r in enumerate(out['nonneg_confounded_range_k_only']):
    print('  storey', j + 1, r)
print('six-storey/six-mode:', out['six_storey_six_modes'])
print('identifiable coherence:\n', coh)
print('storey 1 vs 3,6 coherence', out['storey1_vs_3_6'], 'rank of identifiable signatures', out['rank_identifiable_signatures'])
for nm, v in out['nuisance_cases'].items():
    print('%-55s dim=%d  min ret=%.3f  ret=%s' % (nm, v['dim'], min(v['retention']), np.round(v['retention'], 3)))
