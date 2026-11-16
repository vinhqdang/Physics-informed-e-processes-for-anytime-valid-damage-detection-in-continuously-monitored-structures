"""Detection counts for the benign-change-then-damage experiment (stage 'stiffdam') in two
observation windows after the loss: the nine months (1,095 epochs) used in every other
experiment, and the whole remaining record (3,295 epochs, 824 days).  Reads part_stiffdam.json
(written by `python3 stage.py stiffdam`), writes stiffdam_windows.json.
Epochs are six hours; delays are reported in days (median among detected records)."""
import json
import numpy as np
from scipy.stats import beta

T_STAR, T_LONG, GAP = 1825, 5840, 720
T_LOSS = T_STAR + GAP                 # epoch of the 5 % loss at storey 3 (both scenarios)
NINE = 1095                           # nine months = 4 x 274 days
MON = {'phase_d': 'PHASE', 'phase_r': 'PHASE-R', 'phase_c': 'PHASE-C', 'ctm_f': 'CTM-F',
       'cusum': 'CUSUM', 'phase_omni': 'PHASE-Omega'}


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


def summarise(a):
    a = np.asarray(a); n = len(a)
    pre = (a >= 0) & (a < T_LOSS)
    out = dict(n=n, n_pre=int(pre.sum()),
               pre_med_days=float(np.median(a[pre] - T_STAR) / 4) if pre.any() else None)
    for name, hi in (('nine_months', T_LOSS + NINE), ('to_end', T_LONG + 1)):
        det = (a >= T_LOSS) & (a < hi)
        out[name] = dict(k=int(det.sum()), ci=cp(int(det.sum()), n),
                         med_days=float(np.median(a[det] - T_LOSS) / 4) if det.any() else None)
    return out


if __name__ == '__main__':
    d = json.load(open('part_stiffdam.json'))
    res = dict(loss_epoch=T_LOSS, record_epochs=T_LONG, nine_months_epochs=NINE,
               remaining_epochs=T_LONG - T_LOSS, remaining_days=(T_LONG - T_LOSS) / 4)
    for key, lab in (('plain', 'loss_alone'), ('stiffdam', 'stiffening_then_loss')):
        res[lab] = {m: summarise(d[key]['alarms'][m]) for m in MON}
    json.dump(res, open('stiffdam_windows.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}))
