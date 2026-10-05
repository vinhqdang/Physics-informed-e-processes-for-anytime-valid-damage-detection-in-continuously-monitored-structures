"""Original schematic figures: the PHASE pipeline, the eight-storey frame used in
the simulation study, and an elevation of the Z24 bridge. All drawings are our
own; no third-party imagery is used."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle
import model as M

plt.rcParams.update({'font.size': 8.5, 'font.family': 'serif',
                     'figure.dpi': 300, 'savefig.dpi': 300})
INK = '#1f2933'
ACC = '#c0392b'
BLU = '#2c6fbb'
GRN = '#2e7d52'


# ----------------------------------------------------------------- Fig: pipeline
def pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 2.55))
    ax.set_xlim(0, 100); ax.set_ylim(-6, 34); ax.axis('off')

    def box(x, y, w, h, title, sub, fc='white', ec=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.6',
                                    fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h * 0.66, title, ha='center', va='center',
                fontsize=8.2, weight='bold', color=INK)
        ax.text(x + w / 2, y + h * 0.27, sub, ha='center', va='center',
                fontsize=6.6, color='#4a5568')

    def arrow(x0, y0, x1, y1, style='-|>', col=INK, ls='-'):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                     mutation_scale=9, lw=0.9, color=col,
                                     linestyle=ls, shrinkA=0, shrinkB=0))

    # design-time row
    box(1, 21, 20, 11, 'Digital twin', 'FE model $K_j$, $M$', fc='#eef3fa')
    box(24, 21, 22, 11, 'Physics extraction',
        'sensitivities $S$, $\\kappa$, $\\eta$', fc='#eef3fa')
    box(49, 21, 23, 11, 'Damage cone $\\mathcal{C}$',
        'generators $V$; nuisance $\\mathcal{N}$', fc='#eef3fa')
    box(75, 21, 24, 11, 'Design constants',
        '$\\mu_0$, detectability $\\varrho_j$', fc='#eef3fa')
    for x in (21, 46, 72):
        arrow(x, 26.5, x + 3, 26.5)
    ax.text(1, 33.2, 'design time (before installation)', fontsize=6.8,
            style='italic', color='#4a5568')

    # run-time row
    box(1, 3, 20, 11, 'Monitoring epoch', 'modal id. $y_t$; covariates $z_t$')
    box(24, 3, 22, 11, 'Nuisance surrogate',
        'predictable plug-in $\\hat\\beta_{t-1}$')
    box(49, 3, 23, 11, 'Cone score',
        '$X_t=\\min(u_t^+/c,\\,1)$')
    box(75, 3, 24, 11, 'e-process and alarm',
        '$W_t \\geq 1/\\alpha$ (Ville)', fc='#fdf0ee', ec=ACC)
    for x in (21, 46, 72):
        arrow(x, 8.5, x + 3, 8.5)
    ax.text(1, 15.4, 'run time (every epoch, indefinitely)', fontsize=6.8,
            style='italic', color='#4a5568')

    # couplings between rows
    arrow(35, 21, 35, 14.5, col=BLU, ls=':')
    arrow(60, 21, 60, 14.5, col=BLU, ls=':')
    arrow(87, 21, 87, 14.5, col=BLU, ls=':')
    # feedback: cone-filtered adaptation
    ax.add_patch(FancyArrowPatch((60, 3), (35, 3), arrowstyle='-|>',
                                 mutation_scale=9, lw=0.9, color=GRN,
                                 connectionstyle='arc3,rad=-0.4'))
    ax.text(47.5, -5.0, 'cone-filtered adaptation', fontsize=6.6, color=GRN,
            ha='center')
    fig.tight_layout()
    fig.savefig('fig_pipeline.png', bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------ Fig: shear frame
def frame():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3),
                           gridspec_kw={'width_ratios': [1, 1.35]})
    n = M.N_STOREY
    a = ax[0]
    H, W = 1.0, 2.4
    a.add_patch(Rectangle((-0.7, -0.42), W + 1.4, 0.42, fc='#d7dde3', ec=INK,
                          lw=0.8, hatch='///'))
    for s in range(n):
        y = s * H
        a.plot([0, 0], [y, y + H], color=INK, lw=1.6)
        a.plot([W, W], [y, y + H], color=INK, lw=1.6)
        col = '#f6c8c1' if s == 2 else '#eaeef2'
        a.add_patch(Rectangle((-0.18, y + H - 0.16), W + 0.36, 0.22, fc=col,
                              ec=INK, lw=0.9))
        a.text(W + 0.35, y + H - 0.05, '%d' % (s + 1), fontsize=7,
               va='center', color=INK)
    a.annotate('', xy=(-0.95, 3 * H - 0.05), xytext=(-0.95, 2 * H - 0.05),
               arrowprops=dict(arrowstyle='<->', lw=0.8, color=ACC))
    a.text(-1.05, 2.5 * H, 'storey 3\nstiffness loss\n$\\rho$', fontsize=6.8,
           color=ACC, ha='right', va='center')
    a.text(W + 0.2, n * H + 0.25, 'storey', fontsize=6.8, color=INK)
    a.text(-2.5, n * H + 0.55, '$m_j = 2.5\\times10^{5}$ kg', fontsize=7)
    a.text(-2.5, n * H + 0.15, '$k_j = 6.0\\times10^{8}$ N/m', fontsize=7)
    a.plot(W / 2, n * H - 0.05, marker='v', ms=5, color=BLU)
    a.text(W / 2, -0.75, 'accelerometers', fontsize=6.4,
           color=BLU, va='center', ha='center')
    for s in range(0, n, 2):
        a.plot(W / 2, s * H + H - 0.05, marker='v', ms=5, color=BLU)
    a.set_xlim(-2.6, W + 1.5); a.set_ylim(-1.1, n * H + 1.0)
    a.axis('off')
    a.set_title('(a) eight-storey shear frame', fontsize=8.5)

    b = ax[1]
    phi = M.REF['phi']
    z = np.arange(n + 1)
    for i in range(3):
        v = np.concatenate([[0], phi[:, i]])
        v = v / np.abs(v).max()
        b.plot(v + 2.6 * i, z, '-o', ms=2.6, lw=1.1,
               color=[BLU, GRN, ACC][i],
               label='mode %d: %.2f Hz' % (i + 1, M.REF['f0'][i]))
        b.plot([2.6 * i, 2.6 * i], [0, n], color='0.8', lw=0.6, zorder=0)
    b.set_yticks(range(0, n + 1, 2)); b.set_ylabel('storey')
    b.set_xticks([]); b.legend(fontsize=6.6, loc='upper center', bbox_to_anchor=(0.5,-0.02), ncol=3, frameon=False)
    b.set_title('(b) mode shapes of the reference state', fontsize=8.5)
    for sp in ('top', 'right', 'bottom'):
        b.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig('fig_frame.png', bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------- Fig: Z24 bridge
def z24_elevation(ax):
    L1, L2, L3 = 14.0, 30.0, 14.0
    total = L1 + L2 + L3
    GY = -6.0                                   # ground level
    ax.add_patch(Rectangle((-6, GY - 1.1), total + 12, 1.1, fc='#cbd3da',
                           ec=INK, lw=0.6, hatch='///'))
    ax.add_patch(Rectangle((0, 0), total, 1.5, fc='#e7edf3', ec=INK, lw=1.0))
    for px in (L1, L1 + L2):                    # main piers
        ax.add_patch(Rectangle((px - 0.9, GY), 1.8, -GY, fc='#d7dde3',
                               ec=INK, lw=1.0))
    for cx in (0.9, total - 0.9):               # abutment column triplets
        ax.add_patch(Rectangle((cx - 2.4, GY), 4.8, 0.9, fc='#d7dde3',
                               ec=INK, lw=0.8))
        for d in (-1.5, 0, 1.5):
            ax.plot([cx + d, cx + d], [GY + 0.9, 0], color=INK, lw=1.5)
    for a_, b_, lab in [(0, L1, '14 m'), (L1, L1 + L2, '30 m'),
                        (L1 + L2, total, '14 m')]:
        ax.annotate('', xy=(a_, 3.6), xytext=(b_, 3.6),
                    arrowprops=dict(arrowstyle='<->', lw=0.7, color='#4a5568'))
        ax.text((a_ + b_) / 2, 4.1, lab, fontsize=6.6, ha='center',
                color='#4a5568')
    ax.text(L1 - 2.0, -2.6, 'Koppigen pier', fontsize=6.6, ha='right',
            color=INK)
    ax.text(L1 + L2 + 2.0, -2.6, 'Utzenstorf pier', fontsize=6.6, ha='left',
            color=INK)
    ax.add_patch(FancyArrowPatch((L1 - 2.6, -3.4), (L1 - 2.6, -5.2),
                                 arrowstyle='-|>', mutation_scale=10, lw=1.4,
                                 color=ACC))
    ax.text(L1 - 3.2, -4.3, 'lowered\n20\u201395 mm', fontsize=6.6, color=ACC,
            va='center', ha='right')
    for sx in [L1 * 0.5, L1 + L2 * 0.28, L1 + L2 * 0.5, L1 + L2 * 0.72,
               L1 + L2 + L3 * 0.5]:
        ax.plot(sx, 1.5, marker='v', ms=5, color=BLU, clip_on=False)
    ax.text(L1 + L2 * 0.5, 2.3, 'five permanent reference accelerometers',
            fontsize=6.4, color=BLU, ha='center')
    ax.set_xlim(-14, total + 6); ax.set_ylim(-8.4, 5.2)
    ax.axis('off'); ax.set_aspect('equal')


def bridge():
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    z24_elevation(ax)
    fig.tight_layout()
    fig.savefig('fig_bridge.png', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    pipeline(); frame(); bridge()
    print('wrote fig_pipeline.png, fig_frame.png, fig_bridge.png')
