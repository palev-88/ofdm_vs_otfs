"""Uncoded BER vs SNR figures for §12.2, from uncoded_ref.json (v2).
Grid: TDL-A/B/C/D rows x fD {0, 600, 1000} columns, per bandwidth case."""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
FIGS = os.path.join(os.path.dirname(HERE), 'figures')
d = json.load(open(os.path.join(DATA, 'uncoded_ref.json')))

METH = ['OFDM', 'PCP-guard', 'PCP-orig']
STY = {'OFDM': ('tab:green', 'o-'), 'PCP-guard': ('tab:blue', 's--'),
       'PCP-orig': ('tab:purple', 'd--')}
CH = ['TDL-A', 'TDL-B', 'TDL-C', 'TDL-D']
FDS = [0.0, 600.0, 1000.0]

for bw in ('NB', 'WB'):
    fig, axes = plt.subplots(len(CH), len(FDS), figsize=(11.5, 12.5),
                             sharex=True, sharey=True)
    for r, cm in enumerate(CH):
        for c, fd in enumerate(FDS):
            ax = axes[r, c]
            for nm in METH:
                pts = []
                for k, v in d.items():
                    b2, cm2, fd2, snr, nm2 = k.split('|')
                    if (b2, cm2, nm2) == (bw, cm, nm) and float(fd2) == fd:
                        pts.append((float(snr), v['ber']))
                pts.sort()
                if not pts:
                    continue
                s = np.array([p[0] for p in pts])
                y = np.array([max(p[1], 5e-7) for p in pts])
                col, st = STY[nm]
                ax.semilogy(s, y, st, color=col, label=nm, lw=1.6, ms=5)
            ax.grid(True, which='both', alpha=0.3)
            ax.set_ylim(5e-7, 1)
            if r == 0:
                ax.set_title(f'$f_D$ = {fd:.0f} Hz')
            if c == 0:
                ax.set_ylabel(f'{cm}\nBER')
            if r == len(CH) - 1:
                ax.set_xlabel('per-RE SNR [dB]')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'{bw}: uncoded QPSK BER vs SNR, final receiver chain',
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIGS, f'fig_uncoded_{bw}_ber_vs_snr.png')
    fig.savefig(out, dpi=130)
    print('saved', out)
    plt.close(fig)
