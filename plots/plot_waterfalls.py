"""BLER-vs-SNR and coded-BER-vs-SNR waterfalls from the final coded sweep.

Grid: rows = TDL profiles, cols = selected Dopplers.  Within a cell every
method shares the same SNR points (the fine set is the union of all methods'
knee brackets plus the anchors), so the curves are directly comparable.
Knee-bracket points carry up to 2000 frames (BLER floor 5e-4); anchor points
carry 300 (floor 3.3e-3) -- both floors marked.
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
FIGS = os.path.join(os.path.dirname(HERE), 'figures')
TAG = sys.argv[1] if len(sys.argv) > 1 else 'final_NB'
BW = TAG.split('_')[-1]
d = json.load(open(os.path.join(DATA, TAG + '.json')))

rec = {}
for k, v in d.items():
    bw, cm, fd, snr, nm = k.split('|')
    rec.setdefault((cm, float(fd), nm), []).append((float(snr), v))
for k in rec:
    rec[k].sort()

METHODS = ['OFDM', 'PCP-guard', 'PCP-orig']
STY = {'OFDM': ('tab:green', 'o-'),
       'PCP-guard': ('tab:blue', 's--'), 'PCP-orig': ('tab:purple', 'd--')}
CH = ['TDL-A', 'TDL-B', 'TDL-C', 'TDL-D']
FDS_SEL = [0.0, 600.0, 1000.0]

for metric, ylab, floor_note in [('bler', 'BLER (PER)', True),
                                 ('ber', 'coded BER', False)]:
    fig, axes = plt.subplots(len(CH), len(FDS_SEL),
                             figsize=(4.1 * len(FDS_SEL), 3.1 * len(CH)),
                             sharex=True, sharey=True)
    for r, cm in enumerate(CH):
        for q, fd in enumerate(FDS_SEL):
            ax = axes[r, q]
            for nm in METHODS:
                c, st = STY[nm]
                pts = rec.get((cm, fd, nm), [])
                if not pts:
                    continue
                s = np.array([p[0] for p in pts])
                if metric == 'bler':
                    y = np.array([p[1]['bler'] for p in pts])
                    yf = np.array([0.5 / max(p[1]['blocks'], 1) for p in pts])
                else:
                    y = np.array([p[1]['ber'] for p in pts])
                    yf = np.array([0.5 / max(p[1]['bits'], 1) for p in pts])
                # a zero-error point is an upper limit, not a measurement;
                # plotting it at its MC floor fakes a hook when budgets differ
                y = np.where(y > 0, y, np.nan)
                ax.semilogy(s, y, st, color=c, label=nm, lw=1.6, ms=5)
            if metric == 'bler':
                ax.axhline(0.10, color='gray', ls=':', lw=0.9)
                # 2000 blocks/point -> 1e-3 is ~2 block errors; below that is
                # statistically empty at this budget
                ax.set_ylim(1e-3, 1.1)
            else:
                ax.set_ylim(1e-4, 0.6)
            ax.grid(True, which='both', alpha=0.3)
            if r == 0:
                ax.set_title(f'$f_D$ = {fd:.0f} Hz')
            if q == 0:
                ax.set_ylabel(f'{cm}\n{ylab}')
            if r == len(CH) - 1:
                ax.set_xlabel('per-RE SNR [dB]')
    axes[0, 0].legend(fontsize=8, loc='lower left')
    extra = '  (dotted line: 10% BLER target)' if floor_note else ''
    fig.suptitle(f'{BW}: {ylab} vs SNR -- QPSK, 5G-NR LDPC r=1/2, iso-block, '
                 f'realistic CSI + noise estimation{extra}', fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIGS, f'fig_{TAG}_{metric}_vs_snr.png')
    fig.savefig(out, dpi=130)
    print('saved', out)
