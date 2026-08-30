"""AWGN anchor: coded BLER and BER waterfalls, NB and WB.

The anchor's role is calibration: with identical codewords, identical applied
per-RE SNR and no fading, any residual separation is estimator noise, not
waveform behaviour.
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
FIGS = os.path.join(os.path.dirname(HERE), 'figures')
METH = ['OFDM', 'PCP-guard', 'PCP-orig']
STY = {'OFDM': ('tab:green','o-'), 'PCP-guard': ('tab:blue','s--'),
       'PCP-orig': ('tab:purple','d--')}
# The dense 0.2 dB sweep (awgn_fine.json) resolves the LDPC knee, which the
# adaptive grid in final_*.json samples too coarsely to render; fall back to
# the coarse data only if the dense run is absent.
fine = {}
fp = os.path.join(DATA, 'awgn_fine.json')
if os.path.exists(fp):
    fine = json.load(open(fp))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
for ax, tag in zip(axes, ('final_NB', 'final_WB')):
    bw = tag.split('_')[1]
    d = json.load(open(os.path.join(DATA, tag + '.json')))
    src = {k: v for k, v in fine.items() if k.startswith(f'{bw}|AWGN|')} or d
    rec = {}
    for k, v in src.items():
        _, cm, fd, snr, nm = k.split('|')
        if cm != 'AWGN':
            continue
        rec.setdefault(nm, []).append((float(snr), v))
    for nm in METH:
        pts = sorted(rec.get(nm, []))
        if not pts:
            continue
        s = np.array([p[0] for p in pts])
        y = np.array([p[1]['bler'] for p in pts])
        y = np.where(y > 0, y, np.nan)
        ax.semilogy(s, y, STY[nm][1], color=STY[nm][0], label=nm, lw=1.7, ms=6)
    ax.axhline(0.10, color='gray', ls=':', lw=0.9)
    ax.set_title(f"{tag.split('_')[1]}: AWGN anchor")
    ax.set_xlabel('per-RE SNR [dB]'); ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim(1e-3, 1.1)
axes[0].set_ylabel('BLER'); axes[0].legend(fontsize=9)
fig.suptitle('AWGN anchor: coded BLER vs SNR (QPSK, NR LDPC r=1/2, iso-block) '
             '-- dotted line: 10% BLER target', fontsize=11)
fig.tight_layout()
out = os.path.join(FIGS, 'fig_coded_AWGN.png')
fig.savefig(out, dpi=130); print('saved', out)
