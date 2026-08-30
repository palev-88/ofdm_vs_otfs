"""Figures from the adaptive coded sweep: BLER/BER vs SNR, and vs Doppler."""
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
CH = [c for c in ['TDL-A', 'TDL-B', 'TDL-C', 'TDL-D']
      if any(k[0] == c for k in rec)]
FDS = sorted({k[1] for k in rec if k[0] in CH})
ANCH = [4.0, 8.0, 12.0]


def knee(pts, tgt):
    """SNR at which the log-linear BLER interpolation crosses the target (NaN if never bracketed)."""
    s = np.array([p[0] for p in pts]); b = np.clip([p[1]['bler'] for p in pts], 1e-6, 1)
    lb, lt = np.log10(b), np.log10(tgt)
    for i in range(len(s) - 1):
        if (lb[i] - lt) * (lb[i + 1] - lt) <= 0 and lb[i] != lb[i + 1]:
            return float(s[i] + (lt - lb[i]) / (lb[i + 1] - lb[i]) * (s[i + 1] - s[i]))
    return np.nan


# ---- Fig 1: SNR @ 10% BLER vs Doppler, one panel per channel ----
fig, axes = plt.subplots(1, len(CH), figsize=(4.1 * len(CH), 4.2), sharey=True)
axes = np.atleast_1d(axes)
for ax, cm in zip(axes, CH):
    for nm in METHODS:
        c, st = STY[nm]
        y = [knee(rec.get((cm, fd, nm), []), 0.10) for fd in FDS]
        ax.plot(FDS, y, st, color=c, label=nm, lw=1.7, ms=6)
    ax.set_title(cm); ax.set_xlabel('$f_D$ [Hz]'); ax.grid(alpha=0.3)
axes[0].set_ylabel('SNR for BLER = 10%  [dB]')
axes[0].legend(fontsize=8)
fig.suptitle(f'{BW}: SNR required for 10% BLER vs Doppler '
             f'(QPSK, 5G-NR LDPC r=1/2, iso-block, realistic CSI)', fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, f'fig_{TAG}_snr_vs_fd.png'), dpi=130)

# ---- Fig 2: BLER vs Doppler at the anchor SNRs ----
fig, axes = plt.subplots(len(ANCH), len(CH), figsize=(4.0 * len(CH), 3.3 * len(ANCH)),
                         sharex=True, sharey=True)
axes = np.atleast_2d(axes)
for r, a in enumerate(ANCH):
    for q, cm in enumerate(CH):
        ax = axes[r, q]
        for nm in METHODS:
            c, st = STY[nm]
            y = []
            for fd in FDS:
                pts = dict((p[0], p[1]) for p in rec.get((cm, fd, nm), []))
                y.append(pts[a]['bler'] if a in pts else np.nan)
            ax.semilogy(FDS, np.maximum(y, 2e-3), st, color=c, label=nm, lw=1.6, ms=5)
        ax.grid(alpha=0.3, which='both')
        if r == 0:
            ax.set_title(cm)
        if q == 0:
            ax.set_ylabel(f'BLER @ {a:.0f} dB')
        if r == len(ANCH) - 1:
            ax.set_xlabel('$f_D$ [Hz]')
axes[0, 0].legend(fontsize=7.5)
fig.suptitle(f'{BW}: BLER (PER) vs Doppler at fixed SNR', fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, f'fig_{TAG}_bler_vs_fd.png'), dpi=130)

# ---- Fig 3: coded BER vs Doppler at anchors ----
fig, axes = plt.subplots(len(ANCH), len(CH), figsize=(4.0 * len(CH), 3.3 * len(ANCH)),
                         sharex=True, sharey=True)
axes = np.atleast_2d(axes)
for r, a in enumerate(ANCH):
    for q, cm in enumerate(CH):
        ax = axes[r, q]
        for nm in METHODS:
            c, st = STY[nm]
            y = []
            for fd in FDS:
                pts = dict((p[0], p[1]) for p in rec.get((cm, fd, nm), []))
                y.append(pts[a]['ber'] if a in pts else np.nan)
            ax.semilogy(FDS, np.maximum(y, 1e-6), st, color=c, label=nm, lw=1.6, ms=5)
        ax.grid(alpha=0.3, which='both')
        if r == 0:
            ax.set_title(cm)
        if q == 0:
            ax.set_ylabel(f'coded BER @ {a:.0f} dB')
        if r == len(ANCH) - 1:
            ax.set_xlabel('$f_D$ [Hz]')
axes[0, 0].legend(fontsize=7.5)
fig.suptitle(f'{BW}: coded BER vs Doppler at fixed SNR', fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, f'fig_{TAG}_ber_vs_fd.png'), dpi=130)

# ---- winner count at 10% BLER ----
print(f"\n{BW}: winner count at 10% BLER (lowest required SNR)")
win = {m: 0 for m in METHODS}
for cm in CH:
    for fd in FDS:
        ks = {nm: knee(rec.get((cm, fd, nm), []), 0.10) for nm in METHODS}
        ks = {k: v for k, v in ks.items() if not np.isnan(v)}
        if ks:
            win[min(ks, key=ks.get)] += 1
tot = sum(win.values())
for m in METHODS:
    print(f"  {m:10s} {win[m]:3d}/{tot}  ({win[m]/max(tot,1)*100:5.1f}%)")
print(f"saved fig_{TAG}_*.png")
