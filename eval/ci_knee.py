"""95% confidence half-widths on the 10%-BLER sensitivity knees.

Parametric bootstrap: at every measured SNR point the block-error count is
resampled as Binomial(blocks, bler), the knee is recomputed with the exact
convention of plots/make_tables.py (first log-linear crossing of the 10%
target over ascending SNR), and the half-width is taken as half the
2.5-97.5 percentile span of the resampled knees.  Prints per-cell values
and the per-bandwidth median that the report quotes.

Usage: python eval/ci_knee.py [NB] [WB]
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-30"
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
METHODS = ['OFDM', 'PCP-guard', 'PCP-orig']
CHANNELS = ['TDL-A', 'TDL-B', 'TDL-C', 'TDL-D']
TGT = 0.10
NBOOT = 1000


def knee(snrs, blers, tgt=TGT):
    """First log-linear crossing of the target over ascending SNR (NaN if none)."""
    lb = np.log10(np.clip(blers, 1e-6, 1.0))
    lt = np.log10(tgt)
    for i in range(len(snrs) - 1):
        if (lb[i] - lt) * (lb[i + 1] - lt) <= 0 and lb[i] != lb[i + 1]:
            return snrs[i] + (lt - lb[i]) / (lb[i + 1] - lb[i]) * (snrs[i + 1] - snrs[i])
    return np.nan


def main(bws):
    rng = np.random.default_rng(7)
    for bw in bws:
        d = json.load(open(os.path.join(DATA, f'final_{bw}.json')))
        rec = {}
        for k, v in d.items():
            _bw, cm, fd, snr, nm = k.split('|')
            rec.setdefault((cm, float(fd), nm), []).append(
                (float(snr), v['bler'], v['blocks']))
        half = []
        print(f"\n{bw}: bootstrap 95% half-widths on the {TGT:.0%}-BLER knee "
              f"({NBOOT} resamples/cell)")
        for cm in CHANNELS:
            for fd in (0, 200, 400, 600, 800, 1000):
                for nm in METHODS:
                    pts = sorted(rec.get((cm, float(fd), nm), []))
                    if not pts:
                        continue
                    snrs = np.array([p[0] for p in pts])
                    bler = np.array([p[1] for p in pts])
                    blks = np.array([max(p[2], 1) for p in pts])
                    if np.isnan(knee(snrs, bler)):
                        continue          # no crossing in the swept range
                    ks = []
                    for _ in range(NBOOT):
                        bb = rng.binomial(blks, np.clip(bler, 0, 1)) / blks
                        kk = knee(snrs, bb)
                        if not np.isnan(kk):
                            ks.append(kk)
                    if len(ks) < NBOOT * 0.9:
                        print(f"  {cm} fd={fd:4d} {nm:10s}: crossing unstable "
                              f"({NBOOT - len(ks)} resamples lost)")
                        continue
                    lo, hi = np.percentile(ks, [2.5, 97.5])
                    hw = (hi - lo) / 2
                    half.append(hw)
                    print(f"  {cm} fd={fd:4d} {nm:10s}: +-{hw:.2f} dB")
        half = np.array(half)
        print(f"{bw}: median +-{np.median(half):.2f} dB, "
              f"90th pct +-{np.percentile(half, 90):.2f}, "
              f"max +-{half.max():.2f}  ({len(half)} knees)")


if __name__ == '__main__':
    main(sys.argv[1:] or ['NB', 'WB'])
