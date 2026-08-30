"""Generate tab_final_{NB,WB}.tex sensitivity rows from final_{BW}.json.

Replicates the knee convention of plot_coded.py exactly: first crossing of
log10(BLER) with log10(target) over ascending SNR, log-linear interpolation.
Rows match the report's tab:coded_nb / tab:coded_wb format (AWGN first,
multirow channel blocks, bold minimum per cell, $>$max when the method never
reaches the target inside the swept range).

Usage: python make_tables.py [NB] [WB]     (default: both)
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
METHODS = ['OFDM', 'PCP-guard', 'PCP-orig']
CHANNELS = ['TDL-A', 'TDL-B', 'TDL-C', 'TDL-D']
TGT = 0.10


def knee(pts, tgt=TGT):
    """SNR at which the log-linear BLER interpolation crosses the target (NaN if never bracketed)."""
    if not pts:
        return np.nan
    pts = sorted(pts)
    s = np.array([p[0] for p in pts])
    b = np.clip([p[1]['bler'] for p in pts], 1e-6, 1)
    lb, lt = np.log10(b), np.log10(tgt)
    for i in range(len(s) - 1):
        if (lb[i] - lt) * (lb[i + 1] - lt) <= 0 and lb[i] != lb[i + 1]:
            return float(s[i] + (lt - lb[i]) / (lb[i + 1] - lb[i]) * (s[i + 1] - s[i]))
    # no bracketed crossing: below target everywhere -> flag; above -> >max
    if b.min() < tgt:
        return -np.inf   # rendered as $<$min so it can't pass silently
    return np.nan


def fmt_row(vals):
    """Format one table row: 2-decimal values, bold minimum, $>$max / $<$min flags."""
    reached = [v for v in vals if np.isfinite(v)]
    lo = min(reached) if reached else None
    out = []
    for v in vals:
        if np.isnan(v):
            out.append('$>$max')
        elif v == -np.inf:
            out.append('$<$min')
        elif lo is not None and abs(v - lo) < 5e-3:
            out.append(f'\\textbf{{{v:.2f}}}')
        else:
            out.append(f'{v:.2f}')
    return out


def build(bw):
    """Generate data/tables/tab_final_<bw>.tex from data/final_<bw>.json and print the winner count."""
    d = json.load(open(os.path.join(DATA, f'final_{bw}.json')))
    rec = {}
    for k, v in d.items():
        _bw, cm, fd, snr, nm = k.split('|')
        rec.setdefault((cm, float(fd), nm), []).append((float(snr), v))
    fds = sorted({k[1] for k in rec if k[0] in CHANNELS})

    # AWGN row from the dense 0.2 dB sweep (awgn_fine.json), which is what
    # the report quotes; the adaptive grid samples the steep AWGN knee too
    # coarsely.  Falls back to the adaptive points if the file is absent.
    awgn = {nm: rec.get(('AWGN', 0.0, nm), []) for nm in METHODS}
    fine = os.path.join(DATA, 'awgn_fine.json')
    if os.path.exists(fine):
        fa = json.load(open(fine))
        awgn = {nm: [] for nm in METHODS}
        for k, v in fa.items():
            _bw, cm, fd, snr, nm = k.split('|')
            if _bw == bw and cm == 'AWGN' and nm in awgn:
                awgn[nm].append((float(snr), v))

    lines = ['    AWGN & --- & '
             + ' & '.join(fmt_row([knee(awgn[nm])
                                   for nm in METHODS])) + r' \\']
    for cm in CHANNELS:
        lines.append(r'    \midrule')
        for i, fd in enumerate(fds):
            cells = fmt_row([knee(rec.get((cm, fd, nm), [])) for nm in METHODS])
            head = (f'    \\multirow{{{len(fds)}}}{{*}}{{{cm}}} & {fd:.0f}'
                    if i == 0 else f'     & {fd:.0f}')
            lines.append(head + ' & ' + ' & '.join(cells) + r' \\')

    out = os.path.join(DATA, 'tables', f'tab_final_{bw}.tex')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    win = {m: 0 for m in METHODS}
    tot = 0
    for cm in CHANNELS:
        for fd in fds:
            ks = {nm: knee(rec.get((cm, fd, nm), [])) for nm in METHODS}
            ks = {k: v for k, v in ks.items() if np.isfinite(v)}
            if ks:
                tot += 1
                win[min(ks, key=ks.get)] += 1
    print(f'{bw}: wrote tab_final_{bw}.tex | winners @{TGT:.0%} BLER: '
          + ', '.join(f'{m} {win[m]}/{tot}' for m in METHODS))


if __name__ == '__main__':
    for bw in (sys.argv[1:] or ['NB', 'WB']):
        build(bw)
