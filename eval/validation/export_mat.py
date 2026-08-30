"""Export the evaluation results to .mat for MATLAB postprocessing.

The SNR grids are adaptive (ragged per cell), so the natural MATLAB layout is
LONG FORMAT: parallel column vectors, one row per measured point.  Filter and
pivot in MATLAB with logical indexing, e.g.

    S = load('coded_eval_v2.mat');
    m = strcmp(S.method,'OFDM') & strcmp(S.channel,'TDL-C') & S.fd==1000 ...
        & strcmp(S.bw,'WB');
    semilogy(S.snr_dB(m), S.bler(m), '-o');

Contents
--------
coded_eval_v2.mat     bw, channel, fd, snr_dB, method, ber, bler, bits, blocks
                      + summary arrays: sens10 (SNR @10% BLER) over a regular
                      [bw x channel x fd x method] grid with axis label cells
uncoded_ref_v2.mat    bw, channel, fd, snr_dB, method, ber, bits
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import os, sys, json
import numpy as np
from scipy.io import savemat

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = sys.argv[1] if len(sys.argv) > 1 else HERE
SUFFIX = sys.argv[2] if len(sys.argv) > 2 else ""   # e.g. "_filteredchain" to test


def cellstr(lst):
    """MATLAB cell-array literal from a python string list."""
    return np.array(lst, dtype=object)


def knee(snrs, bler, tgt=0.10):
    """SNR at which the log-linear BLER interpolation crosses the target (NaN if never bracketed)."""
    s = np.asarray(snrs, float)
    b = np.clip(np.asarray(bler, float), 1e-6, 1.0)
    lb, lt = np.log10(b), np.log10(tgt)
    for i in range(len(s) - 1):
        if (lb[i] - lt) * (lb[i + 1] - lt) <= 0 and lb[i] != lb[i + 1]:
            return float(s[i] + (lt - lb[i]) / (lb[i + 1] - lb[i]) * (s[i + 1] - s[i]))
    return np.nan


# ── coded ──
rows = []
for b in ('NB', 'WB'):
    f = os.path.join(HERE, f'final_{b}{SUFFIX}.json')
    if not os.path.exists(f):
        print(f"skip (missing): {f}")
        continue
    for k, v in json.load(open(f)).items():
        bw, cm, fd, snr, nm = k.split('|')
        rows.append((bw, cm, float(fd), float(snr), nm,
                     v['ber'], v['bler'], v['bits'], v['blocks']))
if rows:
    rows.sort()
    out = dict(
        bw=cellstr([r[0] for r in rows]),
        channel=cellstr([r[1] for r in rows]),
        fd=np.array([r[2] for r in rows]),
        snr_dB=np.array([r[3] for r in rows]),
        method=cellstr([r[4] for r in rows]),
        ber=np.array([r[5] for r in rows]),
        bler=np.array([r[6] for r in rows]),
        bits=np.array([r[7] for r in rows], dtype=np.int64),
        blocks=np.array([r[8] for r in rows], dtype=np.int64),
    )
    # regular-grid sensitivity summary
    bws = sorted({r[0] for r in rows})
    chans = ['AWGN', 'TDL-A', 'TDL-B', 'TDL-C', 'TDL-D']
    fds = sorted({r[2] for r in rows})
    meths = sorted({r[4] for r in rows})
    sens = np.full((len(bws), len(chans), len(fds), len(meths)), np.nan)
    cell = {}
    for r in rows:
        cell.setdefault((r[0], r[1], r[2], r[4]), []).append((r[3], r[6]))
    for (b, cm, fd, nm), pts in cell.items():
        pts.sort()
        sens[bws.index(b), chans.index(cm), fds.index(fd), meths.index(nm)] = \
            knee([p[0] for p in pts], [p[1] for p in pts])
    out.update(sens10_dB=sens, sens10_axes_bw=cellstr(bws),
               sens10_axes_channel=cellstr(chans),
               sens10_axes_fd=np.array(fds), sens10_axes_method=cellstr(meths),
               readme=("long-format vectors; sens10_dB is SNR at 10% BLER on "
                       "[bw x channel x fd x method]; NaN = no crossing "
                       "in swept range"))
    p = os.path.join(OUTDIR, f'coded_eval_v2{SUFFIX}.mat')
    savemat(p, out, do_compression=True)
    print(f"wrote {p}  ({len(rows)} points)")

# ── uncoded ──
f = os.path.join(HERE, f'uncoded_ref{SUFFIX}.json')
if os.path.exists(f):
    rows = []
    for k, v in json.load(open(f)).items():
        bw, cm, fd, snr, nm = k.split('|')
        rows.append((bw, cm, float(fd), float(snr), nm, v['ber'], v['bits']))
    rows.sort()
    out = dict(
        bw=cellstr([r[0] for r in rows]),
        channel=cellstr([r[1] for r in rows]),
        fd=np.array([r[2] for r in rows]),
        snr_dB=np.array([r[3] for r in rows]),
        method=cellstr([r[4] for r in rows]),
        ber=np.array([r[5] for r in rows]),
        bits=np.array([r[6] for r in rows], dtype=np.int64),
        readme="uncoded hard-decision QPSK BER, final receiver chain",
    )
    p = os.path.join(OUTDIR, f'uncoded_ref_v2{SUFFIX}.mat')
    savemat(p, out, do_compression=True)
    print(f"wrote {p}  ({len(rows)} points)")
else:
    print(f"skip (missing): {f}")

# ── round-trip self-check ──
from scipy.io import loadmat
p = os.path.join(OUTDIR, f'coded_eval_v2{SUFFIX}.mat')
if os.path.exists(p):
    d = loadmat(p, squeeze_me=True)
    n = len(d['snr_dB'])
    m = (d['method'] == 'OFDM')
    print(f"round-trip OK: {n} rows, OFDM rows={int(np.sum(m))}, "
          f"sens10 shape={d['sens10_dB'].shape}, "
          f"nan-frac={np.mean(np.isnan(d['sens10_dB'])):.2f}")
