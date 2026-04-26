"""
plot_eval_v1.py  --  generate all final-evaluation figures + summary tables
from results/otfs_eval_v1.json.

Outputs (all under results/figures/):
  fig_eval_BERvsSNR_NB.png       4 channels x 4 Dopplers grid, 4 methods
  fig_eval_BERvsSNR_WB.png       same grid for WB
  fig_eval_BERvsFD_NB.png        4 channels x 2 SNR rows (20/30 dB), BER vs fd
  fig_eval_BERvsFD_WB.png        same for WB
  fig_eval_AWGN.png              both BW cases side by side (calibration anchor)
  fig_eval_winner_heatmap.png    NB+WB heatmap, lowest-BER method per cell
  tab_eval_BERat20dB.csv         BER@20 dB per (bw, channel, fd, method)
  tab_eval_SNRforBER.csv         SNR required to hit BER=1e-3 per (bw, channel, fd, method)
  tab_eval_winner.csv            method with lowest BER per (bw, channel, fd) at SNR=20 dB

Also prints a human-readable summary of the winner matrix for each BW case.

Usage
-----
    python plot_results.py

Reads ``results/otfs_eval_v1.json`` (produced by the evaluation driver) and
writes all figures (PNG) and CSV digests into ``results/figures/`` (created
if missing). No command-line arguments — paths are hard-coded near the top.
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "CC-BY-4.0"
__copyright__ = "(c) 2026 Panos N. Alevizos"

import json, os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')                  # non-interactive backend (writes PNGs only)
import matplotlib.pyplot as plt
from collections import defaultdict


# ════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════

RESULTS_FILE = 'results/otfs_eval_v1.json'   # input: aggregated Monte-Carlo results
OUTDIR       = 'results/figures'             # output: figures + CSV digests
os.makedirs(OUTDIR, exist_ok=True)


# Ordered list of waveform/receiver methods compared in every plot/table.
# The order also drives the legend ordering and the column order of the CSVs.
METHODS  = ['ofdm', 'zp', 'pcp_guard', 'pcp_orig']

# Per-method visual style — kept consistent across every figure so the reader
# can recognise a method at a glance regardless of which subplot they look at.
COLORS   = {'ofdm':'#2ca02c', 'zp':'#1f77b4',           # green / blue
            'pcp_guard':'#d62728', 'pcp_orig':'#ff7f0e'} # red / orange
LABELS   = {'ofdm':'OFDM', 'zp':'ZP-OTFS',
            'pcp_guard':'PCP-guard', 'pcp_orig':'PCP-orig'}
MARKERS  = {'ofdm':'s', 'zp':'D',                       # square / diamond
            'pcp_guard':'o', 'pcp_orig':'^'}            # circle / triangle


# ════════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════════

def load():
    """Load the aggregated evaluation JSON and reshape it into a lookup table.

    Returns
    -------
    d : dict
        Raw JSON payload (full file contents) — kept for callers that need
        access to fields not surfaced through the convenience returns.
    snr_lst : list of float
        Ordered list of SNR points (dB) common to every (bw, method, channel,
        fd) cell.
    tbl : dict
        Lookup table keyed by ``(bw_case, method, channel, fd)`` whose value
        is a 1-D ``np.ndarray`` of BERs aligned with ``snr_lst``.
    bw_cases : list of str
        Bandwidth labels, e.g. ``['NB', 'WB']``.
    channels : list of str
        Channel-model labels, e.g. ``['AWGN', 'TDL-A', ..., 'TDL-E']``.
    dopplers : list of int
        Doppler shift values (Hz) sweep, e.g. ``[0, 50, 100, ...]``.

    Notes
    -----
    The flat list-of-records layout in the JSON is convenient for the eval
    driver but awkward for plotting; we re-key into ``tbl`` so figure
    generators can pull a single curve in O(1).
    """
    with open(RESULTS_FILE) as f:
        d = json.load(f)
    snr_lst = d['snr_list']
    res     = d['results']
    # table[(bw, method, channel, fd)] = ber list
    tbl = {}
    for r in res:
        key = (r['bw_case'], r['method'], r['channel'], r['fd'])
        tbl[key] = np.array(r['ber'], dtype=float)
    bw_cases = list(d['configs'].keys())
    channels = d['channels']
    dopplers = d['dopplers']
    return d, snr_lst, tbl, bw_cases, channels, dopplers


def snr_for_target(snrs, bers, target=1e-3):
    """Linear-in-log interpolation of SNR to achieve BER=target.

    Parameters
    ----------
    snrs : array_like
        SNR sweep points (dB), monotonically increasing.
    bers : array_like
        Measured BER at each SNR point (same length as ``snrs``).
    target : float, optional
        Target BER threshold (default ``1e-3``).

    Returns
    -------
    float
        Interpolated SNR (dB) at which the curve first crosses ``target``,
        or ``nan`` if the curve never reaches it within the swept range.

    Notes
    -----
    Interpolation is linear in ``log10(BER)`` vs. SNR (i.e. straight line on
    the standard semilogy BER plot). Zero BERs are floored to 1e-6 before
    taking the log to avoid ``-inf``; this matches the BER floor used
    throughout the rest of the script for visualisation.
    """
    bers = np.asarray(bers, dtype=float)
    snrs = np.asarray(snrs, dtype=float)
    # Replace zeros with small floor for log.
    bers_log = np.log10(np.maximum(bers, 1e-6))
    target_log = np.log10(target)
    # Find first index where BER <= target.
    below = np.where(bers_log <= target_log)[0]
    if len(below) == 0: return float('nan')
    i = below[0]
    if i == 0: return float(snrs[0])
    # Interp between i-1 and i.
    x0, x1 = bers_log[i-1], bers_log[i]
    if x0 == x1: return float(snrs[i])           # degenerate flat segment — pick right edge
    t = (target_log - x0) / (x1 - x0)            # fractional position in log-BER between i-1 and i
    return float(snrs[i-1] + t * (snrs[i] - snrs[i-1]))


# ════════════════════════════════════════════════════════════════════════════
# Figure generators
# ════════════════════════════════════════════════════════════════════════════

def plot_ber_vs_snr(snr_lst, tbl, bw_case, channels, dopplers_show,
                    title_suffix, outname):
    """4 channels (rows) x |dopplers_show| cols; 4 curves per subplot.

    Parameters
    ----------
    snr_lst : list of float
        SNR sweep (dB) shared across every cell.
    tbl : dict
        Result table from :func:`load`.
    bw_case : str
        Bandwidth case label to filter on (e.g. ``'NB'`` or ``'WB'``).
    channels : list of str
        Channel labels to place along the rows (typically the 5 TDL profiles
        excluding AWGN, which has its own dedicated figure).
    dopplers_show : list of int
        Doppler shifts (Hz) to place along the columns. The grid width is
        chosen by the caller (see ``main`` — a 4-Doppler subset is used so
        the figure stays readable on a report page).
    title_suffix : str
        Free-text suffix appended to the figure super-title.
    outname : str
        File name (relative to ``OUTDIR``) for the saved PNG.

    Returns
    -------
    None
        The figure is saved to disk and closed.

    Notes
    -----
    * Sub-axes share both x and y so the eye can compare BER curves across
      Doppler/channel combinations on the same scale.
    * Axis limits ``[0, 30]`` dB on x and ``[1e-5, 1]`` on y were chosen to
      cover the operating regime of interest (link budgets above 0 dB SNR,
      BERs from random guess down to the Monte-Carlo floor).
    * "All-zero" curves (no errors observed in any MC trial — typically
      AWGN-like high-SNR points) are drawn at a 1e-6 floor with a dotted
      line so the method is still visible in the legend region.
    """
    n_ch = len(channels)
    n_fd = len(dopplers_show)
    # 3.2" per Doppler col / 2.8" per channel row keeps the panels close to
    # square at the typical 4x4 grid used in the report.
    fig, axes = plt.subplots(n_ch, n_fd,
                             figsize=(3.2 * n_fd, 2.8 * n_ch),
                             sharex=True, sharey=True, squeeze=False)
    for i, ch in enumerate(channels):
        for j, fd in enumerate(dopplers_show):
            ax = axes[i, j]
            for m in METHODS:
                bers = tbl.get((bw_case, m, ch, fd))
                if bers is None: continue
                mask = bers > 0
                if not mask.any():
                    # all-zero — plot at 1e-6 floor for visibility
                    ax.semilogy(snr_lst, np.full_like(bers, 1e-6),
                                marker=MARKERS[m], color=COLORS[m],
                                label=LABELS[m], ls=':', lw=1.5, ms=5)
                else:
                    # Skip zero-BER points (MC-floor artefacts) but keep the
                    # rest of the curve so the trend remains visible.
                    ax.semilogy([snr_lst[k] for k in range(len(snr_lst)) if mask[k]],
                                bers[mask],
                                marker=MARKERS[m], color=COLORS[m],
                                label=LABELS[m], lw=1.8, ms=6)
            ax.grid(True, which='both', alpha=0.3)
            ax.set_xlim([0, 30]); ax.set_ylim([1e-5, 1])    # SNR window / BER dynamic range
            if i == 0: ax.set_title(f'$f_D = {fd}$ Hz', fontsize=10)
            if j == 0: ax.set_ylabel(f'{ch}\nBER', fontsize=9)
            if i == n_ch - 1: ax.set_xlabel('SNR (dB)')
            if i == 0 and j == 0: ax.legend(loc='lower left', fontsize=7.5)  # one legend for the whole grid
    fig.suptitle(f'BER vs SNR -- {title_suffix}', fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.97])              # leave 3% headroom for suptitle
    out = os.path.join(OUTDIR, outname)
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out}')


def plot_ber_vs_fd(snr_lst, tbl, bw_case, channels, dopplers,
                   target_snrs, title_suffix, outname):
    """|channels| cols, curves per (method, SNR); one combined subplot row.

    Parameters
    ----------
    snr_lst : list of float
        Full SNR sweep (dB) — used to look up the column index of each
        target SNR.
    tbl : dict
        Result table from :func:`load`.
    bw_case : str
        Bandwidth case to filter on.
    channels : list of str
        Channels to lay out across columns.
    dopplers : list of int
        x-axis sweep — the full Doppler set is used here because the panels
        are wide enough to accommodate it.
    target_snrs : list of float
        SNR slices to overlay (typically two: a "moderate" and a "high"
        anchor — drawn solid and dotted respectively).
    title_suffix : str
        Free-text suffix appended to the super-title.
    outname : str
        Output PNG name (relative to ``OUTDIR``).

    Returns
    -------
    None
        Figure saved to disk and closed.

    Notes
    -----
    The 4x2 layout (channel x SNR) is encoded by drawing two curves per
    method per panel — one per SNR slice — distinguished by linestyle. A
    custom two-part legend (method colour + SNR linestyle) is built by hand
    to keep the legend compact.
    """
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, len(channels),
                             figsize=(3.3 * len(channels), 3.5),
                             sharey=True, squeeze=False)
    # Map each requested SNR slice to a linestyle: solid for the lower
    # anchor, dotted for the higher anchor (see suptitle for the legend).
    linestyles = {target_snrs[0]: '-', target_snrs[-1]: ':'}
    for j, ch in enumerate(channels):
        ax = axes[0, j]
        for m in METHODS:
            for snr_t in target_snrs:
                si = snr_lst.index(snr_t) if snr_t in snr_lst else None
                if si is None: continue
                xs, ys = [], []
                for fd in dopplers:
                    bers = tbl.get((bw_case, m, ch, fd))
                    if bers is None: continue
                    # Floor BER at 1e-6 so log axis can plot zero-error MC
                    # cells (otherwise the curve would have gaps).
                    xs.append(fd); ys.append(max(bers[si], 1e-6))
                if not xs: continue
                ax.semilogy(xs, ys,
                            marker=MARKERS[m], color=COLORS[m],
                            ls=linestyles.get(snr_t, '--'),
                            lw=1.6, ms=5)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_ylim([1e-5, 1])
        ax.set_title(ch, fontsize=10)
        ax.set_xlabel('$f_D$ (Hz)')
        if j == 0:
            ax.set_ylabel('BER')
            # Two-part legend: methods + SNR linestyle
            method_handles = [
                Line2D([0], [0], color=COLORS[m], marker=MARKERS[m],
                       lw=1.6, ms=5, label=LABELS[m])
                for m in METHODS
            ]
            style_handles = [
                Line2D([0], [0], color='black', ls='-',  lw=1.6,
                       label=f'{int(target_snrs[0])} dB'),
                Line2D([0], [0], color='black', ls=':',  lw=1.6,
                       label=f'{int(target_snrs[-1])} dB'),
            ]
            # add_artist preserves the first legend so the second call
            # doesn't replace it (matplotlib's default behaviour).
            leg1 = ax.legend(handles=method_handles, loc='lower right',
                             fontsize=7, title='method',
                             title_fontsize=7)
            ax.add_artist(leg1)
            ax.legend(handles=style_handles, loc='upper left',
                      fontsize=7, title='SNR', title_fontsize=7)
    fig.suptitle(f'BER vs Doppler -- {title_suffix}'
                 + f'  (solid = {int(target_snrs[0])} dB, '
                 + f'dotted = {int(target_snrs[-1])} dB)', fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.93])              # extra room for the longer suptitle
    out = os.path.join(OUTDIR, outname)
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out}')


def plot_awgn(snr_lst, tbl, bw_cases):
    """Plot the AWGN-only BER curve for every BW case as a calibration anchor.

    Parameters
    ----------
    snr_lst : list of float
        SNR sweep (dB).
    tbl : dict
        Result table from :func:`load`.
    bw_cases : list of str
        Bandwidth labels — one panel per BW case, side by side.

    Returns
    -------
    None
        Figure saved to ``fig_eval_AWGN.png`` and closed.

    Notes
    -----
    The AWGN curves should land on (or very near) the analytical BER curve
    of the underlying modulation; this figure is the visual sanity check
    that noise calibration is correct across all four methods.
    """
    fig, axes = plt.subplots(1, len(bw_cases),
                             figsize=(4.5 * len(bw_cases), 3.5),
                             sharey=True, squeeze=False)
    for j, bw in enumerate(bw_cases):
        ax = axes[0, j]
        for m in METHODS:
            bers = tbl.get((bw, m, 'AWGN', 0))
            if bers is None: continue
            # Zero-BER points are Monte-Carlo floor artefacts (no errors
            # observed, not a true measurement); replace with NaN so they
            # are not drawn as a spurious "error floor".
            ys = np.where(np.asarray(bers, dtype=float) > 0,
                          np.asarray(bers, dtype=float), np.nan)
            ax.semilogy(snr_lst, ys,
                        marker=MARKERS[m], color=COLORS[m],
                        label=LABELS[m], lw=1.8, ms=6)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_xlim([0, 30]); ax.set_ylim([1e-6, 1])
        ax.set_title(f'{bw} case -- AWGN', fontsize=11)
        ax.set_xlabel('SNR (dB)')
        if j == 0:
            ax.set_ylabel('BER')
            ax.legend(loc='lower left', fontsize=9)
    fig.suptitle('AWGN anchor -- validates noise calibration',
                 fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.94])
    out = os.path.join(OUTDIR, 'fig_eval_AWGN.png')
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out}')


# ════════════════════════════════════════════════════════════════════════════
# CSV digest writers
# ════════════════════════════════════════════════════════════════════════════

def write_ber20_table(snr_lst, tbl, bw_cases, channels, dopplers):
    """Write the BER@20 dB digest CSV.

    Parameters
    ----------
    snr_lst : list of float
        SNR sweep (dB) — must contain ``20.0`` for the table to be written.
    tbl : dict
        Result table from :func:`load`.
    bw_cases, channels, dopplers : list
        Iteration domains for the CSV rows.

    Returns
    -------
    None
        CSV written to ``tab_eval_BERat20dB.csv``; nothing is returned.

    Notes
    -----
    Columns: ``bw, channel, fd, OFDM, ZP-OTFS, PCP-guard, PCP-orig``. AWGN
    rows are emitted only at ``fd=0`` since Doppler is meaningless without
    a multipath profile.
    """
    out = os.path.join(OUTDIR, 'tab_eval_BERat20dB.csv')
    if 20.0 not in snr_lst:
        print(f'  20 dB not in SNR list; skipping BER@20 table')
        return
    si = snr_lst.index(20.0)
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bw', 'channel', 'fd'] + [LABELS[m] for m in METHODS])
        for bw in bw_cases:
            for ch in channels:
                # AWGN doesn't depend on fd → emit a single row at fd=0.
                fds = [0] if ch == 'AWGN' else dopplers
                for fd in fds:
                    row = [bw, ch, fd]
                    for m in METHODS:
                        bers = tbl.get((bw, m, ch, fd))
                        row.append(f'{bers[si]:.4e}' if bers is not None else '')
                    w.writerow(row)
    print(f'  saved {out}')


def write_snrfor_table(snr_lst, tbl, bw_cases, channels, dopplers,
                       target=1e-3):
    """Write the "SNR required to reach target BER" digest CSV.

    Parameters
    ----------
    snr_lst : list of float
        SNR sweep (dB).
    tbl : dict
        Result table from :func:`load`.
    bw_cases, channels, dopplers : list
        Iteration domains.
    target : float, optional
        Target BER (default ``1e-3``); also emitted as the trailing column
        of the CSV so consumers know which threshold was used.

    Returns
    -------
    None
        CSV written to ``tab_eval_SNRforBER.csv``.

    Notes
    -----
    Uses :func:`snr_for_target` for log-linear interpolation. Methods that
    never reach ``target`` within the swept SNR range are emitted as the
    string ``'> 30'`` to make this case obvious in the CSV.
    """
    out = os.path.join(OUTDIR, 'tab_eval_SNRforBER.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bw', 'channel', 'fd'] + [LABELS[m] for m in METHODS]
                   + [f'target_BER={target}'])
        for bw in bw_cases:
            for ch in channels:
                fds = [0] if ch == 'AWGN' else dopplers
                for fd in fds:
                    row = [bw, ch, fd]
                    for m in METHODS:
                        bers = tbl.get((bw, m, ch, fd))
                        if bers is None:
                            row.append(''); continue
                        snr_req = snr_for_target(snr_lst, bers, target)
                        # NaN ⇒ never reached the target within the sweep.
                        row.append(f'{snr_req:.2f}' if not np.isnan(snr_req) else '> 30')
                    row.append(target)
                    w.writerow(row)
    print(f'  saved {out}')


def write_winner_table(snr_lst, tbl, bw_cases, channels, dopplers):
    """At SNR=20 dB pick the lowest-BER method per (bw, channel, fd); also
    note the runner-up and margin.

    Parameters
    ----------
    snr_lst : list of float
        SNR sweep (dB) — must contain ``20.0`` or the function is a no-op.
    tbl : dict
        Result table from :func:`load`.
    bw_cases, channels, dopplers : list
        Iteration domains.

    Returns
    -------
    None
        CSV written to ``tab_eval_winner.csv``; a per-BW summary of win
        counts is also printed to stdout.

    Notes
    -----
    Columns: ``bw, channel, fd, winner, BER_winner, runnerup, BER_runnerup,
    margin_ratio``. The margin is reported as ``BER_runnerup / BER_winner``
    (e.g. ``2.5x`` means the runner-up was 2.5x worse than the winner). To
    avoid declaring a winner solely because its MC sample happened to round
    to zero, all BERs are floored at ``1e-6`` before sorting.
    """
    out = os.path.join(OUTDIR, 'tab_eval_winner.csv')
    if 20.0 not in snr_lst:
        return
    si = snr_lst.index(20.0)
    # Accumulators for summary print.
    win_count = defaultdict(lambda: defaultdict(int))  # [bw][method] -> wins
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['bw', 'channel', 'fd', 'winner', 'BER_winner',
                    'runnerup', 'BER_runnerup', 'margin_ratio'])
        for bw in bw_cases:
            for ch in channels:
                fds = [0] if ch == 'AWGN' else dopplers
                for fd in fds:
                    ranking = []
                    for m in METHODS:
                        bers = tbl.get((bw, m, ch, fd))
                        if bers is None: continue
                        # Use floored BER for sorting so method with
                        # all-zero rows doesn't always "win".
                        v = max(bers[si], 1e-6)
                        ranking.append((v, m))
                    ranking.sort()                       # ascending BER → best first
                    if len(ranking) < 2:
                        continue                         # need at least 2 methods to call a winner
                    (b1, m1), (b2, m2) = ranking[0], ranking[1]
                    margin = b2 / b1 if b1 > 0 else float('inf')
                    win_count[bw][m1] += 1
                    w.writerow([bw, ch, fd, LABELS[m1], f'{b1:.4e}',
                                LABELS[m2], f'{b2:.4e}', f'{margin:.2f}x'])
    print(f'  saved {out}')
    # Print summary
    print('\n=== Winner count (BER @ 20 dB, lowest wins) ===')
    for bw in bw_cases:
        total = sum(win_count[bw].values())
        print(f'  {bw}: {total} cells')
        for m in METHODS:
            c = win_count[bw].get(m, 0)
            if c: print(f'    {LABELS[m]:<10} {c:>3}  ({100*c/total:4.1f}%)')


def plot_winner_heatmap(snr_lst, tbl, bw_cases, tdl_ch, dopplers):
    """2 side-by-side panels (NB/WB); each is a channel x Doppler heatmap
    coloured by winning method at SNR=20 dB; BER value shown in each cell.

    Parameters
    ----------
    snr_lst : list of float
        SNR sweep (dB) — must contain ``20.0`` or the function is a no-op.
    tbl : dict
        Result table from :func:`load`.
    bw_cases : list of str
        Bandwidth cases to draw side by side (one panel each).
    tdl_ch : list of str
        TDL channel labels (AWGN excluded, since it's a degenerate row).
    dopplers : list of int
        Doppler shifts to place along the x-axis of each heatmap.

    Returns
    -------
    None
        Figure saved to ``fig_eval_winner_heatmap.png`` and closed.

    Notes
    -----
    Visual encoding: cell *colour* = winning method, cell *text* = winning
    method's BER (formatted ``1.6e-02`` etc., or ``'0'`` for the
    Monte-Carlo floor). The rendered grid is built by hand (rectangle
    patches + text annotations) rather than ``imshow`` because we want
    discrete categorical colours per method and per-cell text labels.
    """
    if 20.0 not in snr_lst:
        return
    si = snr_lst.index(20.0)

    meth_idx = {m: i for i, m in enumerate(METHODS)}
    # Pale tints of the per-method line colours so the BER annotation text
    # (drawn in black) stays readable on top.
    meth_cmap = {
        'ofdm':      '#a1d99b',   # pale green
        'zp':        '#9ecae1',   # pale blue
        'pcp_guard': '#fcae91',   # pale red
        'pcp_orig':  '#fdd0a2',   # pale orange
    }
    meth_cells = [meth_cmap[m] for m in METHODS]

    fig, axes = plt.subplots(1, len(bw_cases),
                             figsize=(4.8 * len(bw_cases), 3.2),
                             squeeze=False)
    for j, bw in enumerate(bw_cases):
        ax = axes[0, j]
        # Winner matrix (integer indices of method) and BER @ 20 dB matrix.
        W  = np.full((len(tdl_ch), len(dopplers)), -1, dtype=int)
        B  = np.full_like(W, 0, dtype=float)
        for ri, ch in enumerate(tdl_ch):
            for ci, fd in enumerate(dopplers):
                best_m, best_b = None, float('inf')
                for m in METHODS:
                    bers = tbl.get((bw, m, ch, fd))
                    if bers is None: continue
                    v = max(bers[si], 1e-6)              # MC-floor guard (see write_winner_table)
                    if v < best_b:
                        best_b, best_m = v, m
                if best_m is not None:
                    W[ri, ci] = meth_idx[best_m]
                    B[ri, ci] = best_b
        # Paint cells by method colour.
        for ri in range(W.shape[0]):
            for ci in range(W.shape[1]):
                col = meth_cells[W[ri, ci]] if W[ri, ci] >= 0 else 'white'
                ax.add_patch(plt.Rectangle((ci, ri), 1, 1,
                                           facecolor=col, edgecolor='white'))
                # Annotate with BER (consistent format for every cell).
                v = B[ri, ci]
                if v <= 0:
                    txt = '0'
                else:
                    txt = f'{v:.1e}'          # e.g. 4.2e-04, 1.6e-02
                ax.text(ci + 0.5, ri + 0.5, txt,
                        ha='center', va='center', fontsize=7)
        # Manual axis setup (we drew rectangles, not an imshow image, so we
        # must size the axes ourselves). Y is inverted so row 0 (= first
        # channel) appears at the top, matching reading order.
        ax.set_xlim(0, W.shape[1]); ax.set_ylim(W.shape[0], 0)
        ax.set_xticks(np.arange(W.shape[1]) + 0.5)
        ax.set_xticklabels(dopplers, fontsize=8)
        ax.set_yticks(np.arange(W.shape[0]) + 0.5)
        ax.set_yticklabels(tdl_ch, fontsize=9)
        ax.set_xlabel('Doppler $f_D$ (Hz)')
        ax.set_title(f'{bw} case (winners @ SNR = 20 dB)', fontsize=11)
    # Legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=meth_cmap[m], edgecolor='grey',
                            label=LABELS[m]) for m in METHODS]
    # Bottom-centre legend so it doesn't overlap either heatmap panel.
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=len(METHODS), fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Lowest-BER method per (channel, Doppler) cell'
                 ' — cell text shows the winner\'s BER', fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])     # leave room for both legend (bottom) and suptitle (top)
    out = os.path.join(OUTDIR, 'fig_eval_winner_heatmap.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {out}')


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    """Driver: load results, render every figure, write every CSV digest.

    Returns
    -------
    None
        All artefacts are written under ``OUTDIR``. Progress and the winner
        summary are printed to stdout.

    Notes
    -----
    If ``RESULTS_FILE`` does not exist (e.g. the eval driver hasn't run
    yet) the function prints a notice and returns without raising.
    """
    if not os.path.exists(RESULTS_FILE):
        print(f'No {RESULTS_FILE} — eval not finished yet.')
        return
    d, snr_lst, tbl, bw_cases, channels, dopplers = load()
    print(f'Loaded {len(tbl)} result cells from {RESULTS_FILE}')

    # AWGN is plotted in its own dedicated figure — strip it from the per-
    # channel grids so they only show fading channels.
    tdl_ch = [c for c in channels if c != 'AWGN']
    # Restricted Doppler set for the BER-vs-SNR grids: keeps the 4-column
    # layout legible (the full Doppler sweep is still used by the BER-vs-fd
    # figures and the heatmap).
    dopplers_show = [0, 200, 500, 1000]   # restricted set for BER-vs-SNR panels

    # Skip the per-channel grids entirely when no TDL channels are present
    # (e.g. AWGN-only smoke runs).  matplotlib's subplots() rejects a 0-row
    # grid; this guard keeps the rest of the pipeline working on partial data.
    if tdl_ch:
        for bw in bw_cases:
            plot_ber_vs_snr(snr_lst, tbl, bw, tdl_ch, dopplers_show,
                            title_suffix=f'{bw} case',
                            outname=f'fig_eval_BERvsSNR_{bw}.png')
            plot_ber_vs_fd(snr_lst, tbl, bw, tdl_ch, dopplers,
                           target_snrs=[20.0, 30.0],
                           title_suffix=f'{bw} case',
                           outname=f'fig_eval_BERvsFD_{bw}.png')
    else:
        print('  [info] No TDL channels in the merged data — skipping the '
              'BERvsSNR_*, BERvsFD_*, and winner-heatmap figures.')

    plot_awgn(snr_lst, tbl, bw_cases)
    if tdl_ch:
        plot_winner_heatmap(snr_lst, tbl, bw_cases, tdl_ch, dopplers)

    write_ber20_table(snr_lst, tbl, bw_cases, channels, dopplers)
    write_snrfor_table(snr_lst, tbl, bw_cases, channels, dopplers, target=1e-3)
    write_winner_table(snr_lst, tbl, bw_cases, channels, dopplers)

    print('\n  Done.')


if __name__ == '__main__':
    main()
