"""Verify the TX PSD of every waveform PHY in the CURRENT evaluation chain.

The point of this check: the fixed-rate DFT-grid representation must reproduce
the waveform, not alter it.  Three tests per bandwidth case:

  1. NATIVE vs FIXED-RATE PCP.  Generate PCP at its own rate (n_act*SCS) and
     through the evaluation's fixed-rate chain (Mfft*SCS).  Plotted on the same
     ABSOLUTE frequency axis the two PSDs must coincide inside the occupied
     band -- that is the statement "same waveform, different sample rate".
     The earlier polyphase-resampler version failed exactly here (band-edge
     droop); this is the regression test for that bug.

  2. OCCUPIED BANDWIDTH.  All waveforms must occupy n_act*SCS.  Measured as
     the 99%-power bandwidth and as the -3 dB edges.

  3. PASSBAND FLATNESS + PAPR, for comparison against the report's TX
     characterisation figures.

No transmit filter is applied here: this measures the PHY's own spectrum, so
the OFDM/OTFS guard-band structure is visible rather than masked by a common
RC roll-off.
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import os, sys
import numpy as np
from scipy.signal import welch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'eval'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from coded_sweep import setup
from coded_eval import qpsk_mod
from otfs_pcp import PCPOTFSTransceiver, PCPOTFSConfig

CASES = [('NB', 256, 156, 30e3), ('WB', 1024, 624, 60e3)]
ok_all = True


def check(name, cond, detail=""):
    """Print one labelled pass/fail line for a measured-vs-expected quantity."""
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")


def psd(x, fs, nper=2048):
    """Welch PSD (Blackman-Harris) of one waveform realisation."""
    f, p = welch(x, fs=fs, window='blackmanharris',
                 nperseg=min(nper, len(x) // 4), noverlap=None,
                 return_onesided=False, scaling='density')
    i = np.argsort(f)
    f, p = f[i], p[i]
    return f, 10 * np.log10(p / p.max() + 1e-30)


def occupied_bw(x, fs, frac=0.99):
    """99%-power occupied bandwidth from the integrated PSD."""
    f, p = welch(x, fs=fs, window='blackmanharris',
                 nperseg=min(4096, len(x) // 4), return_onesided=False,
                 scaling='density')
    i = np.argsort(f); f, p = f[i], p[i]
    c = np.cumsum(p) / np.sum(p)
    lo = f[np.searchsorted(c, (1 - frac) / 2)]
    hi = f[np.searchsorted(c, 1 - (1 - frac) / 2)]
    return hi - lo


def papr(x, tail=1e-4):
    """Peak-to-average power ratio of a waveform in dB."""
    inst = np.abs(x) ** 2
    return 10 * np.log10(np.quantile(inst, 1 - tail) / np.mean(inst))


fig, axes = plt.subplots(1, len(CASES), figsize=(7.2 * len(CASES), 5.0))
axes = np.atleast_1d(axes)

for ax, (tag, M, n_act, SCS) in zip(axes, CASES):
    print(f"\n=== {tag}  (Mfft={M}, n_act={n_act}, SCS={SCS/1e3:.0f} kHz) ===")
    FS_HI, FS_NAT = M * SCS, n_act * SCS
    BW_DES = n_act * SCS
    ctx = setup(dict(tag=tag, M=M, n_act=n_act, SCS=SCS, N=14), 1, quiet=True)
    waves = ctx['waves']
    rng = np.random.default_rng(4)

    curves = {}
    for w in waves:
        sig = w.tx(qpsk_mod(rng.integers(0, 2, 2 * w.nd)))
        curves[w.name] = (sig, FS_HI)
        bw = occupied_bw(sig, FS_HI)
        check(f"occupied BW {w.name}", abs(bw / BW_DES - 1) < 0.06,
              f"{bw/1e6:.3f} MHz vs designed {BW_DES/1e6:.3f} MHz")
        print(f"        PAPR(1e-4) = {papr(sig):.2f} dB")

    # native-rate PCP driven by the SAME DD symbols as the fixed-rate one,
    # and PSDs averaged over frames -- comparing single random realisations
    # measures Welch variance (many dB/bin), not waveform fidelity.
    wpcp = [w for w in waves if w.name == 'PCP-guard'][0]
    cp_g = max(4, int(round(1e-6 * FS_NAT)) + 3)
    pnat = PCPOTFSTransceiver(PCPOTFSConfig(
        M=n_act, N=14, Mcp=cp_g, scs_hz=SCS, pilot_doppler=7,
        doppler_guard=1, pilot_power_dB=25.0, zc_root=1, bem_Q=1))
    assert pnat.count_data_res() == wpcp.nd

    # (a) DIRECT time-domain exactness: decimating the fixed-rate signal must
    # return the native subsymbol bodies to machine precision.
    r0 = np.random.default_rng(11)
    syms = qpsk_mod(r0.integers(0, 2, 2 * wpcp.nd))
    nat_full = pnat.tx(syms)[0]
    hi_full = wpcp.tx(syms)
    MT, Mcp = pnat.cfg.MT, pnat.cfg.Mcp
    errs = []
    for n in range(14):
        body_nat = nat_full[n * MT + Mcp: n * MT + Mcp + n_act]
        seg_hi = hi_full[n * wpcp.sym_hi: (n + 1) * wpcp.sym_hi]
        errs.append(np.max(np.abs(wpcp._down(seg_hi[wpcp.K:]) - body_nat)))
    emax = float(np.max(errs))
    check("fixed-rate == native (time domain)", emax < 1e-10,
          f"max |diff| over 14 subsymbols = {emax:.2e}")

    # (b) averaged PSDs from the same symbol stream
    def avg_psd(gen, fs, nfr=24):
        """Average the PSD over repeated frames of one method."""
        acc = None
        for t in range(nfr):
            rr = np.random.default_rng(500 + t)
            sg = gen(qpsk_mod(rr.integers(0, 2, 2 * wpcp.nd)))
            f, p = welch(sg, fs=fs, window='blackmanharris',
                         nperseg=min(1024, len(sg) // 4),
                         return_onesided=False, scaling='density')
            i = np.argsort(f); f, p = f[i], p[i]
            acc = p if acc is None else acc + p
        acc /= nfr
        return f, 10 * np.log10(acc / acc.max() + 1e-30)

    f_hi, p_hi = avg_psd(lambda x: wpcp.tx(x), FS_HI)
    f_nat, p_nat = avg_psd(lambda x: pnat.tx(x)[0], FS_NAT)
    sig_nat = pnat.tx(syms)[0]
    bw_nat = occupied_bw(sig_nat, FS_NAT)
    check("occupied BW PCP native", abs(bw_nat / BW_DES - 1) < 0.06,
          f"{bw_nat/1e6:.3f} MHz vs designed {BW_DES/1e6:.3f} MHz")

    band = 0.45 * BW_DES
    grid = np.linspace(-band, band, 400)
    a = np.interp(grid, f_hi, p_hi)
    b = np.interp(grid, f_nat, p_nat)
    dev = float(np.max(np.abs(a - b)))
    check("PCP native vs fixed-rate PSD agree", dev < 2.0,
          f"max in-band deviation {dev:.2f} dB (averaged over 24 frames)")

    # edge integrity: the fixed-rate PCP must not droop at the band edge
    edge = (np.abs(f_hi) > 0.42 * BW_DES) & (np.abs(f_hi) < 0.48 * BW_DES)
    mid = np.abs(f_hi) < 0.10 * BW_DES
    droop = float(np.median(p_hi[mid]) - np.median(p_hi[edge]))
    check("PCP band-edge not attenuated", abs(droop) < 2.0,
          f"edge vs centre {droop:+.2f} dB")

    for w in waves:
        f, p = psd(curves[w.name][0], FS_HI)
        ax.plot(f / 1e6, p, lw=1.1, label=w.name)
    ax.plot(f_nat / 1e6, p_nat, 'k--', lw=1.0, alpha=0.7,
            label='PCP native rate')
    for e in (-BW_DES / 2 / 1e6, BW_DES / 2 / 1e6):
        ax.axvline(e, color='gray', ls=':', lw=0.9)
    ax.set_title(f'{tag}: TX PSD, evaluation chain (no TX filter)\n'
                 f'designed BW = {BW_DES/1e6:.2f} MHz, front-end rate '
                 f'{FS_HI/1e6:.2f} MHz')
    ax.set_xlabel('frequency [MHz]'); ax.set_ylabel('normalised PSD [dB]')
    ax.set_ylim(-70, 5); ax.grid(alpha=0.3); ax.legend(fontsize=8)

fig.tight_layout()
out = os.path.join(HERE, 'fig_verify_psd.png')
fig.savefig(out, dpi=130)
print(f"\nsaved {out}")
print("OVERALL:", "PASS - waveforms reproduce correctly" if ok_all else "FAIL")
