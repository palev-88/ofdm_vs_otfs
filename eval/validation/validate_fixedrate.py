"""Validation gate for the fixed-rate DFT-grid PCP before the overnight rerun.

Checks, in order:
  1. up/down exactness: _down(_up(x)) == x to machine precision
  2. noise-conversion factor: sigma_nat^2 / sigma_hi^2 == Mn/Mfft numerically
  3. calibration: g_sig ~ 1.0 for every method (no band-edge loss)
  4. APPLIED per-RE SNR identical across variants (author's requirement,
     mirrors the report's original <0.01 dB verification): inject at nominal
     SNR through each method's calibrated s2, measure per-data-RE SNR on the
     pre-equalizer cells, require spread < 0.05 dB
  5. AWGN round-trip: noiseless tx->rx recovers symbols; uncoded BER at 10 dB
  6. gamma calibration in fading for the new PCP: E[|z-x|^2 * gamma] ~ 1
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'eval'))

from channel import TDLChannel, TDLChannelConfig
from coded_sweep import setup
from coded_eval import qpsk_mod

ok_all = True


def check(name, cond, detail=""):
    """Assert one property of the fixed-front-end-rate representation and print the measured value."""
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")


for tag, M, n_act, SCS in [('NB', 256, 156, 30e3), ('WB', 1024, 624, 60e3)]:
    print(f"\n=== {tag} ===")
    ctx = setup(dict(tag=tag, M=M, n_act=n_act, SCS=SCS, N=14), 1, quiet=True)
    waves = ctx['waves']
    cal = ctx['cal']
    FS = M * SCS

    w = waves[1]                                    # PCP-guard
    rng = np.random.default_rng(1)

    # 1. up/down exactness
    x = rng.standard_normal(n_act) + 1j * rng.standard_normal(n_act)
    err = np.max(np.abs(w._down(w._up(x)) - x))
    check("up/down exact", err < 1e-10, f"max err {err:.1e}")

    # 2. noise conversion factor
    L = w.frame_len
    trials = []
    for t in range(6):
        nz = (np.random.default_rng(t).standard_normal(L)
              + 1j * np.random.default_rng(t + 50).standard_normal(L)) / np.sqrt(2)
        Y, s2n = w._bodies(nz)
        trials.append(np.mean(np.abs(Y) ** 2))
    meas = float(np.mean(trials))
    check("noise factor Mn/Mfft", abs(meas / (n_act / M) - 1) < 0.05,
          f"measured {meas:.4f} vs {n_act/M:.4f}")

    # 3. g_sig ~ 1 for all methods
    for wv in waves:
        gs, gn = cal[wv.name]
        check(f"g_sig {wv.name}", abs(gs - 1.0) < 0.03,
              f"g_sig={gs:.4f} g_noise={gn:.4f}")

    # 4. applied per-RE SNR identical across variants
    target = 15.0
    snr_meas = {}
    for wv in waves:
        gs, gn = cal[wv.name]
        s2 = gs / (10.0 ** (target / 10.0) * gn)
        ps = pn = 0.0
        for t in range(10):
            r2 = np.random.default_rng(300 + t)
            syms = qpsk_mod(r2.integers(0, 2, 2 * wv.nd))
            sig = wv.tx(syms)
            Lw = max(len(sig), wv.frame_len)
            sig = np.pad(sig, (0, Lw - len(sig)))
            ps += float(np.mean(np.abs(wv.pre_eq(sig)[:wv.nd]) ** 2))
            nz = np.sqrt(s2 / 2) * (r2.standard_normal(Lw)
                                    + 1j * r2.standard_normal(Lw))
            pn += float(np.mean(np.abs(wv.pre_eq(nz)[:wv.nd]) ** 2))
        snr_meas[wv.name] = 10 * np.log10(ps / pn)
    spread = max(snr_meas.values()) - min(snr_meas.values())
    detail = "  ".join(f"{k}={v:.3f}dB" for k, v in snr_meas.items())
    check("applied per-RE SNR equal", spread < 0.05,
          f"spread {spread*1000:.1f} mdB | {detail}")

    # 5. AWGN round-trip + uncoded BER at 10 dB
    for wv in waves[1:]:                            # the two PCP variants
        r2 = np.random.default_rng(7)
        bits = r2.integers(0, 2, 2 * wv.nd)
        syms = qpsk_mod(bits)
        sig = wv.tx(syms)
        z, _ = wv.rx(sig)                           # noiseless
        hb = np.empty(2 * wv.nd, dtype=int)
        hb[0::2] = (np.real(z[:wv.nd]) < 0)
        hb[1::2] = (np.imag(z[:wv.nd]) < 0)
        check(f"noiseless round-trip {wv.name}", np.sum(hb != bits) == 0,
              f"{np.sum(hb != bits)} bit errors")
        gs, gn = cal[wv.name]
        s2 = gs / (10.0 ** (10.0 / 10.0) * gn)
        r = sig + np.sqrt(s2 / 2) * (r2.standard_normal(len(sig))
                                     + 1j * r2.standard_normal(len(sig)))
        z, _ = wv.rx(r)
        hb[0::2] = (np.real(z[:wv.nd]) < 0)
        hb[1::2] = (np.imag(z[:wv.nd]) < 0)
        ber = np.mean(hb != bits)
        check(f"AWGN 10 dB BER {wv.name}", 1e-4 < ber < 2e-2, f"BER {ber:.2e}")

    # 6. gamma calibration in fading (PCP-guard), TDL-C fD=1000
    ez = eg = 0.0
    nsym = 0
    for t in range(3):
        ch = TDLChannel(TDLChannelConfig('TDL-C', 300e-9, 1000.0, FS,
                                         seed=880 + t, use_fdf=True))
        r2 = np.random.default_rng(600 + t)
        syms = qpsk_mod(r2.integers(0, 2, 2 * w.nd))
        sig = w.tx(syms)
        c, _ = ch.apply(sig, snr_dB=None)
        gs, gn = cal[w.name]
        s2 = gs / (10.0 ** (15.0 / 10.0) * gn)
        r = c + np.sqrt(s2 / 2) * (r2.standard_normal(len(c))
                                   + 1j * r2.standard_normal(len(c)))
        z, g = w.rx(r)
        n = min(len(z), w.nd)
        d = z[:n] - syms[:n]
        ez += float(np.sum(np.abs(d) ** 2 * g[:n]))
        nsym += n
    ratio = ez / nsym
    check("gamma calibration PCP (fading)", 0.4 < ratio < 2.5,
          f"E[|e|^2*gamma] = {ratio:.2f}")

print("\nOVERALL:", "PASS - safe to launch overnight rerun" if ok_all
      else "FAIL - do not launch")
sys.exit(0 if ok_all else 1)
