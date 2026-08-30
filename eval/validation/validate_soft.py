"""Validate the soft output of every receiver before trusting any coded result.

For each waveform we transmit KNOWN QPSK, receive, and check three things:

  1. alignment   : does z line up with the transmitted symbols at all?
                   (uncoded BER from hard decisions on z)
  2. calibration : does the claimed per-symbol SINR gamma predict the ACTUAL
                   error variance?   ratio = mean|z-x|^2 / mean(1/gamma)
                   ratio ~ 1 means the LLRs handed to the decoder are honest;
                   ratio >> 1 means over-confident LLRs (decoder will fail),
                   ratio << 1 means under-confident (decoder loses gain).
  3. SNR calibration: measured per-RE SNR vs the requested one.

Also cross-checks the ZP-OTFS Hutchinson trace estimate against a brute-force
dense diag(A^-1).
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
import coded_eval as CE
from coded_eval import WOFDM, WZP, WPCP, calibrate, qpsk_mod

M, n_act, SCS, N = 256, 156, 30e3, 14
FS = M * SCS
CP = max(round(144 * M / 2048), 4)
FS_NAT = n_act * SCS
chc = TDLChannel(TDLChannelConfig('TDL-C', 300e-9, 0, FS_NAT, seed=1, use_fdf=False))
ZPLEN = max(chc.max_delay_samples, 8)
cp_g = max(4, int(round(1e-6 * FS_NAT)) + 3)
cp_o = max(round(144 * n_act / 2048), 4)

waves = [WOFDM(M, n_act, SCS, N, CP),
         WZP(M, n_act, SCS, N, ZPLEN),
         WPCP(M, n_act, SCS, N, cp_g, 'PCP-guard')]

rng = np.random.default_rng(3)
cal = {w.name: calibrate(w, rng) for w in waves}
print("calibration (g_sig, g_noise):")
for w in waves:
    print(f"  {w.name:10s} {cal[w.name][0]:.4f}  {cal[w.name][1]:.4f}")

print("\n--- AWGN only (no fading): isolates the soft-output maths ---")
print(f"{'wave':11s}{'SNR':>5}{'uncodedBER':>12}{'meas SNR dB':>13}"
      f"{'E|z-x|^2':>11}{'E[|e|^2*g]':>11}{'ratio':>8}")
for w in waves:
    for snr in [4.0, 10.0, 16.0]:
        gs, gn = cal[w.name]
        s2 = gs / (10.0 ** (snr / 10.0) * gn)
        errs = ez = eg = 0.0
        nb = nsym = 0
        for t in range(4):
            r2 = np.random.default_rng(100 + t)
            bits = r2.integers(0, 2, 2 * w.nd)
            x = qpsk_mod(bits)
            sig = w.tx(x)
            L = max(len(sig), w.frame_len)
            sig = np.pad(sig, (0, L - len(sig)))
            rx = sig + np.sqrt(s2 / 2) * (r2.standard_normal(L)
                                          + 1j * r2.standard_normal(L))
            z, g = w.rx(rx)
            n = min(len(z), w.nd)
            d = z[:n] - x[:n]
            ez += float(np.sum(np.abs(d) ** 2))
            eg += float(np.sum(np.abs(d) ** 2 * g[:n]))   # per-RE normalised
            hb = np.empty(2 * n, dtype=int)
            hb[0::2] = (np.real(z[:n]) < 0).astype(int)
            hb[1::2] = (np.imag(z[:n]) < 0).astype(int)
            errs += int(np.sum(hb != bits[:2 * n])); nb += 2 * n; nsym += n
        mse = ez / nsym; norm = eg / nsym
        meas = 10 * np.log10(1.0 / mse) if mse > 0 else np.inf
        print(f"{w.name:11s}{snr:5.0f}{errs/nb:12.2e}{meas:13.2f}"
              f"{mse:11.3e}{norm:11.3f}{norm:8.2f}")

print("\n--- TDL-C, fD=1000 Hz (fading + estimation) ---")
print(f"{'wave':11s}{'SNR':>5}{'uncodedBER':>12}{'E|z-x|^2':>11}{'E[|e|^2*g]':>11}{'ratio':>8}")
for w in waves:
    for snr in [10.0, 20.0]:
        gs, gn = cal[w.name]
        s2 = gs / (10.0 ** (snr / 10.0) * gn)
        errs = ez = eg = 0.0
        nb = nsym = 0
        for t in range(4):
            ch = TDLChannel(TDLChannelConfig('TDL-C', 300e-9, 1000.0, FS,
                                             seed=700 + t, use_fdf=True))
            r2 = np.random.default_rng(200 + t)
            bits = r2.integers(0, 2, 2 * w.nd)
            x = qpsk_mod(bits)
            sig = w.tx(x)
            L = max(len(sig), w.frame_len)
            sig = np.pad(sig, (0, L - len(sig)))
            c, _ = ch.apply(sig, snr_dB=None)
            rx = c + np.sqrt(s2 / 2) * (r2.standard_normal(len(c))
                                        + 1j * r2.standard_normal(len(c)))
            z, g = w.rx(rx)
            n = min(len(z), w.nd)
            d = z[:n] - x[:n]
            ez += float(np.sum(np.abs(d) ** 2))
            eg += float(np.sum(np.abs(d) ** 2 * g[:n]))
            hb = np.empty(2 * n, dtype=int)
            hb[0::2] = (np.real(z[:n]) < 0).astype(int)
            hb[1::2] = (np.imag(z[:n]) < 0).astype(int)
            errs += int(np.sum(hb != bits[:2 * n])); nb += 2 * n; nsym += n
        mse = ez / nsym; norm = eg / nsym
        print(f"{w.name:11s}{snr:5.0f}{errs/nb:12.2e}{mse:11.3e}{norm:11.3f}{norm:8.2f}")

# ---- Hutchinson vs brute force for ZP ----
print("\n--- ZP-OTFS: Hutchinson trace vs dense diag(A^-1) ---")
from scipy import sparse
from scipy.sparse.linalg import spsolve
w = waves[1]
cfg = w.z.cfg
ch = TDLChannel(TDLChannelConfig('TDL-C', 300e-9, 1000.0, FS, seed=701, use_fdf=True))
r2 = np.random.default_rng(5)
x = qpsk_mod(r2.integers(0, 2, 2 * w.nd))
sig = w.tx(x)
c, _ = ch.apply(sig, snr_dB=None)
gs, gn = cal[w.name]
s2t = gs / (10.0 ** (10.0 / 10.0) * gn)
rx = c + np.sqrt(s2t / 2) * (r2.standard_normal(len(c)) + 1j * r2.standard_normal(len(c)))
rxn = w.to_native(rx)
rxn = np.pad(rxn, (0, max(0, cfg.frame_samples - len(rxn))))
y_dt = np.zeros((cfg.M, cfg.N), dtype=complex)
for n_ in range(cfg.N):
    y_dt[:, n_] = rxn[n_ * cfg.Meff: n_ * cfg.Meff + cfg.M]
Y_dd = np.fft.fft(y_dt, axis=1) / np.sqrt(cfg.N)
pre = w.z._estimate_noise_blind(rxn)
taps = w.z._estimate_channel_dd(Y_dd, pre)
s2 = w.z._estimate_noise(rxn, taps)
G = w.z._build_G_from_dd(taps)
NM = cfg.frame_samples
A = (G.conj().T @ G + max(s2, 1e-6) * sparse.eye(NM)).tocsc()
Ad = np.linalg.inv(A.toarray())
exact = float(np.real(np.trace(Ad))) / NM
print(f"  exact mean diag(A^-1) = {exact:.6e}   (s2_hat={s2:.3e})")
for K in (1, 2, 4, 8, 16):
    r3 = np.random.default_rng(0)
    acc = 0.0
    for _ in range(K):
        v = r3.integers(0, 2, NM) * 2.0 - 1.0
        acc += float(np.real(v @ spsolve(A, v))) / NM
    est = acc / K
    print(f"  Hutchinson K={K:2d}: {est:.6e}   rel.err {abs(est/exact-1)*100:6.2f}%")
