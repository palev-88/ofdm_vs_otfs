"""Smoke test: epsilon^2 (channel-estimation error) term in PCP-OTFS's gamma.

Mechanism (mirrors the OFDM fix of sec:ofdm_rel): the Stage-2 GCE-BEM
projection leaves a residual r[l,:] = h_hat[l,:] - h_smooth[l,:] in the
(N-(2Q+1))-dimensional complement of the basis, per delay tap.  Its energy
estimates the per-sample error on the raw Stage-1 estimate -- which includes
BOTH pilot-observation noise AND the data-leakage/Doppler-smear interference
diagnosed earlier (Stage-1 SIR +5.6 dB NB / -1.4 dB WB).  The smoothed tap
retains (2Q+1)/N of that error, and the M-point FFT sums tap errors per bin:

    eps2_Hf = (2Q+1) / (N (N-2Q-1)) * sum_l || h_hat[l,:] - h_sm[l,:] ||^2

The fix replaces s2 -> s2 + eps2_Hf in the MMSE gain mu, the FDE, and gamma,
so the decoder is told about estimation error instead of being promised
noise-limited reliability that does not materialise at high Doppler (the
BLER upturn in the waterfalls).

Smoke grid: the cells where PCP fails hardest, coded BLER at 2-3 SNR points
near PCP's knee, old gamma vs new gamma, plus calibration ratio.
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

import sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'eval'))

from channel import TDLChannel, TDLChannelConfig
from coded_sweep import setup, DS_MAP
from coded_eval import WPCP, qpsk_mod, qpsk_llr, strided
import nr_ldpc


class WPCP_EPS(WPCP):
    """PCP with estimation-error-aware reliability (and MMSE regularisation)."""

    def rx(self, sig):
        """PCP receive with the eps^2 term applied to gamma (smoke-test variant)."""
        cfg = self.p.cfg
        Mp, Np = cfg.M, cfg.N
        Q = cfg.bem_Q
        Y, s2 = self._bodies(sig)
        h_hat = self.p._estimate_stage1(Y, s2)
        h_sm = self.p._estimate_stage2(h_hat, s2)
        # epsilon^2 (variant B): per-tap MEDIAN of the out-of-basis BEM
        # spectrum -> robust white floor; in-basis LS coefficient variance
        F = np.fft.fft(h_hat, axis=1)
        qs = np.arange(Np); dist = np.minimum(qs, Np - qs)
        floor = np.median(np.abs(F[:, dist > Q]) ** 2, axis=1)
        eps2 = float((2 * Q + 1) * np.sum(floor) / Np ** 2)
        s2t = s2 + eps2
        H_f = np.zeros((Mp, Np), dtype=complex)
        Y_f = np.zeros((Mp, Np), dtype=complex)
        for n in range(Np):
            H_f[:, n] = np.fft.fft(h_sm[:, n], Mp)
            Y_f[:, n] = np.fft.fft(Y[:, n])
        pf = np.fft.fft(self.p._pilot_row)
        for n in range(Np):
            ph = np.exp(1j * 2 * np.pi * cfg.pilot_doppler * n / Np) / np.sqrt(Np)
            Y_f[:, n] -= H_f[:, n] * pf * ph
        pw = np.abs(H_f) ** 2
        mu = pw / (pw + s2t)
        X_f = np.conj(H_f) / (pw + s2t) * Y_f
        D = np.fft.fft(np.fft.ifft(X_f, axis=0), axis=1) / np.sqrt(Np)
        mb = float(mu.mean())
        gamma = mb ** 2 / max(float((mu ** 2).mean()) - mb ** 2
                              + float((mu * (1 - mu)).mean()), 1e-15)
        z = D[self.p._data_pos[:, 0], self.p._data_pos[:, 1]] / max(mb, 1e-12)
        return z, np.full(len(z), gamma)


CELLS = {  # decisive subset: worst cells + the regression cell + a low-fd cell
    ('NB', 'TDL-A', 1000.0): [14.0],
    ('WB', 'TDL-A', 1000.0): [14.0],
    ('WB', 'TDL-C', 1000.0): [14.0, 16.0],
    ('WB', 'TDL-C', 600.0):  [11.0, 13.0],
    ('WB', 'TDL-B', 0.0):    [9.0, 10.0],
}
FRAMES = 150
BWS = {'NB': dict(tag='NB', M=256, n_act=156, SCS=30e3, N=14),
       'WB': dict(tag='WB', M=1024, n_act=624, SCS=60e3, N=14)}

t0 = time.time()
print(f"{'cell':24s}{'SNR':>5} | {'BLER old':>9}{'BLER eps':>9} | "
      f"{'calib old':>10}{'calib eps':>10} | {'eps2/s2':>8}")
for b in ('NB', 'WB'):
    ctx = setup(BWS[b], 20, quiet=True)
    import sys as _sys
    variant = _sys.argv[1] if len(_sys.argv) > 1 else 'PCP-guard'
    old = [w for w in ctx['waves'] if w.name == variant][0]
    new = WPCP_EPS(BWS[b]['M'], BWS[b]['n_act'], BWS[b]['SCS'], 14,
                   old.p.cfg.Mcp, variant + '-eps')
    gs, gn = ctx['cal'][variant]
    sel = strided(old.nd, ctx['n_common'])
    bank, PERM, IPERM = ctx['bank'], ctx['perm'], ctx['iperm']
    for (bw, cm, fd), snrs in CELLS.items():
        if bw != b:
            continue
        ds = DS_MAP[cm]
        acc = {(s, k): [0, 0, 0.0, 0.0, 0.0] for s in snrs for k in ('o', 'e')}
        for fr in range(FRAMES):
            ch = TDLChannel(TDLChannelConfig(cm, ds, fd, ctx['FS'],
                                             seed=700_000 + 977 * fr + int(fd),
                                             use_fdf=True))
            tb, coded, prm = bank[fr % len(bank)]
            syms = qpsk_mod(coded[PERM].astype(int))
            frng = np.random.default_rng(5_000 + 13 * fr + int(fd))
            pay = np.zeros(old.nd, dtype=complex)
            pay[sel] = syms
            other = np.setdiff1d(np.arange(old.nd), sel)
            pay[other] = qpsk_mod(frng.integers(0, 2, 2 * len(other)))
            sig = old.tx(pay)
            c, _ = ch.apply(sig, snr_dB=None)
            for snr in snrs:
                s2 = gs / (10.0 ** (snr / 10.0) * gn)
                r = c + np.sqrt(s2 / 2) * (frng.standard_normal(len(c))
                                           + 1j * frng.standard_normal(len(c)))
                for key, wv in (('o', old), ('e', new)):
                    z, g = wv.rx(r)
                    zz, gg = z[sel], g[sel]
                    dec = nr_ldpc.ldpc_decode(qpsk_llr(zz, gg)[IPERM], prm,
                                              n_iter=20)
                    v = acc[(snr, key)]
                    v[0] += 0 if dec['tb_crc_ok'] else 1
                    v[1] += 1
                    d = zz - syms
                    v[2] += float(np.sum(np.abs(d) ** 2 * gg))
                    v[3] += len(zz)
                    if key == 'e':
                        v[4] += 1.0  # placeholder
        for snr in snrs:
            vo, ve = acc[(snr, 'o')], acc[(snr, 'e')]
            print(f"{b+' '+cm+' fD='+str(int(fd)):24s}{snr:5.0f} | "
                  f"{vo[0]/vo[1]:9.3f}{ve[0]/ve[1]:9.3f} | "
                  f"{vo[2]/vo[3]:10.2f}{ve[2]/ve[3]:10.2f} |", flush=True)
print(f"\ntotal {time.time()-t0:.0f}s")
