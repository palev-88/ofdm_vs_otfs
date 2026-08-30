"""Upgraded OFDM receiver: delay-domain DMRS estimation + calibrated LLRs.

Changes versus the report's receiver, which uses per-subcarrier LS, linear
interpolation in frequency and time, dmrs_power_diff noise estimation:

 1. DELAY-DOMAIN FIT.  Per DMRS symbol, fit an L-tap CIR to the K pilot
    observations by least squares (L = N_CP).  Denoises by ~10log10(K/L) dB and
    replaces frequency-domain linear interpolation.
 2. RESIDUAL IMPAIRMENT ESTIMATE.  sigma2 = ||r||^2/(K-L) from a SINGLE symbol,
    so it carries no Doppler bias, unlike differencing symbols 2 and 11.
 3. CALIBRATED gamma.  A noise-only SINR |H|^2/sigma2 is wrong at high Doppler
    because the dominant error is the LINEAR TIME INTERPOLATION across the
    9-symbol DMRS gap, not thermal noise.  We add that term:

      eps2_ce(l) = P*[1 + (1-a)^2 + a^2 - 2(1-a)rho(u) - 2a*rho(v)
                        + 2a(1-a)rho(dl)]        (Jakes, a=u/dl, u=l-l1, v=l2-l)

    rho(dl) is MEASURED from the pilot pair, rho(dl) = 1 - dch/(2P) with
    dch = mean|H2-H1|^2 - 2*sigma2*L/K, then inverted through J0 to get f_D and
    hence rho(u), rho(v).  No genie, no assumed Doppler.

      gamma(l,n) = |H(l,n)|^2 / (sigma2 + eps2_ce(l))
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import numpy as np
from scipy.special import j0

_J0X = np.linspace(0.0, 2.4048, 2049)          # J0 monotone decreasing here
_J0Y = j0(_J0X)


def _inv_j0(y):
    """Smallest x>=0 with J0(x)=y, clamped to the monotone branch."""
    y = np.clip(y, _J0Y[-1], 1.0)
    return float(np.interp(y, _J0Y[::-1], _J0X[::-1]))


class OFDMTimeDomainRx:
    """Delay-domain OFDM receiver: LS CIR fit at the DMRS bins, single-symbol residual noise estimate, and interpolation-error-aware reliability (report Sec. 3.4-3.7)."""
    def __init__(self, trx):
        self.t = trx
        c = trx.cfg
        self.M, self.CP, self.N = c.n_fft, c.n_cp, c.n_symbols_per_slot
        self.SYM = self.M + self.CP
        self.dmrs = sorted(c.dmrs_symbol_indices)
        self.bins = trx._sc_indices() % self.M
        self.pil = trx._dmrs_sc_indices_within_active()
        pb = self.bins[self.pil].astype(float)
        self.K = len(self.pil)
        self.L = self.CP
        F = np.exp(-1j * 2 * np.pi * np.outer(pb, np.arange(self.L)) / self.M)
        self.FP = np.linalg.pinv(F)
        self.PROJ = F @ self.FP
        self.dof = max(self.K - self.L, 1)
        self.FA = np.exp(-1j * 2 * np.pi *
                         np.outer(self.bins.astype(float), np.arange(self.L)) / self.M)
        self.Tsym = self.SYM / c.sample_rate

    def grid(self, rx):
        """CP-strip and FFT each symbol; return the active-subcarrier grid Y[l, n]."""
        M, CP, N = self.M, self.CP, self.N
        Y = np.zeros((N, self.t.cfg.n_active), dtype=complex)
        for l in range(N):
            s = l * self.SYM + CP
            seg = rx[s:s + M]
            if len(seg) < M:
                seg = np.pad(seg, (0, M - len(seg)))
            Y[l] = (np.fft.fft(seg) / np.sqrt(M))[self.bins]
        return Y

    def estimate(self, Y):
        """Returns (H_est, sigma2, eps2_ce_per_symbol)."""
        Hp, res = [], 0.0
        for l in self.dmrs:
            h_ls = Y[l, self.pil] / self.t._dmrs_seq
            res += float(np.sum(np.abs(h_ls - self.PROJ @ h_ls) ** 2))
            Hp.append(self.FA @ (self.FP @ h_ls))
        s2 = max(res / (len(self.dmrs) * self.dof), 1e-12)

        l1, l2 = self.dmrs[0], self.dmrs[-1]
        dl = l2 - l1
        s2_est = s2 * self.L / self.K                    # per-RE estimator noise
        P = max(float(np.mean(np.abs(Hp[0]) ** 2 + np.abs(Hp[1]) ** 2) / 2)
                - s2_est, 1e-12)
        dch = max(float(np.mean(np.abs(Hp[1] - Hp[0]) ** 2)) - 2 * s2_est, 0.0)
        rho_dl = np.clip(1.0 - dch / (2 * P), -1.0, 1.0)
        x_dl = _inv_j0(rho_dl)                            # 2*pi*fD*dl*Tsym
        w = x_dl / max(dl, 1)                             # 2*pi*fD*Tsym

        H = np.zeros((self.N, self.t.cfg.n_active), dtype=complex)
        eps2 = np.zeros(self.N)
        for l in range(self.N):
            a = np.clip((l - l1) / dl, 0.0, 1.0)
            H[l] = (1 - a) * Hp[0] + a * Hp[1]
            u, v = abs(l - l1), abs(l2 - l)
            r_u, r_v, r_d = j0(w * u), j0(w * v), j0(w * dl)
            e = (1 + (1 - a) ** 2 + a ** 2 - 2 * (1 - a) * r_u
                 - 2 * a * r_v + 2 * a * (1 - a) * r_d)
            eps2[l] = P * max(e, 0.0) + s2_est
        return H, s2, eps2

    def rx(self, rx_signal):
        """Returns (z, gamma) on the data REs, bias-corrected."""
        Y = self.grid(rx_signal)
        H, s2, eps2 = self.estimate(Y)
        Hs = np.where(np.abs(H) < 1e-9, 1e-9, H)
        z = self.t._extract_data(Y / Hs)
        den = s2 + eps2[:, None]
        g = self.t._extract_data(np.abs(H) ** 2 / den)
        return z, g

    def rx_hard(self, rx_signal):
        """MMSE-equalised data symbols (for uncoded BER)."""
        Y = self.grid(rx_signal)
        H, s2, eps2 = self.estimate(Y)
        den = np.abs(H) ** 2 + s2 + eps2[:, None]
        return self.t._extract_data(np.conj(H) * Y / den)
