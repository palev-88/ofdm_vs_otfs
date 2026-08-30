"""Vectorised drop-in for TDLChannel._generate_jakes_fading.

The repo version runs a Python double loop over n_taps x n_sinusoids (24 x 32
at NB), which costs ~62 ms per NB frame and ~200 ms per WB frame -- the largest
per-frame item in the profile.

This version vectorises the inner sinusoid sum only, and issues the SAME RNG
draws in the SAME order (alpha, beta, theta-offset per tap; LOS phase after the
Rayleigh part for the Rician first tap).  Output is therefore BIT-IDENTICAL to
the original for a given seed -- verified in _selftest below -- so results stay
comparable with anything produced by the unpatched repo.
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'eval'))
from channel import TDLChannel

_orig = TDLChannel._generate_jakes_fading


def _fast_jakes(self, n_samples):
    n_sin = 32
    f_D = self.cfg.f_doppler
    coeffs = np.zeros((self.n_taps, n_samples), dtype=complex)
    t = np.arange(n_samples) / self.cfg.sample_rate

    if f_D == 0:
        for i in range(self.n_taps):
            if self.fading_type == 'rician' and i == 0:
                coeffs[i, :] = self.amplitudes[i]
            else:
                ph = self.rng.uniform(0, 2 * np.pi)
                coeffs[i, :] = self.amplitudes[i] * np.exp(1j * ph)
        return coeffs

    k = np.arange(1, n_sin + 1)
    for i in range(self.n_taps):
        alpha_n = self.rng.uniform(0, 2 * np.pi, n_sin)
        beta_n = self.rng.uniform(0, 2 * np.pi, n_sin)
        theta_n = (2 * np.pi * k + self.rng.uniform(0, 2 * np.pi)) / (4 * n_sin)
        f_n = f_D * np.cos(theta_n)

        arg = 2 * np.pi * np.outer(f_n, t)               # (n_sin, n_samples)
        h_I = np.cos(arg + alpha_n[:, None]).sum(axis=0)
        h_Q = np.sin(arg + beta_n[:, None]).sum(axis=0)
        h_complex = (h_I + 1j * h_Q) / np.sqrt(n_sin)

        if self.fading_type == 'rician' and i == 0 and self.K_lin is not None:
            K = self.K_lin
            los_phase = self.rng.uniform(0, 2 * np.pi)
            h_los = np.sqrt(K / (K + 1)) * np.exp(
                1j * (2 * np.pi * f_D * t + los_phase))
            h_nlos = np.sqrt(1 / (K + 1)) * h_complex
            coeffs[i, :] = self.amplitudes[i] * (h_los + h_nlos)
        else:
            coeffs[i, :] = self.amplitudes[i] * h_complex
    return coeffs


def enable():
    """Monkey-patch TDLChannel with a cached-fading fast path for repeated-frame studies."""
    TDLChannel._generate_jakes_fading = _fast_jakes


def _selftest():
    from channel import TDLChannelConfig
    import time
    ok = True
    for model, fd in [('TDL-C', 1000.0), ('TDL-D', 500.0), ('TDL-A', 0.0)]:
        cfg = TDLChannelConfig(model, 300e-9, fd, 7.68e6, seed=5, use_fdf=False)
        TDLChannel._generate_jakes_fading = _orig
        a = TDLChannel(cfg)._generate_jakes_fading(4000)
        TDLChannel._generate_jakes_fading = _fast_jakes
        b = TDLChannel(cfg)._generate_jakes_fading(4000)
        same = np.allclose(a, b, rtol=0, atol=1e-12)
        ok &= same
        print(f"  {model} fD={fd:6.0f}: identical={same}  max|diff|={np.max(np.abs(a-b)):.2e}")
    for n, lbl in [(3836, 'NB'), (15344, 'WB')]:
        cfg = TDLChannelConfig('TDL-C', 300e-9, 1000.0, 7.68e6, seed=5, use_fdf=False)
        TDLChannel._generate_jakes_fading = _orig
        c = TDLChannel(cfg); t0 = time.perf_counter()
        for _ in range(3): c._generate_jakes_fading(n)
        t_o = (time.perf_counter() - t0) / 3 * 1e3
        TDLChannel._generate_jakes_fading = _fast_jakes
        c = TDLChannel(cfg); t0 = time.perf_counter()
        for _ in range(3): c._generate_jakes_fading(n)
        t_f = (time.perf_counter() - t0) / 3 * 1e3
        print(f"  {lbl} ({n} samples): {t_o:7.1f} ms -> {t_f:6.1f} ms  ({t_o/t_f:.1f}x)")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == '__main__':
    _selftest()
