"""
Fractional Delay Filter — Modified Farrow Structure

Ported from KDDI's FractionalDelayFilter.m (arXiv:2010.15396).
Uses Lagrange interpolation via Cramer's rule on a Vandermonde matrix
to compute FIR filter coefficients for arbitrary sub-sample delays.

Reference:
    Valimaki & Laakso, "Fractional Delay Filters — Design and Applications,"
    in Nonuniform Sampling, Springer, 2001.
"""

__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "CC-BY-4.0"
__copyright__ = "(c) 2026 Panos N. Alevizos"

import numpy as np
from math import comb


def _matlab_round(x):
    """MATLAB-style rounding: half away from zero (not Python's banker's rounding)."""
    return int(np.floor(x + 0.5))


def _invcramer(M):
    """Matrix inverse via Cramer's rule (stable for small Vandermonde matrices)."""
    n = M.shape[0]
    det_M = np.linalg.det(M)
    if abs(det_M) < 1e-15:
        return np.linalg.inv(M)
    adj = np.zeros_like(M, dtype=float)
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(M, i, axis=0), j, axis=1)
            adj[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
    return adj.T / det_M


class FractionalDelayFilter:
    """
    Farrow-structure fractional delay filter.

    For each tap delay, computes an FIR filter that implements the
    exact fractional part via polynomial interpolation. Filter order
    is capped at `filter_order` (default 8; >9 may lose accuracy).
    """

    def __init__(self, filter_order=8):
        self.filter_order = filter_order
        self.fdf_coeffs = []       # list of 1-D coefficient arrays
        self.nonzero_indices = []   # start index for each tap's output

    def init(self, delays):
        """Pre-compute FIR coefficients for each delay value (in samples)."""
        self.fdf_coeffs = []
        self.nonzero_indices = []

        for delay in delays:
            # For near-zero delays, use identity (no interpolation needed)
            if delay < 0.5:
                self.fdf_coeffs.append(np.array([1.0]))
                self.nonzero_indices.append(int(round(delay)))
                continue

            N = min(int(np.floor(2 * delay + 1)), self.filter_order)
            if N < 1:
                N = 1

            # Vandermonde matrix (increasing powers) and its inverse via Cramer's rule
            U2 = np.zeros((N + 1, N + 1), dtype=float)
            for r in range(N + 1):
                for c in range(N + 1):
                    U2[r, c] = r ** c
            Q = _invcramer(U2)

            # Transition matrix T
            T = np.zeros((N + 1, N + 1), dtype=float)
            for n in range(N + 1):
                for m in range(N + 1):
                    if n >= m:
                        T[m, n] = _matlab_round(N / 2) ** (n - m) * comb(n, m)
            Qf = T @ Q

            # FIR coefficients for this delay's fractional part
            frac_delay = delay % 1.0
            delay_vec = np.array([frac_delay ** p for p in range(N + 1)])
            h = delay_vec @ Qf
            self.fdf_coeffs.append(h)

            # Non-zero start index (causality)
            if N % 2 == 0:
                start = _matlab_round(delay + 0.5) - N // 2
            else:
                start = int(np.floor(delay)) - (N - 1) // 2
            self.nonzero_indices.append(start)

    def apply(self, x):
        """
        Apply fractional delay filters to input signal.

        Args:
            x: 1-D complex input signal (length L)

        Returns:
            out: (L + tail, num_delays) array — one column per tap delay
        """
        num_delays = len(self.fdf_coeffs)
        if num_delays == 0:
            return np.zeros((len(x), 0), dtype=complex)

        L = len(x)
        max_tail = max(
            self.nonzero_indices[i] + len(self.fdf_coeffs[i]) - 1
            for i in range(num_delays)
        )
        out_len = L + max_tail
        out = np.zeros((out_len, num_delays), dtype=complex)

        for i in range(num_delays):
            h = self.fdf_coeffs[i]
            filterout = np.convolve(h, x)  # length = len(h) + L - 1
            start = self.nonzero_indices[i]

            if len(h) == 1:
                # Identity or single-tap filter — place directly, no skip
                n = min(L, out_len - max(start, 0))
                if start >= 0:
                    out[start:start + n, i] = h[0] * x[:n]
                else:
                    out[:n, i] = h[0] * x[-start:-start + n]
            elif start > 0:
                # Multi-tap: MATLAB filterout(2:end) placed at out(start+1:...)
                src = filterout[1:]
                n = min(len(src), out_len - start)
                out[start:start + n, i] = src[:n]
            else:
                # Multi-tap: MATLAB filterout(-start+2:end) placed at out(1:...)
                skip = -start + 1
                src = filterout[skip:]
                n = min(len(src), out_len)
                out[:n, i] = src[:n]

        return out


if __name__ == '__main__':
    print("=== FDF Smoke Test ===\n")

    # Test: integer delay should match np.roll exactly
    fdf = FractionalDelayFilter(filter_order=8)
    delays = np.array([0.0, 1.0, 2.0, 3.0, 5.5, 7.3])
    fdf.init(delays)

    L = 256
    rng = np.random.default_rng(42)
    x = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) / np.sqrt(2)
    out = fdf.apply(x)

    # For integer delays, compare with simple shift
    for idx, d in enumerate([0, 1, 2, 3]):
        d_int = int(delays[idx])
        ref = np.zeros(out.shape[0], dtype=complex)
        ref[d_int:d_int + L] = x
        err = np.max(np.abs(out[:L, idx] - ref[:L]))
        print(f"  delay={delays[idx]:.1f}: max_err vs shift = {err:.2e}")

    # For fractional delays, just check energy is preserved
    for idx in [4, 5]:
        energy_ratio = np.sum(np.abs(out[:, idx]) ** 2) / np.sum(np.abs(x) ** 2)
        print(f"  delay={delays[idx]:.1f}: energy_ratio = {energy_ratio:.4f}")

    print("\n  done")
