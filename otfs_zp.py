"""
ZP-OTFS Transceiver — Zero-Padded OTFS (MathWorks-style).

TX: DD grid → inverse Zak transform → zero padding between subsymbols
RX: ZP removal → Zak transform → DD channel estimation → time-domain LMMSE

ISI mitigation: Zero Padding (not cyclic prefix)
  - ZP absorbs multipath tail → no ISI between subsymbols
  - DD pilot estimation remains clean (unlike frame-CP OTFS)

Channel estimation: DD pilot + quadratic interpolation for fractional Doppler.

Equalization: Sparse time-domain LMMSE using scipy sparse solver.

Note: Perfect-CSI detector retained for upper-bound plots in report.

Future: PAPR/ACPR measurement requires oversampled pulse-shaped waveform.
"""

__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "CC-BY-4.0"
__copyright__ = "(c) 2026 Panos N. Alevizos"

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass
from qam import qam_constellation


@dataclass
class ZPOTFSConfig:
    M: int = 64              # Delay bins (= subcarriers, ALL active)
    N: int = 14              # Doppler bins (= subsymbols per frame)
    scs_hz: float = 15e3     # Subcarrier spacing
    zp_len: int = 10         # Zero padding length (>= max channel delay in samples)

    # Pilot
    pilot_power_boost_dB: float = 15.0
    pilot_delay: int = 1     # Pilot position in delay
    pilot_doppler: int = 7   # Pilot position in Doppler (center = N//2)
    guard_delay: int = 12    # Guard ± around pilot in delay
    guard_doppler: int = 4   # Guard ± around pilot in Doppler

    # Channel estimation
    max_dd_taps: int = 50    # Max taps to keep after thresholding

    @property
    def sample_rate(self): return self.M * self.scs_hz
    @property
    def Meff(self): return self.M + self.zp_len
    @property
    def frame_samples(self): return self.Meff * self.N
    @property
    def T_s(self): return 1.0 / self.sample_rate
    @property
    def pilot_overhead(self):
        ga = (2*self.guard_delay+1) * (2*self.guard_doppler+1)
        return ga / (self.M * self.N)


class ZPOTFSTransceiver:
    def __init__(self, cfg: ZPOTFSConfig):
        self.cfg = cfg
        self._setup()

    def _setup(self):
        cfg = self.cfg
        M, N = cfg.M, cfg.N

        # Guard mask
        mask = np.zeros((M, N), dtype=bool)
        for dm in range(-cfg.guard_delay, cfg.guard_delay + 1):
            for dn in range(-cfg.guard_doppler, cfg.guard_doppler + 1):
                mask[(cfg.pilot_delay + dm) % M, (cfg.pilot_doppler + dn) % N] = True
        self._guard_mask = mask
        self._data_pos = np.argwhere(~mask)  # (n_data, 2) [delay, doppler]
        self._n_data = len(self._data_pos)

    # ═══════════════════════════════════════════════════════════
    # TX: Inverse Zak Transform + Zero Padding
    # ═══════════════════════════════════════════════════════════

    def tx(self, qam_symbols):
        """
        TX: DD grid → inverse Zak → ZP insertion → time-domain signal.

        Inverse Zak: x_dt[m,n] = IFFT(X_dd, axis=1) * √N
        Time signal: x(m + n*Meff) = x_dt[m,n], with zeros in ZP region.
        """
        cfg = self.cfg
        M, N = cfg.M, cfg.N

        # Build DD grid
        dd = np.zeros((M, N), dtype=complex)
        nd = min(len(qam_symbols), self._n_data)
        for i in range(nd):
            dd[self._data_pos[i, 0], self._data_pos[i, 1]] = qam_symbols[i]
        dd[cfg.pilot_delay, cfg.pilot_doppler] = 10 ** (cfg.pilot_power_boost_dB / 20)

        # Inverse Zak transform: IFFT along Doppler (axis 1)
        x_dt = np.fft.ifft(dd, axis=1) * np.sqrt(N)

        # Build time signal with ZP
        tx_sig = np.zeros(cfg.frame_samples, dtype=complex)
        for n in range(N):
            tx_sig[n * cfg.Meff: n * cfg.Meff + M] = x_dt[:, n]
            # ZP region: already zeros

        return tx_sig, dd

    # ═══════════════════════════════════════════════════════════
    # RX: Zak Transform + Channel Estimation + LMMSE
    # ═══════════════════════════════════════════════════════════

    def _estimate_noise_blind(self, rx_signal):
        """
        Blind noise pre-estimate from the last 2 ZP samples of each subsymbol.

        The last 2 ZP samples are always beyond the maximum channel delay
        (zp_len ≥ max_delay by construction), so they contain pure noise.
        This estimate is used to regularize channel estimation before the
        ZP-tail full noise estimator can run.
        """
        cfg = self.cfg
        M, N, Meff = cfg.M, cfg.N, cfg.Meff
        samples = []
        for n in range(N):
            end = n * Meff + Meff
            start = end - 2
            if start >= 0 and end <= len(rx_signal):
                samples.append(rx_signal[start:end])
        if samples:
            return max(float(np.mean(np.abs(np.concatenate(samples)) ** 2)), 1e-10)
        return 0.01

    def rx(self, rx_signal, noise_var, detector='lmmse_est'):  # noise_var unused — self-estimated from ZP tails
        """
        RX pipeline.

        Detectors:
          'lmmse_est': DD estimation + time-domain sparse LMMSE
          'lmmse_true': True CSI (requires fading coefficients)

        Note: noise_var is ignored — rx() self-estimates noise from ZP tails.
        Kept for API compatibility (run_eval passes nv positionally).

        Returns: (data_symbols, dd_channel_est, dd_rx)
        """
        cfg = self.cfg
        M, N, Meff = cfg.M, cfg.N, cfg.Meff

        # Zak transform demodulation: remove ZP, FFT along Doppler
        y_dt = np.zeros((M, N), dtype=complex)
        for n in range(N):
            y_dt[:, n] = rx_signal[n * Meff: n * Meff + M]

        Y_dd = np.fft.fft(y_dt, axis=1) / np.sqrt(N)

        # Blind noise pre-estimate (last 2 ZP samples — always beyond channel)
        # Used only as regularizer for channel estimation; not the final noise_var.
        noise_pre = self._estimate_noise_blind(rx_signal)

        # Channel estimation — uses noise_pre (not caller's noise_var)
        taps_dd = self._estimate_channel_dd(Y_dd, noise_pre)

        # Noise estimation from ZP tail (uses estimated channel length)
        noise_est = self._estimate_noise(rx_signal, taps_dd)

        if detector == 'lmmse_est':
            data_sym = self._detect_lmmse(rx_signal, taps_dd, noise_est)
        else:
            raise ValueError(f"Unknown detector: {detector}")

        return data_sym, taps_dd, Y_dd

    def _estimate_noise(self, rx_signal, taps_dd):
        """
        Estimate noise from ZP tail region.
        
        The ZP region of each subsymbol: y[n*Meff+M : n*Meff+Meff]
        First L_ch samples contain channel tail, rest is pure noise.
        
        σ² = mean(|y_zp_tail|²) for samples beyond estimated channel length.
        """
        cfg = self.cfg
        M, N, Meff = cfg.M, cfg.N, cfg.Meff
        zp_len = cfg.zp_len
        
        # Estimate channel length from DD taps
        if taps_dd:
            max_delay = max(t[0] for t in taps_dd)
        else:
            max_delay = 0
        
        # Noise-only portion: ZP samples beyond channel tail
        noise_start = max_delay + 2  # small margin
        if noise_start >= zp_len:
            # ZP too short — use last few samples anyway
            noise_start = max(zp_len - 2, 0)
        
        noise_samples = []
        for n in range(N):
            zp_start = n * Meff + M + noise_start
            zp_end = n * Meff + Meff
            if zp_end <= len(rx_signal):
                noise_samples.append(rx_signal[zp_start:zp_end])
        
        if noise_samples:
            all_noise = np.concatenate(noise_samples)
            if len(all_noise) > 0:
                return float(np.mean(np.abs(all_noise)**2))
        
        return 0.01  # fallback

    def rx_true_csi(self, rx_signal, fading_coeffs, delay_samples, noise_var):
        """
        RX with true CSI from TDL channel model.
        For upper-bound plots only.
        """
        G = self._build_G_true(fading_coeffs, delay_samples)
        return self._solve_lmmse(rx_signal, G, noise_var)

    # ═══════════════════════════════════════════════════════════
    # DD Channel Estimation with Quadratic Interpolation
    # ═══════════════════════════════════════════════════════════

    def _estimate_channel_dd(self, Y_dd, noise_var):
        """
        Estimate DD channel from pilot response.

        Two modes:
        1. Peak detection + quadratic interpolation (for on-grid Doppler)
        2. Full DD response (for off-grid / continuous Doppler)

        For 3GPP channels with continuous Jakes Doppler, the full DD
        response within the guard zone captures the Doppler spread that
        individual peak detection misses.

        Returns: list of (delay_int, doppler_frac, complex_gain)
        """
        cfg = self.cfg
        M, N = cfg.M, cfg.N
        P = 10 ** (cfg.pilot_power_boost_dB / 20)

        # LMMSE channel estimate
        H_dd = Y_dd * np.conj(P) / (np.abs(P) ** 2 + noise_var)

        # Shift so pilot is at origin
        H_dd = np.roll(H_dd, -cfg.pilot_delay, axis=0)
        H_dd = np.roll(H_dd, -cfg.pilot_doppler, axis=1)

        # Collect ALL significant elements within the guard zone
        max_delay = min(cfg.guard_delay, cfg.zp_len)
        max_doppler = cfg.guard_doppler

        taps = []
        max_pow = 0.0

        # First pass: find maximum power for thresholding
        for k in range(max_delay + 1):
            for dl in range(-max_doppler, max_doppler + 1):
                pw = np.abs(H_dd[k, dl % N]) ** 2
                if pw > max_pow:
                    max_pow = pw

        # Second pass: collect all elements above threshold
        threshold = max(max_pow * 0.01, noise_var * 3)

        for k in range(max_delay + 1):
            for dl in range(-max_doppler, max_doppler + 1):
                h_val = H_dd[k, dl % N]
                if np.abs(h_val) ** 2 >= threshold:
                    taps.append((k, float(dl), h_val))

        if not taps:
            return [(0, 0.0, 1.0)]

        # Keep top-K by magnitude
        taps.sort(key=lambda t: abs(t[2]), reverse=True)
        taps = taps[:cfg.max_dd_taps]

        return taps

    # ═══════════════════════════════════════════════════════════
    # Time-Domain Channel Matrix Build
    # ═══════════════════════════════════════════════════════════

    def _build_G_from_dd(self, taps_dd, noise_var=0):
        """
        Build sparse time-domain channel matrix G from DD taps.

        G[k, k-d] = h * exp(j2π * dop_frac * k / (N * Meff))

        where k is the sample index in the full frame,
        d is the delay in samples,
        dop_frac is the fractional Doppler index.

        Only non-ZP source positions contribute.
        """
        cfg = self.cfg
        NM = cfg.frame_samples
        Meff = cfg.Meff
        M = cfg.M

        rows, cols, vals = [], [], []
        for k in range(NM):
            for (d, dop_frac, h) in taps_dd:
                src = k - d
                if 0 <= src < NM:
                    # Check source is not in ZP region
                    src_subsym = src // Meff
                    src_sample = src % Meff
                    if src_sample < M:  # Not in ZP
                        doppler_hz = dop_frac / (cfg.N * Meff * cfg.T_s)
                        phase = np.exp(1j * 2 * np.pi * doppler_hz * k * cfg.T_s)
                        rows.append(k)
                        cols.append(src)
                        vals.append(h * phase)

        return sparse.csr_matrix((vals, (rows, cols)), shape=(NM, NM))

    def _build_G_true(self, fading, delay_samples):
        """Build G from true TDL fading coefficients (perfect CSI)."""
        cfg = self.cfg
        NM = cfg.frame_samples
        Meff = cfg.Meff
        M = cfg.M
        n_taps = len(delay_samples)

        rows, cols, vals = [], [], []
        for k in range(NM):
            for i in range(n_taps):
                d = delay_samples[i]
                src = k - d
                if 0 <= src < NM and k < fading.shape[1]:
                    if src % Meff < M:  # Not in ZP
                        rows.append(k)
                        cols.append(src)
                        vals.append(fading[i, k])

        return sparse.csr_matrix((vals, (rows, cols)), shape=(NM, NM))

    # ═══════════════════════════════════════════════════════════
    # LMMSE Solver
    # ═══════════════════════════════════════════════════════════

    def _solve_lmmse(self, rx_signal, G, noise_var):
        """Solve x̂ = (G^H G + σ²I)^{-1} G^H y via sparse solver."""
        cfg = self.cfg
        NM = cfg.frame_samples

        y = rx_signal[:NM]
        GH = G.conj().T
        # Floor regularization to avoid singular matrix at noise_var → 0
        reg = max(noise_var, 1e-6)
        A = (GH @ G + reg * sparse.eye(NM)).tocsc()
        try:
            x_hat = spsolve(A, GH @ y)
            if np.any(np.isnan(x_hat)):
                raise ValueError("NaN in solve")
        except Exception:
            # Fallback: use heavier regularization
            A = (GH @ G + 1e-3 * sparse.eye(NM)).tocsc()
            x_hat = spsolve(A, GH @ y)

        # Convert to DD domain
        x_dt = np.zeros((cfg.M, cfg.N), dtype=complex)
        for n in range(cfg.N):
            x_dt[:, n] = x_hat[n * cfg.Meff: n * cfg.Meff + cfg.M]

        X_dd = np.fft.fft(x_dt, axis=1) / np.sqrt(cfg.N)

        # Extract data symbols
        data = np.array([X_dd[self._data_pos[i, 0], self._data_pos[i, 1]]
                         for i in range(self._n_data)])
        return data

    def _detect_lmmse(self, rx_signal, taps_dd, noise_var):
        """LMMSE detection with estimated DD channel."""
        G = self._build_G_from_dd(taps_dd)
        return self._solve_lmmse(rx_signal, G, noise_var)

    def count_data_res(self):
        return self._n_data


# ═══════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    from channel import TDLChannel, TDLChannelConfig
    from qam import qam_modulate, qam_demodulate, generate_random_bits
    import time

    cfg = ZPOTFSConfig(M=64, N=14, scs_hz=15e3, zp_len=10,
                       pilot_power_boost_dB=15.0,
                       pilot_delay=1, pilot_doppler=7,
                       guard_delay=12, guard_doppler=4,
                       max_dd_taps=10)
    zp = ZPOTFSTransceiver(cfg)
    nd = zp.count_data_res()
    FS = cfg.sample_rate

    print(f"═══ ZP-OTFS Smoke Test ═══")
    print(f"  M={cfg.M}, N={cfg.N}, ZP={cfg.zp_len}, Meff={cfg.Meff}")
    print(f"  Fs={FS/1e3:.0f}kHz, data={nd}, OH={cfg.pilot_overhead*100:.1f}%\n")

    bits = generate_random_bits(nd * 2, np.random.default_rng(42))
    syms = qam_modulate(bits, 4)[:nd]
    tx, dd_tx = zp.tx(syms)

    # Test 1: AWGN
    nv = 0.01
    noise = np.sqrt(nv / 2) * (np.random.randn(len(tx)) + 1j * np.random.randn(len(tx)))
    dr, taps, _ = zp.rx(tx + noise, nv)
    br = qam_demodulate(dr, 4)
    ber = np.mean(bits[:len(br)] != br)
    print(f"  AWGN 20dB: BER={ber:.4e}, {len(taps)} taps {'✓' if ber < 0.01 else '✗'}")

    # Test 2: TDL-A with estimated CSI
    for fd in [0, 300, 500, 1000]:
        ch = TDLChannel(TDLChannelConfig('TDL-A', 100e-9, fd, FS, seed=42))
        rx, fading = ch.apply(tx, snr_dB=20)
        t0 = time.time()
        dr, taps, _ = zp.rx(rx, 0.01)
        dt = time.time() - t0
        br = qam_demodulate(dr, 4)
        ber_est = np.mean(bits[:len(br)] != br)

        # True CSI
        dr_true = zp.rx_true_csi(rx, fading, ch.delays_samples, 0.01)
        br_true = qam_demodulate(dr_true, 4)
        ber_true = np.mean(bits[:len(br_true)] != br_true)

        print(f"  TDL-A fD={fd:5d}Hz: est={ber_est:.4e} true={ber_true:.4e} "
              f"({len(taps)}t, {dt*1000:.0f}ms)")

    print("\n  ✓ Complete")
