"""
OTFS Deconvolutional Transceiver — KDDI/Hashimoto method.

Implements the CP-OTFS pipeline from:
    Hashimoto et al., "Channel Estimation and Equalization for CP-OFDM-based
    OTFS in Fractional Doppler Channels," ICC Workshops 2021.
    arXiv:2010.15396

TX path: identical to OTFSTransceiver (ISFFT, SC mapping, CP insertion).
RX path:
    1. OFDM demod (CP removal, FFT, SC extract)
    2. SFFT -> delay-Doppler domain
    3. Upsilon-basis channel estimation (fractional Doppler resolution)
    4. Per-delay-bin deconvolutional Wiener equalization via 2D FFT
"""

__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "CC-BY-4.0"
__copyright__ = "(c) 2026 Panos N. Alevizos"

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from otfs import OTFSTransceiver, OTFSConfig


@dataclass
class DeconvOTFSConfig:
    n_fft: int = 256
    n_active: int = 156
    scs_hz: float = 15e3
    n_cp: int = 18
    n_symbols_per_frame: int = 14
    pilot_power_boost_dB: float = 15.0
    pilot_guard_delay: int = 8
    pilot_guard_doppler: int = 5
    doppler_guard_edge: int = 1
    delay_guard_edge: int = 2
    # Upsilon estimator params
    dividing_number: int = 10       # fractional Doppler resolution = 1/div
    alpha: float = 1.0 / 50        # magnitude threshold
    beta: float = 1.0 / 10         # noise-floor threshold

    @property
    def sample_rate(self): return self.n_fft * self.scs_hz
    @property
    def symbol_duration_samples(self): return self.n_fft + self.n_cp
    @property
    def frame_duration_samples(self): return self.symbol_duration_samples * self.n_symbols_per_frame
    @property
    def M(self): return self.n_active
    @property
    def N(self): return self.n_symbols_per_frame
    @property
    def pilot_doppler_idx(self): return self.N // 2
    @property
    def pilot_delay_idx(self): return self.delay_guard_edge + self.pilot_guard_delay + 1
    @property
    def pilot_overhead(self):
        pg = (2 * self.pilot_guard_delay + 1) * (2 * self.pilot_guard_doppler + 1)
        return pg / (self.M * self.N)


class DeconvOTFSTransceiver:
    """CP-OTFS with Upsilon-basis estimation and deconvolutional Wiener EQ."""

    def __init__(self, cfg: DeconvOTFSConfig):
        self.cfg = cfg
        self._setup_grid()
        self._setup_upsilon()
        # Backing OTFSTransceiver for cross_domain / lmmse detection
        self._otfs = OTFSTransceiver(OTFSConfig(
            n_fft=cfg.n_fft, n_active=cfg.n_active, scs_hz=cfg.scs_hz,
            n_cp=cfg.n_cp, n_symbols_per_frame=cfg.n_symbols_per_frame,
            pilot_power_boost_dB=cfg.pilot_power_boost_dB,
            pilot_guard_delay=cfg.pilot_guard_delay,
            pilot_guard_doppler=cfg.pilot_guard_doppler,
            doppler_guard_edge=cfg.doppler_guard_edge,
            delay_guard_edge=cfg.delay_guard_edge,
        ))

    # ---------------------------------------------------------------
    # Grid setup (identical to OTFSTransceiver)
    # ---------------------------------------------------------------

    def _setup_grid(self):
        cfg = self.cfg
        N, M = cfg.N, cfg.M
        mask = np.zeros((N, M), dtype=bool)

        for n in range(cfg.doppler_guard_edge):
            mask[n, :] = True
            mask[N - 1 - n, :] = True
        for m in range(cfg.delay_guard_edge):
            mask[:, m] = True
            mask[:, M - 1 - m] = True
        for dn in range(-cfg.pilot_guard_doppler, cfg.pilot_guard_doppler + 1):
            for dm in range(-cfg.pilot_guard_delay, cfg.pilot_guard_delay + 1):
                mask[(cfg.pilot_doppler_idx + dn) % N, (cfg.pilot_delay_idx + dm) % M] = True

        self._guard_mask = mask
        self._data_idx = np.argwhere(~mask)
        self._data_n = self._data_idx[:, 0]
        self._data_m = self._data_idx[:, 1]
        self._n_data = len(self._data_idx)

        # SC mapping (DC-centered)
        half = M // 2
        sc = np.zeros(M, dtype=int)
        sc[:half] = np.arange(-half, 0)
        sc[half:] = np.arange(1, half + 1)
        self._sc_indices = sc

    # ---------------------------------------------------------------
    # Upsilon basis pre-computation
    # ---------------------------------------------------------------

    def _setup_upsilon(self):
        """
        Pre-compute Upsilon basis matrix (N x DividingNumber).

        Upsilon_k[n] = (1/N) sum_{i=0}^{N-1} exp(j*2*pi*i*(kappa_k - n)/N)

        This is a fractional-Doppler sinc-like kernel that decorrelates
        paths at sub-integer Doppler offsets.
        """
        N = self.cfg.N
        div = self.cfg.dividing_number
        kappa = np.arange(0, 1, 1.0 / div)  # [0, 1/div, ..., (div-1)/div]

        self._upsilon = np.zeros((N, div), dtype=complex)
        i_vec = np.arange(N, dtype=float)  # [0, 1, ..., N-1]

        for k in range(div):
            x = kappa[k] - np.arange(N, dtype=float)  # (N,)
            # Sum over i: (1/N) * sum_i exp(j*2*pi*i*x/N)
            # = (1/N) * sum_i [ exp(j*2*pi*x/N) ]^i
            # Vectorize: phase_matrix[i, n] = exp(j*2*pi*i*x[n]/N)
            phase = np.exp(1j * 2 * np.pi * np.outer(i_vec, x) / N)  # (N, N)
            self._upsilon[:, k] = np.sum(phase, axis=0) / N

    # ---------------------------------------------------------------
    # ISFFT / SFFT
    # ---------------------------------------------------------------

    def _isfft(self, x_dd):
        N, M = x_dd.shape
        return np.fft.fft(np.fft.ifft(x_dd, axis=0) * np.sqrt(N), axis=1) / np.sqrt(M)

    def _sfft(self, X_tf):
        N, M = X_tf.shape
        return np.fft.ifft(np.fft.fft(X_tf, axis=0) / np.sqrt(N), axis=1) * np.sqrt(M)

    # ---------------------------------------------------------------
    # TX (identical to OTFSTransceiver.tx)
    # ---------------------------------------------------------------

    def tx(self, qam_symbols):
        cfg = self.cfg
        N, M = cfg.N, cfg.M

        dd = np.zeros((N, M), dtype=complex)
        n_data = min(len(qam_symbols), self._n_data)
        dd[self._data_n[:n_data], self._data_m[:n_data]] = qam_symbols[:n_data]
        dd[cfg.pilot_doppler_idx, cfg.pilot_delay_idx] = 10 ** (cfg.pilot_power_boost_dB / 20)

        tf = self._isfft(dd)

        tx_sig = np.zeros(cfg.frame_duration_samples, dtype=complex)
        for l in range(N):
            freq = np.zeros(cfg.n_fft, dtype=complex)
            freq[self._sc_indices % cfg.n_fft] = tf[l, :]
            tsym = np.fft.ifft(freq) * np.sqrt(cfg.n_fft)
            start = l * cfg.symbol_duration_samples
            tx_sig[start:start + cfg.n_cp] = tsym[-cfg.n_cp:]
            tx_sig[start + cfg.n_cp:start + cfg.symbol_duration_samples] = tsym
        return tx_sig, dd

    # ---------------------------------------------------------------
    # RX
    # ---------------------------------------------------------------

    def _estimate_noise_cp(self, rx_signal):
        """
        Estimate noise variance from CP redundancy.

        The CP copies the last n_cp samples of the IFFT output to the front of
        each symbol. After the channel both copies carry the same ISI-free
        convolution product (within one symbol), so their difference is pure
        noise: diff = w_cp - w_tail → σ² = mean(|diff|²) / 2.
        """
        cfg = self.cfg
        sym_dur = cfg.symbol_duration_samples   # n_fft + n_cp
        n_cp = cfg.n_cp
        n_fft = cfg.n_fft
        diffs = []
        for l in range(cfg.N):
            start = l * sym_dur
            cp   = rx_signal[start         : start + n_cp]
            tail = rx_signal[start + n_fft : start + n_fft + n_cp]
            if len(cp) == n_cp and len(tail) == n_cp:
                diffs.append(cp - tail)
        if diffs:
            all_diffs = np.concatenate(diffs)
            return max(float(np.mean(np.abs(all_diffs) ** 2) / 2), 1e-10)
        return 0.01

    def rx(self, rx_signal, noise_var=0.01, detector='cross_domain', n_iter=5):  # noise_var unused — self-estimated from CP redundancy
        """
        Full RX pipeline: OFDM demod -> SFFT -> Upsilon estimation -> detection.

        Args:
            rx_signal: received time-domain signal
            noise_var: ignored — noise variance is estimated internally from CP
                       redundancy. Argument kept for call-site compatibility.
            detector: 'cross_domain' (default, best), 'deconv' (KDDI original),
                      or 'lmmse' (full matrix)
            n_iter: iterations for cross_domain detector

        Returns:
            data_sym: equalized data symbols (n_data,)
            paths: list of (gain, doppler, delay, phase)
            dd_rx: raw DD-domain received grid (N, M)
        """
        cfg = self.cfg
        N, M = cfg.N, cfg.M

        noise_var = self._estimate_noise_cp(rx_signal)

        # OFDM demod
        tf_rx = np.zeros((N, M), dtype=complex)
        for l in range(N):
            start = l * cfg.symbol_duration_samples + cfg.n_cp
            if start + cfg.n_fft > len(rx_signal):
                break
            freq = np.fft.fft(rx_signal[start:start + cfg.n_fft]) / np.sqrt(cfg.n_fft)
            tf_rx[l, :] = freq[self._sc_indices % cfg.n_fft]

        # SFFT -> DD domain
        dd_rx = self._sfft(tf_rx)

        # Channel estimation:
        #   cross_domain / lmmse: use the parabolic DD-pilot estimator (same as CP-OTFS /
        #     MC-OTFS). Reliable for 3GPP TDL-Jakes channels — the Jakes Doppler spectrum
        #     is continuous and does NOT match the Upsilon point-path model.
        #   deconv: Upsilon estimator preserved for discrete-path geometric channels where
        #     the point-path assumption holds. Not suitable for TDL-Jakes at high Doppler.
        if detector == 'deconv':
            paths = self._estimate_channel_upsilon(dd_rx, noise_var)
            taps = self._paths_to_taps(paths)
            data_sym = self._detect_deconv(dd_rx, paths, noise_var)
        else:
            # DD-grid estimator: use all integer Doppler bins in the pilot guard zone.
            # Each (dn, dm) bin becomes one tap; _build_tf_channel sums them as a partial
            # 2D IDFT, capturing the Jakes Doppler spread that single-peak parabolic misses.
            taps = self._estimate_channel_dd_grid(dd_rx, noise_var)
            paths = []  # not used by cross_domain / lmmse
            if detector == 'cross_domain':
                data_sym = self._otfs._detect_cross_domain(tf_rx, dd_rx, taps, noise_var, n_iter)
            elif detector == 'lmmse':
                data_sym = self._otfs._detect_lmmse(dd_rx, taps, noise_var)
            else:
                raise ValueError(f"Unknown detector: {detector}")

        return data_sym, paths, dd_rx

    def _estimate_channel_dd_grid(self, dd_rx, noise_var):
        """
        Estimate channel as a full Doppler-grid tap set from the pilot guard zone.

        For each integer Doppler offset dn ∈ {-gn,...,gn} and causal delay offset
        dm ∈ {0,...,gd}, extract:
            h_val = dd_rx[(np_idx + dn) % N, (mp_idx + dm) % M] / pilot_amplitude

        Each surviving (dn, dm) becomes one tap (n_frac=dn, m_int=dm, h_val).
        _build_tf_channel then computes:
            H_tf[l,k] = Σ h_val · exp(j2π·dn·l/N) · exp(-j2π·dm·k/M)

        This partial 2D IDFT is a Fourier-series fit to the N-sample per-symbol
        channel sequence. For fD=500 Hz (ν_max≈0.47 DD bins), the dn=±1 bins carry
        Clarke-spectrum leakage energy from sinusoids near ±ν_max, improving the
        time-varying channel model vs the static single-tap from parabolic.
        At fD=0 all dn≠0 bins are noise — the noise-adaptive threshold removes them,
        avoiding regression on static channels.

        Threshold: max of 0.5% relative and 5× per-bin pilot-normalized noise floor.
        """
        cfg = self.cfg
        N, M = cfg.N, cfg.M
        pv = 10 ** (cfg.pilot_power_boost_dB / 20)
        gd = cfg.pilot_guard_delay
        gn = cfg.pilot_guard_doppler
        np_idx = cfg.pilot_doppler_idx
        mp_idx = cfg.pilot_delay_idx

        # Per-bin noise power in the pilot-normalised domain: W[n,m]/pv has noise var = noise_var/pv²
        noise_per_bin = noise_var / (pv ** 2)

        # Collect all (dn, dm) responses and track peak power
        candidates = []
        max_pow = 0.0
        for dn in range(-gn, gn + 1):
            for dm in range(gd + 1):   # causal only: dm >= 0
                n_idx = (np_idx + dn) % N
                m_idx = (mp_idx + dm) % M
                h_val = dd_rx[n_idx, m_idx] / pv
                pw = float(np.abs(h_val) ** 2)
                candidates.append((float(dn), dm, h_val, pw))
                if pw > max_pow:
                    max_pow = pw

        if max_pow < 1e-15:
            return [(0.0, 0, 1.0)]

        # Two-part threshold: relative (0.5% of peak) OR absolute (5× noise floor).
        # The absolute floor prevents noisy bins from being included at moderate SNR
        # with low Doppler, where dn≠0 bins carry no signal (only noise).
        threshold = max(max_pow * 0.005, 5.0 * noise_per_bin)
        taps = [(dn, dm, h_val) for dn, dm, h_val, pw in candidates if pw >= threshold]

        if not taps:
            best = max(candidates, key=lambda x: x[3])
            taps = [(best[0], best[1], best[2])]

        return taps

    def _paths_to_taps(self, paths):
        """Convert Upsilon-estimated paths to (n_frac, m_int, h_complex) tap format."""
        taps = []
        for gain, dop, delay_bin, phase in paths:
            h_complex = gain * np.exp(1j * phase)
            taps.append((dop, delay_bin, h_complex))
        return taps

    # ---------------------------------------------------------------
    # Upsilon-basis channel estimation (Hashimoto et al.)
    # ---------------------------------------------------------------

    def _estimate_channel_upsilon(self, dd_rx, noise_var):
        """
        Estimate channel paths from pilot response using Upsilon basis
        cross-correlation with iterative path extraction and dual thresholding.

        Port of OtfsPilotResponseBasedPathParameterEstimator.estimate().

        The pilot at (pilot_doppler, pilot_delay) creates a channel response
        in the DD grid: a tap with delay d and Doppler k appears at
        (pilot_doppler + k, pilot_delay + d). We extract this response
        into Hdd_full[d, k] for d in [0, gd], k in [-gn, gn].

        Returns:
            paths: list of (gain, doppler_in_sample, delay_in_sample, phase_offset)
                   where doppler_in_sample is in units of DD grid bins (fractional),
                   and delay_in_sample is integer delay bin index.
        """
        cfg = self.cfg
        N, M = cfg.N, cfg.M
        Ncp = cfg.n_cp
        div = cfg.dividing_number
        alpha = cfg.alpha
        beta = cfg.beta
        pv = 10 ** (cfg.pilot_power_boost_dB / 20)
        gd = cfg.pilot_guard_delay
        gn = cfg.pilot_guard_doppler

        # Extract pilot response: Hdd_full[d, n] = channel response at delay d, Doppler n
        # After dividing by pilot value, entries are channel gains directly.
        # Only positive delays (causal channel) and the full Doppler range.
        Hdd_full = np.zeros((gd + 1, N), dtype=complex)
        for dm in range(gd + 1):  # delay offsets 0..gd
            for dn in range(-gn, gn + 1):  # Doppler offsets
                n_idx = (cfg.pilot_doppler_idx + dn) % N
                m_idx = (cfg.pilot_delay_idx + dm) % M
                Hdd_full[dm, dn % N] = dd_rx[n_idx, m_idx] / pv

        # Scale noise variance to match the pilot-normalized signal level
        # The pilot response has been divided by pv, so the effective noise
        # on Hdd_full is noise_var / pv^2 (in power), sqrt of that for amplitude
        noise_var_eff = noise_var / (pv ** 2)

        paths = []
        n_delays = gd + 1  # only scan delay bins with potential energy

        for delay in range(n_delays):
            Hprime = Hdd_full[delay, :].copy()
            sum_magnitude = np.abs(np.sum(Hdd_full[delay, :]))

            if sum_magnitude < 1e-15:
                continue

            prev_gain = np.inf

            while True:
                # Cross-correlate with each Upsilon basis vector via FFT
                crosscorr = np.zeros((div, N), dtype=complex)
                fft_Hp = np.fft.fft(Hprime)
                for k in range(div):
                    crosscorr[k, :] = np.fft.ifft(
                        fft_Hp * np.conj(np.fft.fft(self._upsilon[:, k]))
                    )

                # Flatten and circshift to center (Doppler 0 in the middle)
                crosscorrvec = np.roll(crosscorr.flatten(), N * div // 2)

                # Sort by magnitude, apply dual threshold
                magnitudes = np.abs(crosscorrvec)
                sorted_idx = np.argsort(magnitudes)[::-1]

                # Condition 1: magnitude > alpha * sum_magnitude
                valid = magnitudes[sorted_idx] >= alpha * sum_magnitude
                sorted_idx = sorted_idx[valid]

                # Condition 2: magnitude > beta * sqrt(effective noise variance)
                if noise_var_eff > 0:
                    valid2 = magnitudes[sorted_idx] >= beta * np.sqrt(noise_var_eff)
                    sorted_idx = sorted_idx[valid2]

                if len(sorted_idx) == 0:
                    break

                largest_idx = sorted_idx[0]
                est_gain = magnitudes[largest_idx]

                # Stop if gain is increasing (diverging — indicates residual noise)
                if est_gain >= prev_gain:
                    break
                prev_gain = est_gain

                # Convert flat index to Doppler value
                est_doppler = largest_idx / div - N / 2  # centered Doppler index

                # Phase offset: account for CP-induced Doppler shift
                est_doppler_shift = np.exp(
                    1j * 2 * np.pi * est_doppler * (Ncp - delay) / ((cfg.n_fft + Ncp) * N)
                )
                est_initial_phase = (
                    crosscorrvec[largest_idx] / np.abs(crosscorrvec[largest_idx])
                    * np.conj(est_doppler_shift)
                )

                paths.append((
                    est_gain,
                    est_doppler,       # in DD grid bins (fractional)
                    delay,             # delay bin index
                    np.angle(est_initial_phase),
                ))

                # Subtract found path from Hprime and continue
                int_doppler = int(np.floor(est_doppler))
                frac_part = est_doppler % 1.0
                frac_idx = int(np.floor(frac_part * div + 1e-9)) % div
                Hprime = Hprime - (
                    est_gain
                    * np.exp(1j * np.angle(est_initial_phase))
                    * np.roll(self._upsilon[:, frac_idx], int_doppler)
                )

        return paths

    # ---------------------------------------------------------------
    # Deconvolutional Wiener equalizer (Hashimoto et al.)
    # ---------------------------------------------------------------

    def _detect_deconv(self, dd_rx, paths, noise_var):
        """
        Per-delay-bin 2D deconvolutional equalization via FFT.

        Port of OtfsDeconvolutionalEqualizer.equalize().

        For each delay bin l:
            1. Reconstruct the effective DD channel Hdd[l,:] from all paths
            2. 2D FFT of both Hdd and Ydd
            3. Wiener filter: X = ifft2(conj(H)*Y / (|H|^2 + sigma^2))
            4. Extract row l from the result
        """
        cfg = self.cfg
        N, M = cfg.N, cfg.M
        Ncp = cfg.n_cp
        div = cfg.dividing_number

        if len(paths) == 0:
            return dd_rx[self._data_n, self._data_m]

        # Transpose dd_rx to (M, N) — KDDI convention (delay, doppler)
        Ydd = dd_rx.T  # our convention is (N, M) = (doppler, delay); KDDI is (M, N) = (delay, doppler)
        # Wait — our dd_rx is (N, M) where axis 0=doppler, axis 1=delay
        # KDDI's Ydd is (M, N) where axis 0=delay, axis 1=doppler
        # So: Ydd_kddi[m, n] = dd_rx[n, m]  => Ydd = dd_rx.T
        Ydd_dft = np.fft.fft2(Ydd)

        # Pre-compute Lambda: effective channel per path per delay bin
        # lambda[p, l, n] = gain_p * delta(delay_p == l) * circshift(upsilon_frac, int_doppler)
        P = len(paths)
        lam = np.zeros((P, M, N), dtype=complex)

        for p, (gain, dop, delay_bin, phase) in enumerate(paths):
            int_dop = int(np.floor(round(dop % N + 1e-9, 1))) % N
            frac_part = (dop % N) - int_dop
            frac_idx = int(round(frac_part * div + 1e-9)) % div
            # Only non-zero at the path's delay bin
            lam[p, delay_bin, :] = gain * np.roll(self._upsilon[:, frac_idx], int_dop)

        # Equalize per delay bin
        Xdd = np.zeros((M, N), dtype=complex)
        for l in range(M):
            # Build Hdd for this delay bin from all paths
            # psi_p = exp(j*2*pi * dop_p * (Ncp - delay_p + l) / ((M+Ncp)*N))
            # phi_p = exp(j * phase_p)
            # Hdd[l] = sum_p psi_p * phi_p * lambda_p[l,:]
            Hdd_l = np.zeros((M, N), dtype=complex)
            for p, (gain, dop, delay_bin, phase) in enumerate(paths):
                psi = np.exp(1j * 2 * np.pi * dop * (Ncp - delay_bin + l) / ((cfg.n_fft + Ncp) * N))
                phi = np.exp(1j * phase)
                # lam[p] is (M, N) but only row delay_bin is non-zero
                Hdd_l += psi * phi * lam[p]

            Hdd_dft = np.fft.fft2(Hdd_l)

            # Wiener equalization with estimation error correction (KDDI Experimental 3)
            denom = np.abs(Hdd_dft) ** 2 + noise_var
            threshold = noise_var * 0.1
            if np.any(denom < threshold):
                # Estimate channel estimation error variance
                Pt = np.sqrt(M * N) * 0.9472  # approx for 16QAM; close enough for QPSK
                Pr = np.mean(np.abs(Ydd_dft))
                Ph = Pr - noise_var * np.sqrt(M * N)
                Ph_est = np.mean(np.abs(Hdd_dft))
                est_err = max((Ph_est * Pt - Ph) / np.sqrt(M * N) / 2, 0.0)
                denom = denom + est_err

            eq = np.conj(Hdd_dft) / denom
            Xdd_tmp = np.fft.ifft2(Ydd_dft * eq)
            Xdd[l, :] = Xdd_tmp[l, :]

        # Convert back to our (doppler, delay) convention and extract data
        dd_eq = Xdd.T  # (N, M)
        return dd_eq[self._data_n, self._data_m]

    def count_data_res(self):
        return self._n_data


# ---------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------

if __name__ == '__main__':
    from channel import TDLChannel, TDLChannelConfig
    from qam import qam_modulate, qam_demodulate, generate_random_bits
    import time

    print("=== Deconv OTFS Smoke Test ===\n")

    cfg = DeconvOTFSConfig(
        n_fft=256, n_active=156, scs_hz=60e3, n_cp=18,
        n_symbols_per_frame=14, pilot_power_boost_dB=15.0,
        pilot_guard_delay=8, pilot_guard_doppler=5,
        doppler_guard_edge=1, delay_guard_edge=2,
        dividing_number=10, alpha=1.0 / 50, beta=1.0 / 10,
    )
    trx = DeconvOTFSTransceiver(cfg)
    n_data = trx.count_data_res()
    fs = cfg.sample_rate
    print(f"  n_data={n_data}, OH={cfg.pilot_overhead * 100:.1f}%")

    bits = generate_random_bits(n_data * 2, np.random.default_rng(42))
    syms = qam_modulate(bits, 4)[:n_data]
    tx_sig, _ = trx.tx(syms)

    # AWGN-only test
    rng = np.random.RandomState(42)
    sp = np.mean(np.abs(tx_sig) ** 2)
    nv_20 = sp * 10 ** (-20 / 10)
    rx_awgn = tx_sig + np.sqrt(nv_20 / 2) * (rng.randn(len(tx_sig)) + 1j * rng.randn(len(tx_sig)))
    dr, paths, _ = trx.rx(rx_awgn, nv_20 / sp)
    br = qam_demodulate(dr[:n_data], 4)
    n = min(len(bits), len(br))
    ber_awgn = np.mean(bits[:n] != br[:n])
    print(f"  AWGN 20dB: BER={ber_awgn:.4e}, {len(paths)} paths")

    # TDL-A sweep: compare cross_domain (default) vs deconv
    print("\n  Doppler sweep (TDL-A, DS=30ns, SNR=20dB):")
    print(f"  {'fD':>6s}  {'CrossDomain':>12s}  {'Deconv':>10s}  {'paths':>6s}")
    for fd in [0, 100, 300, 500, 1000]:
        ch = TDLChannel(TDLChannelConfig('TDL-A', 30e-9, fd, fs, seed=42))
        rx, _ = ch.apply(tx_sig, snr_dB=20)
        nv = 10 ** (-20 / 10)

        dr_cd, paths, _ = trx.rx(rx, nv, detector='cross_domain', n_iter=5)
        br_cd = qam_demodulate(dr_cd[:n_data], 4)
        n = min(len(bits), len(br_cd))
        ber_cd = np.mean(bits[:n] != br_cd[:n])

        ch2 = TDLChannel(TDLChannelConfig('TDL-A', 30e-9, fd, fs, seed=42))
        rx2, _ = ch2.apply(tx_sig, snr_dB=20)
        dr_dc, _, _ = trx.rx(rx2, nv, detector='deconv')
        br_dc = qam_demodulate(dr_dc[:n_data], 4)
        ber_dc = np.mean(bits[:n] != br_dc[:n])

        print(f"  {fd:5d}Hz  {ber_cd:12.4e}  {ber_dc:10.4e}  {len(paths):6d}")

    # SNR sweep with cross_domain
    print("\n  SNR sweep (TDL-A, DS=30ns, fD=500Hz, cross_domain):")
    for snr in [5, 10, 15, 20, 25, 30]:
        ch = TDLChannel(TDLChannelConfig('TDL-A', 30e-9, 500, fs, seed=42))
        rx, _ = ch.apply(tx_sig, snr_dB=snr)
        dr, paths, _ = trx.rx(rx, 10 ** (-snr / 10), detector='cross_domain', n_iter=5)
        br = qam_demodulate(dr[:n_data], 4)
        n = min(len(bits), len(br))
        ber = np.mean(bits[:n] != br[:n])
        print(f"    SNR={snr:2d}dB: BER={ber:.4e} ({len(paths)} paths)")
