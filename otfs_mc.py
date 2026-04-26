"""
MC-OTFS Transceiver -- Multi-Carrier OTFS with TF-domain equalization.

Based on: Garcia Astudillo, "Performance comparison of OTFS vs OFDM
wireless signals in high-mobility environments," UPC, 2026.

Key idea: OTFS as precoded OFDM. DD-domain pilot for channel estimation,
but equalization in TF domain (per-subcarrier MMSE). The DD diversity
comes from the ISFFT spreading, not from DD-domain equalization.

Architecture:
  TX: DD grid (impulse pilot + guard) -> ISFFT -> TF -> IFFT+CP -> time
  RX: CP removal -> FFT -> SFFT -> DD pilot est. -> build H_TF ->
      per-SC MMSE in TF -> SFFT -> DD data

Differences from CP-OTFS (otfs.py):
  - No slow DD-LMMSE or cross-domain iteration
  - Per-subcarrier MMSE in TF domain (O(NM) complexity, same as OFDM)
  - Same pilot structure (DD impulse) for fair comparison
"""

__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "CC-BY-4.0"
__copyright__ = "(c) 2026 Panos N. Alevizos"

import numpy as np
from dataclasses import dataclass


@dataclass
class MCOTFSConfig:
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

    @property
    def sample_rate(self): return self.n_fft * self.scs_hz
    @property
    def sym_samples(self): return self.n_fft + self.n_cp
    @property
    def frame_samples(self): return self.sym_samples * self.n_symbols_per_frame
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


class MCOTFSTransceiver:
    """
    MC-OTFS with DD pilot estimation and TF-domain equalization.

    Uses the same DD impulse pilot + guard zone as CP-OTFS for channel
    estimation, but equalizes in the TF domain via per-subcarrier MMSE.
    This captures the "OTFS as precoded OFDM" viewpoint from the thesis.
    """

    def __init__(self, cfg: MCOTFSConfig):
        self.cfg = cfg
        self._setup()

    def _setup(self):
        cfg = self.cfg
        N, M = cfg.N, cfg.M

        # Guard mask (same as CP-OTFS)
        mask = np.zeros((N, M), dtype=bool)
        for n in range(cfg.doppler_guard_edge):
            mask[n, :] = True
            mask[N - 1 - n, :] = True
        for m in range(cfg.delay_guard_edge):
            mask[:, m] = True
            mask[:, M - 1 - m] = True
        for dn in range(-cfg.pilot_guard_doppler, cfg.pilot_guard_doppler + 1):
            for dm in range(-cfg.pilot_guard_delay, cfg.pilot_guard_delay + 1):
                mask[(cfg.pilot_doppler_idx + dn) % N,
                     (cfg.pilot_delay_idx + dm) % M] = True
        self._guard_mask = mask
        self._data_idx = np.argwhere(~mask)
        self._n_data = len(self._data_idx)

        # SC mapping (DC-centered, skip DC)
        half = M // 2
        sc = np.zeros(M, dtype=int)
        sc[:half] = np.arange(-half, 0)
        sc[half:] = np.arange(1, half + 1)
        self._sc_indices = sc

    def _isfft(self, x_dd):
        N, M = x_dd.shape
        return np.fft.fft(np.fft.ifft(x_dd, axis=0) * np.sqrt(N), axis=1) / np.sqrt(M)

    def _sfft(self, X_tf):
        N, M = X_tf.shape
        return np.fft.ifft(np.fft.fft(X_tf, axis=0) / np.sqrt(N), axis=1) * np.sqrt(M)

    # === TX ===

    def tx(self, qam_symbols):
        cfg = self.cfg
        N, M = cfg.N, cfg.M

        dd = np.zeros((N, M), dtype=complex)
        nd = min(len(qam_symbols), self._n_data)
        dd[self._data_idx[:nd, 0], self._data_idx[:nd, 1]] = qam_symbols[:nd]
        dd[cfg.pilot_doppler_idx, cfg.pilot_delay_idx] = 10 ** (cfg.pilot_power_boost_dB / 20)

        tf = self._isfft(dd)

        tx_sig = np.zeros(cfg.frame_samples, dtype=complex)
        for l in range(N):
            freq = np.zeros(cfg.n_fft, dtype=complex)
            freq[self._sc_indices % cfg.n_fft] = tf[l, :]
            tsym = np.fft.ifft(freq) * np.sqrt(cfg.n_fft)
            start = l * cfg.sym_samples
            tx_sig[start:start + cfg.n_cp] = tsym[-cfg.n_cp:]
            tx_sig[start + cfg.n_cp:start + cfg.sym_samples] = tsym
        return tx_sig, dd

    # === RX ===

    def _estimate_noise_cp(self, rx_signal):
        """
        Estimate noise variance from CP redundancy.

        The CP copies the last n_cp samples of the IFFT output to the front of
        each symbol. After the channel both copies carry the same ISI-free
        convolution product (within one symbol), so their difference is pure
        noise: diff = w_cp - w_tail → σ² = mean(|diff|²) / 2.
        """
        cfg = self.cfg
        sym_samples = cfg.sym_samples   # n_fft + n_cp
        n_cp = cfg.n_cp
        n_fft = cfg.n_fft
        diffs = []
        for l in range(cfg.N):
            start = l * sym_samples
            cp   = rx_signal[start          : start + n_cp]
            tail = rx_signal[start + n_fft  : start + n_fft + n_cp]
            if len(cp) == n_cp and len(tail) == n_cp:
                diffs.append(cp - tail)
        if diffs:
            all_diffs = np.concatenate(diffs)
            return max(float(np.mean(np.abs(all_diffs) ** 2) / 2), 1e-10)
        return 0.01

    def rx(self, rx_signal, noise_var=0.01):  # noise_var unused — self-estimated from CP redundancy
        cfg = self.cfg
        N, M = cfg.N, cfg.M

        noise_var = self._estimate_noise_cp(rx_signal)

        # OFDM demod
        tf_rx = np.zeros((N, M), dtype=complex)
        for l in range(N):
            start = l * cfg.sym_samples + cfg.n_cp
            if start + cfg.n_fft > len(rx_signal):
                break
            freq = np.fft.fft(rx_signal[start:start + cfg.n_fft]) / np.sqrt(cfg.n_fft)
            tf_rx[l, :] = freq[self._sc_indices % cfg.n_fft]

        # SFFT -> DD domain for channel estimation
        dd_rx = self._sfft(tf_rx)

        # DD channel estimation (parabolic interpolation in guard zone)
        taps = self._estimate_channel(dd_rx)

        # Build TF channel from DD taps
        H_tf = self._build_tf_channel(taps)

        # Per-subcarrier MMSE equalization in TF domain
        X_eq = np.conj(H_tf) / (np.abs(H_tf) ** 2 + noise_var) * tf_rx

        # SFFT -> DD domain
        dd_eq = self._sfft(X_eq)

        # Extract data symbols
        data_sym = dd_eq[self._data_idx[:, 0], self._data_idx[:, 1]]
        return data_sym, taps, dd_eq

    def _estimate_channel_dd_ls(self, dd_rx):
        """
        Build H_dd (N×M) from pilot using DD-domain LS (thesis §5.1.7).

        1. Circshift dd_rx to align the pilot to (0,0).
        2. Divide by pilot amplitude x_p → LS channel estimate.
        3. Keep guard zone only: delay dm ∈ {0…g_delay},
           Doppler dn ∈ {-g_doppler…+g_doppler} (wrapped mod N).
           Zero outside guard (data-contaminated for in-frame pilot).
        """
        cfg = self.cfg
        N, M = cfg.N, cfg.M
        pv = 10 ** (cfg.pilot_power_boost_dB / 20)
        np_idx = cfg.pilot_doppler_idx
        mp_idx = cfg.pilot_delay_idx
        gd = cfg.pilot_guard_delay
        gn = cfg.pilot_guard_doppler

        # Align pilot to (0,0)
        Y_circ = np.roll(dd_rx, (-np_idx, -mp_idx), axis=(0, 1))

        # LS estimate: keep guard zone only (causal delays, ±Doppler guard)
        H_dd = np.zeros((N, M), dtype=complex)
        for dm in range(gd + 1):
            for dn in range(-gn, gn + 1):
                row = dn % N
                H_dd[row, dm] = Y_circ[row, dm] / pv
        return H_dd

    def _estimate_channel(self, dd_rx):
        """DD channel estimation from impulse pilot (same as CP-OTFS)."""
        cfg = self.cfg
        N, M = cfg.N, cfg.M
        pv = 10 ** (cfg.pilot_power_boost_dB / 20)
        gd, gn = cfg.pilot_guard_delay, cfg.pilot_guard_doppler
        np_idx, mp_idx = cfg.pilot_doppler_idx, cfg.pilot_delay_idx

        raw = np.zeros((2 * gn + 1, 2 * gd + 1), dtype=complex)
        for dn in range(-gn, gn + 1):
            for dm in range(-gd, gd + 1):
                raw[dn + gn, dm + gd] = dd_rx[(np_idx + dn) % N, (mp_idx + dm) % M] / pv

        taps = []
        raw_pow = np.abs(raw) ** 2
        max_pow = np.max(raw_pow)
        if max_pow < 1e-15:
            return [(0.0, 0, 1.0)]

        for dm_idx in range(2 * gd + 1):
            col = raw[:, dm_idx]
            col_pow = np.abs(col) ** 2
            if np.max(col_pow) < max_pow * 0.015:
                continue
            for dn_idx in range(1, len(col) - 1):
                L, C, R = col_pow[dn_idx - 1], col_pow[dn_idx], col_pow[dn_idx + 1]
                if C < max_pow * 0.02:
                    continue
                if C >= L and C >= R:
                    denom = L - 2 * C + R
                    delta = 0.5 * (L - R) / denom if abs(denom) > 1e-15 else 0.0
                    delta = np.clip(delta, -0.5, 0.5)
                    dn = dn_idx - gn
                    n_frac = dn + delta
                    dm = dm_idx - gd
                    if delta >= 0:
                        h_corr = (1 - delta) * col[dn_idx] + delta * col[min(dn_idx + 1, len(col) - 1)]
                    else:
                        h_corr = (1 + delta) * col[dn_idx] + (-delta) * col[max(dn_idx - 1, 0)]
                    taps.append((n_frac, dm % M, h_corr))
            for dn_idx in [0, len(col) - 1]:
                if col_pow[dn_idx] >= max_pow * 0.05:
                    dn = dn_idx - gn
                    taps.append((float(dn), (dm_idx - gd) % M, col[dn_idx]))

        if not taps:
            pk = np.unravel_index(np.argmax(raw_pow), raw.shape)
            taps.append((float(pk[0] - gn), (pk[1] - gd) % M, raw[pk[0], pk[1]]))
        return taps

    def _build_tf_channel(self, taps):
        """Build TF channel H[l,k] from DD taps."""
        N, M = self.cfg.N, self.cfg.M
        H_tf = np.zeros((N, M), dtype=complex)
        l_vec = np.arange(N, dtype=float)
        k_vec = np.arange(M, dtype=float)
        for (n_frac, m_int, h_val) in taps:
            doppler_phase = np.exp(1j * 2 * np.pi * n_frac * l_vec / N)
            delay_phase = np.exp(-1j * 2 * np.pi * m_int * k_vec / M)
            H_tf += h_val * np.outer(doppler_phase, delay_phase)
        return H_tf

    def count_data_res(self):
        return self._n_data


# === Smoke test ===
if __name__ == '__main__':
    from channel import TDLChannel, TDLChannelConfig
    from qam import qam_modulate, qam_demodulate, generate_random_bits
    import time

    cfg = MCOTFSConfig(n_fft=256, n_active=156, scs_hz=60e3, n_cp=18,
                       n_symbols_per_frame=14, pilot_power_boost_dB=15.0,
                       pilot_guard_delay=8, pilot_guard_doppler=5,
                       doppler_guard_edge=1, delay_guard_edge=2)
    mc = MCOTFSTransceiver(cfg)
    nd = mc.count_data_res()
    fs = cfg.sample_rate
    print(f"=== MC-OTFS Smoke Test ===")
    print(f"  n_data={nd}, OH={cfg.pilot_overhead*100:.1f}%")

    bits = generate_random_bits(nd * 2, np.random.default_rng(42))
    tx, _ = mc.tx(qam_modulate(bits, 4)[:nd])

    # AWGN
    nv = 0.01
    rng = np.random.default_rng(777)
    noise = np.sqrt(nv / 2) * (rng.standard_normal(len(tx)) + 1j * rng.standard_normal(len(tx)))
    dr, _, _ = mc.rx(tx + noise, nv)
    br = qam_demodulate(dr[:nd], 4)
    ber = np.mean(bits[:min(len(bits), len(br))] != br[:min(len(bits), len(br))])
    print(f"  AWGN 20dB: BER={ber:.4e}")

    # Fading
    print("\n  Doppler sweep (TDL-A, DS=30ns, 20dB):")
    for fd in [0, 100, 300, 500, 1000]:
        ch = TDLChannel(TDLChannelConfig('TDL-A', 30e-9, fd, fs, seed=42))
        rx, _ = ch.apply(tx, snr_dB=20)
        t0 = time.time()
        dr, taps, _ = mc.rx(rx, 0.01)
        dt = time.time() - t0
        br = qam_demodulate(dr[:nd], 4)
        ber = np.mean(bits[:min(len(bits), len(br))] != br[:min(len(bits), len(br))])
        print(f"  fD={fd:5d}Hz: BER={ber:.4e} ({len(taps)} taps, {dt*1000:.0f}ms)")
