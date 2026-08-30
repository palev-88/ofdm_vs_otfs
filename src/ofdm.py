"""
OFDM Transceiver — 5G NR style.

TX: Map to DC-centered TF grid → IFFTSHIFT → IFFT → CP
RX: Remove CP → FFT → FFTSHIFT → Channel Est → Equalize → Demap

Channel estimation strategies:
    A: Linear interpolation (baseline)
    B: Wiener interpolation (MMSE optimal for Jakes)
    C: Phase ramp tracking (per-subcarrier Doppler tracker)

Equalization:
    ZF or MMSE single-tap per subcarrier
    Optional ICI cancellation (iterative)
"""

__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from qam import qam_modulate, qam_demodulate, generate_random_bits, qam_constellation


@dataclass
class OFDMConfig:
    """OFDM waveform configuration."""
    n_fft: int = 1024               # FFT size
    n_active: int = 624             # Active subcarriers (excl. DC)
    scs_hz: float = 15e3            # Subcarrier spacing [Hz]
    n_cp: int = 72                  # CP length [samples]
    n_symbols_per_slot: int = 14    # OFDM symbols per slot (5G NR)
    
    # DMRS configuration
    dmrs_symbol_indices: List[int] = field(default_factory=lambda: [2, 11])
    dmrs_comb_size: int = 2         # Pilot every K subcarriers
    
    # Derived
    @property
    def sample_rate(self) -> float:
        return self.n_fft * self.scs_hz
    
    @property
    def symbol_duration_samples(self) -> int:
        return self.n_fft + self.n_cp
    
    @property
    def slot_duration_samples(self) -> int:
        return self.symbol_duration_samples * self.n_symbols_per_slot
    
    @property
    def bw_hz(self) -> float:
        return self.n_active * self.scs_hz
    
    @property
    def n_data_symbols(self) -> int:
        return self.n_symbols_per_slot - len(self.dmrs_symbol_indices)
    
    @property 
    def n_dmrs_sc(self) -> int:
        return self.n_active // self.dmrs_comb_size
    
    @property
    def n_data_sc_per_dmrs_sym(self) -> int:
        return self.n_active - self.n_dmrs_sc
    
    @property
    def n_data_res(self) -> int:
        """Total data resource elements per slot."""
        return (self.n_data_symbols * self.n_active + 
                len(self.dmrs_symbol_indices) * self.n_data_sc_per_dmrs_sym)
    
    @property
    def pilot_overhead(self) -> float:
        total = self.n_symbols_per_slot * self.n_active
        pilots = len(self.dmrs_symbol_indices) * self.n_dmrs_sc
        return pilots / total


class OFDMTransceiver:
    """OFDM transceiver with 5G NR-style resource mapping."""
    
    def __init__(self, cfg: OFDMConfig):
        self.cfg = cfg
        
        # Generate DMRS sequence (QPSK, known)
        self._dmrs_seq = self._generate_dmrs()
    
    def _generate_dmrs(self) -> np.ndarray:
        """Generate DMRS pilot sequence (QPSK, deterministic)."""
        rng = np.random.default_rng(31415)  # Fixed seed for pilots
        n_pilots = self.cfg.n_dmrs_sc
        bits = rng.integers(0, 2, size=n_pilots * 2)
        return qam_modulate(bits, 4)
    
    def _sc_indices(self) -> np.ndarray:
        """Active subcarrier indices in DC-centered grid."""
        n = self.cfg.n_active
        # Centered around DC, excluding DC (index 0)
        # Negative: -n//2 to -1, Positive: +1 to +n//2
        neg = np.arange(-n // 2, 0)
        pos = np.arange(1, n // 2 + 1)
        return np.concatenate([neg, pos])
    
    def _dmrs_sc_indices_within_active(self) -> np.ndarray:
        """DMRS subcarrier indices within the active subcarrier range."""
        return np.arange(0, self.cfg.n_active, self.cfg.dmrs_comb_size)
    
    def _data_sc_indices_in_dmrs_symbol(self) -> np.ndarray:
        """Data subcarrier indices within a DMRS symbol."""
        dmrs_idx = set(self._dmrs_sc_indices_within_active())
        return np.array([i for i in range(self.cfg.n_active) if i not in dmrs_idx])
    
    # ═══════════════════════════════════════════════════════════
    # TRANSMITTER
    # ═══════════════════════════════════════════════════════════
    
    def tx(self, qam_symbols: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        OFDM transmitter.
        
        Args:
            qam_symbols: Data symbols to transmit
        
        Returns:
            tx_signal: Time-domain baseband signal
            tf_grid: (n_symbols, n_active) TF grid for reference
        """
        cfg = self.cfg
        n_sym = cfg.n_symbols_per_slot
        n_act = cfg.n_active
        n_fft = cfg.n_fft
        
        # Build TF resource grid (DC-centered, active SCs only)
        tf_grid = np.zeros((n_sym, n_act), dtype=complex)
        
        # Map data and DMRS
        data_idx = 0
        dmrs_sc_idx = self._dmrs_sc_indices_within_active()
        data_sc_in_dmrs = self._data_sc_indices_in_dmrs_symbol()
        
        for l in range(n_sym):
            if l in cfg.dmrs_symbol_indices:
                # DMRS symbol: pilots at comb positions, data elsewhere
                tf_grid[l, dmrs_sc_idx] = self._dmrs_seq
                n_data_here = len(data_sc_in_dmrs)
                if data_idx + n_data_here <= len(qam_symbols):
                    tf_grid[l, data_sc_in_dmrs] = qam_symbols[data_idx:data_idx + n_data_here]
                    data_idx += n_data_here
            else:
                # Data symbol: all active SCs carry data
                n_data_here = n_act
                if data_idx + n_data_here <= len(qam_symbols):
                    tf_grid[l, :] = qam_symbols[data_idx:data_idx + n_data_here]
                    data_idx += n_data_here
        
        # Modulate: for each OFDM symbol, map to FFT bins, IFFT, add CP
        tx_signal = np.zeros(cfg.slot_duration_samples, dtype=complex)
        sc_indices = self._sc_indices()  # DC-centered indices
        
        for l in range(n_sym):
            # Map active SCs to FFT grid (DC-centered → IFFT order)
            freq_grid = np.zeros(n_fft, dtype=complex)
            for i, sc in enumerate(sc_indices):
                freq_grid[sc % n_fft] = tf_grid[l, i]  # Wrap negative to end
            
            # IFFT (no need for explicit ifftshift — we mapped directly)
            time_sym = np.fft.ifft(freq_grid) * np.sqrt(n_fft)
            
            # Add CP
            cp = time_sym[-cfg.n_cp:]
            ofdm_sym = np.concatenate([cp, time_sym])
            
            # Place in output
            start = l * cfg.symbol_duration_samples
            tx_signal[start:start + cfg.symbol_duration_samples] = ofdm_sym
        
        return tx_signal, tf_grid
    
    # ═══════════════════════════════════════════════════════════
    # RECEIVER
    # ═══════════════════════════════════════════════════════════
    
    def rx(self, rx_signal: np.ndarray, 
           ch_est_method: str = 'linear',
           eq_method: str = 'mmse',
           noise_var: float = 0.01,
           f_doppler: float = 0.0,
           noise_est_method: str = None
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        OFDM receiver.
        
        Args:
            rx_signal: Received time-domain signal
            ch_est_method: 'linear', 'wiener', 'phase_ramp'
            eq_method: 'zf' or 'mmse'
            noise_var: Noise variance (used if noise_est_method is None)
            f_doppler: Max Doppler freq for Wiener interpolation
            noise_est_method: 'dmrs_residual', 'dmrs_diff', 'dmrs_power'
                             If set, overrides noise_var with estimate.
        
        Returns:
            data_symbols: Equalized data symbols
            H_est: (n_symbols, n_active) estimated channel
            tf_grid_rx: (n_symbols, n_active) received TF grid
        """
        cfg = self.cfg
        n_sym = cfg.n_symbols_per_slot
        n_fft = cfg.n_fft
        sc_indices = self._sc_indices()
        
        # Demodulate: remove CP, FFT, extract active SCs
        tf_grid_rx = np.zeros((n_sym, cfg.n_active), dtype=complex)
        
        for l in range(n_sym):
            start = l * cfg.symbol_duration_samples + cfg.n_cp  # Skip CP
            if start + n_fft > len(rx_signal):
                break
            time_sym = rx_signal[start:start + n_fft]
            
            # FFT
            freq_grid = np.fft.fft(time_sym) / np.sqrt(n_fft)
            
            # Extract active SCs
            for i, sc in enumerate(sc_indices):
                tf_grid_rx[l, i] = freq_grid[sc % n_fft]
        
        # Channel estimation
        H_est = self._channel_estimate(tf_grid_rx, ch_est_method, 
                                        noise_var, f_doppler)
        
        # Noise estimation (if requested, overrides passed noise_var)
        if noise_est_method is not None:
            noise_var = self._estimate_noise(tf_grid_rx, H_est, noise_est_method)
        
        # Equalization
        tf_eq = self._equalize(tf_grid_rx, H_est, eq_method, noise_var)
        
        # Extract data symbols (skip DMRS positions)
        data_symbols = self._extract_data(tf_eq)
        
        return data_symbols, H_est, tf_grid_rx
    
    def _channel_estimate(self, tf_rx: np.ndarray, method: str,
                          noise_var: float, f_doppler: float) -> np.ndarray:
        """
        Estimate channel H[l,k] from DMRS observations.
        
        1. LS at DMRS positions
        2. Frequency interpolation within each DMRS symbol
        3. Time interpolation between DMRS symbols
        """
        cfg = self.cfg
        n_sym = cfg.n_symbols_per_slot
        n_act = cfg.n_active
        dmrs_sc_idx = self._dmrs_sc_indices_within_active()
        dmrs_syms = cfg.dmrs_symbol_indices
        
        # Step 1: LS estimation at pilot positions
        H_ls = {}  # {symbol_idx: full interpolated channel}
        for l in dmrs_syms:
            # LS: H = Y/X at pilot positions
            h_pilots = tf_rx[l, dmrs_sc_idx] / self._dmrs_seq
            
            # Step 2: Frequency interpolation (linear between pilots)
            h_full = np.zeros(n_act, dtype=complex)
            h_full[dmrs_sc_idx] = h_pilots
            
            # Linear interpolation in frequency
            pilot_positions = dmrs_sc_idx.astype(float)
            for k in range(n_act):
                if k in dmrs_sc_idx:
                    continue
                # Find surrounding pilots
                left = pilot_positions[pilot_positions <= k]
                right = pilot_positions[pilot_positions > k]
                if len(left) == 0:
                    h_full[k] = h_pilots[0]
                elif len(right) == 0:
                    h_full[k] = h_pilots[-1]
                else:
                    l_idx = int(np.argmin(np.abs(dmrs_sc_idx - k)))
                    r_idx = min(l_idx + 1, len(dmrs_sc_idx) - 1)
                    if l_idx == r_idx:
                        h_full[k] = h_pilots[l_idx]
                    else:
                        alpha = (k - dmrs_sc_idx[l_idx]) / max(
                            dmrs_sc_idx[r_idx] - dmrs_sc_idx[l_idx], 1)
                        h_full[k] = (1 - alpha) * h_pilots[l_idx] + alpha * h_pilots[r_idx]
            
            H_ls[l] = h_full
        
        # Step 3: Time interpolation
        H_est = np.zeros((n_sym, n_act), dtype=complex)
        
        if method == 'linear':
            H_est = self._time_interp_linear(H_ls, n_sym, n_act, dmrs_syms)
        elif method == 'wiener':
            H_est = self._time_interp_wiener(H_ls, n_sym, n_act, dmrs_syms,
                                              f_doppler, noise_var)
        elif method == 'phase_ramp':
            H_est = self._time_interp_phase_ramp(H_ls, n_sym, n_act, dmrs_syms)
        else:
            raise ValueError(f"Unknown ch_est_method: {method}")
        
        return H_est
    
    def _time_interp_linear(self, H_ls: dict, n_sym: int, n_act: int,
                            dmrs_syms: list) -> np.ndarray:
        """Linear interpolation in time between DMRS symbols."""
        H_est = np.zeros((n_sym, n_act), dtype=complex)
        
        sorted_dmrs = sorted(dmrs_syms)
        
        for l in range(n_sym):
            if l in H_ls:
                H_est[l, :] = H_ls[l]
            else:
                # Find surrounding DMRS symbols
                before = [d for d in sorted_dmrs if d <= l]
                after = [d for d in sorted_dmrs if d > l]
                
                if not before:
                    H_est[l, :] = H_ls[sorted_dmrs[0]]
                elif not after:
                    H_est[l, :] = H_ls[sorted_dmrs[-1]]
                else:
                    l1 = before[-1]
                    l2 = after[0]
                    alpha = (l - l1) / (l2 - l1)
                    H_est[l, :] = (1 - alpha) * H_ls[l1] + alpha * H_ls[l2]
        
        return H_est
    
    def _time_interp_wiener(self, H_ls: dict, n_sym: int, n_act: int,
                            dmrs_syms: list, f_doppler: float,
                            noise_var: float) -> np.ndarray:
        """Wiener (MMSE) interpolation using Jakes autocorrelation."""
        from scipy.special import j0 as bessel_j0
        
        H_est = np.zeros((n_sym, n_act), dtype=complex)
        sorted_dmrs = sorted(dmrs_syms)
        n_pilots_time = len(sorted_dmrs)
        
        T_sym = (self.cfg.n_fft + self.cfg.n_cp) / self.cfg.sample_rate
        
        # Autocorrelation matrix of channel at pilot times
        R_pp = np.zeros((n_pilots_time, n_pilots_time), dtype=complex)
        for i in range(n_pilots_time):
            for j in range(n_pilots_time):
                dt = abs(sorted_dmrs[i] - sorted_dmrs[j]) * T_sym
                R_pp[i, j] = bessel_j0(2 * np.pi * f_doppler * dt)
        
        # Add noise
        R_pp += noise_var * np.eye(n_pilots_time)
        
        # For each data symbol, compute Wiener weights
        R_pp_inv = np.linalg.inv(R_pp)
        
        # Pilot channel values
        H_pilot_matrix = np.zeros((n_pilots_time, n_act), dtype=complex)
        for idx, l in enumerate(sorted_dmrs):
            H_pilot_matrix[idx, :] = H_ls[l]
        
        for l in range(n_sym):
            if l in H_ls:
                H_est[l, :] = H_ls[l]
            else:
                # Cross-correlation between target and pilots
                r_dp = np.zeros(n_pilots_time, dtype=complex)
                for idx, lp in enumerate(sorted_dmrs):
                    dt = abs(l - lp) * T_sym
                    r_dp[idx] = bessel_j0(2 * np.pi * f_doppler * dt)
                
                # Wiener weights
                w = R_pp_inv @ r_dp
                H_est[l, :] = w @ H_pilot_matrix
        
        return H_est
    
    def _time_interp_phase_ramp(self, H_ls: dict, n_sym: int, n_act: int,
                                 dmrs_syms: list) -> np.ndarray:
        """Phase ramp tracking: estimate per-subcarrier phase slope."""
        H_est = np.zeros((n_sym, n_act), dtype=complex)
        sorted_dmrs = sorted(dmrs_syms)
        
        if len(sorted_dmrs) < 2:
            # Fallback to nearest
            for l in range(n_sym):
                nearest = min(sorted_dmrs, key=lambda d: abs(d - l))
                H_est[l, :] = H_ls[nearest]
            return H_est
        
        # Compute phase slope between first two DMRS symbols
        l1, l2 = sorted_dmrs[0], sorted_dmrs[1]
        h1, h2 = H_ls[l1], H_ls[l2]
        
        # Per-subcarrier phase rate
        phase_diff = np.angle(h2 * np.conj(h1))
        phase_rate = phase_diff / (l2 - l1)  # radians per OFDM symbol
        
        # Amplitude: linear interpolation
        amp1 = np.abs(h1)
        amp2 = np.abs(h2)
        
        for l in range(n_sym):
            if l in H_ls:
                H_est[l, :] = H_ls[l]
            else:
                # Find nearest DMRS
                nearest = min(sorted_dmrs, key=lambda d: abs(d - l))
                dl = l - nearest
                
                # Amplitude: linear interp
                alpha = (l - l1) / max(l2 - l1, 1)
                alpha = np.clip(alpha, 0, 1)
                amp = (1 - alpha) * amp1 + alpha * amp2
                
                # Phase: extrapolate from nearest
                phase = np.angle(H_ls[nearest]) + phase_rate * dl
                H_est[l, :] = amp * np.exp(1j * phase)
        
        return H_est
    
    def _equalize(self, tf_rx: np.ndarray, H_est: np.ndarray,
                  method: str, noise_var: float) -> np.ndarray:
        """Single-tap equalization."""
        if method == 'zf':
            # Avoid division by zero
            H_safe = np.where(np.abs(H_est) > 1e-10, H_est, 1e-10)
            return tf_rx / H_safe
        elif method == 'mmse':
            return (np.conj(H_est) * tf_rx) / (np.abs(H_est)**2 + noise_var)
        else:
            raise ValueError(f"Unknown eq_method: {method}")

    def _estimate_noise(self, tf_rx: np.ndarray, H_est: np.ndarray,
                        method: str = 'dmrs_diff') -> float:
        """
        Estimate noise variance from decorrelated DMRS observations.

        First decorrelate: Z[l,k] = Y[l,k] · X_p*[k] = H[l,k] + W'[l,k]
        where W' has same variance σ² as W (since |X_p|=1).

        Methods on Z[l1,k], Z[l2,k]:

          'dmrs_power_avg': Per-tone power averaging.
              σ² = (1/K) Σ_k [ (|Z1|²+|Z2|²)/2 - |(Z1+Z2)/2|² ]
              Depends on channel frequency response cancellation.

          'dmrs_diff': IQ difference of decorrelated DMRS / 2.
              σ² = (1/2K) Σ_k |Z1[k] - Z2[k]|²
              Biased by Doppler (channel change between DMRS symbols).

          'dmrs_power_diff': Absolute power difference.
              σ² = (1/2K) Σ_k | |Z1[k]|² - |Z2[k]|² |
              Less accurate due to cross-term variance.
        """
        cfg = self.cfg
        dmrs_sc = self._dmrs_sc_indices_within_active()
        dmrs_syms = cfg.dmrs_symbol_indices

        if len(dmrs_syms) < 2:
            return 0.01

        l1, l2 = dmrs_syms[0], dmrs_syms[1]

        # Decorrelate: Z = Y · X_p* (remove known DMRS modulation)
        Z1 = tf_rx[l1, dmrs_sc] * np.conj(self._dmrs_seq)
        Z2 = tf_rx[l2, dmrs_sc] * np.conj(self._dmrs_seq)
        K = len(dmrs_sc)

        if method == 'dmrs_power_avg':
            # σ² = mean[ (|Z1|²+|Z2|²)/2 - |(Z1+Z2)/2|² ]
            avg_power = (np.abs(Z1)**2 + np.abs(Z2)**2) / 2
            coherent_power = np.abs((Z1 + Z2) / 2)**2
            sigma2 = np.mean(avg_power - coherent_power)
            return max(float(sigma2), 1e-10)

        elif method == 'dmrs_diff':
            # σ² = (1/2K) Σ |Z1 - Z2|²
            diff = Z1 - Z2
            sigma2 = np.mean(np.abs(diff)**2) / 2
            return max(float(sigma2), 1e-10)

        elif method == 'dmrs_power_diff':
            # Average squared power difference and average channel power
            # across all subcarriers, then single division:
            # σ² = (1/4) · mean_k[(|Z1|²-|Z2|²)²] / mean_k[(|Z1|²+|Z2|²)/2]
            num = np.mean((np.abs(Z1)**2 - np.abs(Z2)**2)**2)
            den = np.mean((np.abs(Z1)**2 + np.abs(Z2)**2) / 2)
            if den > 1e-15:
                sigma2 = num / (4 * den)
            else:
                sigma2 = 1e-10
            return max(float(sigma2), 1e-10)

        else:
            raise ValueError(f"Unknown noise_est method: {method}")
    
    def _extract_data(self, tf_eq: np.ndarray) -> np.ndarray:
        """Extract data symbols from equalized TF grid."""
        cfg = self.cfg
        data_sc_in_dmrs = self._data_sc_indices_in_dmrs_symbol()
        
        data = []
        for l in range(cfg.n_symbols_per_slot):
            if l in cfg.dmrs_symbol_indices:
                data.append(tf_eq[l, data_sc_in_dmrs])
            else:
                data.append(tf_eq[l, :])
        
        return np.concatenate(data)
    
    def count_data_res(self) -> int:
        """Count total data resource elements per slot."""
        return self.cfg.n_data_res


# ═══════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    from channel import TDLChannel, TDLChannelConfig
    
    print("═══ OFDM Transceiver Smoke Test ═══\n")
    
    cfg = OFDMConfig(n_fft=256, n_active=156, scs_hz=60e3, n_cp=18,
                     n_symbols_per_slot=14, dmrs_symbol_indices=[2, 11],
                     dmrs_comb_size=2)
    
    ofdm = OFDMTransceiver(cfg)
    print(f"  Config: FFT={cfg.n_fft}, active={cfg.n_active}, "
          f"SCS={cfg.scs_hz/1e3:.0f}kHz, Fs={cfg.sample_rate/1e6:.1f}MHz")
    print(f"  Data REs: {ofdm.count_data_res()}, "
          f"Pilot overhead: {cfg.pilot_overhead*100:.1f}%")
    
    # Generate data
    n_data = ofdm.count_data_res()
    bits_tx = generate_random_bits(n_data * 2)
    qam_syms = qam_modulate(bits_tx, 4)
    
    # TX
    tx_sig, tf_grid_tx = ofdm.tx(qam_syms[:n_data])
    print(f"  TX signal: {len(tx_sig)} samples, "
          f"duration={len(tx_sig)/cfg.sample_rate*1e3:.2f} ms")
    
    # Test 1: AWGN only (no fading, no Doppler) — should give low BER
    noise_var = 0.01
    noise = np.sqrt(noise_var / 2) * (
        np.random.randn(len(tx_sig)) + 1j * np.random.randn(len(tx_sig)))
    rx_sig = tx_sig + noise
    
    data_rx, H_est, _ = ofdm.rx(rx_sig, ch_est_method='linear',
                                  noise_var=noise_var)
    bits_rx = qam_demodulate(data_rx[:n_data], 4)
    ber = np.mean(bits_tx[:len(bits_rx)] != bits_rx[:len(bits_tx)])
    print(f"\n  Test 1 (AWGN 20dB, no fading): BER = {ber:.4e} "
          f"({'✓' if ber < 0.01 else '✗'})")
    
    # Test 2: TDL-A channel, low Doppler
    ch_cfg = TDLChannelConfig(model='TDL-A', delay_spread=30e-9,
                               f_doppler=100, sample_rate=cfg.sample_rate, seed=42)
    ch = TDLChannel(ch_cfg)
    rx_sig_ch, _ = ch.apply(tx_sig, snr_dB=20)
    
    for method in ['linear', 'phase_ramp']:
        data_rx, _, _ = ofdm.rx(rx_sig_ch, ch_est_method=method,
                                 noise_var=0.01, f_doppler=100)
        bits_rx = qam_demodulate(data_rx[:n_data], 4)
        ber = np.mean(bits_tx[:len(bits_rx)] != bits_rx[:len(bits_tx)])
        print(f"  Test 2 (TDL-A, fD=100Hz, {method:12s}): BER = {ber:.4e}")
    
    # Test 3: Higher Doppler
    ch_cfg_hd = TDLChannelConfig(model='TDL-A', delay_spread=30e-9,
                                  f_doppler=1000, sample_rate=cfg.sample_rate, seed=42)
    ch_hd = TDLChannel(ch_cfg_hd)
    rx_sig_hd, _ = ch_hd.apply(tx_sig, snr_dB=20)
    
    for method in ['linear', 'phase_ramp']:
        data_rx, _, _ = ofdm.rx(rx_sig_hd, ch_est_method=method,
                                 noise_var=0.01, f_doppler=1000)
        bits_rx = qam_demodulate(data_rx[:n_data], 4)
        ber = np.mean(bits_tx[:len(bits_rx)] != bits_rx[:len(bits_tx)])
        print(f"  Test 3 (TDL-A, fD=1000Hz, {method:12s}): BER = {ber:.4e}")
    
    print("\n  ✓ OFDM transceiver smoke test complete")
