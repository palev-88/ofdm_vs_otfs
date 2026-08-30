"""
3GPP TDL Channel Model (TR 38.901 §7.7.2)

Physical time-domain tapped delay line:
    r(t) = Σ αᵢ(t) · x(t - τᵢ) + n(t)

Each tap has:
    - Fixed delay τᵢ (from table, scaled by desired DS)
    - Time-varying gain αᵢ(t) with Jakes Doppler spectrum
    - Rayleigh fading (TDL-A/B/C) or Rician (TDL-D/E, first tap)

References:
    3GPP TR 38.901 V17.0.0, Tables 7.7.2-1 through 7.7.2-5
    3GPP TS 38.141-1 §G.2 (simplified profiles for conformance)
"""

__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

# ═══════════════════════════════════════════════════════════════
# 3GPP TDL Tap Tables (TR 38.901 Tables 7.7.2-1 to 7.7.2-5)
# Normalized delays (multiply by desired DS to get actual delays)
# Powers in dB (relative)
# ═══════════════════════════════════════════════════════════════

TDL_PROFILES = {
    'TDL-A': {
        'delays_norm': np.array([
            0.0000, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375,
            0.6708, 0.5750, 0.7618, 1.5375, 1.8978, 2.2242,
            2.1717, 2.4942, 2.5119, 3.0582, 4.0810, 4.4579,
            4.5695, 4.7966, 5.0066, 5.3043, 9.6586
        ]),
        'powers_dB': np.array([
            -13.4, 0.0, -2.2, -4.0, -6.0, -8.2,
            -9.9, -10.5, -7.5, -15.9, -6.6, -16.7,
            -12.4, -15.2, -10.8, -11.3, -12.7, -16.2,
            -18.3, -18.9, -16.6, -19.9, -29.7
        ]),
        'fading': 'rayleigh',
        'K_dB': None,
        'n_taps': 23,
    },
    'TDL-B': {
        'delays_norm': np.array([
            0.0000, 0.1072, 0.2155, 0.2095, 0.2870, 0.2986,
            0.3752, 0.5055, 0.3681, 0.3697, 0.5700, 0.5283,
            1.1021, 1.2756, 1.5474, 1.7842, 2.0169, 2.8294,
            3.0219, 3.6187, 4.1067, 4.2790, 4.7834
        ]),
        'powers_dB': np.array([
            0.0, -2.2, -4.0, -3.2, -9.8, -1.2,
            -3.4, -5.2, -7.6, -3.0, -8.9, -9.0,
            -4.8, -5.7, -7.5, -1.9, -7.6, -12.2,
            -9.8, -11.4, -14.9, -9.2, -11.3
        ]),
        'fading': 'rayleigh',
        'K_dB': None,
        'n_taps': 23,
    },
    'TDL-C': {
        'delays_norm': np.array([
            0.0000, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366,
            0.6448, 0.6560, 0.6584, 0.7935, 0.8213, 0.9336,
            1.2285, 1.3083, 2.1704, 2.7105, 4.2589, 4.6003,
            5.4902, 5.6077, 6.3065, 6.6374, 7.0427, 8.6523
        ]),
        'powers_dB': np.array([
            -4.4, -1.2, -3.5, -5.2, -2.5, 0.0,
            -2.2, -3.9, -7.4, -7.1, -10.7, -11.1,
            -5.1, -6.8, -8.7, -13.2, -13.9, -13.9,
            -15.8, -17.1, -16.0, -15.7, -21.6, -22.8
        ]),
        'fading': 'rayleigh',
        'K_dB': None,
        'n_taps': 24,
    },
    'TDL-D': {
        'delays_norm': np.array([
            0.0000, 0.0000, 0.0350, 0.6120, 1.3630,
            1.4050, 1.8040, 2.5960, 1.7750, 4.0420,
            7.9370, 9.4240, 9.7080
        ]),
        'powers_dB': np.array([
            -0.2, -13.5, -18.8, -21.0, -22.8,
            -17.9, -20.1, -21.9, -22.9, -27.8,
            -23.6, -24.8, -30.0
        ]),
        'fading': 'rician',
        'K_dB': 13.3,  # K-factor of first tap
        'n_taps': 13,
    },
    'TDL-E': {
        'delays_norm': np.array([
            0.0000, 0.0000, 0.5133, 0.5440, 0.5630,
            0.5440, 0.7112, 1.9092, 1.9293, 1.9589,
            2.6426, 3.7136, 5.4524, 12.0034
        ]),
        'powers_dB': np.array([
            -0.03, -22.03, -15.8, -18.1, -19.8,
            -22.9, -22.4, -18.6, -20.8, -22.6,
            -22.3, -25.6, -20.2, -29.8
        ]),
        'fading': 'rician',
        'K_dB': 22.0,
        'n_taps': 14,
    },
}


@dataclass
class TDLChannelConfig:
    """Configuration for TDL channel model."""
    model: str = 'TDL-A'            # TDL-A/B/C (NLOS) or TDL-D/E (LOS)
    delay_spread: float = 30e-9     # Desired RMS delay spread [s]
    f_doppler: float = 100.0        # Maximum Doppler frequency [Hz]
    sample_rate: float = 10e6       # Sample rate [Hz]
    seed: Optional[int] = None      # Random seed for reproducibility
    use_fdf: bool = True            # Use fractional delay filter (Farrow structure)


class TDLChannel:
    """
    3GPP TDL channel model.
    
    Implements the physical TDL:
        r(t) = Σ αᵢ(t) · x(t - τᵢ) + n(t)
    
    Each tap gain αᵢ(t) has Jakes Doppler spectrum with max Doppler f_D.
    Rayleigh fading for TDL-A/B/C, Rician for TDL-D/E (first tap).
    """
    
    def __init__(self, cfg: TDLChannelConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        
        profile = TDL_PROFILES[cfg.model]
        self.n_taps = profile['n_taps']
        self.fading_type = profile['fading']
        self.K_dB = profile['K_dB']
        
        # Scale delays by desired delay spread
        # TR 38.901 §7.7.3: τ_scaled = τ_normalized × DS_desired
        self.delays_s = profile['delays_norm'] * cfg.delay_spread
        self.delays_samples_frac = self.delays_s * cfg.sample_rate  # exact fractional
        self.delays_samples = np.round(self.delays_samples_frac).astype(int)
        self.max_delay_samples = int(np.max(self.delays_samples))

        # Fractional delay filter (Farrow structure) — optional
        self._fdf = None
        if cfg.use_fdf and np.max(self.delays_samples_frac) > 0:
            from fdf import FractionalDelayFilter
            self._fdf = FractionalDelayFilter(filter_order=8)
            self._fdf.init(self.delays_samples_frac)
        
        # Convert powers from dB to linear, normalize to unit total power
        powers_lin = 10 ** (profile['powers_dB'] / 10)
        self.powers_lin = powers_lin / np.sum(powers_lin)
        self.amplitudes = np.sqrt(self.powers_lin)
        
        # For Rician: split first tap into LOS + NLOS components
        if self.fading_type == 'rician' and self.K_dB is not None:
            K_lin = 10 ** (self.K_dB / 10)
            # First tap (index 0) is LOS (fixed), second tap (index 1) is
            # the NLOS component at the same delay
            # LOS power fraction = K/(K+1), NLOS = 1/(K+1)
            self.K_lin = K_lin
        else:
            self.K_lin = None
        
        # Fading state (generated per call)
        self._fading_coeffs = None
    
    def _generate_jakes_fading(self, n_samples: int) -> np.ndarray:
        """
        Generate Rayleigh fading with Jakes Doppler spectrum for each tap.
        
        Uses sum-of-sinusoids method (Zheng & Xiao, 2003).
        
        Returns:
            coeffs: (n_taps, n_samples) complex fading coefficients
        """
        n_sinusoids = 32  # Number of sinusoids per tap (accuracy vs speed)
        f_D = self.cfg.f_doppler
        
        coeffs = np.zeros((self.n_taps, n_samples), dtype=complex)
        t = np.arange(n_samples) / self.cfg.sample_rate
        
        if f_D == 0:
            # Static channel — random but constant phase per tap
            for i in range(self.n_taps):
                if self.fading_type == 'rician' and i == 0:
                    # LOS tap: deterministic
                    coeffs[i, :] = self.amplitudes[i]
                else:
                    phase = self.rng.uniform(0, 2 * np.pi)
                    coeffs[i, :] = self.amplitudes[i] * np.exp(1j * phase)
            return coeffs
        
        for i in range(self.n_taps):
            # Generate complex Rayleigh fading using sum-of-sinusoids
            alpha_n = self.rng.uniform(0, 2 * np.pi, n_sinusoids)
            beta_n = self.rng.uniform(0, 2 * np.pi, n_sinusoids)
            
            # Angles of arrival (uniformly distributed for Jakes)
            theta_n = (2 * np.pi * np.arange(1, n_sinusoids + 1) + 
                       self.rng.uniform(0, 2 * np.pi)) / (4 * n_sinusoids)
            
            # Doppler frequencies for each sinusoid
            f_n = f_D * np.cos(theta_n)
            
            # Complex fading
            h_I = np.zeros(n_samples)
            h_Q = np.zeros(n_samples)
            for s in range(n_sinusoids):
                h_I += np.cos(2 * np.pi * f_n[s] * t + alpha_n[s])
                h_Q += np.sin(2 * np.pi * f_n[s] * t + beta_n[s])
            
            h_complex = (h_I + 1j * h_Q) / np.sqrt(n_sinusoids)
            
            # Handle Rician first tap
            if self.fading_type == 'rician' and i == 0 and self.K_lin is not None:
                K = self.K_lin
                # LOS component (deterministic, with Doppler)
                los_phase = self.rng.uniform(0, 2 * np.pi)
                los_doppler = f_D  # LOS typically at max Doppler
                h_los = np.sqrt(K / (K + 1)) * np.exp(
                    1j * (2 * np.pi * los_doppler * t + los_phase))
                # NLOS component (Rayleigh)
                h_nlos = np.sqrt(1 / (K + 1)) * h_complex
                coeffs[i, :] = self.amplitudes[i] * (h_los + h_nlos)
            else:
                coeffs[i, :] = self.amplitudes[i] * h_complex
        
        return coeffs
    
    def apply(self, tx_signal: np.ndarray, snr_dB: Optional[float] = None
              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply TDL channel to transmit signal.
        
        Args:
            tx_signal: Complex baseband transmit signal
            snr_dB: If provided, add AWGN at this Es/N0 [dB].
                     If None, no noise added.
        
        Returns:
            rx_signal: Received signal (same length as tx_signal)
            fading_coeffs: (n_taps, n_samples) fading coefficients for reference
        """
        N = len(tx_signal)
        
        # Generate fading coefficients
        fading = self._generate_jakes_fading(N)
        self._fading_coeffs = fading
        
        # Apply TDL: r(t) = Σ αᵢ(t) · x(t - τᵢ)
        rx_signal = np.zeros(N, dtype=complex)
        if self._fdf is not None:
            # Fractional delay filter path — sub-sample accuracy
            delayed = self._fdf.apply(tx_signal)  # (N+tail, n_taps)
            for i in range(self.n_taps):
                col = delayed[:N, i] if delayed.shape[0] >= N else np.pad(delayed[:, i], (0, N - delayed.shape[0]))
                rx_signal += fading[i, :] * col
        else:
            for i in range(self.n_taps):
                delay = self.delays_samples[i]
                if delay == 0:
                    rx_signal += fading[i, :] * tx_signal
                elif delay < N:
                    rx_signal[delay:] += fading[i, delay:] * tx_signal[:N - delay]
        
        # Add AWGN
        if snr_dB is not None:
            sig_power = np.mean(np.abs(rx_signal) ** 2)
            noise_power = sig_power / (10 ** (snr_dB / 10))
            noise = np.sqrt(noise_power / 2) * (
                self.rng.standard_normal(N) + 1j * self.rng.standard_normal(N))
            rx_signal += noise
        
        return rx_signal, fading
    
    def get_freq_response(self, n_fft: int, n_symbols: int,
                          symbol_duration_samples: int) -> np.ndarray:
        """
        Compute the true channel frequency response H[l,k] for reference.
        
        Args:
            n_fft: FFT size
            n_symbols: Number of OFDM symbols
            symbol_duration_samples: Total samples per OFDM symbol (FFT + CP)
        
        Returns:
            H: (n_symbols, n_fft) true channel frequency response
        """
        if self._fading_coeffs is None:
            raise RuntimeError("Call apply() first to generate fading coefficients")
        
        H = np.zeros((n_symbols, n_fft), dtype=complex)
        
        for l in range(n_symbols):
            # Sample index at the center of OFDM symbol l
            t_idx = l * symbol_duration_samples + symbol_duration_samples // 2
            if t_idx >= self._fading_coeffs.shape[1]:
                break
            
            # Build frequency response from tap gains at this time instant
            for i in range(self.n_taps):
                delay_s = self.delays_s[i]
                gain = self._fading_coeffs[i, t_idx]
                # H(f) = Σ αᵢ · exp(-j2πfτᵢ)
                freqs = np.arange(n_fft) - n_fft // 2  # DC-centered
                H[l, :] += gain * np.exp(
                    -1j * 2 * np.pi * freqs * self.cfg.sample_rate / n_fft * delay_s)
        
        return H
    
    def __repr__(self):
        return (f"TDLChannel(model={self.cfg.model}, DS={self.cfg.delay_spread*1e9:.0f}ns, "
                f"fD={self.cfg.f_doppler:.0f}Hz, {self.n_taps} taps, "
                f"max_delay={self.max_delay_samples} samples)")


# ═══════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("═══ TDL Channel Model Smoke Test ═══\n")
    
    for model in ['TDL-A', 'TDL-B', 'TDL-C', 'TDL-D', 'TDL-E']:
        cfg = TDLChannelConfig(model=model, delay_spread=100e-9,
                               f_doppler=300, sample_rate=10e6, seed=42)
        ch = TDLChannel(cfg)
        
        # Generate test signal
        N = 10000
        x = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)
        
        # Apply channel
        y, fading = ch.apply(x, snr_dB=20)
        
        # Check power preservation (approximately)
        pwr_in = np.mean(np.abs(x) ** 2)
        pwr_out = np.mean(np.abs(y) ** 2)
        pwr_ratio_dB = 10 * np.log10(pwr_out / pwr_in)
        
        print(f"  {model}: {ch.n_taps} taps, max_delay={ch.max_delay_samples} samples, "
              f"power_ratio={pwr_ratio_dB:+.2f} dB, "
              f"fading_shape={fading.shape}")
    
    print("\n  ✓ All models instantiated and applied successfully")
