"""
PCP-OTFS v3 — Zadoff-Chu pilot with CP and GCE-BEM estimation.

Ref: "A Practical Pilot for Channel Estimation of OTFS" (IEEE 2023)

Key: ZC sequence of length L (=CP length), NOT M. Concentrates pilot
power in L samples for high per-sample SNR.

TX: DD grid with ZC+CP pilot → IDFT along Doppler → CP per subsymbol
RX: CP removal → ZC correlation (Stage 1) → BEM smoothing (Stage 2) → FDE
"""

__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "CC-BY-4.0"
__copyright__ = "(c) 2026 Panos N. Alevizos"
import numpy as np
from dataclasses import dataclass
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass
class PCPOTFSConfig:
    M: int = 256
    N: int = 14
    Mcp: int = 18
    scs_hz: float = 60e3
    pilot_doppler: int = 7
    doppler_guard: int = 1
    pilot_power_dB: float = 30.0
    zc_root: int = 1
    bem_Q: int = 2

    @property
    def L(self): return self.Mcp
    @property
    def Fs(self): return self.M * self.scs_hz
    @property
    def Ts(self): return 1.0 / self.Fs
    @property
    def MT(self): return self.M + self.Mcp
    @property
    def frame_samples(self): return self.MT * self.N
    @property
    def pilot_overhead(self):
        pilot_delay_bins = 3 * self.L - 2
        guard_cols = 2 * self.doppler_guard
        return (pilot_delay_bins + self.M * guard_cols) / (self.M * self.N)


def zadoff_chu(length, root=1):
    n = np.arange(length)
    cf = length % 2
    return np.exp(-1j * np.pi * root * n * (n + cf) / length)


class PCPOTFSTransceiver:
    def __init__(self, cfg: PCPOTFSConfig):
        self.cfg = cfg
        self._setup()

    def _setup(self):
        cfg = self.cfg
        M, N, L = cfg.M, cfg.N, cfg.L
        gamma = 10 ** (cfg.pilot_power_dB / 10)
        zc_raw = zadoff_chu(L, cfg.zc_root)
        self._zc = zc_raw * np.sqrt(gamma / L)
        self._zc_fft_L = np.fft.fft(self._zc)

        # Pilot row: ZC at [0..L-1], CP at [M-L+1..M-1]
        self._pilot_row = np.zeros(M, dtype=complex)
        for i in range(L):
            self._pilot_row[i] = self._zc[i]
        for i in range(L - 1):
            self._pilot_row[M - L + 1 + i] = self._zc[i + 1]

        # Data positions
        pilot_cols = set()
        for dg in range(-cfg.doppler_guard, cfg.doppler_guard + 1):
            pilot_cols.add((cfg.pilot_doppler + dg) % N)

        self._data_pos = []
        for n in range(N):
            if n in pilot_cols:
                if n == cfg.pilot_doppler:
                    for m in range(2 * L - 1, M - L + 1):
                        self._data_pos.append((m, n))
            else:
                for m in range(M):
                    self._data_pos.append((m, n))
        self._data_pos = np.array(self._data_pos) if self._data_pos else np.zeros((0, 2), dtype=int)
        self._n_data = len(self._data_pos)

    def tx(self, qam_symbols):
        cfg = self.cfg
        M, N, MT = cfg.M, cfg.N, cfg.MT
        D = np.zeros((M, N), dtype=complex)
        nd = min(len(qam_symbols), self._n_data)
        for i in range(nd):
            D[self._data_pos[i, 0], self._data_pos[i, 1]] = qam_symbols[i]
        D[:, cfg.pilot_doppler] = self._pilot_row
        for i in range(nd):
            m, n = self._data_pos[i]
            if n == cfg.pilot_doppler:
                D[m, n] = qam_symbols[i]
        X = np.fft.ifft(D, axis=1) * np.sqrt(N)
        tx_sig = np.zeros(cfg.frame_samples, dtype=complex)
        for n in range(N):
            x_n = X[:, n]
            tx_sig[n * MT: n * MT + cfg.Mcp] = x_n[-cfg.Mcp:]
            tx_sig[n * MT + cfg.Mcp: (n + 1) * MT] = x_n
        return tx_sig, D

    def rx(self, rx_signal, noise_var):  # noise_var unused — rx() self-estimates from CP redundancy; kept for call-site compatibility
        cfg = self.cfg
        M, N, MT, L = cfg.M, cfg.N, cfg.MT, cfg.L
        Y = np.zeros((M, N), dtype=complex)
        for n in range(N):
            Y[:, n] = rx_signal[n * MT + cfg.Mcp: n * MT + cfg.Mcp + M]
        
        # Noise estimation from CP redundancy
        noise_est = self._estimate_noise(rx_signal)
        
        h_hat = self._estimate_stage1(Y, noise_est)
        h_smooth = self._estimate_stage2(h_hat, noise_est)
        H_f = np.zeros((M, N), dtype=complex)
        for n in range(N):
            H_f[:, n] = np.fft.fft(h_smooth[:, n], M)
        data_sym = self._detect_fde(Y, H_f, noise_est)
        return data_sym, H_f, None

    def _estimate_noise(self, rx_signal):
        """
        Estimate noise from CP redundancy.
        
        CP copies the last Mcp samples of each subsymbol to the front.
        After channel: both copies see the same channel (within one symbol).
        
        received_cp[i]   = h * x_tail[i] + w_cp[i]
        received_tail[i] = h * x_tail[i] + w_tail[i]
        
        Difference = w_cp - w_tail → σ² = mean(|diff|²) / 2
        
        Low Doppler bias because CP and tail are separated by only M 
        samples (within one symbol), unlike OFDM DMRS separated by 9 symbols.
        """
        cfg = self.cfg
        M, N, MT, Mcp = cfg.M, cfg.N, cfg.MT, cfg.Mcp
        
        diffs = []
        for n in range(N):
            # CP region: first Mcp samples of subsymbol
            cp_start = n * MT
            cp = rx_signal[cp_start: cp_start + Mcp]
            # Corresponding tail: last Mcp samples of data portion
            tail_start = n * MT + M  # = n*MT + Mcp + (M - Mcp)... 
            tail = rx_signal[tail_start: tail_start + Mcp]
            if len(cp) == Mcp and len(tail) == Mcp:
                diffs.append(cp - tail)
        
        if diffs:
            all_diffs = np.concatenate(diffs)
            sigma2 = np.mean(np.abs(all_diffs)**2) / 2
            return max(float(sigma2), 1e-10)
        
        return 0.01

    def rx_true_csi(self, rx_signal, fading, delay_samples, noise_var):
        cfg = self.cfg
        M, N, MT = cfg.M, cfg.N, cfg.MT
        H_f = np.zeros((M, N), dtype=complex)
        for n in range(N):
            h_time = np.zeros(M, dtype=complex)
            t_mid = n * MT + cfg.Mcp + M // 2
            for i in range(len(delay_samples)):
                d = delay_samples[i]
                if d < M and t_mid < fading.shape[1]:
                    h_time[d] += fading[i, t_mid]
            H_f[:, n] = np.fft.fft(h_time)
        Y = np.zeros((M, N), dtype=complex)
        for n in range(N):
            Y[:, n] = rx_signal[n * MT + cfg.Mcp: n * MT + cfg.Mcp + M]
        return self._detect_fde(Y, H_f, noise_var)

    def _estimate_stage1(self, Y, noise_var):
        """ZC correlation per subsymbol → h[l, n] for l=0..L-1."""
        cfg = self.cfg
        M, N, L = cfg.M, cfg.N, cfg.L
        n_p = cfg.pilot_doppler
        h_hat = np.zeros((L, N), dtype=complex)
        for n in range(N):
            y_pilot = Y[:L, n]
            Y_f = np.fft.fft(y_pilot)
            phase = np.exp(1j * 2 * np.pi * n_p * n / N) / np.sqrt(N)
            P_f = self._zc_fft_L * phase
            H_f = Y_f * np.conj(P_f) / (np.abs(P_f) ** 2 + noise_var)
            h_hat[:, n] = np.fft.ifft(H_f)
        return h_hat

    def _estimate_stage2(self, h_hat, noise_var):
        """GCE-BEM smoothing: h[l,n] = Σ_q c[l,q] exp(j2πqn/N)."""
        cfg = self.cfg
        L, N = h_hat.shape
        Q = cfg.bem_Q
        if Q <= 0 or 2 * Q + 1 > N:
            return self._pad(h_hat)
        n_basis = 2 * Q + 1
        B = np.zeros((N, n_basis), dtype=complex)
        for idx, q in enumerate(range(-Q, Q + 1)):
            B[:, idx] = np.exp(1j * 2 * np.pi * q * np.arange(N, dtype=float) / N)
        reg = noise_var * 0.01 * np.eye(n_basis)
        B_pinv = np.linalg.inv(B.conj().T @ B + reg) @ B.conj().T
        h_pow = np.mean(np.abs(h_hat) ** 2, axis=1)
        max_pow = np.max(h_pow) if np.max(h_pow) > 0 else 1
        active = np.where(h_pow > max_pow * 0.005)[0]
        if len(active) == 0:
            active = np.array([0])
        h_smooth = np.zeros((L, N), dtype=complex)
        for l in active:
            c = B_pinv @ h_hat[l, :]
            h_smooth[l, :] = B @ c
        return self._pad(h_smooth)

    def _pad(self, h_L):
        M = self.cfg.M
        h_M = np.zeros((M, h_L.shape[1]), dtype=complex)
        h_M[:h_L.shape[0], :] = h_L
        return h_M

    def _detect_fde(self, Y, H_f, noise_var):
        """FDE with pilot cancellation."""
        cfg = self.cfg
        M, N = cfg.M, cfg.N
        n_p = cfg.pilot_doppler
        Y_f = np.zeros((M, N), dtype=complex)
        for n in range(N):
            Y_f[:, n] = np.fft.fft(Y[:, n])
        pilot_f = np.fft.fft(self._pilot_row)
        for n in range(N):
            phase = np.exp(1j * 2 * np.pi * n_p * n / N) / np.sqrt(N)
            Y_f[:, n] -= H_f[:, n] * pilot_f * phase
        X_f = np.conj(H_f) / (np.abs(H_f) ** 2 + noise_var) * Y_f
        X_hat = np.zeros((M, N), dtype=complex)
        for n in range(N):
            X_hat[:, n] = np.fft.ifft(X_f[:, n])
        D_hat = np.fft.fft(X_hat, axis=1) / np.sqrt(N)
        return np.array([D_hat[self._data_pos[i, 0], self._data_pos[i, 1]]
                         for i in range(self._n_data)])

    def count_data_res(self):
        return self._n_data


if __name__ == '__main__':
    from channel import TDLChannel, TDLChannelConfig
    from qam import qam_modulate, qam_demodulate, generate_random_bits
    import time

    for M in [64, 256]:
        CP = max(round(144 * M / 2048), 4)
        SCS = 60e3; FS = M * SCS
        cfg = PCPOTFSConfig(M=M, N=14, Mcp=CP, scs_hz=SCS,
            pilot_doppler=7, doppler_guard=1, pilot_power_dB=30.0,
            zc_root=1, bem_Q=2)
        pcp = PCPOTFSTransceiver(cfg)
        nd = pcp.count_data_res()
        print(f"\n═══ PCP-OTFS v3 | M={M}, CP={CP}, L={cfg.L} ═══")
        print(f"  Data={nd}/{M*14}, OH={cfg.pilot_overhead*100:.1f}%")
        print(f"  Per-sample pilot/data ratio: {10**(cfg.pilot_power_dB/10)/CP:.0f}×")

        bits = generate_random_bits(nd * 2, np.random.default_rng(42))
        tx, _ = pcp.tx(qam_modulate(bits, 4)[:nd])

        nv = 0.01; rng = np.random.default_rng(777)
        noise = np.sqrt(nv/2)*(rng.standard_normal(len(tx))+1j*rng.standard_normal(len(tx)))
        dr, _, _ = pcp.rx(tx + noise, nv)
        br = qam_demodulate(dr, 4)
        ber = np.mean(bits[:len(br)] != br)
        print(f"  AWGN 20dB: BER={ber:.4e} {'✓' if ber < 0.01 else '✗'}")

        for fd in [0, 300, 500, 1000, 2000]:
            ch = TDLChannel(TDLChannelConfig('TDL-A', 30e-9, fd, FS, seed=42))
            rx, fading = ch.apply(tx, snr_dB=20)
            dr, _, _ = pcp.rx(rx, 0.01)
            br = qam_demodulate(dr, 4)
            ber_e = np.mean(bits[:len(br)] != br)
            dr_t = pcp.rx_true_csi(rx, fading, ch.delays_samples, 0.01)
            br_t = qam_demodulate(dr_t, 4)
            ber_t = np.mean(bits[:len(br_t)] != br_t)
            print(f"  fD={fd:5d}Hz: est={ber_e:.4e} true={ber_t:.4e}")
    print("\n  ✓ Done")
