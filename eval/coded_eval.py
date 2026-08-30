"""Coded OFDM-vs-OTFS evaluation with the real 3GPP NR LDPC.

Compares the report's own receivers -- OFDM with linear time/frequency DMRS
interpolation, ZP-OTFS, PCP-guard, PCP-orig -- now with channel coding.

Design decisions settled with the author:

  FIXED FRONT-END RATE / SAME SHAPE.  Practical receivers run one fixed
  sample rate, and all waveforms occupy the same bandwidth (report TX-PSD
  figures).  The Zak-domain waveforms are therefore carried on the front-end
  DFT grid: each cyclic subsymbol body is placed bin-for-bin on the Mfft grid
  (exact band-limited interpolation, brick-wall by construction, no filter),
  with the CP taken at the fixed rate at the same guard duration.  The RX
  front-end is one Mfft-FFT + bin extraction, structurally identical to
  OFDM's.  An earlier revision used a polyphase resampler here; it attenuated
  the critically-sampled band edge (g_sig ~ 0.94) and added filter ISI to the
  OTFS paths only, and was replaced by this exact representation.

  ISO-BLOCK CODING.  All methods carry the identical codeword: same A, same E,
  same base graph, same block length, same interleaver.  E = min(n_re)*Qm over
  the methods.  A method with surplus REs fills them with UNCOUNTED dummy QPSK,
  so its waveform statistics and PAPR are unchanged and only the accounting
  differs.  The carrier REs are a uniformly STRIDED subset -- taking the first
  n_common in natural order would put OFDM's dummy REs at the end of the slot,
  i.e. on the symbols furthest from the DMRS and worst estimated at high
  Doppler, silently favouring OFDM.

  CALIBRATED PER-DATA-RE SNR.  Signal and noise gains through each method's
  full chain (resample -> channel -> decimate -> RX transform) are measured
  once per configuration, so the injected time-domain noise yields exactly the
  target SNR per data RE for every waveform.  This is essential here because
  decimation removes out-of-band noise and would otherwise hand the resampled
  waveforms a ~2.2 dB bonus.

Soft output.  Each receiver returns an unbiased estimate z and a per-symbol
SINR gamma, so the LDPC decoder gets calibrated
LLRs.  The rule differs by receiver structure, as it must:
  OFDM  per-RE, exact for the one-tap model: z = Y/H_hat, gamma = |H_hat|^2/s2
  PCP   per-cell MMSE followed by a unitary despread, so the DD symbol sees the
        AVERAGED noise plus self-interference from the varying MMSE gain mu:
            gamma = mean(mu)^2 / (mean(mu^2)-mean(mu)^2 + mean(mu(1-mu)))
  ZP    joint time-domain LMMSE; the DD error variance averages to
        s2*trace(A^-1)/n, estimated with Hutchinson probes (A = G^H G + s2 I).
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "2"

import sys, json, time, argparse
import numpy as np
from scipy.signal import resample_poly
from scipy import sparse
from scipy.sparse.linalg import spsolve, splu

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
PANOS = os.path.join(os.path.dirname(HERE), 'src')
sys.path.insert(0, PANOS)
sys.path.insert(0, HERE)

from channel import TDLChannel, TDLChannelConfig
from ofdm import OFDMTransceiver, OFDMConfig
from otfs_pcp import PCPOTFSTransceiver, PCPOTFSConfig
from otfs_zp import ZPOTFSTransceiver, ZPOTFSConfig
import nr_ldpc

def qpsk_mod(bits):
    """Unit-power Gray QPSK: b0 -> real, b1 -> imag, bit 0 -> +1.
    The mapping convention must match qpsk_llr below."""
    b = np.asarray(bits).reshape(-1, 2)
    return ((1 - 2 * b[:, 0]) + 1j * (1 - 2 * b[:, 1])) / np.sqrt(2)


def qpsk_llr(z, gamma):
    """Per-bit LLRs from an unbiased symbol estimate z and per-symbol SINR
    gamma (effective 1/N0): L = 2*sqrt(2)*gamma*component, Gray mapping as
    in qpsk_mod."""
    k = 2.0 * np.sqrt(2.0) * np.asarray(gamma)
    llr = np.empty(2 * len(z))
    llr[0::2] = k * np.real(z)
    llr[1::2] = k * np.imag(z)
    return llr

QM = 2                      # QPSK
DS_MAP = {'TDL-A': 30e-9, 'TDL-B': 100e-9, 'TDL-C': 300e-9, 'TDL-D': 30e-9}


# ════════════════════════════════════════════════════════════════════
#  Waveform wrappers: native-rate TX/RX + common-rate resampling
# ════════════════════════════════════════════════════════════════════
class Wave:
    """Common interface: tx(symbols) -> common-rate signal; rx(sig) -> (z, gamma)."""
    name = '?'
    up = 1; dn = 1                    # resample_poly(up, dn) to reach common rate

    def to_common(self, x):
        """Resample from this waveform's native rate to the common front-end rate."""
        return x if self.up == self.dn else resample_poly(x, self.up, self.dn)

    def to_native(self, x):
        """Resample from the common front-end rate back to the native rate."""
        return x if self.up == self.dn else resample_poly(x, self.dn, self.up)


class WOFDM(Wave):
    """Classical 5G-NR OFDM baseline: LS + linear interpolation, DMRS power-difference noise estimate, noise-only reliability.  Kept for reference; superseded by WOFDM_TD in the evaluation."""
    name = 'OFDM'

    def __init__(self, M, n_act, SCS, N, CP):
        self.t = OFDMTransceiver(OFDMConfig(
            n_fft=M, n_active=n_act, scs_hz=SCS, n_cp=CP, n_symbols_per_slot=N,
            dmrs_symbol_indices=[2, 11], dmrs_comb_size=2))
        self.nd = self.t.count_data_res()
        self.frame_len = self.t.cfg.slot_duration_samples

    def tx(self, syms):
        return self.t.tx(syms)[0]                 # already at the common rate

    def pre_eq(self, sig):
        """Data REs after the RX transform only (no estimator, no equaliser).

        Calibration must not run through the channel estimator: feeding it pure
        noise makes H_hat noise too, so Y/H_hat is a ratio of noises and its
        mean square measures estimator blow-up, not the chain's noise gain.
        """
        n = self.t.cfg.n_symbols_per_slot
        M, CP = self.t.cfg.n_fft, self.t.cfg.n_cp
        bins = self.t._sc_indices() % M
        Y = np.zeros((n, self.t.cfg.n_active), dtype=complex)
        for l in range(n):
            s = l * (M + CP) + CP
            seg = sig[s:s + M]
            if len(seg) < M:
                seg = np.pad(seg, (0, M - len(seg)))
            Y[l] = (np.fft.fft(seg) / np.sqrt(M))[bins]
        return self.t._extract_data(Y)

    def rx(self, sig):
        _, H, Y = self.t.rx(sig, 'linear', 'mmse', 1e-3, 0.0)
        s2 = self.t._estimate_noise(Y, H, 'dmrs_power_diff')
        Hs = np.where(np.abs(H) < 1e-9, 1e-9, H)
        z = self.t._extract_data(Y / Hs)
        g = self.t._extract_data(np.abs(H) ** 2 / s2)
        return z, g


class WOFDM_TD(WOFDM):
    """The OFDM baseline: identical 5G-NR waveform and DMRS as WOFDM (same
    pilots, same 7.14% overhead, same data REs), but with the delay-domain
    estimator of ofdm_td.py -- CIR least-squares fit, single-symbol residual
    impairment estimate, and a gamma that prices the time-interpolation error.

    WOFDM (per-subcarrier LS + linear frequency interpolation +
    dmrs_power_diff) is retained for reference/regression only; it is 1.5-5 dB
    worse and its noise estimator is Doppler-biased by up to 76x."""
    name = 'OFDM'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        from ofdm_td import OFDMTimeDomainRx
        self.td = OFDMTimeDomainRx(self.t)

    def rx(self, sig):
        return self.td.rx(sig)


class WPCP(Wave):
    """PCP-OTFS at the fixed front-end rate via DFT-grid representation.

    No resampler: each subsymbol body is carried on the front-end DFT grid
    (exact band-limited interpolation for the cyclic body), the CP is taken
    at the fixed rate with the same guard duration, and the receiver
    front-end is one Mfft-FFT + bin extraction -- structurally identical to
    OFDM's.  The native PCP estimator/FDE pipeline then runs unchanged in
    its own M_nat domain.
    """
    name = 'PCP'

    def __init__(self, M, n_act, SCS, N, Mcp, label):
        self.name = label
        self.p = PCPOTFSTransceiver(PCPOTFSConfig(
            M=n_act, N=N, Mcp=Mcp, scs_hz=SCS, pilot_doppler=N // 2,
            doppler_guard=1, pilot_power_dB=25.0, zc_root=1, bem_Q=1))
        self.nd = self.p.count_data_res()
        self.Mfft, self.Mn, self.N = M, n_act, N
        k = np.arange(n_act)
        # natural (fftfreq) mapping of the critically-sampled band onto the
        # front-end grid; PCP has no structural DC null (report Sec. 12.10)
        self.bins = np.where(k < n_act // 2, k, k - n_act) % M
        self.K = int(round(Mcp * M / n_act))        # CP at fixed rate, same duration
        self.sym_hi = M + self.K
        self.frame_len = N * self.sym_hi
        self.Mcp_nat = Mcp

    # exact interpolation of a cyclic body to the front-end rate
    def _up(self, x):
        F = np.zeros(self.Mfft, dtype=complex)
        F[self.bins] = np.fft.fft(x)
        return np.fft.ifft(F) * (self.Mfft / self.Mn)

    def _down(self, y):
        return np.fft.ifft(np.fft.fft(y)[self.bins]) * (self.Mn / self.Mfft)

    def tx(self, syms):
        nat = self.p.tx(syms)[0]
        MT, Mcp, Mn = self.p.cfg.MT, self.p.cfg.Mcp, self.Mn
        out = np.zeros(self.frame_len, dtype=complex)
        for n in range(self.N):
            body = nat[n * MT + Mcp: n * MT + Mcp + Mn]
            hi = self._up(body)
            out[n * self.sym_hi: n * self.sym_hi + self.K] = hi[-self.K:]
            out[n * self.sym_hi + self.K: (n + 1) * self.sym_hi] = hi
        return out

    def _bodies(self, sig):
        """Per-subsymbol native-domain bodies + high-rate CP noise estimate."""
        Y = np.zeros((self.Mn, self.N), dtype=complex)
        diffs = []
        for n in range(self.N):
            seg = sig[n * self.sym_hi: (n + 1) * self.sym_hi]
            if len(seg) < self.sym_hi:
                seg = np.pad(seg, (0, self.sym_hi - len(seg)))
            body = seg[self.K:]
            Y[:, n] = self._down(body)
            diffs.append(seg[:self.K] - body[-self.K:])
        s2_hi = max(float(np.mean(np.abs(np.concatenate(diffs)) ** 2) / 2), 1e-12)
        s2_nat = s2_hi * self.Mn / self.Mfft   # verified numerically (validation gate)
        return Y, s2_nat

    def pre_eq(self, sig):
        Y, _ = self._bodies(sig)
        D = np.fft.fft(Y, axis=1) / np.sqrt(self.N)
        return D[self.p._data_pos[:, 0], self.p._data_pos[:, 1]]

    def rx(self, sig):
        cfg = self.p.cfg
        Mp, Np = cfg.M, cfg.N
        Y, s2 = self._bodies(sig)
        h_hat = self.p._estimate_stage1(Y, s2)
        h_sm = self.p._estimate_stage2(h_hat, s2)
        H_f = np.zeros((Mp, Np), dtype=complex)
        Y_f = np.zeros((Mp, Np), dtype=complex)
        for n in range(Np):
            H_f[:, n] = np.fft.fft(h_sm[:, n], Mp)
            Y_f[:, n] = np.fft.fft(Y[:, n])
        pf = np.fft.fft(self.p._pilot_row)
        for n in range(Np):
            ph = np.exp(1j * 2 * np.pi * cfg.pilot_doppler * n / Np) / np.sqrt(Np)
            Y_f[:, n] -= H_f[:, n] * pf * ph            # pilot cancellation
        # epsilon^2: channel-estimation-error power, from the per-tap MEDIAN
        # of the out-of-basis GCE-BEM spectrum (robust to the concentrated
        # data-leakage spikes at |q|>=2).  Mirrors the OFDM reliability fix
        # (sec:ofdm_rel): s2 -> s2 + eps2 in the MMSE gain, FDE and gamma,
        # so the decoder is told about estimation error.  Validated by smoke
        # tests on both PCP variants: BLER halves at high-Doppler cells,
        # calibration 2.3-3.0 -> 0.5-1.1 (conservative side).
        Qb = cfg.bem_Q
        F_bem = np.fft.fft(h_hat, axis=1)
        qs = np.arange(Np); dist = np.minimum(qs, Np - qs)
        floor = np.median(np.abs(F_bem[:, dist > Qb]) ** 2, axis=1)
        eps2 = float((2 * Qb + 1) * np.sum(floor) / Np ** 2)
        s2 = s2 + eps2
        pw = np.abs(H_f) ** 2
        mu = pw / (pw + s2)
        X_f = np.conj(H_f) / (pw + s2) * Y_f
        D = np.fft.fft(np.fft.ifft(X_f, axis=0), axis=1) / np.sqrt(Np)
        mb = float(mu.mean())
        gamma = mb ** 2 / max(float((mu ** 2).mean()) - mb ** 2
                              + float((mu * (1 - mu)).mean()), 1e-15)
        z = D[self.p._data_pos[:, 0], self.p._data_pos[:, 1]] / max(mb, 1e-12)
        return z, np.full(len(z), gamma)


class WZP(Wave):
    """ZP-OTFS wrapper (screening only; excluded from the coded evaluation - its joint-LMMSE soft output fails the calibration test)."""
    name = 'ZP-OTFS'

    def __init__(self, M, n_act, SCS, N, zp_len, n_probe=1):
        self.z = ZPOTFSTransceiver(ZPOTFSConfig(
            M=n_act, N=N, scs_hz=SCS, zp_len=zp_len, pilot_delay=1,
            pilot_doppler=N // 2, guard_delay=zp_len + 2, guard_doppler=4,
            max_dd_taps=50))
        self.nd = self.z.count_data_res()
        self.up, self.dn = M, n_act
        self.n_probe = n_probe
        self.frame_len = len(self.to_common(np.zeros(self.z.cfg.frame_samples)))

    def tx(self, syms):
        return self.to_common(self.z.tx(syms)[0])

    def pre_eq(self, sig):
        """DD data cells after ZP removal + Zak only (H=1, no equaliser)."""
        rxn = self.to_native(sig)
        cfg = self.z.cfg
        if len(rxn) < cfg.frame_samples:
            rxn = np.pad(rxn, (0, cfg.frame_samples - len(rxn)))
        y = np.zeros((cfg.M, cfg.N), dtype=complex)
        for n in range(cfg.N):
            y[:, n] = rxn[n * cfg.Meff: n * cfg.Meff + cfg.M]
        Y = np.fft.fft(y, axis=1) / np.sqrt(cfg.N)
        return np.array([Y[self.z._data_pos[i, 0], self.z._data_pos[i, 1]]
                         for i in range(self.nd)])

    def rx(self, sig):
        rxn = self.to_native(sig)
        cfg = self.z.cfg
        need = cfg.frame_samples
        if len(rxn) < need:
            rxn = np.pad(rxn, (0, need - len(rxn)))
        y_dt = np.zeros((cfg.M, cfg.N), dtype=complex)
        for n in range(cfg.N):
            y_dt[:, n] = rxn[n * cfg.Meff: n * cfg.Meff + cfg.M]
        Y_dd = np.fft.fft(y_dt, axis=1) / np.sqrt(cfg.N)
        pre = self.z._estimate_noise_blind(rxn)
        taps = self.z._estimate_channel_dd(Y_dd, pre)
        s2 = self.z._estimate_noise(rxn, taps)
        G = self.z._build_G_from_dd(taps)
        NM = cfg.frame_samples
        GH = G.conj().T
        A = (GH @ G + max(s2, 1e-6) * sparse.eye(NM)).tocsc()
        # ONE factorisation reused for the data solve and the trace probe;
        # spsolve would refactorise on every call.
        lu = splu(A)
        xh = lu.solve(np.asarray(GH @ rxn[:NM], dtype=complex))
        if np.any(~np.isfinite(xh)):
            xh = np.nan_to_num(xh)
        # Hutchinson mean diag(A^-1). Validated against a dense diag(A^-1):
        # exact to 0.00% at K=1 here (A^-1 is strongly diagonally dominant),
        # so one probe suffices.
        rng = np.random.default_rng(0)
        acc = 0.0
        for _ in range(self.n_probe):
            v = (rng.integers(0, 2, NM) * 2.0 - 1.0).astype(complex)
            acc += float(np.real(v @ lu.solve(v))) / NM
        sig_e = np.clip(s2 * acc / self.n_probe, 1e-12, 1 - 1e-12)
        mu = 1.0 - sig_e
        gamma = mu / sig_e
        x_dt = np.zeros((cfg.M, cfg.N), dtype=complex)
        for n in range(cfg.N):
            x_dt[:, n] = xh[n * cfg.Meff: n * cfg.Meff + cfg.M]
        X = np.fft.fft(x_dt, axis=1) / np.sqrt(cfg.N)
        z = np.array([X[self.z._data_pos[i, 0], self.z._data_pos[i, 1]]
                      for i in range(self.nd)]) / mu
        return z, np.full(len(z), gamma)


# ════════════════════════════════════════════════════════════════════
#  Calibration + iso-block plumbing
# ════════════════════════════════════════════════════════════════════
def strided(nd, k):
    """k uniformly spread indices out of nd (monotone, unique for nd>=k)."""
    return (np.arange(k) * nd) // k


def calibrate(w, rng, n_trial=16):
    """Per-data-RE signal and noise gain through the LINEAR chain only.

    Measured on pre_eq() (RX transform, no estimator, no equaliser), so the
    numbers reflect the transform + resampling chain.  Decimating back to the
    native rate discards out-of-band noise -- for the resampled waveforms
    g_noise ~ n_act/M ~ 0.61, i.e. a 2.2 dB bonus that must be calibrated out.
    """
    gs, gn = [], []
    for _ in range(n_trial):
        s = qpsk_mod(rng.integers(0, 2, 2 * w.nd))
        sig = w.tx(s)                                  # identity channel, no noise
        gs.append(np.mean(np.abs(w.pre_eq(sig)[:w.nd]) ** 2))
        L = max(len(sig), w.frame_len)
        nz = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) / np.sqrt(2)
        gn.append(np.mean(np.abs(w.pre_eq(nz)[:w.nd]) ** 2))
    return float(np.mean(gs)), float(np.mean(gn))


def run(bw, args):
    """Fixed-grid coded evaluation over (channel, Doppler, SNR) for one bandwidth case; returns accumulated bit/block error counts per method."""
    M, n_act, SCS = bw['M'], bw['n_act'], bw['SCS']
    N, tag = bw['N'], bw['tag']
    FS = M * SCS
    CP = max(round(144 * M / 2048), 4)
    # ZP length must be sized at ZP-OTFS's OWN sample rate (n_act*SCS), not the
    # common rate, or it is over-sized by M/n_act = 1.64 and inflates the frame.
    FS_NAT = n_act * SCS
    chc = TDLChannel(TDLChannelConfig('TDL-C', 300e-9, 0, FS_NAT, seed=1,
                                      use_fdf=False))
    ZPLEN = max(chc.max_delay_samples, 8)
    cp_guard = max(4, int(round(1e-6 * (n_act * SCS))) + 3)
    cp_orig = max(round(144 * n_act / 2048), 4)

    waves = [WOFDM(M, n_act, SCS, N, CP),
             WOFDM_TD(M, n_act, SCS, N, CP),
             WZP(M, n_act, SCS, N, ZPLEN),
             WPCP(M, n_act, SCS, N, cp_guard, 'PCP-guard'),
             WPCP(M, n_act, SCS, N, cp_orig, 'PCP-orig')]
    n_common = min(w.nd for w in waves)
    E = n_common * QM
    A = E // 2                                   # rate 1/2 iso-block
    sel = {w.name: strided(w.nd, n_common) for w in waves}

    rng = np.random.default_rng(11)
    cal = {w.name: calibrate(w, rng) for w in waves}

    print(f"\n=== {tag}: M={M} n_act={n_act} SCS={SCS/1e3:.0f}k Fs={FS/1e6:.2f}M "
          f"CP={CP} ZP={ZPLEN} cp_guard={cp_guard} cp_orig={cp_orig}")
    for w in waves:
        gs, gn = cal[w.name]
        print(f"    {w.name:10s} dataRE={w.nd:5d} resample={w.up}/{w.dn} "
              f"g_sig={gs:.3f} g_noise={gn:.3f}")
    print(f"    iso-block: n_common={n_common} A={A} E={E}")

    # LDPC codeword bank (encode once, reuse)
    bank = []
    brng = np.random.default_rng(99)
    for _ in range(args.bank):
        tb = brng.integers(0, 2, A).astype(np.uint8)
        coded, prm = nr_ldpc.ldpc_encode(tb, 0.5, QM, n_common)
        bank.append((tb, coded, prm))
    p0 = bank[0][2]
    print(f"    LDPC: BG{p0['bg']} Z={p0['Z']} C={p0['C']} N_ldpc={p0['N_ldpc']} "
          f"coded={len(bank[0][1])} bank={len(bank)}")
    PERM = np.random.default_rng(7).permutation(E)
    IPERM = np.argsort(PERM)

    R = {}
    t0 = time.time()
    for cm in args.channels:
        ds = DS_MAP[cm]
        for fd in args.fds:
            for fr in range(args.frames):
                ch = TDLChannel(TDLChannelConfig(cm, ds, fd, FS,
                                                 seed=900_000 + 977 * fr + int(fd),
                                                 use_fdf=True))
                tb, coded, prm = bank[fr % len(bank)]
                syms = qpsk_mod(coded[PERM].astype(int))
                frng = np.random.default_rng(5_000 + 13 * fr + int(fd))
                # frame lengths differ slightly across waveforms (guard-length
                # differences), so draw ONE fading realisation spanning the
                # longest and slice it -- all methods then see the same channel.
                Lmax = max(w.frame_len for w in waves)
                fading = ch._generate_jakes_fading(Lmax)
                fdf = ch._fdf
                clean = {}
                for w in waves:
                    pay = np.zeros(w.nd, dtype=complex)
                    pay[sel[w.name]] = syms
                    other = np.setdiff1d(np.arange(w.nd), sel[w.name])
                    if len(other):                      # uncounted dummy
                        pay[other] = qpsk_mod(frng.integers(0, 2, 2 * len(other)))
                    s = w.tx(pay)
                    L = len(s)
                    c = np.zeros(L, dtype=complex)
                    if fdf is not None:
                        dl = fdf.apply(s)               # fractional delays
                        for i in range(ch.n_taps):
                            col = dl[:L, i] if dl.shape[0] >= L else np.pad(
                                dl[:, i], (0, L - dl.shape[0]))
                            c += fading[i, :L] * col
                    else:
                        for i in range(ch.n_taps):
                            d = int(ch.delays_samples[i])
                            if d == 0:
                                c += fading[i, :L] * s
                            elif d < L:
                                c[d:] += fading[i, d:L] * s[:L - d]
                    clean[w.name] = c
                for snr in args.snrs:
                    for w in waves:
                        gs, gn = cal[w.name]
                        s2 = gs / (10.0 ** (snr / 10.0) * gn)   # calibrated per-RE
                        c = clean[w.name]
                        r = c + np.sqrt(s2 / 2) * (frng.standard_normal(len(c))
                                                   + 1j * frng.standard_normal(len(c)))
                        z, g = w.rx(r)
                        zz, gg = z[sel[w.name]], g[sel[w.name]]
                        llr = qpsk_llr(zz, gg)[IPERM]
                        dec = nr_ldpc.ldpc_decode(llr, prm, n_iter=args.iters)
                        nerr = int(np.sum(dec['bits'][:len(tb)] != tb))
                        k = (cm, fd, snr, w.name)
                        v = R.setdefault(k, [0, 0, 0, 0])
                        v[0] += nerr; v[1] += len(tb)
                        v[2] += 0 if dec['tb_crc_ok'] else 1; v[3] += 1
            print(f"  {tag} {cm} fD={fd:5.0f}  ({time.time()-t0:.0f}s)", flush=True)
    return R, [w.name for w in waves], dict(n_common=n_common, A=A, E=E)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=30)
    ap.add_argument('--bank', type=int, default=20)
    ap.add_argument('--iters', type=int, default=25)
    ap.add_argument('--snrs', type=float, nargs='+', default=[0, 2, 4, 6, 8, 10])
    ap.add_argument('--fds', type=float, nargs='+', default=[0, 500, 1000])
    ap.add_argument('--channels', nargs='+', default=['TDL-C'])
    ap.add_argument('--bw', nargs='+', default=['NB'])
    ap.add_argument('--tag', default='codedeval')
    args = ap.parse_args()

    BWS = {'NB': dict(tag='NB', M=256, n_act=156, SCS=30e3, N=14),
           'WB': dict(tag='WB', M=1024, n_act=624, SCS=60e3, N=14)}
    out = {}
    for b in args.bw:
        R, names, meta = run(BWS[b], args)
        for (cm, fd, snr, nm), v in R.items():
            out[f"{b}|{cm}|{fd:g}|{snr:g}|{nm}"] = dict(
                ber=v[0] / v[1], bler=v[2] / v[3], bits=v[1], blocks=v[3])
        print(f"\n--- {b}: coded BER (BLER) ---")
        for cm in args.channels:
            for fd in args.fds:
                print(f"  {cm} fD={fd:g}")
                print("    SNR " + "".join(f"{n:>22}" for n in names))
                for snr in args.snrs:
                    row = ""
                    for n in names:
                        v = R[(cm, fd, snr, n)]
                        row += f"{v[0]/v[1]:>13.2e}({v[2]/v[3]:4.2f})"
                    print(f"    {snr:3.0f} " + row)
    with open(os.path.join(DATA, args.tag + '.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.tag}.json")
