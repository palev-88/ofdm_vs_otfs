"""
tx_spectrum_papr.py -- PSD / PAPR / ACLR characterization of OTFS waveforms.

Models a 3GPP TX chain:

    baseband waveform -> 4x upsampling (FFT zero-pad / polyphase resample)
                      -> 3GPP-style raised-cosine TX filter
                         (flat passband + raised-cosine roll-off in the
                          guard band, zero stopband beyond the channel edge)
                      -> PSD via Welch with a Blackman-Harris window.

Outputs (written to ``results/figures/``)
-----------------------------------------
* ``fig_tx_spectrum_NB-A.png`` -- PSD overlay for the NB-A bandwidth class.
* ``fig_tx_spectrum_WB-B.png`` -- PSD overlay for the WB-B bandwidth class.
* ``fig_tx_papr_ccdf_NB-A.png`` -- PAPR CCDF for all five waveforms at NB-A.
* ``fig_tx_papr_ccdf_WB-B.png`` -- PAPR CCDF for all five waveforms at WB-B.

Tests performed
---------------
- Spectrum plot at NB-A and WB-B (all 5 methods overlay) -- proves equal BW.
- PAPR CCDF per method (BW, method grid).
- ACLR (NR adjacent channel) at NB-A and WB-B.

Notes on signal processing choices
----------------------------------
- The TX filter is a 3GPP-compliant "brick-wall-with-rolloff" mask:
  unity gain inside the occupied bandwidth, raised-cosine roll-off across
  the guard band, then zero in the stopband beyond the channel edge.
  This mirrors the spectrum mask language in 3GPP TS 38.104 without
  committing to any specific RRC/FIR realisation.
- The Welch PSD uses a Blackman-Harris window (very low side-lobes,
  ~-92 dB) so that the >60 dB stopband attenuation produced by the TX
  filter remains visible above the spectral-estimator floor.
- 4x upsampling is used so that we can observe the first few adjacent
  channels (needed for ACLR) without aliasing energy back into the
  in-band region. 4x is the standard rate used by 3GPP conformance
  measurements at moderate SCS values.
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "CC-BY-4.0"
__copyright__ = "(c) 2026 Panos N. Alevizos"

import os, sys
import numpy as np

# Pin OpenBLAS to a single thread; numpy/scipy use BLAS internally for FFT
# helpers, and over-subscribing threads slows down a process that is already
# generating many short waveforms in a serial loop.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

# Import scipy.signal FIRST before mocking scipy.sparse (which OTFS needs).
# scipy.signal pulls in pieces of scipy that, in turn, do legitimate
# `import scipy.sparse`. If we mocked scipy.sparse before that import, the
# mock would shadow the real submodule and scipy.signal would fail to load.
import matplotlib
matplotlib.use('Agg')                      # headless backend; no display server
import matplotlib.pyplot as plt
from scipy import signal as scisig

# Now mock scipy.sparse so OTFS modules import without that dep.
# The OTFS reference implementations only touch scipy.sparse symbols inside
# methods that this characterisation script never calls (channel estimation
# helpers). A MagicMock for the module + its `linalg` submodule is enough to
# satisfy `from scipy.sparse import ...` at import time.
from unittest.mock import MagicMock
sys.modules.setdefault('scipy.sparse', MagicMock())
sys.modules.setdefault('scipy.sparse.linalg', MagicMock())

from ofdm      import OFDMTransceiver, OFDMConfig
from otfs      import OTFSTransceiver, OTFSConfig
# Note: in the current otfs_pcp API both PCP variants (orig and guard) use the
# same PCPOTFSTransceiver class — the variant is selected purely by the Mcp
# value in PCPOTFSConfig (full NR CP for "orig", channel-matched short CP for
# "guard").  An older API exposed a separate GuardedPCPOTFS class; aliased here
# for backward compatibility with this script's existing constructor calls.
from otfs_pcp  import PCPOTFSConfig, PCPOTFSTransceiver
GuardedPCPOTFS = PCPOTFSTransceiver  # alias; "guarded" behaviour encoded in the Mcp arg
from otfs_zp   import ZPOTFSConfig, ZPOTFSTransceiver
from qam       import qam_modulate, generate_random_bits


# ═══════════════════════════ Configuration ═══════════════════════════
OSR = 4                # oversampling ratio applied before the TX filter
N_SLOTS = 20           # number of slots per method for PSD averaging

# Two bandwidth classes used by the rest of the study:
#   NB-A : narrowband, M=256 sub-carriers at 15 kHz SCS  (~3.84 MHz BW class)
#   WB-B : wideband,   M=1024 sub-carriers at 60 kHz SCS (~61.44 MHz BW class)
# ``Mcp_g`` and ``ZP`` carry the per-class CP / zero-pad lengths that the
# transceiver factories below consume.
BW_CONFIGS = {
    'NB-A': dict(M=256,  SCS=15e3, Mcp_g=6,  ZP=8),
    'WB-B': dict(M=1024, SCS=60e3, Mcp_g=29, ZP=29),
}

# Per-method colour and label tables -- shared by both the PSD overlay and
# the PAPR CCDF plot so the legend stays consistent across figures.
COLORS  = {'ofdm':'#2ca02c', 'zp':'#1f77b4', 'pcp_guard':'#d62728',
           'pcp_orig':'#ff7f0e', 'mc_mp':'#9467bd'}
LABELS  = {'ofdm':'OFDM', 'zp':'ZP-OTFS', 'pcp_guard':'PCP-guard',
           'pcp_orig':'PCP-orig', 'mc_mp':'MC-OTFS'}


# ════════════════════ TX waveform generation per method ════════════════════
def build(method, cfg_label, M_fft, SCS, Mcp_g, ZP):
    """Instantiate a transceiver object for one waveform / BW combination.

    Parameters
    ----------
    method : {'ofdm', 'zp', 'pcp_guard', 'pcp_orig', 'mc_mp'}
        Which waveform implementation to construct.
    cfg_label : str
        Bandwidth-class tag (currently ``'NB-A'`` or ``'WB-B'``); kept
        for symmetry with future multi-class extensions but unused here.
    M_fft : int
        FFT length of the OFDM grid for this BW class (e.g. 256, 1024).
    SCS : float
        Sub-carrier spacing in Hz.
    Mcp_g : int
        Cyclic-prefix length (in samples) used by guarded PCP-OTFS; the
        non-guarded variants derive their CP from ``M_fft`` directly.
    ZP : int
        Zero-pad length (in samples) used by ZP-OTFS.

    Returns
    -------
    obj : object
        Transceiver instance whose ``.tx(qam)`` method maps complex QAM
        symbols to a baseband sample vector.
    Fs : float
        Baseband sample rate in Hz (= ``M_wave * SCS``).
    n_act : int
        Number of active (non-DC, non-guard) sub-carriers actually
        carrying user data -- used later to compute the occupied BW.
    M_wave : int
        Effective FFT / waveform length used by the chosen method.
        Equals ``n_act`` for ZP/PCP variants (they don't oversample with
        guard tones) and ``M_fft`` for OFDM and MC-OTFS.

    Notes
    -----
    The "same occupied BW" branch ensures that ZP-OTFS and PCP-OTFS run
    with ``M = n_act`` so their occupied bandwidth matches OFDM's
    ``n_act * SCS``. Without this trick they would either over-fill or
    under-fill the channel relative to OFDM.
    """
    N = 14                                          # OFDM symbols per slot
    CP_M  = max(round(144 * M_fft / 2048), 4)       # 3GPP CP length scaling
    n_act = min(M_fft - 2, int(round(156 * M_fft / 256)))
                                                     # active SC count, guard-band aware

    # Same occupied BW: PCP/ZP use M = n_act
    if method in ('zp', 'pcp_guard', 'pcp_orig'):
        M_wave = n_act
        CP_wave = max(round(144 * M_wave / 2048), 4)
    else:
        M_wave = M_fft
        CP_wave = CP_M
    Fs = M_wave * SCS                               # baseband sample rate

    if method == 'ofdm':
        # Standard CP-OFDM with two DM-RS-bearing symbols (idx 2 and 11).
        obj = OFDMTransceiver(OFDMConfig(
            n_fft=M_wave, n_active=n_act, scs_hz=SCS, n_cp=CP_wave,
            n_symbols_per_slot=N, dmrs_symbol_indices=[2,11], dmrs_comb_size=2))
    elif method == 'zp':
        # Zero-padded OTFS: explicit zero-tail per delay block + DD-domain
        # pilot with a small guard region around it.
        Fs_wave = M_wave * SCS
        ZP_wave = max(4, int(round(0.033 * M_wave)))    # ~3.3% zero-pad
        guard_d = int(round(0.04 * M_wave))             # ~4% delay guard
        obj = ZPOTFSTransceiver(ZPOTFSConfig(
            M=M_wave, N=N, scs_hz=SCS, zp_len=ZP_wave,
            pilot_delay=1, pilot_doppler=N//2,
            guard_delay=guard_d, guard_doppler=9, max_dd_taps=50))
    elif method == 'pcp_guard':
        # Guarded PCP-OTFS: pilot CP scaled to ~1 us absolute time so the
        # guard region absorbs realistic channel delay spreads independent
        # of M / SCS.
        Fs_wave = M_wave * SCS
        Mcp_new = max(4, int(round(1.0e-6 * Fs_wave)) + 3)
        obj = GuardedPCPOTFS(PCPOTFSConfig(
            M=M_wave, N=N, Mcp=Mcp_new, scs_hz=SCS,
            pilot_doppler=N//2, doppler_guard=1,
            pilot_power_dB=25.0, zc_root=1, bem_Q=1))
    elif method == 'pcp_orig':
        # Original PCP-OTFS: pilot CP set to the OFDM CP -- baseline before
        # the "guarded" enhancement.
        obj = PCPOTFSTransceiver(PCPOTFSConfig(
            M=M_wave, N=N, Mcp=CP_wave, scs_hz=SCS,
            pilot_doppler=N//2, doppler_guard=1,
            pilot_power_dB=25.0, zc_root=1, bem_Q=1))
    elif method == 'mc_mp':
        # Multi-carrier multi-pilot OTFS: spreads pilots across the DD grid
        # with a delay guard sized at ~2.5% of M.
        pgd = max(4, int(round(0.025 * M_wave)))
        obj = OTFSTransceiver(OTFSConfig(
            n_fft=M_wave, n_active=n_act, scs_hz=SCS, n_cp=CP_wave,
            n_symbols_per_frame=N,
            pilot_guard_delay=pgd, pilot_guard_doppler=6,
            pilot_power_boost_dB=12.0))
    return obj, Fs, n_act, M_wave


# ═══════════════════════════ TX filter design ═══════════════════════════
def rrc_transmit_filter_up(sig, osr, M_wave, n_act, rolloff=0.1,
                            filt_beta=0.22, filt_span=32):
    """3GPP-style TX chain: time-domain upsample + flat-passband RC filter.

    Pipeline applied to the baseband waveform:

    1. Upsample by ``osr`` using ``scipy.signal.resample_poly`` (polyphase
       FIR upsampler = direct zero-insertion + anti-alias FIR in time).
    2. Apply a frequency-domain transmit mask shaped as

           H(f) = 1                              for |f| <= f_occ
                  raised-cosine taper            for f_occ < |f| < f_ch
                  0                              for |f| >= f_ch

       which mirrors the 3GPP TS 38.104 transmit spectrum mask (flat
       passband, smooth roll-off across the guard band, zero stopband).

    Parameters
    ----------
    sig : array_like, complex
        Complex baseband samples at the original rate ``M_wave * SCS``.
    osr : int
        Oversampling ratio applied by the polyphase upsampler (typically
        4 for these characterisation runs).
    M_wave : int
        Effective FFT length of the input waveform; determines where the
        Nyquist boundary sits in the normalised-frequency axis.
    n_act : int
        Number of active sub-carriers; together with ``M_wave`` it sets
        the occupied-bandwidth fraction ``f_occ``.
    rolloff : float, optional
        Fractional roll-off width of the transmit mask, expressed
        relative to the occupied-band edge (default ``0.1`` -> 10 %).
    filt_beta, filt_span : float, int, optional
        Reserved for future RRC-based realisations; the current
        implementation uses a frequency-domain mask and ignores them.

    Returns
    -------
    sig_up : ndarray, complex
        Filtered, upsampled baseband signal at rate ``osr * M_wave * SCS``.

    Notes
    -----
    The frequency-domain mask is constructed in the FFT-shifted ordering
    (DC at the centre) for readability and converted back with
    ``ifftshift`` before multiplying the FFT of the upsampled signal.
    """
    sig = np.asarray(sig).ravel()
    # Direct time-domain polyphase upsampling (inserts zeros + FIR anti-alias)
    sig_up = scisig.resample_poly(sig, osr, 1)
    N_up = len(sig_up)

    # Frequency-domain 3GPP TX filter (flat passband + RC rolloff)
    # ----------------------------------------------------------------
    # Frequencies are normalised so that 1.0 == upsampled sample rate.
    # f_occ : half of the occupied bandwidth at the *upsampled* rate.
    # f_nyq : Nyquist of the *original* rate (= 1/(2 osr) at the up rate).
    # f_ch  : channel edge, capped at f_nyq so we never try to specify a
    #         passband edge beyond the original Nyquist.
    f_occ = n_act / (2.0 * M_wave * osr)  # normalized occupied half-BW
    f_nyq = 0.5 / osr                     # original-rate Nyquist
    f_ch  = min(f_occ * (1 + rolloff), f_nyq)
    trans_bw = f_ch - f_occ

    # Build the mask H(f) on a centred (fftshift) frequency axis. This
    # makes the three regions (passband / RC roll-off / stopband) easy to
    # express as logical masks over |f|.
    f_norm = np.fft.fftshift(np.fft.fftfreq(N_up, d=1.0))
    fa = np.abs(f_norm)
    H = np.zeros(N_up)
    H[fa <= f_occ] = 1.0                                       # flat passband
    if trans_bw > 1e-10:
        # Raised-cosine taper: 0.5 * (1 + cos(pi * (|f| - f_occ) / trans_bw))
        # equals 1 at |f| = f_occ and 0 at |f| = f_ch, with smooth slope.
        mask = (fa > f_occ) & (fa < f_occ + trans_bw)
        H[mask] = 0.5 * (1 + np.cos(np.pi * (fa[mask] - f_occ) / trans_bw))
    # |f| >= f_ch is left at 0 -> hard stopband beyond the channel edge.

    # Apply the mask: ifftshift undoes the centring so that H lines up
    # with numpy's natural FFT ordering (DC at index 0).
    sig_up = np.fft.ifft(np.fft.fft(sig_up) * np.fft.ifftshift(H))
    return sig_up


# ═══════════════════════════ PSD computation ═══════════════════════════
def psd_welch(sig, fs, nperseg=None):
    """Welch PSD, Blackman-Harris, two-sided (centered).

    Parameters
    ----------
    sig : array_like, complex
        Time-domain signal samples.
    fs : float
        Sample rate of ``sig`` in Hz.
    nperseg : int, optional
        Welch segment length. Defaults to ``min(4096, len(sig)//8)`` so
        we always get at least 8 averaging segments while bounding
        per-segment FFT cost.

    Returns
    -------
    f : ndarray
        Frequency axis in Hz, centred at DC and monotonically increasing
        from ``-fs/2`` to ``+fs/2``.
    Pxx : ndarray
        Two-sided PSD aligned with ``f``.

    Notes
    -----
    Blackman-Harris is chosen for its very low side-lobe level
    (~-92 dB), which lets the plot show the >60 dB TX-mask stopband
    attenuation without contamination from window leakage.
    """
    if nperseg is None:
        nperseg = min(4096, len(sig) // 8)
    f, Pxx = scisig.welch(sig, fs=fs, window='blackmanharris',
                          nperseg=nperseg, noverlap=nperseg//2,
                          return_onesided=False, scaling='density')
    # Center around DC
    # scipy returns frequencies in [0, fs); we want them in [-fs/2, +fs/2)
    # for a symmetric plot around DC.
    idx = np.argsort(f)
    f = f[idx]; Pxx = Pxx[idx]
    # Shift so DC is at center
    mask_neg = f > fs/2 - 1
    f[f > fs/2] -= fs
    idx2 = np.argsort(f)
    f = f[idx2]; Pxx = Pxx[idx2]
    return f, Pxx


# ═══════════════════════════ PAPR CCDF computation ═══════════════════════════
def compute_papr_ccdf(sig, n_bins=200):
    """CCDF of instantaneous power / mean power (PAPR).

    Parameters
    ----------
    sig : array_like, complex
        Complex baseband (or upsampled) signal samples.
    n_bins : int, optional
        Number of equally-spaced PAPR thresholds (in dB) over [0, 15 dB]
        at which the empirical CCDF is evaluated.

    Returns
    -------
    x_db : ndarray
        PAPR threshold axis in dB.
    ccdf : ndarray
        Empirical CCDF values, ``Pr{ PAPR > x }``, one per threshold.

    Notes
    -----
    PAPR is defined sample-wise as ``|x[n]|^2 / mean(|x[n]|^2)`` and
    converted to dB. The 0.01 % CCDF (``ccdf == 1e-4``) is the standard
    figure of merit reported in 3GPP and is later extracted from this
    curve by interpolation.
    """
    inst_pow = np.abs(sig)**2
    mean_pow = np.mean(inst_pow)
    # +1e-30 avoids log(0) for occasional exact-zero samples (e.g. ZP tail).
    papr_db = 10 * np.log10(inst_pow / mean_pow + 1e-30)
    # CCDF axis in dB
    # CCDF at threshold x is the empirical Pr{PAPR > x}.
    x_db = np.linspace(0, 15, n_bins)
    ccdf = np.array([np.mean(papr_db > x) for x in x_db])
    return x_db, ccdf


# ═══════════════════════════ ACLR computation ═══════════════════════════
def measure_aclr(f, Pxx, occ_bw, ch_bw):
    """NR-ACLR: ratio of in-channel power to adjacent-channel power.

    Parameters
    ----------
    f : ndarray
        Frequency axis in Hz (DC-centred, monotonically increasing).
    Pxx : ndarray
        Two-sided PSD (same length as ``f``) in linear power units.
    occ_bw : float
        Occupied bandwidth in Hz (currently unused -- kept in the
        signature for callers that want to pass both numbers).
    ch_bw : float
        Channel bandwidth in Hz used for both the in-band and the two
        adjacent-channel integration windows.

    Returns
    -------
    aclr_hi : float
        ACLR for the upper adjacent channel, in dB
        (``10*log10(P_main / P_adj_+)``).
    aclr_lo : float
        ACLR for the lower adjacent channel, in dB
        (``10*log10(P_main / P_adj_-)``).

    Notes
    -----
    Power inside a band is computed by Riemann-summing ``Pxx`` over the
    matching frequency bins, multiplied by the bin spacing ``df`` to
    convert PSD (W/Hz) into power (W). The classic NR-ACLR definition
    uses identical bandwidths for the in-band and adjacent-channel
    integrals -- both are set to ``ch_bw`` here.
    """
    def band_pow(f_center, bw):
        # Half-open interval [f_center - bw/2, f_center + bw/2) keeps
        # adjacent windows from double-counting the boundary bin.
        mask = (f >= f_center - bw/2) & (f < f_center + bw/2)
        df = f[1] - f[0]
        return np.sum(Pxx[mask]) * df
    P_main  = band_pow(0, ch_bw)            # in-band power
    P_adj_p = band_pow(+ch_bw, ch_bw)       # upper adjacent channel
    P_adj_n = band_pow(-ch_bw, ch_bw)       # lower adjacent channel
    # Floor adjacent-channel power at 1e-30 to avoid log(0) when the TX
    # filter has driven it below numerical resolution.
    return 10 * np.log10(P_main / max(P_adj_p, 1e-30)), \
           10 * np.log10(P_main / max(P_adj_n, 1e-30))


def generate_waveform(obj, n_slots, seed_base=42):
    """Concatenate N_slots worth of TX signal.

    Parameters
    ----------
    obj : object
        Transceiver instance returned by :func:`build`. Must expose
        ``count_data_res()`` (number of data resource elements per slot)
        and ``tx(qam) -> (samples, meta)``.
    n_slots : int
        How many slots to concatenate. More slots -> smoother PSD
        averages and longer PAPR tail samples.
    seed_base : int, optional
        Base seed; per-slot RNGs use ``seed_base + slot_index`` so the
        run is bit-reproducible across invocations.

    Returns
    -------
    ndarray, complex
        Concatenated baseband samples spanning all ``n_slots`` slots.
    """
    nd = obj.count_data_res()
    frames = []
    for s in range(n_slots):
        # Per-slot deterministic RNG so the spectrum / PAPR estimates are
        # reproducible across runs.
        rng = np.random.default_rng(seed_base + s)
        bits = generate_random_bits(nd * 2, rng)        # 2 bits / QPSK symbol
        qam = qam_modulate(bits, 4)[:nd]                # QPSK = 4-QAM
        tx, _ = obj.tx(qam)
        frames.append(tx)
    return np.concatenate(frames)


def run_one_config(cfg_label, cfg, outdir):
    """Run all 5 waveforms at one bandwidth class and collect results.

    Parameters
    ----------
    cfg_label : str
        Bandwidth-class tag (e.g. ``'NB-A'``, ``'WB-B'``).
    cfg : dict
        BW-class configuration dictionary from :data:`BW_CONFIGS`.
    outdir : str
        Output directory for figures (passed through; not used for any
        side effect inside this function).

    Returns
    -------
    dict
        Mapping ``method -> result_dict`` where each ``result_dict``
        contains ``f_psd``, ``Pxx``, ``fs_up``, ``x_db``, ``ccdf``,
        ``occ_bw``, ``ch_bw``, ``aclr_lo``, ``aclr_hi`` and the
        interpolated ``papr_001`` (PAPR at 0.01 % CCDF, in dB).
    """
    methods = ['ofdm', 'zp', 'pcp_guard', 'pcp_orig', 'mc_mp']
    M_fft, SCS = cfg['M'], cfg['SCS']

    results = {}
    for method in methods:
        obj, Fs, n_act, M_wave = build(method, cfg_label, M_fft, SCS,
                                        cfg['Mcp_g'], cfg['ZP'])
        print(f'  [{cfg_label}/{method:10s}] M_wave={M_wave} Fs={Fs/1e6:.3f}MHz, generating {N_SLOTS} slots ...', flush=True)
        sig = generate_waveform(obj, N_SLOTS)
        sig_up = rrc_transmit_filter_up(sig, OSR, M_wave, n_act)
        fs_up = Fs * OSR
        f_psd, Pxx = psd_welch(sig_up, fs_up)
        x_db, ccdf = compute_papr_ccdf(sig_up)
        # ACLR: channel BW = 1.1× occupied BW, per practice
        # (10 % guard between active SCs and the channel edge is the
        # value used by 3GPP NR conformance tests for these BW classes.)
        occ_bw = n_act * SCS
        ch_bw  = 1.1 * occ_bw
        aclr_lo, aclr_hi = measure_aclr(f_psd, Pxx, occ_bw, ch_bw)
        results[method] = {
            'f_psd': f_psd, 'Pxx': Pxx, 'fs_up': fs_up,
            'x_db': x_db, 'ccdf': ccdf,
            'occ_bw': occ_bw, 'ch_bw': ch_bw,
            'aclr_lo': aclr_lo, 'aclr_hi': aclr_hi,
            # PAPR at the 0.01 % CCDF point, found by interpolating the
            # monotonically decreasing CCDF (passed reversed so x is
            # ascending in y).
            'papr_001': float(np.interp(1e-4, ccdf[::-1], x_db[::-1]))
                        if np.any(ccdf > 0) else float('nan'),
        }
    return results


# ═══════════════════════════ Plotting ═══════════════════════════
def plot_psd(results, cfg_label, outdir):
    """Single-panel PSD overlay with clear region annotations so the reader
    immediately sees: (i) the flat passband, (ii) the RC-rolloff edge, and
    (iii) the stopband where the 3GPP TX filter has knocked the signal down
    by >60 dB.  x-axis spans ±1.3 × BW/2; y-axis covers the full dynamic
    range so passband flatness AND edge roll-off are simultaneously visible.

    Parameters
    ----------
    results : dict
        Per-method results dict produced by :func:`run_one_config`.
    cfg_label : str
        Bandwidth-class tag used in the title and output filename.
    outdir : str
        Directory in which to save the PNG.

    Returns
    -------
    None
        Side effect: writes ``fig_tx_spectrum_<cfg_label>.png``.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    occ_bw = list(results.values())[0]['occ_bw']
    occ_mhz = occ_bw / 1e6
    x_span = occ_mhz * 0.85   # ~±0.85×BW

    # ── Shaded region annotations (below curves, above grid) ────────────
    # Passband: -occ/2 to +occ/2
    ax.axvspan(-occ_mhz/2, occ_mhz/2, alpha=0.12, color='#2ca02c', zorder=0)
    # Stopband: beyond ±occ/2
    ax.axvspan(-x_span, -occ_mhz/2, alpha=0.12, color='#d62728', zorder=0)
    ax.axvspan( occ_mhz/2,  x_span, alpha=0.12, color='#d62728', zorder=0)

    # Annotate regions
    ax.text(0, 2.5, 'PASSBAND\n(flat, all methods\nsuperimposed)',
            ha='center', va='top', fontsize=10, color='#1a6a1a',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#2ca02c'))
    ax.text(-x_span * 0.8, -30, 'STOPBAND\n(>45 dB\nattenuation)',
            ha='center', va='center', fontsize=9, color='#8b1a1a',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#d62728'))
    ax.text( x_span * 0.8, -30, 'STOPBAND\n(>45 dB\nattenuation)',
            ha='center', va='center', fontsize=9, color='#8b1a1a',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#d62728'))
    # Edge transition label
    ax.annotate('RC rolloff\nedge', xy=(occ_mhz/2, -20), xytext=(occ_mhz*0.65, -10),
                fontsize=9, ha='center', color='#444',
                arrowprops=dict(arrowstyle='->', color='#444', lw=0.8))

    # ── Reference level lines (-3 dB and -30 dB) ───────────────────────
    ax.axhline(-3,  color='gray', linestyle=':', alpha=0.6, linewidth=0.8)
    ax.axhline(-30, color='gray', linestyle=':', alpha=0.6, linewidth=0.8)
    ax.text(x_span * 0.98, -2,  '−3 dB',  fontsize=8, ha='right', color='gray')
    ax.text(x_span * 0.98, -29, '−30 dB', fontsize=8, ha='right', color='gray')

    # ── PSD curves ──────────────────────────────────────────────────────
    for m, r in results.items():
        Pxx_db = 10 * np.log10(r['Pxx'] + 1e-30)
        peak = np.max(Pxx_db)
        f_mhz = r['f_psd'] / 1e6
        mask = np.abs(f_mhz) <= x_span
        ax.plot(f_mhz[mask], Pxx_db[mask] - peak, color=COLORS[m],
                linewidth=1.5, zorder=3,
                label=f"{LABELS[m]}  (ACLR={results[m]['aclr_hi']:.1f} dB, "
                      f"PAPR$_{{0.01\\%}}$={results[m]['papr_001']:.2f} dB)")

    # ── BW edge markers ─────────────────────────────────────────────────
    ax.axvline( occ_mhz/2, color='k', linestyle='--', alpha=0.7, linewidth=1.0,
                label=f'BW edges: ±{occ_mhz/2:.3f} MHz (= {occ_mhz:.2f} MHz total)')
    ax.axvline(-occ_mhz/2, color='k', linestyle='--', alpha=0.7, linewidth=1.0)

    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Normalized PSD (dB, peak = 0)')
    ax.set_title(f'TX Power Spectral Density — {cfg_label}   '
                 f'(equal-BW verification: passband flatness + edge rolloff)',
                 fontsize=12)
    ax.set_ylim(-72, 5)
    ax.set_xlim(-x_span, x_span)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8.5, loc='lower center', ncol=2, framealpha=0.92)

    outpath = os.path.join(outdir, f'fig_tx_spectrum_{cfg_label}.png')
    fig.tight_layout()
    fig.savefig(outpath, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {outpath}')


def plot_papr_ccdf(results, cfg_label, outdir):
    """Plot the per-method PAPR CCDF on a semilog-y axis.

    Parameters
    ----------
    results : dict
        Per-method results dict produced by :func:`run_one_config`.
    cfg_label : str
        Bandwidth-class tag used in the title and output filename.
    outdir : str
        Directory in which to save the PNG.

    Returns
    -------
    None
        Side effect: writes ``fig_tx_papr_ccdf_<cfg_label>.png``.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    for m, r in results.items():
        ax.semilogy(r['x_db'], r['ccdf'], color=COLORS[m],
                    label=f"{LABELS[m]} (PAPR@0.01%={r['papr_001']:.2f} dB)",
                    linewidth=1.8)
    # Horizontal reference line at the standard 0.01 % CCDF cut.
    ax.axhline(1e-4, color='k', linestyle=':', alpha=0.5, label='0.01% CCDF')
    ax.set_xlabel('PAPR (dB)'); ax.set_ylabel('CCDF')
    ax.set_title(f'PAPR CCDF — {cfg_label}')
    ax.set_xlim(0, 14); ax.set_ylim(1e-5, 1)
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=9)
    outpath = os.path.join(outdir, f'fig_tx_papr_ccdf_{cfg_label}.png')
    fig.tight_layout()
    fig.savefig(outpath, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {outpath}')


def print_summary(all_results):
    """Print a compact tabular summary of PAPR and ACLR across configs.

    Parameters
    ----------
    all_results : dict
        Mapping ``cfg_label -> per-method-results-dict``, exactly the
        shape assembled by :func:`main`.

    Returns
    -------
    None
        Side effect: writes a formatted table to stdout.
    """
    print()
    print(f'{"Config":7s} {"Method":11s} {"PAPR@0.01%":>12s} {"ACLR (dB)":>12s} {"occ_BW_MHz":>11s}')
    print('-' * 60)
    for cl, res in all_results.items():
        for m, r in res.items():
            print(f'{cl:7s} {m:11s} {r["papr_001"]:>10.2f} dB   '
                  f'{r["aclr_hi"]:>9.2f} / {r["aclr_lo"]:.2f}   '
                  f'{r["occ_bw"]/1e6:>9.3f}')


# ═══════════════════════════ Main ═══════════════════════════
def main():
    """Top-level driver: iterate BW classes, run characterisation, plot.

    Returns
    -------
    None
        Side effects: creates ``results/figures``, writes four PNGs, and
        prints a per-method summary table.
    """
    outdir = 'results/figures'
    os.makedirs(outdir, exist_ok=True)
    all_results = {}
    for cl, cfg in BW_CONFIGS.items():
        print(f'\n=== {cl} ===')
        res = run_one_config(cl, cfg, outdir)
        plot_psd(res, cl, outdir)
        plot_papr_ccdf(res, cl, outdir)
        all_results[cl] = res
    print_summary(all_results)


if __name__ == '__main__':
    main()
