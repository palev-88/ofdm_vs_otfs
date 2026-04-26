"""
run_eval.py — BER evaluation for one FFT/SCS/channel config.

Usage: python3 run_eval.py <M> <SCS_kHz> [NMC] [channel] [FD_RANGE] [N_ACT]
  M:        FFT size (64, 128, 256, 512)
  SCS_kHz:  Subcarrier spacing in kHz (15, 30, 60)
  NMC:      Monte Carlo trials (default 20)
  channel:  TDL-A, TDL-D, TDL-C, AWGN, or ALL (default ALL)
  FD_RANGE: low | mid | high | full (default full)
  N_ACT:    Active subcarriers (default: scales ~0.61 * M, 3GPP-like)

Examples:
  python3 run_eval.py 256 60                    # All channels, 20 MC, n_act=156
  python3 run_eval.py 256 60 50 ALL full 120    # n_act=120 override
  python3 run_eval.py 128 60 20 TDL-A full 78   # M=128, n_act=78

Saves JSON to results/ber_M{M}_SCS{SCS}k_{channel}.json
"""

__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "CC-BY-4.0"
__copyright__ = "(c) 2026 Panos N. Alevizos"
import numpy as np, sys, time, json, os
from channel import TDLChannel, TDLChannelConfig
from ofdm import OFDMTransceiver, OFDMConfig
from otfs_zp import ZPOTFSTransceiver, ZPOTFSConfig
from otfs_pcp import PCPOTFSTransceiver, PCPOTFSConfig
from otfs_mc import MCOTFSTransceiver, MCOTFSConfig
from otfs_deconv import DeconvOTFSTransceiver, DeconvOTFSConfig
from qam import qam_modulate, qam_demodulate, generate_random_bits

# ═══ Parse args ═══
M = int(sys.argv[1])
SCS = float(sys.argv[2]) * 1e3
NMC = int(sys.argv[3]) if len(sys.argv) > 3 else 20
CH_ARG = sys.argv[4] if len(sys.argv) > 4 else 'ALL'
FD_RANGE = sys.argv[5] if len(sys.argv) > 5 else 'full'  # 'low','mid','high','full'

FS = M * SCS
N = 14

# OFDM uses 5G NR Normal CP (standardized, don't change)
CP_OFDM = max(round(144 * M / 2048), 4)

# OTFS methods — each method's guard length tuned to minimum that doesn't hurt BER
# (empirically verified via sweep across [6,8,10,12,14,16] at M=256 SCS=60kHz)
ch_c = TDLChannel(TDLChannelConfig('TDL-C', 100e-9, 0, FS, seed=42))
MAX_DELAY = ch_c.max_delay_samples  # 13 for M=256 SCS=60kHz TDL-C

# CP-OTFS/Deconv: cross-domain detector captures ISI in DD channel matrix, so
# CP can be much shorter than max channel delay without BER impact.
CP_CPOTFS = max(MAX_DELAY // 2, 4)  # ~6-8 for typical channels

# PCP-OTFS — TWO VARIANTS evaluated separately:
#   pcp_guard: channel-matched short CP sized to absorb ~1us delay spread,
#              independent of M / SCS.  Lower overhead, comparable BER on
#              short-delay channels.
#   pcp_orig:  full 5G-NR Normal CP scaled to PCP's M=n_act.  Higher overhead
#              but cleaner on long-delay-spread channels (e.g. TDL-C).
# (CP values defined below, after n_act is known.)

# ZP-OTFS: needs to cover max delay but can be slightly less (edge taps have
# low power). Sweep shows zp=10-12 performs as well or better than zp=15.
ZP = max(MAX_DELAY, 8)  # 13 for M=256 SCS=60kHz (was max_delay+2)

CP = CP_OFDM  # kept for backward-compat printouts

# ═══ Build transceivers ═══

# n_active: configurable via CLI, default scales ~0.61 * M (3GPP-like)
# Round down to multiple of 12 for NR consistency, capped at M-2
if len(sys.argv) > 6:
    n_act = int(sys.argv[6])
else:
    n_act = (int(M * 0.61) // 12) * 12
n_act = min(n_act, M - 2)

# PCP-OTFS CP values (depend on n_act, so computed here)
CP_PCP_GUARD = max(4, int(round(1.0e-6 * (n_act * SCS))) + 3)  # channel-matched short CP
CP_PCP_ORIG  = max(round(144 * n_act / 2048), 4)               # full 5G-NR Normal CP

ofdm = OFDMTransceiver(OFDMConfig(n_fft=M, n_active=n_act, scs_hz=SCS, n_cp=CP_OFDM,
    n_symbols_per_slot=N, dmrs_symbol_indices=[2, 11], dmrs_comb_size=2))
zp = ZPOTFSTransceiver(ZPOTFSConfig(M=n_act, N=N, scs_hz=SCS, zp_len=ZP,
    pilot_delay=1, pilot_doppler=N//2, guard_delay=ZP+2, guard_doppler=4, max_dd_taps=50))
pcp_guard = PCPOTFSTransceiver(PCPOTFSConfig(M=n_act, N=N, Mcp=CP_PCP_GUARD, scs_hz=SCS,
    pilot_doppler=N//2, doppler_guard=1, pilot_power_dB=25.0, zc_root=1, bem_Q=1))
pcp_orig  = PCPOTFSTransceiver(PCPOTFSConfig(M=n_act, N=N, Mcp=CP_PCP_ORIG,  scs_hz=SCS,
    pilot_doppler=N//2, doppler_guard=1, pilot_power_dB=25.0, zc_root=1, bem_Q=1))
# Pilot guard zones sized to realistic channel (empirically tuned):
#   guard_delay=6: covers max delay ~6 bins (TDL-C DS=100ns at 156 SCs) with margin
#   guard_doppler=2: 8x safety over fD=1000Hz Doppler spread (~0.25 bins)
#   doppler_guard_edge=1, delay_guard_edge=1: minimal SFFT edge protection
# Total overhead: ~18% (vs 25% original)
GUARD_DELAY = 6
GUARD_DOPPLER = 2
EDGE_D = 1
EDGE_L = 1
mc_otfs = MCOTFSTransceiver(MCOTFSConfig(n_fft=M, n_active=n_act, scs_hz=SCS,
    n_cp=CP_CPOTFS, n_symbols_per_frame=N, pilot_power_boost_dB=15.0,
    pilot_guard_delay=GUARD_DELAY, pilot_guard_doppler=GUARD_DOPPLER,
    doppler_guard_edge=EDGE_D, delay_guard_edge=EDGE_L))
deconv = DeconvOTFSTransceiver(DeconvOTFSConfig(n_fft=M, n_active=n_act, scs_hz=SCS,
    n_cp=CP_CPOTFS, n_symbols_per_frame=N, pilot_power_boost_dB=15.0,
    pilot_guard_delay=GUARD_DELAY, pilot_guard_doppler=GUARD_DOPPLER,
    doppler_guard_edge=EDGE_D, delay_guard_edge=EDGE_L,
    dividing_number=10, alpha=1.0/50, beta=1.0/10))

nd_o  = ofdm.count_data_res()
nd_z  = zp.count_data_res()
nd_pg = pcp_guard.count_data_res()
nd_po = pcp_orig.count_data_res()
nd_m  = mc_otfs.count_data_res()
nd_d  = deconv.count_data_res()

# ═══ Test scenarios ═══
SNR_LIST = [0, 5, 10, 15, 20, 25, 30]
DS_MAP = {'TDL-A': 30e-9, 'TDL-B': 100e-9, 'TDL-C': 100e-9, 'TDL-D': 30e-9, 'TDL-E': 30e-9}
ALL_DOPPLERS = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

if FD_RANGE == 'low':
    DOPPLERS = [0, 100, 200, 300]
elif FD_RANGE == 'mid':
    DOPPLERS = [400, 500, 600]
elif FD_RANGE == 'high':
    DOPPLERS = [700, 800, 900, 1000]
elif FD_RANGE == 'smoke':
    DOPPLERS = [0, 500]
else:
    DOPPLERS = ALL_DOPPLERS

if CH_ARG == 'ALL':
    ch_list = [('AWGN', 0), ('TDL-A', 30e-9), ('TDL-B', 100e-9), ('TDL-C', 100e-9), ('TDL-D', 30e-9), ('TDL-E', 30e-9)]
elif CH_ARG == 'AWGN':
    ch_list = [('AWGN', 0)]
else:
    ch_list = [(CH_ARG, DS_MAP[CH_ARG])]

# ═══ Run ═══
t0 = time.time()
print(f"=== BER Eval: M={M} SCS={SCS/1e3:.0f}kHz Fs={FS/1e6:.1f}MHz n_act={n_act} NMC={NMC} ===")
print(f"  OFDM:  data={nd_o},  CP={CP_OFDM},  OH={ofdm.cfg.pilot_overhead*100:.1f}%")
print(f"  ZP:    data={nd_z},  ZP={ZP},  OH={zp.cfg.pilot_overhead*100:.1f}%")
print(f"  PCP-G: data={nd_pg}, Mcp={CP_PCP_GUARD}, OH={pcp_guard.cfg.pilot_overhead*100:.1f}%")
print(f"  PCP-O: data={nd_po}, Mcp={CP_PCP_ORIG},  OH={pcp_orig.cfg.pilot_overhead*100:.1f}%")
print(f"  MC:    data={nd_m},  CP={CP_CPOTFS},  OH={mc_otfs.cfg.pilot_overhead*100:.1f}%")
print(f"  DCV:   data={nd_d},  CP={CP_CPOTFS},  OH={deconv.cfg.pilot_overhead*100:.1f}%\n")

R = {}

for cm, ds in ch_list:
    fd_list = [0] if cm == 'AWGN' else DOPPLERS
    for fd in fd_list:
        for snr in SNR_LIST:
            nv = 10 ** (-snr / 10)
            for nm, mt, obj, nd in [('ofdm',     'o', ofdm,      nd_o),
                                     ('zp',       'z', zp,        nd_z),
                                     ('pcp_guard','p', pcp_guard, nd_pg),
                                     ('pcp_orig', 'p', pcp_orig,  nd_po),
                                     ('mc',       'm', mc_otfs,   nd_m),
                                     ('deconv',   'd', deconv,    nd_d)]:
                errs, nb = 0, 0
                for t in range(NMC):
                    s = t * 1000 + 42
                    bits = generate_random_bits(nd * 2, np.random.default_rng(s))
                    tx, _ = obj.tx(qam_modulate(bits, 4)[:nd])

                    if cm == 'AWGN':
                        sp_ = np.mean(np.abs(tx)**2)
                        rng = np.random.RandomState(s + 77)
                        rx = tx + np.sqrt(sp_*nv/2) * (
                            rng.randn(len(tx)) + 1j * rng.randn(len(tx)))
                    else:
                        ch = TDLChannel(TDLChannelConfig(cm, ds, fd, FS, seed=s))
                        rx, _ = ch.apply(tx, snr_dB=snr)

                    if mt == 'o':
                        dr = obj.rx(rx, 'linear', 'mmse', nv, 0,
                                    noise_est_method='dmrs_power_diff')[0]
                    elif mt == 'd':
                        dr, _, _ = obj.rx(rx, nv, detector='cross_domain', n_iter=3)
                    else:
                        dr = obj.rx(rx, nv)[0]

                    br = qam_demodulate(dr[:nd], 4)
                    n = min(len(bits), len(br))
                    errs += int(np.sum(bits[:n] != br[:n]))
                    nb += n

                R[f'{cm}_{fd}_{snr}_{nm}'] = errs / nb if nb > 0 else 0.0

            print(f"  {cm:>5s} fD={fd:4d} SNR={snr:2d}dB: "
                  f"O={R[f'{cm}_{fd}_{snr}_ofdm']:.2e} "
                  f"Z={R[f'{cm}_{fd}_{snr}_zp']:.2e} "
                  f"PG={R[f'{cm}_{fd}_{snr}_pcp_guard']:.2e} "
                  f"PO={R[f'{cm}_{fd}_{snr}_pcp_orig']:.2e} "
                  f"M={R[f'{cm}_{fd}_{snr}_mc']:.2e} "
                  f"D={R[f'{cm}_{fd}_{snr}_deconv']:.2e} "
                  f"({time.time()-t0:.0f}s)")

# ═══ Save (merge with existing) ═══
os.makedirs('results', exist_ok=True)
fname = f"results/ber_M{M}_SCS{int(SCS/1e3)}k_{CH_ARG}.json"

# Load existing results if any
existing_ber = {}
if os.path.exists(fname):
    with open(fname) as f:
        old = json.load(f)
    existing_ber = old.get('ber', {})

# Merge
existing_ber.update(R)

out = {
    'config': {
        'M': M, 'SCS_kHz': SCS / 1e3, 'Fs_MHz': FS / 1e6,
        'CP': CP, 'ZP': ZP, 'N': N, 'NMC': NMC,
        'n_active': n_act,
    },
    'methods': {
        'ofdm':      {'data_res': nd_o,  'overhead_pct': ofdm.cfg.pilot_overhead * 100},
        'zp':        {'data_res': nd_z,  'overhead_pct': zp.cfg.pilot_overhead * 100},
        'pcp_guard': {'data_res': nd_pg, 'overhead_pct': pcp_guard.cfg.pilot_overhead * 100,
                      'mcp': CP_PCP_GUARD},
        'pcp_orig':  {'data_res': nd_po, 'overhead_pct': pcp_orig.cfg.pilot_overhead * 100,
                      'mcp': CP_PCP_ORIG},
        'mc':        {'data_res': nd_m,  'overhead_pct': mc_otfs.cfg.pilot_overhead * 100},
        'deconv':    {'data_res': nd_d,  'overhead_pct': deconv.cfg.pilot_overhead * 100},
    },
    'snr_list': SNR_LIST,
    'ber': existing_ber,
}

fname = f"results/ber_M{M}_SCS{int(SCS/1e3)}k_{CH_ARG}.json"
with open(fname, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n[OK] Saved {fname} ({time.time()-t0:.0f}s)")
