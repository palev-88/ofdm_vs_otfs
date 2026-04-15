"""
run_eval.py — BER evaluation for one FFT/SCS/channel config.

Usage: python3 run_eval.py <M> <SCS_kHz> [NMC] [channel]
  M:       FFT size (64, 128, 256, 512)
  SCS_kHz: Subcarrier spacing in kHz (15, 30, 60)
  NMC:     Monte Carlo trials (default 20)
  channel: TDL-A, TDL-D, TDL-C, AWGN, or ALL (default ALL)

Examples:
  python3 run_eval.py 256 60           # All channels, 20 MC
  python3 run_eval.py 256 60 50        # All channels, 50 MC
  python3 run_eval.py 256 60 20 TDL-A  # TDL-A only, 20 MC

Saves JSON to results/ber_M{M}_SCS{SCS}k_{channel}.json
"""
import numpy as np, sys, time, json, os
from channel import TDLChannel, TDLChannelConfig
from ofdm import OFDMTransceiver, OFDMConfig
from otfs_zp import ZPOTFSTransceiver, ZPOTFSConfig
from otfs_pcp import PCPOTFSTransceiver, PCPOTFSConfig
from qam import qam_modulate, qam_demodulate, generate_random_bits

# ═══ Parse args ═══
M = int(sys.argv[1])
SCS = float(sys.argv[2]) * 1e3
NMC = int(sys.argv[3]) if len(sys.argv) > 3 else 20
CH_ARG = sys.argv[4] if len(sys.argv) > 4 else 'ALL'
FD_RANGE = sys.argv[5] if len(sys.argv) > 5 else 'full'  # 'low','high','full'

FS = M * SCS
CP = max(round(144 * M / 2048), 4)
N = 14

# ═══ Build transceivers ═══
ch_c = TDLChannel(TDLChannelConfig('TDL-C', 100e-9, 0, FS, seed=42))
ZP = max(8, ch_c.max_delay_samples + 2)
n_act = min(M - 2, 156)

ofdm = OFDMTransceiver(OFDMConfig(n_fft=M, n_active=n_act, scs_hz=SCS, n_cp=CP,
    n_symbols_per_slot=N, dmrs_symbol_indices=[2, 11], dmrs_comb_size=2))
zp = ZPOTFSTransceiver(ZPOTFSConfig(M=M, N=N, scs_hz=SCS, zp_len=ZP,
    pilot_delay=1, pilot_doppler=N//2, guard_delay=ZP+2, guard_doppler=4, max_dd_taps=50))
pcp = PCPOTFSTransceiver(PCPOTFSConfig(M=M, N=N, Mcp=CP, scs_hz=SCS,
    pilot_doppler=N//2, doppler_guard=1, pilot_power_dB=25.0, zc_root=1, bem_Q=1))

nd_o = ofdm.count_data_res()
nd_z = zp.count_data_res()
nd_p = pcp.count_data_res()

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
print(f"═══ BER Eval: M={M} SCS={SCS/1e3:.0f}kHz Fs={FS/1e6:.1f}MHz CP={CP} ZP={ZP} NMC={NMC} ═══")
print(f"  OFDM:  data={nd_o}, OH={ofdm.cfg.pilot_overhead*100:.1f}%")
print(f"  ZP:    data={nd_z}, OH={zp.cfg.pilot_overhead*100:.1f}%")
print(f"  PCP:   data={nd_p}, OH={pcp.cfg.pilot_overhead*100:.1f}%\n")

R = {}

for cm, ds in ch_list:
    fd_list = [0] if cm == 'AWGN' else DOPPLERS
    for fd in fd_list:
        for snr in SNR_LIST:
            nv = 10 ** (-snr / 10)
            for nm, mt, obj, nd in [('ofdm','o',ofdm,nd_o),
                                     ('zp','z',zp,nd_z),
                                     ('pcp','p',pcp,nd_p)]:
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
                  f"P={R[f'{cm}_{fd}_{snr}_pcp']:.2e} "
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
        'ofdm': {'data_res': nd_o, 'overhead_pct': ofdm.cfg.pilot_overhead * 100},
        'zp':   {'data_res': nd_z, 'overhead_pct': zp.cfg.pilot_overhead * 100},
        'pcp':  {'data_res': nd_p, 'overhead_pct': pcp.cfg.pilot_overhead * 100},
    },
    'snr_list': SNR_LIST,
    'ber': existing_ber,
}

fname = f"results/ber_M{M}_SCS{int(SCS/1e3)}k_{CH_ARG}.json"
with open(fname, 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n✓ Saved {fname} ({time.time()-t0:.0f}s)")
