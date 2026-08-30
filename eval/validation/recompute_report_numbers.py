"""Recompute every configuration-derived number the report quotes, at the
ACTUAL evaluated geometry (common grid, N=14), replacing the legacy
Zhang-pick (M=512,N=7) values.  Prints a block ready for the report pass.
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import json, os, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data')

CASES = {
    'NB': dict(M=256, n_act=156, SCS=30e3, N=14, CP=18, cp_g=8,  cp_o=11,
               nd_o=2028, nd_g=1850, nd_p=1841, A=1841, E=3682,  BG=2, Z=192),
    'WB': dict(M=1024, n_act=624, SCS=60e3, N=14, CP=72, cp_g=40, cp_o=44,
               nd_o=8112, nd_g=7370, nd_p=7358, A=7358, E=14716, BG=1, Z=352),
}


def fft_cmac(n):
    """Radix-2 equivalent butterfly count; mixed-radix for non-power-of-2
    carries ~10% overhead."""
    if n <= 1:
        return 0
    p = 1.0 if (n & (n - 1)) == 0 else 1.10
    return p * (n / 2.0) * math.log2(n)


print("=" * 72)
print("1. OVERHEAD AND THROUGHPUT (actual geometry)")
print("=" * 72)
for tag, c in CASES.items():
    Mn, N = c['n_act'], c['N']
    grid = Mn * N
    print(f"\n{tag}: DD/TF grid = {Mn}x{N} = {grid} RE")
    for name, nd, cp in (('OFDM', c['nd_o'], c['CP']),
                         ('PCP-guard', c['nd_g'], c['cp_g']),
                         ('PCP-orig', c['nd_p'], c['cp_o'])):
        o_grid = 1 - nd / grid
        o_time = cp / (Mn + cp) if name != 'OFDM' else c['CP'] / (c['M'] + c['CP'])
        tot = 1 - (1 - o_grid) * (1 - o_time)
        eta = nd / c['nd_o']
        print(f"  {name:10s} dataRE={nd:5d}  grid-ov={o_grid*100:5.2f}%  "
              f"time-ov={o_time*100:5.2f}%  total={tot*100:5.2f}%  eta={eta:.3f}")
    # structural formula check for PCP
    for name, L in (('PCP-guard', c['cp_g']), ('PCP-orig', c['cp_o'])):
        pred = (N - 3) * Mn + (Mn - 3 * L + 2)
        formula = (2 * Mn + 3 * L - 2) / (Mn * N)
        print(f"  [{name}] data-RE formula (N-3)M+(M-3L+2) = {pred}  |  "
              f"grid-ov (2M+3L-2)/(MN) = {formula*100:.2f}%  "
              f"| report's 3/N = {3/N*100:.2f}%")

print()
print("=" * 72)
print("2. RECEIVER CMAC AT THE ACTUAL GEOMETRY (per slot)")
print("=" * 72)
for tag, c in CASES.items():
    Mn, N, M = c['n_act'], c['N'], c['M']
    K = Mn // 2            # comb-2 DMRS pilots per DMRS symbol
    L_td = c['CP']         # delay window = CP length
    print(f"\n{tag}  (n_act={Mn}, N={N}, K={K}, L={L_td})")

    # --- OFDM classical baseline: FFT + LS + linear interp + MMSE ---
    o_fft = N * fft_cmac(M)
    o_ls = 2 * K
    o_interp = Mn * N * 0.5          # linear interp (freq) + time interp
    o_mmse = Mn * N                  # scalar MMSE per data RE
    o_base = o_fft + o_ls + o_interp + o_mmse
    print(f"  OFDM baseline : FFT {o_fft:9.0f} + est {o_ls+o_interp:8.0f} "
          f"+ MMSE {o_mmse:7.0f} = {o_base:9.0f}")

    # --- OFDM-TD (evaluated): delay-domain pinv + residual noise + gamma ---
    td_pinv = 2 * (L_td * K)          # F+ applied at each DMRS symbol
    td_recon = 2 * (Mn * L_td)        # H reconstruction on active SCs
    td_resid = 2 * (K * L_td)         # residual r = h - F g
    o_td = o_fft + td_pinv + td_recon + td_resid + o_interp + o_mmse
    print(f"  OFDM-TD (eval): FFT {o_fft:9.0f} + est {td_pinv+td_recon+td_resid+o_interp:8.0f} "
          f"+ MMSE {o_mmse:7.0f} = {o_td:9.0f}   ({o_td/o_base:.1f}x baseline)")

    # --- PCP-OTFS receiver ---
    for name, L in (('PCP-guard', c['cp_g']), ('PCP-orig', c['cp_o'])):
        p_fe = N * fft_cmac(M)                     # front-end FFT (fixed rate)
        p_body = N * fft_cmac(Mn)                  # Y -> freq  (native grid)
        p_s1 = N * (2 * fft_cmac(L) + 4 * L)       # ZC correlation
        p_s2 = L * (2 * (2 * 1 + 1) * N)           # GCE-BEM fit + rebuild, Q=1
        p_hf = N * fft_cmac(Mn)                    # h_smooth -> H_f
        p_fde = N * Mn * 3                         # pilot cancel + MMSE
        p_ifft = N * fft_cmac(Mn)                  # back to time
        p_zak = Mn * fft_cmac(N)                   # Doppler DFT
        p_noise = N * L
        p_tot_nat = p_body + p_s1 + p_s2 + p_hf + p_fde + p_ifft + p_zak + p_noise
        p_tot_fe = p_tot_nat + p_fe
        print(f"  {name:10s}: body {p_body:8.0f} S1 {p_s1:6.0f} S2 {p_s2:6.0f} "
              f"Hf {p_hf:8.0f} FDE {p_fde:7.0f} IFFT {p_ifft:8.0f} "
              f"Zak {p_zak:7.0f} = {p_tot_nat:9.0f}")
        print(f"              -> {p_tot_nat/o_base:.1f}x OFDM baseline, "
              f"{p_tot_nat/o_td:.2f}x OFDM-TD  "
              f"(+front-end FFT: {p_tot_fe/o_base:.1f}x / {p_tot_fe/o_td:.2f}x)")

        # critical path = stages spanning the Doppler axis
        crit = p_s2 + p_hf + p_fde + p_ifft + p_zak
        print(f"              critical path (BEM+Hf+FDE+IFFT+Zak) = {crit:.0f} CMAC "
              f"= {crit/1e10*1e6:.1f} us @10GCMAC/s")

print()
print("=" * 72)
print("3. SRAM (corner-turn frame buffer, 8 B/complex, double-buffered)")
print("=" * 72)
for tag, c in CASES.items():
    Mn, N, M = c['n_act'], c['N'], c['M']
    pcp = 2 * Mn * N * 8 / 1024
    ofdm_stream = M * 8 / 1024                       # streaming FFT working set
    ofdm_buf = 10 * Mn * 8 / 1024                    # DMRS-gap datapath buffer
    print(f"{tag}: PCP frame buffer {pcp:7.1f} kB | OFDM FFT {ofdm_stream:5.1f} kB "
          f"+ DMRS-gap buffer {ofdm_buf:5.1f} kB = {ofdm_stream+ofdm_buf:5.1f} kB "
          f"| ratio {pcp/(ofdm_stream+ofdm_buf):.1f}x")

print()
print("=" * 72)
print("4. eps^2 EFFECT (paired ablation: *_preeps vs v2) + uncoded ref")
print("=" * 72)


def knee(pts, tgt=0.10):
    """SNR at which the log-linear BLER interpolation crosses the target (NaN if never bracketed)."""
    pts = sorted(pts)
    s = np.array([p[0] for p in pts])
    b = np.clip([p[1]['bler'] for p in pts], 1e-6, 1)
    lb, lt = np.log10(b), np.log10(tgt)
    for i in range(len(s) - 1):
        if (lb[i] - lt) * (lb[i + 1] - lt) <= 0 and lb[i] != lb[i + 1]:
            return float(s[i] + (lt - lb[i]) / (lb[i + 1] - lb[i]) * (s[i + 1] - s[i]))
    return float('nan')


def load(tag):
    """Load data/<tag>.json into {(channel, fd, method): [(snr, rec), ...]}."""
    p = os.path.join(DATA, tag + '.json')
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    rec = {}
    for k, v in d.items():
        bw, cm, fd, snr, nm = k.split('|')
        rec.setdefault((cm, float(fd), nm), []).append((float(snr), v))
    return rec


for bw in ('NB', 'WB'):
    old, new = load(f'final_{bw}_preeps'), load(f'final_{bw}')
    if not old or not new:
        print(f"{bw}: missing pair"); continue
    print(f"\n{bw}: eps^2 sensitivity change (dB, negative = improvement)")
    rows = []
    for (cm, fd, nm) in sorted(new):
        if nm == 'OFDM' or cm == 'AWGN':
            continue
        a, b = knee(old.get((cm, fd, nm), [(0, {'bler': 1})])), knee(new[(cm, fd, nm)])
        if not (math.isnan(a) or math.isnan(b)):
            rows.append((b - a, cm, fd, nm, a, b))
    rows.sort()
    for d, cm, fd, nm, a, b in rows[:6]:
        print(f"   BEST  {cm} fD={fd:4.0f} {nm:10s} {a:6.2f} -> {b:6.2f}  ({d:+.2f})")
    for d, cm, fd, nm, a, b in rows[-3:]:
        print(f"   WORST {cm} fD={fd:4.0f} {nm:10s} {a:6.2f} -> {b:6.2f}  ({d:+.2f})")
    if rows:
        print(f"   median change {np.median([r[0] for r in rows]):+.2f} dB "
              f"over {len(rows)} cells")

u = os.path.join(DATA, 'uncoded_ref.json')
if os.path.exists(u):
    d = json.load(open(u))
    print(f"\nuncoded_ref.json: {len(d)} entries (v2). Sample high-Doppler:")
    for k in sorted(d)[:6]:
        print(f"   {k}: {d[k]}")
