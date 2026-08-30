"""Why does PCP-OTFS floor at high Doppler?  Measure, don't guess.

Hypothesis: Stage-1 ZC correlation reads the first L delay samples of each
subsymbol.  In the DD grid, data occupies ALL delays in every non-guard
Doppler column, and the inverse Zak (an IDFT along Doppler only) preserves the
delay index -- so those data cells land on exactly the delays the pilot
occupies.  The Doppler guard protects the pilot in the DD domain but not in
the per-subsymbol time domain where Stage 1 operates.
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import os, sys
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="2"
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'eval'))
from otfs_pcp import PCPOTFSTransceiver, PCPOTFSConfig
from qam import qam_modulate, generate_random_bits

for tag, M, SCS in [('NB', 156, 30e3), ('WB', 624, 60e3)]:
    FS = M*SCS
    Mcp = max(4, int(round(1e-6*FS))+3)
    p = PCPOTFSTransceiver(PCPOTFSConfig(M=M, N=14, Mcp=Mcp, scs_hz=SCS,
        pilot_doppler=7, doppler_guard=1, pilot_power_dB=25.0, zc_root=1, bem_Q=1))
    nd, L, N = p.count_data_res(), p.cfg.L, 14
    rng = np.random.default_rng(3)
    # pilot-only frame (zero data) and data-only frame (zero pilot)
    zero = np.zeros(nd, dtype=complex)
    tx_p, D_p = p.tx(zero)                      # pilot alone
    syms = qam_modulate(generate_random_bits(nd*2, rng), 4)[:nd]
    tx_b, _ = p.tx(syms)                        # pilot + data
    MT, Mcp_ = p.cfg.MT, p.cfg.Mcp
    Pp = Pd = 0.0
    for n in range(N):
        body_p = tx_p[n*MT+Mcp_ : n*MT+Mcp_+M][:L]     # pilot-only, pilot delays
        body_b = tx_b[n*MT+Mcp_ : n*MT+Mcp_+M][:L]     # both
        Pp += np.sum(np.abs(body_p)**2)
        Pd += np.sum(np.abs(body_b - body_p)**2)       # data leakage there
    sir = 10*np.log10(Pp/Pd)
    kappa = lambda fd: fd * N * (M/FS)
    print(f"{tag}: M={M} L={Mcp} Fs={FS/1e6:.2f}MHz  data cells={nd}")
    print(f"   pilot power at pilot delays : {Pp/(N*L):.3f} per sample")
    print(f"   data leakage on same delays : {Pd/(N*L):.3f} per sample")
    print(f"   --> Stage-1 SIR = {sir:5.2f} dB  (SNR-independent floor)")
    print(f"   BEM: Q=1 -> 3 coeffs from N={N} samples; averaging {10*np.log10(N/3):.1f} dB")
    for fd in (0, 500, 1000):
        print(f"   fD={fd:5d}Hz: Doppler spread = {kappa(fd):.3f} bins "
              f"(BEM Q=1 spans +-1.0)")
