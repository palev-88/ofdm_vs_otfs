"""Ablation: how much does the eps^2 (interpolation-error) term matter?

Compares three reliability rules on the SAME receiver output:
  (a) BASELINE   gamma = |H|^2 / sigma^2_dmrs_power_diff   (report v3.1)
  (b) TD, no eps gamma = |H|^2 / sigma^2_TD                (fixed noise only)
  (c) TD + eps   gamma = |H|^2 / (sigma^2_TD + eps^2)      (deployed)
Metric: E[|z-x|^2 * gamma], which must equal 1 for calibrated LLRs.
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="2"
import sys, numpy as np
sys.path.insert(0,'.'); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'eval'))
from channel import TDLChannel, TDLChannelConfig
from ofdm import OFDMTransceiver, OFDMConfig
from qam import qam_modulate, generate_random_bits
from ofdm_td import OFDMTimeDomainRx

M,NA,N,SCS = 256,156,14,30e3; FS=M*SCS; CP=max(round(144*M/2048),4)
trx=OFDMTransceiver(OFDMConfig(n_fft=M,n_active=NA,scs_hz=SCS,n_cp=CP,
    n_symbols_per_slot=N,dmrs_symbol_indices=[2,11],dmrs_comb_size=2))
td=OFDMTimeDomainRx(trx); ND=trx.count_data_res()
print(f"{'chan':7s}{'fD':>6}{'SNR':>5} | {'(a) baseline':>13}{'(b) TD no eps':>15}{'(c) TD + eps':>14}")
for model,ds in [('TDL-C',300e-9),('TDL-D',30e-9)]:
    for fd in [0,500,1000]:
        for snr in [20]:
            acc={k:0.0 for k in 'abc'}; nsym=0
            for t in range(8):
                ch=TDLChannel(TDLChannelConfig(model,ds,fd,FS,seed=9100+91*t+fd,use_fdf=True))
                rng=np.random.default_rng(700+t)
                bits=generate_random_bits(ND*2,np.random.default_rng(t+1))
                x=qam_modulate(bits,4)[:ND]
                tx,_=trx.tx(x); clean,_=ch.apply(tx,snr_dB=None)
                s2t=np.mean(np.abs(clean)**2)/10**(snr/10)
                rx=clean+np.sqrt(s2t/2)*(rng.standard_normal(len(clean))+1j*rng.standard_normal(len(clean)))
                # (a) baseline
                _,Hr,Yr=trx.rx(rx,'linear','mmse',s2t,0.0)
                s2r=trx._estimate_noise(Yr,Hr,'dmrs_power_diff')
                Hs=np.where(np.abs(Hr)<1e-9,1e-9,Hr)
                za=trx._extract_data(Yr/Hs); ga=trx._extract_data(np.abs(Hr)**2/s2r)
                acc['a']+=float(np.sum(np.abs(za[:ND]-x)**2*ga[:ND]))
                # (b),(c) share the TD estimate
                Y=td.grid(rx); H,s2,eps2=td.estimate(Y)
                Hs2=np.where(np.abs(H)<1e-9,1e-9,H)
                z=trx._extract_data(Y/Hs2)
                gb=trx._extract_data(np.abs(H)**2/s2)
                gc=trx._extract_data(np.abs(H)**2/(s2+eps2[:,None]))
                acc['b']+=float(np.sum(np.abs(z[:ND]-x)**2*gb[:ND]))
                acc['c']+=float(np.sum(np.abs(z[:ND]-x)**2*gc[:ND]))
                nsym+=ND
            print(f"{model:7s}{fd:6d}{snr:5d} | {acc['a']/nsym:13.2f}{acc['b']/nsym:15.2f}{acc['c']/nsym:14.2f}")
