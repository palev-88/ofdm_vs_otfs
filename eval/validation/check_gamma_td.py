"""Spot-check of the delay-domain receiver's reliability calibration E[|z-d|^2 gamma]."""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="2"
import sys; import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'eval')); sys.path.insert(0,".")
from channel import TDLChannel, TDLChannelConfig
from ofdm import OFDMTransceiver, OFDMConfig
from qam import qam_modulate, qam_demodulate, generate_random_bits
from ofdm_td import OFDMTimeDomainRx
M,NA,N,SCS=256,156,14,30e3; FS=M*SCS; CP=max(round(144*M/2048),4)
trx=OFDMTransceiver(OFDMConfig(n_fft=M,n_active=NA,scs_hz=SCS,n_cp=CP,
    n_symbols_per_slot=N,dmrs_symbol_indices=[2,11],dmrs_comb_size=2))
td=OFDMTimeDomainRx(trx); ND=trx.count_data_res()
print(f"{'chan':7s}{'fD':>6}{'SNR':>5} | {'REPORT ratio':>13}{'TD ratio':>10} | {'BER rep':>10}{'BER TD':>10}")
for model,ds in [('TDL-A',30e-9),('TDL-C',300e-9)]:
    for fd in [0,500,1000,2000]:
        for snr in [10,20,30]:
            nr=nt=0.0; er=et=0; nb=0
            for t in range(6):
                ch=TDLChannel(TDLChannelConfig(model,ds,fd,FS,seed=8800+91*t+fd,use_fdf=True))
                rng=np.random.default_rng(500+t)
                bits=generate_random_bits(ND*2,np.random.default_rng(t+1))
                x=qam_modulate(bits,4)[:ND]
                tx,_=trx.tx(x); clean,_=ch.apply(tx,snr_dB=None)
                s2=np.mean(np.abs(clean)**2)/10**(snr/10)
                rx=clean+np.sqrt(s2/2)*(rng.standard_normal(len(clean))+1j*rng.standard_normal(len(clean)))
                # report rx
                dr,Hr,Yr=trx.rx(rx,'linear','mmse',s2,0.0)
                s2r=trx._estimate_noise(Yr,Hr,'dmrs_power_diff')
                Hs=np.where(np.abs(Hr)<1e-9,1e-9,Hr)
                zr=trx._extract_data(Yr/Hs); gr=trx._extract_data(np.abs(Hr)**2/s2r)
                dd=zr[:ND]-x; nr+=float(np.sum(np.abs(dd)**2*gr[:ND]))
                br=qam_demodulate(dr[:ND],4); er+=int(np.sum(bits[:len(br)]!=br)); nb+=len(br)
                # TD rx
                zt,gt=td.rx(rx); dd=zt[:ND]-x; nt+=float(np.sum(np.abs(dd)**2*gt[:ND]))
                bt=qam_demodulate(td.rx_hard(rx)[:ND],4); et+=int(np.sum(bits[:len(bt)]!=bt))
            n=6*ND
            print(f"{model:7s}{fd:6d}{snr:5d} | {nr/n:13.2f}{nt/n:10.2f} | {er/nb:10.2e}{et/nb:10.2e}")
