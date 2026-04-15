"""Fast evaluation: OFDM vs PCP-OTFS only (skip ZP-OTFS).
Usage: python3 run_fast.py <M> <SCS_kHz> <NMC> <channel> <fD_start> <fD_end>
"""
import numpy as np, sys, time, json, os
from channel import TDLChannel, TDLChannelConfig
from ofdm import OFDMTransceiver, OFDMConfig
from otfs_pcp import PCPOTFSTransceiver, PCPOTFSConfig
from qam import qam_modulate, qam_demodulate, generate_random_bits

M=int(sys.argv[1]); SCS=float(sys.argv[2])*1e3; NMC=int(sys.argv[3])
CH=sys.argv[4]; FD_S=int(sys.argv[5]); FD_E=int(sys.argv[6])

FS=M*SCS; CP=max(round(144*M/2048),4); N=14
ch_c=TDLChannel(TDLChannelConfig('TDL-C',100e-9,0,FS,seed=42))
ZP=max(8,ch_c.max_delay_samples+2); n_act=min(M-2,156)

ofdm=OFDMTransceiver(OFDMConfig(n_fft=M,n_active=n_act,scs_hz=SCS,n_cp=CP,
    n_symbols_per_slot=N,dmrs_symbol_indices=[2,11],dmrs_comb_size=2))
pcp=PCPOTFSTransceiver(PCPOTFSConfig(M=M,N=N,Mcp=CP,scs_hz=SCS,
    pilot_doppler=N//2,doppler_guard=1,pilot_power_dB=25.0,zc_root=1,bem_Q=1))

nd_o=ofdm.count_data_res(); nd_p=pcp.count_data_res()
DS={'TDL-A':30e-9,'TDL-B':100e-9,'TDL-C':100e-9,'TDL-D':30e-9,'TDL-E':30e-9}
SNR_LIST=[0,5,10,15,20,25,30]

fname=f"results/ber_M{M}_SCS{int(SCS/1e3)}k_{CH}.json"
if os.path.exists(fname):
    with open(fname) as f: out=json.load(f)
else:
    out={'M':M,'SCS':SCS/1e3,'CH':CH,'NMC':NMC,'CP':CP,
         'nd_o':nd_o,'nd_p':nd_p,'ber':{}}

ds=DS.get(CH,30e-9); t0=time.time()
fd_list=list(range(FD_S,FD_E+1,100))

for fd in fd_list:
    for snr in SNR_LIST:
        nv=10**(-snr/10)
        for nm,mt,obj,nd in [('ofdm','o',ofdm,nd_o),('pcp','p',pcp,nd_p)]:
            errs,nb=0,0
            for t in range(NMC):
                s=t*1000+42
                bits=generate_random_bits(nd*2,np.random.default_rng(s))
                tx,_=obj.tx(qam_modulate(bits,4)[:nd])
                ch=TDLChannel(TDLChannelConfig(CH,ds,fd,FS,seed=s))
                rx,_=ch.apply(tx,snr_dB=snr)
                if mt=='o':
                    dr=obj.rx(rx,'linear','mmse',nv,0,noise_est_method='dmrs_power_diff')[0]
                else:
                    dr=obj.rx(rx,nv)[0]
                br=qam_demodulate(dr[:nd],4)
                n=min(len(bits),len(br))
                errs+=int(np.sum(bits[:n]!=br[:n]));nb+=n
            out['ber'][f'{CH}_{fd}_{snr}_{nm}']=errs/nb if nb>0 else 0
        print(f"  {CH} fD={fd} SNR={snr}: O={out['ber'][f'{CH}_{fd}_{snr}_ofdm']:.2e} P={out['ber'][f'{CH}_{fd}_{snr}_pcp']:.2e} ({time.time()-t0:.0f}s)")

os.makedirs('results',exist_ok=True)
with open(fname,'w') as f: json.dump(out,f)
print(f"Saved {fname}")
