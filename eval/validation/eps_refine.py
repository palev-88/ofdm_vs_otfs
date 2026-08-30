"""Select the eps^2 estimator by calibration, not by taste.

A: mean residual, in-basis fraction (2Q+1)/N          [current: overcharges ~2x
   because the q=+-2 data-leakage spikes are OUT of basis yet counted]
B: per-tap MEDIAN of out-of-basis BEM spectrum -> white floor only; in-basis
   error per bin = sum_l (2Q+1) * floor_l / N^2  (LS coefficient variance)
Metric: E[|z-x|^2 gamma] target 1.0, six cells, PCP-guard, 100 frames.
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): os.environ[v]="2"
import sys, time
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'eval'))
from channel import TDLChannel, TDLChannelConfig
from coded_sweep import setup, DS_MAP
from coded_eval import WPCP, qpsk_mod, strided

def rx_gamma(w, sig, mode):
    """Receive with eps^2 formula A (mean residual) or B (median out-of-basis); returns (z, gamma, eps2/sigma2)."""
    cfg=w.p.cfg; Mp,Np,Q=cfg.M,cfg.N,cfg.bem_Q
    Y,s2=w._bodies(sig)
    h_hat=w.p._estimate_stage1(Y,s2); h_sm=w.p._estimate_stage2(h_hat,s2)
    L=h_hat.shape[0]
    if mode=='A':
        r=h_hat-h_sm[:L,:]
        eps2=float((2*Q+1)/(Np*(Np-2*Q-1))*np.sum(np.abs(r)**2))
    else:
        F=np.fft.fft(h_hat,axis=1)                    # (L,N) BEM spectrum
        qs=np.arange(Np); dist=np.minimum(qs,Np-qs)
        out=dist>Q                                    # out-of-basis bins
        floor=np.median(np.abs(F[:,out])**2,axis=1)   # per-tap white floor
        eps2=float((2*Q+1)*np.sum(floor)/Np**2)
    s2t=s2+eps2
    H_f=np.fft.fft(h_sm,Mp,axis=0) if h_sm.shape[0]==Mp else None
    H_f=np.zeros((Mp,Np),dtype=complex); Y_f=np.zeros((Mp,Np),dtype=complex)
    for n in range(Np):
        H_f[:,n]=np.fft.fft(h_sm[:,n],Mp); Y_f[:,n]=np.fft.fft(Y[:,n])
    pf=np.fft.fft(w.p._pilot_row)
    for n in range(Np):
        ph=np.exp(1j*2*np.pi*cfg.pilot_doppler*n/Np)/np.sqrt(Np)
        Y_f[:,n]-=H_f[:,n]*pf*ph
    pw=np.abs(H_f)**2; mu=pw/(pw+s2t)
    X_f=np.conj(H_f)/(pw+s2t)*Y_f
    D=np.fft.fft(np.fft.ifft(X_f,axis=0),axis=1)/np.sqrt(Np)
    mb=float(mu.mean())
    g=mb**2/max(float((mu**2).mean())-mb**2+float((mu*(1-mu)).mean()),1e-15)
    z=D[w.p._data_pos[:,0],w.p._data_pos[:,1]]/max(mb,1e-12)
    return z,g,eps2/s2

CELLS=[('NB','TDL-C',1000.0,14.0),('NB','TDL-A',1000.0,14.0),
       ('NB','TDL-D',1000.0,8.0),('WB','TDL-C',1000.0,14.0),
       ('WB','TDL-A',1000.0,13.0),('WB','TDL-C',600.0,12.0),
       ('NB','TDL-C',0.0,10.0),('WB','TDL-B',0.0,9.0)]
BWS={'NB':dict(tag='NB',M=256,n_act=156,SCS=30e3,N=14),
     'WB':dict(tag='WB',M=1024,n_act=624,SCS=60e3,N=14)}
t0=time.time()
print(f"{'cell':22s} | {'calib A':>8}{'calib B':>8} | {'e/s A':>7}{'e/s B':>7}")
for b in ('NB','WB'):
    ctx=setup(BWS[b],1,quiet=True)
    w=[x for x in ctx['waves'] if x.name=='PCP-guard'][0]
    gs,gn=ctx['cal']['PCP-guard']
    for (bw,cm,fd,snr) in CELLS:
        if bw!=b: continue
        ds=DS_MAP[cm]; accA=accB=0.0; ns=0; rA=rB=0.0
        for fr in range(100):
            ch=TDLChannel(TDLChannelConfig(cm,ds,fd,ctx['FS'],
                seed=700_000+977*fr+int(fd),use_fdf=True))
            rng=np.random.default_rng(5_000+13*fr+int(fd))
            syms=qpsk_mod(rng.integers(0,2,2*w.nd))
            sig=w.tx(syms)
            if cm=='AWGN': c=sig
            else: c,_=ch.apply(sig,snr_dB=None)
            s2=gs/(10.0**(snr/10.0)*gn)
            r=c+np.sqrt(s2/2)*(rng.standard_normal(len(c))+1j*rng.standard_normal(len(c)))
            for mode in 'AB':
                z,g,es=rx_gamma(w,r,mode)
                d=z[:w.nd]-syms[:w.nd]
                if mode=='A': accA+=float(np.sum(np.abs(d)**2)*g); rA+=es
                else: accB+=float(np.sum(np.abs(d)**2)*g); rB+=es
            ns+=w.nd
        print(f"{b+' '+cm+' f'+str(int(fd))+' s'+str(int(snr)):22s} | "
              f"{accA/ns:8.2f}{accB/ns:8.2f} | {rA/100:7.2f}{rB/100:7.2f}",flush=True)
print(f"total {time.time()-t0:.0f}s")
