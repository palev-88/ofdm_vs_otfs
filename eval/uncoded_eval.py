"""Uncoded BER reference with the FINAL receiver chain.

Same waveforms, calibration, channel conventions and receivers as the coded
sweep (OFDM = delay-domain estimator; PCP-guard/orig), but no code: random
QPSK on ALL data REs, hard decisions on the receivers' unbiased outputs z.
This is the Sec. eval_uncoded subset for the report: the only difference from
the coded evaluation is the absence of the LDPC code.
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
    os.environ[_v] = "1"

import sys, json, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))

from channel import TDLChannel, TDLChannelConfig
from coded_sweep import setup, DS_MAP
from coded_eval import qpsk_mod

BWS = {'NB': dict(tag='NB', M=256, n_act=156, SCS=30e3, N=14),
       'WB': dict(tag='WB', M=1024, n_act=624, SCS=60e3, N=14)}

_W = {}


def _init(bw):
    _W['ctx'] = setup(bw, n_bank=1, quiet=True)   # bank unused here


def _work(job):
    cm, fd, snrs, ids = job
    ctx = _W['ctx']
    waves, cal = ctx['waves'], ctx['cal']
    ds = DS_MAP.get(cm, 0.0)
    acc = {(s, w.name): [0, 0] for s in snrs for w in waves}
    Lmax = max(w.frame_len for w in waves)
    for fr in ids:
        if cm == 'AWGN':
            fading, ch = None, None
        else:
            ch = TDLChannel(TDLChannelConfig(cm, ds, fd, ctx['FS'],
                                             seed=810_000 + 977 * fr + int(fd),
                                             use_fdf=True))
            fading = ch._generate_jakes_fading(Lmax)
        frng = np.random.default_rng(9_000 + 13 * fr + int(fd))
        payload, clean = {}, {}
        for w in waves:
            bits = frng.integers(0, 2, 2 * w.nd)
            payload[w.name] = bits
            sig = w.tx(qpsk_mod(bits))
            L = len(sig)
            if fading is None:
                clean[w.name] = sig
                continue
            c = np.zeros(L, dtype=complex)
            dl = ch._fdf.apply(sig) if ch._fdf is not None else None
            for i in range(ch.n_taps):
                if dl is not None:
                    col = dl[:L, i] if dl.shape[0] >= L else np.pad(
                        dl[:, i], (0, L - dl.shape[0]))
                    c += fading[i, :L] * col
                else:
                    d = int(ch.delays_samples[i])
                    if d == 0:
                        c += fading[i, :L] * sig
                    elif d < L:
                        c[d:] += fading[i, d:L] * sig[:L - d]
            clean[w.name] = c
        for snr in snrs:
            for w in waves:
                gs, gn = cal[w.name]
                s2 = gs / (10.0 ** (snr / 10.0) * gn)
                c = clean[w.name]
                r = c + np.sqrt(s2 / 2) * (frng.standard_normal(len(c))
                                           + 1j * frng.standard_normal(len(c)))
                z, _ = w.rx(r)
                n = min(len(z), w.nd)
                hb = np.empty(2 * n, dtype=int)
                hb[0::2] = (np.real(z[:n]) < 0).astype(int)
                hb[1::2] = (np.imag(z[:n]) < 0).astype(int)
                v = acc[(snr, w.name)]
                v[0] += int(np.sum(hb != payload[w.name][:2 * n]))
                v[1] += 2 * n
    return acc


def main():
    """Hard-decision QPSK BER with the identical chain and calibration as the coded evaluation, code removed; writes data/<tag>.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--bw', nargs='+', default=['NB', 'WB'])
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--snrs', type=float, nargs='+',
                    default=[0, 5, 10, 15, 20, 25, 30])
    ap.add_argument('--fds', type=float, nargs='+', default=[0, 600, 1000])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--tag', default='uncoded_ref')
    args = ap.parse_args()

    out = {}
    t0 = time.time()
    for b in args.bw:
        import multiprocessing as mp
        pool = mp.Pool(args.workers, initializer=_init, initargs=(BWS[b],))
        names = None
        for cm in ['AWGN', 'TDL-A', 'TDL-B', 'TDL-C', 'TDL-D']:
            fds = [0.0] if cm == 'AWGN' else [float(f) for f in args.fds]
            for fd in fds:
                ids = list(range(args.frames))
                chunks = [ids[i::args.workers] for i in range(args.workers)]
                jobs = [(cm, fd, [float(s) for s in args.snrs], c)
                        for c in chunks if c]
                tot = {}
                for part in pool.map(_work, jobs):
                    for k, v in part.items():
                        d = tot.setdefault(k, [0, 0])
                        d[0] += v[0]; d[1] += v[1]
                names = sorted({k[1] for k in tot})
                for (snr, nm), v in tot.items():
                    out[f"{b}|{cm}|{fd:g}|{snr:g}|{nm}"] = dict(
                        ber=v[0] / max(v[1], 1), bits=v[1])
                print(f"  {b} {cm} fD={fd:6.0f}  ({time.time()-t0:.0f}s)",
                      flush=True)
        pool.close(); pool.join()
        print(f"\n--- {b}: uncoded BER ---")
        for cm in ['AWGN', 'TDL-A', 'TDL-B', 'TDL-C', 'TDL-D']:
            fds = [0.0] if cm == 'AWGN' else [float(f) for f in args.fds]
            for fd in fds:
                print(f"  {cm} fD={fd:g}")
                print("    SNR " + "".join(f"{n:>12}" for n in names))
                for snr in args.snrs:
                    row = "".join(
                        f"{out[f'{b}|{cm}|{fd:g}|{snr:g}|{nm}']['ber']:>12.2e}"
                        for nm in names)
                    print(f"    {snr:3.0f} " + row)
    with open(os.path.join(DATA, args.tag + '.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.tag}.json  total {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
