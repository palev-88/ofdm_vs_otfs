"""Dense AWGN anchor sweep.

The adaptive sweep places SNR points around a knee located on a coarse
(3 dB) grid.  That is adequate for fading cells, whose waterfalls span several
dB, but not for AWGN: with an NR-LDPC code the transition from BLER 1 to
below 1e-3 occupies roughly one decibel, so 0.5-1 dB spacing renders it as two
or three points and the characteristic steep knee is invisible.

This sweep uses a fixed 0.2 dB grid over the region containing both bandwidth
cases' knees, at a frame count deep enough to resolve 1e-3.  AWGN costs little:
with no fading the clean signal is generated once per frame and reused across
every SNR point.
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

import sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))

from coded_sweep import setup, measure_par, _pool_init

BWS = {'NB': dict(tag='NB', M=256, n_act=156, SCS=30e3, N=14),
       'WB': dict(tag='WB', M=1024, n_act=624, SCS=60e3, N=14)}
SNRS = [round(x, 2) for x in np.arange(0.0, 3.81, 0.2)]
FRAMES = 4000
BANK = 40
NW = 8


def main():
    """Dense 0.2 dB AWGN anchor (4000 transport blocks per point) for both bandwidth cases; writes data/awgn_fine.json."""
    out = {}
    t0 = time.time()
    for b in ('NB', 'WB'):
        ctx = setup(BWS[b], BANK, quiet=True)
        import multiprocessing as mp
        pool = mp.Pool(NW, initializer=_pool_init, initargs=(BWS[b], BANK))
        acc, used = measure_par(pool, NW, ctx, 'AWGN', 0.0, SNRS,
                                FRAMES, 10 ** 9, FRAMES, 950_000)
        pool.close(); pool.join()
        for (snr, nm), v in acc.items():
            out[f"{b}|AWGN|0|{snr:g}|{nm}"] = dict(
                ber=v[0] / max(v[1], 1), bler=v[2] / max(v[3], 1),
                bits=v[1], blocks=v[3])
        print(f"  {b}: {len(SNRS)} points x {used} frames "
              f"({time.time()-t0:.0f}s)", flush=True)
        names = ctx['names']
        print(f"    SNR " + "".join(f"{n:>12}" for n in names))
        for snr in SNRS:
            row = "".join(
                f"{acc[(snr, n)][2]/max(acc[(snr, n)][3],1):12.2e}"
                for n in names)
            print(f"    {snr:4.1f} " + row)
    with open(os.path.join(DATA, 'awgn_fine.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote awgn_fine.json  total {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
