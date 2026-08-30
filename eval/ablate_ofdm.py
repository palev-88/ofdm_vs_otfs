"""Which of the three OFDM estimator changes carries the gain?

Nested paired ablation, identical channels / codewords / seeds throughout:

  A  baseline      linear freq+time interpolation, DMRS power-difference
                   noise, gamma = |H|^2 / sigma^2
  B  +CIR fit      delay-domain least-squares channel estimate; noise and
                   gamma still as in A
  C  +residual     noise from the single-symbol fit residual; gamma still
                   noise-only
  D  +eps^2        gamma = |H|^2 / (sigma^2 + eps^2[l])   <- evaluated receiver

Each step adds exactly one element, so D-A decomposes into the three
contributions.  Usage: python ablate_ofdm.py [NB|WB ...]
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import sys, json, time
import numpy as np
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))

from coded_eval import WOFDM, strided, calibrate, QM
from coded_sweep import setup, measure, knee
import nr_ldpc

BWS = {'NB': dict(tag='NB', M=256, n_act=156, SCS=30e3, N=14),
       'WB': dict(tag='WB', M=1024, n_act=624, SCS=60e3, N=14)}

# cells chosen to span the space: static, mid and high Doppler, on the
# short-delay Rayleigh profile and the long-delay one
CELLS = [('TDL-A', 0.0), ('TDL-A', 1000.0),
         ('TDL-C', 0.0), ('TDL-C', 1000.0),
         ('TDL-D', 1000.0)]
SNRS = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
NFRAMES = 260


class Abl(WOFDM):
    """One receiver, three switches."""
    def __init__(self, M, n_act, SCS, N, CP, label, cir, resid, eps):
        super().__init__(M, n_act, SCS, N, CP)
        self.name = label
        self.cir, self.resid, self.eps = cir, resid, eps
        from ofdm_td import OFDMTimeDomainRx
        self.td = OFDMTimeDomainRx(self.t)

    def rx(self, sig):
        """Receive with the configured subset of estimator elements (cir/resid/eps switches)."""
        if not self.cir:                      # variant A: stock receiver
            return WOFDM.rx(self, sig)

        Y = self.td.grid(sig)
        H, s2_res, eps2 = self.td.estimate(Y)     # delay-domain CIR fit

        if self.resid:
            s2 = s2_res
        else:                                  # DMRS power-difference instead
            s2 = self.t._estimate_noise(Y, H, 'dmrs_power_diff')

        Hs = np.where(np.abs(H) < 1e-9, 1e-9, H)
        z = self.t._extract_data(Y / Hs)
        den = (s2 + eps2[:, None]) if self.eps else np.full_like(eps2, s2)[:, None]
        g = self.t._extract_data(np.abs(H) ** 2 / den)
        return z, g


VARIANTS = [('A base',    False, False, False),
            ('B +CIR',    True,  False, False),
            ('C +resid',  True,  True,  False),
            ('D +eps2',   True,  True,  True)]


def build(bw):
    """setup() for the iso-block config, then swap in the four variants."""
    ctx = setup(BWS[bw], 24, quiet=True)
    b, CP = BWS[bw], max(round(144 * BWS[bw]['M'] / 2048), 4)
    waves = [Abl(b['M'], b['n_act'], b['SCS'], b['N'], CP, nm, c, r, e)
             for nm, c, r, e in VARIANTS]
    n_common = ctx['n_common']                 # keep the report's codeword
    ctx['waves'] = waves
    ctx['names'] = [w.name for w in waves]
    ctx['sel'] = {w.name: strided(w.nd, n_common) for w in waves}
    rng = np.random.default_rng(11)
    ctx['cal'] = {w.name: calibrate(w, rng) for w in waves}
    return ctx


_W = {}


def _init(bw):
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = "1"
    _W['ctx'] = build(bw)


def _work(job):
    cm, fd, ids = job
    acc, _ = measure(_W['ctx'], cm, fd, SNRS, 0, seed0=4242, frame_ids=ids)
    return acc


def main(bws):
    """Nested ablation A->B->C->D of the OFDM estimator elements at identical seeds; prints per-cell knees and writes data/ablate_ofdm.json."""
    # merge into the existing JSON so a single-bandwidth run cannot
    # silently drop the other bandwidth's cells
    path = os.path.join(DATA, 'ablate_ofdm.json')
    out = json.load(open(path)) if os.path.exists(path) else {}
    for bw in bws:
        ctx = build(bw)
        print(f"\n{'='*74}\n{bw}: OFDM estimator ablation "
              f"(n_common={ctx['n_common']}, {NFRAMES} frames/point)\n{'='*74}")
        nw = 8
        pool = mp.Pool(nw, initializer=_init, initargs=(bw,))
        try:
            for cm, fd in CELLS:
                t0 = time.time()
                ids = list(range(NFRAMES))
                jobs = [(cm, fd, ids[i::nw]) for i in range(nw)]
                acc = {}
                for part in pool.map(_work, jobs):
                    for k, v in part.items():
                        d = acc.setdefault(k, [0, 0, 0, 0])
                        for q in range(4):
                            d[q] += v[q]
                ks = {}
                for nm in ctx['names']:
                    bler = [acc[(s, nm)][2] / max(acc[(s, nm)][3], 1) for s in SNRS]
                    ks[nm] = knee(SNRS, bler, 0.10)
                    out[f'{bw}|{cm}|{fd:g}|{nm}'] = ks[nm]
                base = ks['A base']
                line = f"  {cm:6s} fD={fd:5.0f} | "
                for nm in ctx['names']:
                    v = ks[nm]
                    line += f"{nm}={'--' if v is None else f'{v:5.2f}'} "
                if base is not None:
                    line += "| gains: "
                    prev = base
                    for nm in ctx['names'][1:]:
                        v = ks[nm]
                        line += (f"{nm.split()[1]}:{'--':>6} " if v is None
                                 else f"{nm.split()[1]}:{prev - v:+5.2f} ")
                        if v is not None:
                            prev = v
                    line += f"| total:{base - prev:+5.2f}"
                print(line + f"  ({time.time()-t0:.0f}s)", flush=True)
        finally:
            pool.close(); pool.join()
    json.dump(out, open(path, 'w'), indent=1)
    print("\nwrote ablate_ofdm.json")


if __name__ == '__main__':
    mp.freeze_support()
    main(sys.argv[1:] or ['NB', 'WB'])
