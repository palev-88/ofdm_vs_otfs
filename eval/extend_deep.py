"""Deep-tail extension for the coded waterfalls.

The adaptive sweep concentrated frames at the 10%-BLER knee (+-1 dB), so the
plotted waterfalls stop near BLER ~ 1e-1 in easy cells.  This driver extends
the PLOTTED cells (AWGN + 4 TDL x fd in {0,600,1000}) with SNR points from the
knee down to knee+4 dB at 2000 frames each, so curves reach the 1e-3 axis
floor.  Same seeds as the fine stage (seed0=700000) -> same channel
realisations.  Results are merged into final_{bw}.json (original backed up to
final_{bw}.orig.json).
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import os, sys, json, shutil, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))

from coded_sweep import setup, measure_par, _pool_init, knee, BLER_TARGETS

BWS = {'NB': dict(tag='NB', M=256, n_act=156, SCS=30e3, N=14),
       'WB': dict(tag='WB', M=1024, n_act=624, SCS=60e3, N=14)}
CELLS = [('AWGN', 0.0)] + [(cm, fd) for cm in ('TDL-A', 'TDL-B', 'TDL-C', 'TDL-D')
                           for fd in (0.0, 600.0, 1000.0)]
FRAMES = 2000
BANK = 40
NW = 8


def main(bws):
    """Extend each stored knee with deeper-BLER SNR points so the 1%-BLER interpolation is supported; merges in place into data/final_<bw>.json."""
    for b in bws:
        path = os.path.join(DATA, f'final_{b}.json')
        orig = os.path.join(DATA, f'final_{b}.orig.json')
        if not os.path.exists(orig):
            shutil.copy(path, orig)
        data = json.load(open(path))
        rec = {}
        for k, v in data.items():
            _, cm, fd, snr, nm = k.split('|')
            rec.setdefault((cm, float(fd)), {}).setdefault(nm, []).append(
                (float(snr), v))
        ctx = setup(BWS[b], BANK)
        import multiprocessing as mp
        pool = mp.Pool(NW, initializer=_pool_init, initargs=(BWS[b], BANK))
        t0 = time.time()
        for cm, fd in CELLS:
            cell = rec.get((cm, fd), {})
            if not cell:
                continue
            have = {s for pts in cell.values() for s, _ in pts}
            knees = []
            for nm, pts in cell.items():
                pts.sort()
                k = knee([s for s, _ in pts],
                         [v['bler'] for _, v in pts], BLER_TARGETS[0]) \
                    if False else None
                # knee() in coded_sweep takes (snrs, bler, target)
                k = knee([s for s, _ in pts], [v['bler'] for _, v in pts],
                         BLER_TARGETS[0])
                if k is not None:
                    knees.append(k)
            if not knees:
                lo = min(have)
            else:
                lo = min(knees)
            hi = (max(knees) if knees else lo) + 4.0
            ext = set()
            s = np.floor(lo * 2) / 2
            while s <= hi + 1e-9:
                ext.add(round(float(s), 1))
                s += 1.0
            new = sorted(x for x in ext
                         if all(abs(x - h) > 0.26 for h in have) and -2 <= x <= 26)
            if not new:
                print(f"  {b} {cm} fD={fd:g}: nothing to add", flush=True)
                continue
            acc, used = measure_par(pool, NW, ctx, cm, fd, new, FRAMES,
                                    10 ** 9, FRAMES, 700_000)
            for (snr, nm), v in acc.items():
                data[f"{b}|{cm}|{fd:g}|{snr:g}|{nm}"] = dict(
                    ber=v[0] / max(v[1], 1), bler=v[2] / max(v[3], 1),
                    bits=v[1], blocks=v[3])
            print(f"  {b} {cm} fD={fd:g}: +{len(new)} pts x {used} frames "
                  f"({time.time()-t0:.0f}s)", flush=True)
        pool.close(); pool.join()
        json.dump(data, open(path, 'w'), indent=1)
        print(f"merged into final_{b}.json")


if __name__ == '__main__':
    main(sys.argv[1:] or ['NB', 'WB'])
