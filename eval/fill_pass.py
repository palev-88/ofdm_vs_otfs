"""Targeted deep-tail fill for the coded waterfalls.

After the knee->+4dB extension, some shallow-sloped cells (typically PCP at
WB high Doppler) still end above BLER 5e-3.  For every PLOTTED cell whose
deepest measured point of ANY method is above that threshold, add up to two
more +1 dB points at 2000 frames, so every curve reaches the 1e-3 axis or its
Monte-Carlo floor.  Merges into final_{bw}.json like extend_deep.
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import os, sys, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))

from coded_sweep import setup, measure_par, _pool_init

BWS = {'NB': dict(tag='NB', M=256, n_act=156, SCS=30e3, N=14),
       'WB': dict(tag='WB', M=1024, n_act=624, SCS=60e3, N=14)}
CELLS = [('AWGN', 0.0)] + [(cm, fd) for cm in ('TDL-A', 'TDL-B', 'TDL-C', 'TDL-D')
                           for fd in (0.0, 600.0, 1000.0)]
THRESH = 5e-3
FRAMES = 2000
BANK = 40
NW = 8


def main(bws):
    """Fill any (cell, anchor-SNR) combinations missing from the adaptive sweep so constant-SNR views have a complete grid; merges in place."""
    for b in bws:
        path = os.path.join(DATA, f'final_{b}.json')
        data = json.load(open(path))
        rec = {}
        for k, v in data.items():
            _, cm, fd, snr, nm = k.split('|')
            rec.setdefault((cm, float(fd)), {}).setdefault(nm, []).append(
                (float(snr), v))
        todo = []
        for cm, fd in CELLS:
            cell = rec.get((cm, fd), {})
            if not cell:
                continue
            smax = max(s for pts in cell.values() for s, _ in pts)
            worst = max(min(v['bler'] for s, v in pts if abs(s - smax) < 0.26)
                        if any(abs(s - smax) < 0.26 for s, _ in pts) else 1.0
                        for pts in cell.values())
            if worst > THRESH and smax < 25.5:
                pts_new = [round(smax + 1, 1)]
                if worst > 3 * THRESH and smax + 2 <= 26:
                    pts_new.append(round(smax + 2, 1))
                todo.append((cm, fd, pts_new, worst))
        if not todo:
            print(f"{b}: nothing to fill")
            continue
        print(f"{b}: {len(todo)} cells to fill:")
        for cm, fd, pts, worst in todo:
            print(f"    {cm} fD={fd:g}: worst deep BLER {worst:.3f} -> add {pts}")
        ctx = setup(BWS[b], BANK, quiet=True)
        import multiprocessing as mp
        pool = mp.Pool(NW, initializer=_pool_init, initargs=(BWS[b], BANK))
        t0 = time.time()
        for cm, fd, pts, _ in todo:
            acc, _ = measure_par(pool, NW, ctx, cm, fd, pts, FRAMES,
                                 10 ** 9, FRAMES, 700_000)
            for (snr, nm), v in acc.items():
                data[f"{b}|{cm}|{fd:g}|{snr:g}|{nm}"] = dict(
                    ber=v[0] / max(v[1], 1), bler=v[2] / max(v[3], 1),
                    bits=v[1], blocks=v[3])
            print(f"    {b} {cm} fD={fd:g}: +{len(pts)} pts ({time.time()-t0:.0f}s)",
                  flush=True)
        pool.close(); pool.join()
        json.dump(data, open(path, 'w'), indent=1)
        print(f"{b}: merged")


if __name__ == '__main__':
    main(sys.argv[1:] or ['NB', 'WB'])
