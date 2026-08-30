"""Mid-gap refinement for knees interpolated across the 8-12 dB anchor gap.

In a few TDL-B / TDL-C cells the adaptive fine grid (placed at the
coarse-stage knee) landed entirely above the 10%-BLER level, so the
published knee was interpolated between the 8 and 12 dB anchors, whose
deep side carries very few block errors.  This pass measures 9, 10 and
11 dB at the full 2000-frame budget in exactly those cells, so the knee
interpolation runs between adjacent, well-populated points.  Merges into
final_{bw}.json like extend_deep / fill_pass.
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-30"
import os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))

from coded_sweep import setup, measure_par, _pool_init

BWS = {'NB': dict(tag='NB', M=256, n_act=156, SCS=30e3, N=14),
       'WB': dict(tag='WB', M=1024, n_act=624, SCS=60e3, N=14)}

# (bandwidth, profile, Doppler) cells whose knee lay in the anchor gap
CELLS = {'NB': [('TDL-B', 800.0), ('TDL-C', 400.0)],
         'WB': [('TDL-B', 400.0), ('TDL-B', 800.0)]}
POINTS = [9.0, 10.0, 11.0]
FRAMES = 2000
BANK = 40
NW = 8


def main(bws):
    """Add 9/10/11 dB points (2000 frames each) in the anchor-gap cells; merges in place."""
    import multiprocessing as mp
    for b in bws:
        if b not in CELLS:
            continue
        path = os.path.join(DATA, f'final_{b}.json')
        data = json.load(open(path))
        have = {}
        for k in data:
            _, cm, fd, snr, nm = k.split('|')
            have.setdefault((cm, float(fd)), set()).add(float(snr))
        todo = []
        for cm, fd in CELLS[b]:
            pts = [s for s in POINTS if s not in have.get((cm, fd), set())]
            if pts:
                todo.append((cm, fd, pts))
        if not todo:
            print(f"{b}: nothing to refine")
            continue
        print(f"{b}: refining {todo}")
        ctx = setup(BWS[b], BANK, quiet=True)
        pool = mp.Pool(NW, initializer=_pool_init, initargs=(BWS[b], BANK))
        t0 = time.time()
        try:
            for cm, fd, pts in todo:
                acc, _ = measure_par(pool, NW, ctx, cm, fd, pts, FRAMES,
                                     10 ** 9, FRAMES, 700_000)
                for (snr, nm), v in acc.items():
                    data[f"{b}|{cm}|{fd:g}|{snr:g}|{nm}"] = dict(
                        ber=v[0] / max(v[1], 1), bler=v[2] / max(v[3], 1),
                        bits=v[1], blocks=v[3])
                print(f"    {b} {cm} fD={fd:g}: +{len(pts)} pts "
                      f"({time.time()-t0:.0f}s)", flush=True)
        finally:
            pool.close(); pool.join()
        json.dump(data, open(path, 'w'), indent=1)
        print(f"{b}: merged")


if __name__ == '__main__':
    main(sys.argv[1:] or ['NB', 'WB'])
