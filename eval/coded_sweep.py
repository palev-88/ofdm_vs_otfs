"""Adaptive coded sweep: coded BER + BLER/PER, with SNR points placed at the knee.

Two stages per (bandwidth, channel, Doppler) cell:

  STAGE 1 - COARSE.  All methods at a wide, sparse SNR grid with few frames.
    Locate each method's knee: the SNR where BLER crosses the target (10%).

  STAGE 2 - FINE.  Simulate the UNION over methods of {knee-2 .. knee+2} dB,
    plus a few fixed ANCHOR SNRs that are present in every cell.  The anchors
    are what make BLER-vs-Doppler and BER-vs-Doppler plots possible: those need
    the same SNR column in every cell, which a purely adaptive grid cannot give.

Frame count is error-driven, not fixed: keep drawing frames until every method
has at least ERR_TARGET block errors, or N_MAX frames are used.  At BLER = 0.5
that stops in ~200 frames; near BLER = 0.01 it runs to the cap.  This is where
most of the saving comes from -- a fixed 200-frame grid over-simulates the
saturated points and under-simulates the interesting ones.

Outputs per (bw, channel, fd, method):
  coded BER, BLER/PER, and the interpolated SNR needed for BLER = 10% and 1%.
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
    os.environ[_v] = "2"

import sys, json, time, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'src'))

from channel import TDLChannel, TDLChannelConfig
import nr_ldpc
from coded_eval import (WOFDM, WOFDM_TD, WZP, WPCP, calibrate, strided,
                        qpsk_mod, qpsk_llr, QM, DS_MAP)

BLER_TARGETS = (0.10, 0.01)


def setup(bw, n_bank, seed=11, quiet=False):
    """Build the evaluation context for one bandwidth case: transceivers on the common grid, iso-block LDPC codeword bank, calibration factors, strided data-RE selections."""
    M, n_act, SCS, N = bw['M'], bw['n_act'], bw['SCS'], bw['N']
    FS = M * SCS
    CP = max(round(144 * M / 2048), 4)
    FS_NAT = n_act * SCS
    # guard sizing at each waveform's OWN rate, with the report's +2 margin
    # (report S9: L_ZP = max(8, L_ch + 2)).  Zero margin lets the Farrow
    # fractional-delay tail spill past the ZP and alias into the next subsymbol.
    ch0 = TDLChannel(TDLChannelConfig('TDL-C', 300e-9, 0, FS_NAT, seed=1,
                                      use_fdf=False))
    ZPLEN = max(8, ch0.max_delay_samples + 2)
    cp_g = max(4, int(round(1e-6 * FS_NAT)) + 3)
    cp_o = max(round(144 * n_act / 2048), 4)

    # ZP-OTFS is excluded from the CODED comparison: the uncoded sweep already
    # establishes that it is not competitive (it wins no cell in the report and
    # floors in fading here), and its joint-LMMSE soft output was the one
    # receiver whose gamma could not be validated (4-7x over-confident in
    # fading).  Publishing a coded ZP number on an unvalidated soft output
    # would attribute a harness artefact to the waveform.  Coded comparison is
    # therefore OFDM (report + improved estimator) vs the two PCP variants.
    waves = [WOFDM_TD(M, n_act, SCS, N, CP),          # OFDM, delay-domain est.
             WPCP(M, n_act, SCS, N, cp_g, 'PCP-guard'),
             WPCP(M, n_act, SCS, N, cp_o, 'PCP-orig')]
    n_common = min(w.nd for w in waves)
    E = n_common * QM
    A = E // 2
    sel = {w.name: strided(w.nd, n_common) for w in waves}
    rng = np.random.default_rng(seed)
    cal = {w.name: calibrate(w, rng) for w in waves}

    bank, brng = [], np.random.default_rng(99)
    for _ in range(n_bank):
        tb = brng.integers(0, 2, A).astype(np.uint8)
        coded, prm = nr_ldpc.ldpc_encode(tb, 0.5, QM, n_common)
        bank.append((tb, coded, prm))
    PERM = np.random.default_rng(7).permutation(E)
    ctx = dict(waves=waves, cal=cal, sel=sel, bank=bank, perm=PERM,
               iperm=np.argsort(PERM), n_common=n_common, A=A, E=E,
               FS=FS, CP=CP, ZPLEN=ZPLEN, tag=bw['tag'], M=M, n_act=n_act,
               SCS=SCS, N=N, names=[w.name for w in waves])
    if not quiet:
        print(f"\n=== {bw['tag']}: M={M} n_act={n_act} SCS={SCS/1e3:.0f}k "
              f"Fs={FS/1e6:.2f}M CP={CP} ZP={ZPLEN} cp_g={cp_g} cp_o={cp_o}")
        for w in waves:
            gs, gn = cal[w.name]
            print(f"    {w.name:10s} dataRE={w.nd:6d} g_sig={gs:.3f} "
                  f"g_noise={gn:.3f}")
        p0 = bank[0][2]
        print(f"    iso-block n_common={n_common} A={A} E={E} | "
              f"LDPC BG{p0['bg']} Z={p0['Z']} C={p0['C']}")
    return ctx


def measure(ctx, cm, fd, snrs, n_frames, err_target=None, n_min=0, seed0=0,
            frame_ids=None):
    """Run n_frames (or until err_target block errors on every method).

    Returns acc[(snr, name)] = [bit_err, bits, blk_err, blocks].
    """
    waves, cal, sel = ctx['waves'], ctx['cal'], ctx['sel']
    bank, PERM, IPERM = ctx['bank'], ctx['perm'], ctx['iperm']
    ds = DS_MAP.get(cm, 0.0)
    acc = {(s, w.name): [0, 0, 0, 0] for s in snrs for w in waves}
    Lmax = max(w.frame_len for w in waves)
    used = 0
    ids = range(n_frames) if frame_ids is None else frame_ids
    for nfr, fr in enumerate(ids):
        used = nfr + 1
        if cm == 'AWGN':
            fading = None
        else:
            ch = TDLChannel(TDLChannelConfig(cm, ds, fd, ctx['FS'],
                                             seed=seed0 + 977 * fr + int(fd),
                                             use_fdf=True))
            fading = ch._generate_jakes_fading(Lmax)
        tb, coded, prm = bank[fr % len(bank)]
        syms = qpsk_mod(coded[PERM].astype(int))
        frng = np.random.default_rng(seed0 + 13 * fr + int(fd) + 7)

        clean = {}
        for w in waves:
            pay = np.zeros(w.nd, dtype=complex)
            pay[sel[w.name]] = syms
            other = np.setdiff1d(np.arange(w.nd), sel[w.name])
            if len(other):
                pay[other] = qpsk_mod(frng.integers(0, 2, 2 * len(other)))
            s = w.tx(pay)
            L = len(s)
            if fading is None:
                clean[w.name] = s
                continue
            c = np.zeros(L, dtype=complex)
            dl = ch._fdf.apply(s) if ch._fdf is not None else None
            for i in range(ch.n_taps):
                if dl is not None:
                    col = dl[:L, i] if dl.shape[0] >= L else np.pad(
                        dl[:, i], (0, L - dl.shape[0]))
                    c += fading[i, :L] * col
                else:
                    d = int(ch.delays_samples[i])
                    if d == 0:
                        c += fading[i, :L] * s
                    elif d < L:
                        c[d:] += fading[i, d:L] * s[:L - d]
            clean[w.name] = c

        for snr in snrs:
            for w in waves:
                gs, gn = cal[w.name]
                s2 = gs / (10.0 ** (snr / 10.0) * gn)
                c = clean[w.name]
                r = c + np.sqrt(s2 / 2) * (frng.standard_normal(len(c))
                                           + 1j * frng.standard_normal(len(c)))
                z, g = w.rx(r)
                zz, gg = z[sel[w.name]], g[sel[w.name]]
                dec = nr_ldpc.ldpc_decode(qpsk_llr(zz, gg)[IPERM], prm, n_iter=20)
                v = acc[(snr, w.name)]
                v[0] += int(np.sum(dec['bits'][:len(tb)] != tb)); v[1] += len(tb)
                v[2] += 0 if dec['tb_crc_ok'] else 1; v[3] += 1
        if err_target is not None and used >= max(n_min, 1):
            if all(acc[(s, w.name)][2] >= err_target
                   for s in snrs for w in waves):
                break
    return acc, used



# ════════════════════════════════════════════════════════════════════
#  Frame-level parallelism
# ════════════════════════════════════════════════════════════════════
_W = {}


def _pool_init(bw, n_bank):
    """Each worker rebuilds the context once (Windows spawns, so transceiver
    objects cannot be inherited).  The LDPC bank is seeded, so every worker
    holds identical codewords."""
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = "1"          # one BLAS thread per worker
    _W['ctx'] = setup(bw, n_bank, quiet=True)


def _pool_work(job):
    cm, fd, snrs, ids, seed0 = job
    acc, _ = measure(_W['ctx'], cm, fd, snrs, 0, seed0=seed0, frame_ids=ids)
    return acc


def _merge(dst, src):
    for k, v in src.items():
        d = dst.setdefault(k, [0, 0, 0, 0])
        for i in range(4):
            d[i] += v[i]
    return dst


def measure_par(pool, nw, ctx, cm, fd, snrs, n_max, err_target, n_min, seed0,
                batch=None):
    """Adaptive measurement, parallel over frames; adaptive at batch granularity."""
    if pool is None:
        return measure(ctx, cm, fd, snrs, n_max, err_target=err_target,
                       n_min=n_min, seed0=seed0)
    names = ctx['names']
    batch = batch or max(nw * 4, 8)
    acc, done = {}, 0
    while done < n_max:
        take = min(batch, n_max - done)
        ids = list(range(done, done + take))
        chunks = [ids[i::nw] for i in range(nw)]
        jobs = [(cm, fd, snrs, c, seed0) for c in chunks if c]
        for part in pool.map(_pool_work, jobs):
            _merge(acc, part)
        done += take
        if done >= max(n_min, 1) and all(
                acc[(s, nm)][2] >= err_target for s in snrs for nm in names):
            break
    return acc, done


def knee(snrs, bler, target):
    """SNR where BLER crosses `target`, by linear interp in log-BLER."""
    s = np.asarray(snrs, float)
    b = np.clip(np.asarray(bler, float), 1e-6, 1.0)
    lb, lt = np.log10(b), np.log10(target)
    for i in range(len(s) - 1):
        if (lb[i] - lt) * (lb[i + 1] - lt) <= 0 and lb[i] != lb[i + 1]:
            f = (lt - lb[i]) / (lb[i + 1] - lb[i])
            return float(s[i] + f * (s[i + 1] - s[i]))
    return None


def main():
    """CLI driver: coarse knee location then fine knee-centred measurement per (channel, Doppler) cell; writes data/<tag>.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--bw', nargs='+', default=['NB'])
    ap.add_argument('--channels', nargs='+', default=['TDL-C'])
    ap.add_argument('--fds', type=float, nargs='+',
                    default=[0, 200, 400, 600, 800, 1000])
    ap.add_argument('--coarse-snrs', type=float, nargs='+',
                    default=[0, 3, 6, 9, 12, 15, 18])
    ap.add_argument('--anchors', type=float, nargs='+', default=[4, 8, 12])
    ap.add_argument('--coarse-frames', type=int, default=20)
    ap.add_argument('--nmax', type=int, default=400)
    ap.add_argument('--nmin', type=int, default=40)
    ap.add_argument('--errtarget', type=int, default=60)
    ap.add_argument('--bank', type=int, default=20)
    ap.add_argument('--maxfine', type=int, default=10)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--anchor-frames', type=int, default=300)
    ap.add_argument('--tag', default='sweep')
    args = ap.parse_args()

    BWS = {'NB': dict(tag='NB', M=256, n_act=156, SCS=30e3, N=14),
           'WB': dict(tag='WB', M=1024, n_act=624, SCS=60e3, N=14)}
    out = {}
    t_start = time.time()
    for b in args.bw:
        ctx = setup(BWS[b], args.bank)
        nw = args.workers
        pool = None
        if nw > 1:
            import multiprocessing as mp
            pool = mp.Pool(nw, initializer=_pool_init,
                           initargs=(BWS[b], args.bank))
            print(f"    pool: {nw} workers")
        names = ctx['names']
        for cm in args.channels:
            fds = [0.0] if cm == 'AWGN' else args.fds
            for fd in fds:
                t0 = time.time()
                # ---- stage 1: coarse, locate knees ----
                acc, _ = measure_par(pool, nw, ctx, cm, fd, args.coarse_snrs,
                                     args.coarse_frames, 10**9,
                                     args.coarse_frames, 300_000)
                knees = []
                for nm in names:
                    bl = [acc[(s, nm)][2] / max(acc[(s, nm)][3], 1)
                          for s in args.coarse_snrs]
                    k = knee(args.coarse_snrs, bl, BLER_TARGETS[0])
                    if k is not None:
                        knees.append(k)
                # ---- stage 2: fine, union of knee brackets + anchors ----
                # tight bracket (+-1 dB) so the frames land where the knee is;
                # spreading them over +-2 dB was what gave +-1.1 dB CIs.
                fine = set()
                for k in knees:
                    for d in (-1, 0, 1):
                        fine.add(round(2 * (k + d)) / 2.0)
                fine = sorted(x for x in fine if -2 <= x <= 26)
                if len(fine) > args.maxfine:
                    fine = sorted(fine, key=lambda x: min(abs(x - k)
                                  for k in knees))[:args.maxfine]
                    fine = sorted(fine)
                # knee points: many frames, error-driven (target ~200 block
                # errors -> ~+-0.4 dB on the 10% sensitivity)
                acc2, used = measure_par(pool, nw, ctx, cm, fd, fine,
                                         args.nmax, args.errtarget,
                                         args.nmin, 700_000)
                # anchors: fixed low frame count, they only feed the
                # BLER/BER-vs-Doppler plots and need no precision
                anch = [float(a) for a in args.anchors if a not in fine]
                if anch:
                    acc_a, _ = measure_par(pool, nw, ctx, cm, fd, anch,
                                           args.anchor_frames, 10**9,
                                           args.anchor_frames, 500_000)
                    acc2.update(acc_a)
                    fine = sorted(set(fine) | set(anch))
                for s in fine:
                    for nm in names:
                        v = acc2[(s, nm)]
                        out[f"{b}|{cm}|{fd:g}|{s:g}|{nm}"] = dict(
                            ber=v[0] / max(v[1], 1), bler=v[2] / max(v[3], 1),
                            bits=v[1], blocks=v[3])
                kd = "  ".join(f"{k:.1f}" for k in knees) if knees else "none"
                print(f"  {b} {cm} fD={fd:6.0f}  knees[{kd}]  "
                      f"fine={len(fine)}pts frames={used}  ({time.time()-t0:.0f}s)",
                      flush=True)

        if pool is not None:
            pool.close(); pool.join()

        # ---- per-cell summary: SNR at BLER targets ----
        print(f"\n--- {b}: SNR (dB) for BLER targets ---")
        for cm in args.channels:
            fds = [0.0] if cm == 'AWGN' else args.fds
            for tgt in BLER_TARGETS:
                print(f"  target BLER={tgt:.0%}   " +
                      "".join(f"{n:>12}" for n in names))
                for fd in fds:
                    row = ""
                    for nm in names:
                        ks = sorted(float(k.split('|')[3]) for k in out
                                    if k.startswith(f"{b}|{cm}|{fd:g}|")
                                    and k.endswith(f"|{nm}"))
                        bl = [out[f"{b}|{cm}|{fd:g}|{s:g}|{nm}"]['bler'] for s in ks]
                        k2 = knee(ks, bl, tgt)
                        row += f"{k2:>12.2f}" if k2 is not None else f"{'>max':>12}"
                    print(f"    {cm} fD={fd:6.0f} {row}")
    with open(os.path.join(DATA, args.tag + '.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.tag}.json   total {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
