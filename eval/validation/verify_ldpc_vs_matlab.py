"""Bit-exact verification of the Python NR-LDPC port against the MATLAB source.

Stage 1 (this script, --gen): write test transport blocks to ldpc_tv_in.mat
Stage 2 (MATLAB, author-side): the reference codec encodes it and writes
                             ldpc_tv_out.mat
Stage 3 (this script, --cmp): encode the same TBs in Python and compare the
                             rate-matched coded bits BIT BY BIT, then decode
                             MATLAB's own coded bits with the Python decoder.

A silent 2Z / filler / interleaver error still round-trips inside one
implementation, so cross-checking against the reference is the only way to
catch it.
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import sys, os
import numpy as np
from scipy.io import savemat, loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'src'))
IN_MAT = os.path.join(HERE, 'ldpc_tv_in.mat')
OUT_MAT = os.path.join(HERE, 'ldpc_tv_out.mat')

# (n_bytes, code_rate, Qm, n_re) -- span BG2/BG1, single & multi code block
CASES = [
    (253,  0.5, 2, 2028),    # NB OFDM  : BG2, C=1
    (231,  0.5, 2, 1850),    # NB PCP   : BG2, C=1
    (780,  0.5, 2, 8112),    # WB OFDM  : larger TB
    (120,  0.5, 2, 1000),    # small TB : BG2, low Kb
    (1100, 0.8, 2, 5600),    # high rate: forces BG1
]


def gen():
    """Write the LDPC test-vector .mat consumed by the MATLAB reference encoder."""
    rng = np.random.default_rng(12345)
    tbs = [rng.integers(0, 256, nb, dtype=np.uint8) for nb, *_ in CASES]
    savemat(IN_MAT, {
        'n_cases': len(CASES),
        'tb': np.array([np.pad(t, (0, max(c[0] for c in CASES) - len(t)))
                        for t in tbs], dtype=np.uint8),
        'tb_len': np.array([len(t) for t in tbs], dtype=np.float64),
        'code_rate': np.array([c[1] for c in CASES], dtype=np.float64),
        'Qm': np.array([c[2] for c in CASES], dtype=np.float64),
        'n_re': np.array([c[3] for c in CASES], dtype=np.float64),
    })
    print(f"wrote {IN_MAT} with {len(CASES)} cases")


def cmp():
    """Compare MATLAB's rate-matched output bit-for-bit against nr_ldpc."""
    sys.path.insert(0, HERE)
    import nr_ldpc
    d = loadmat(OUT_MAT)
    din = loadmat(IN_MAT)          # TBs live in the input file
    ok_all = True
    for i, (nb, R, Qm, n_re) in enumerate(CASES):
        tb_bytes = np.array(din['tb'][i][:nb], dtype=np.uint8)
        cm = np.array(d['coded'][i][0]).ravel().astype(np.uint8)
        bgM, ZM, KM, CM = (int(d['bg'][0, i]), int(d['Z'][0, i]),
                           int(d['K'][0, i]), int(d['C'][0, i]))

        tb_bits = np.unpackbits(tb_bytes)
        cp, p = nr_ldpc.ldpc_encode(tb_bits, R, Qm, n_re)

        same_par = (p['bg'] == bgM and p['Z'] == ZM and p['K'] == KM and p['C'] == CM)
        same_len = (len(cp) == len(cm))
        same_bits = same_len and np.array_equal(cp, cm)
        nmis = int(np.sum(cp[:min(len(cp), len(cm))] != cm[:min(len(cp), len(cm))]))

        # decode MATLAB's coded bits with the Python decoder (moderate noise)
        rng = np.random.default_rng(7 + i)
        x = 1.0 - 2.0 * cm.astype(float)
        s = 0.45
        y = x + s * rng.standard_normal(len(x))
        llr = 2.0 * y / (s ** 2)
        r = nr_ldpc.ldpc_decode(llr, p)
        xdec = np.array_equal(r['bits'][:len(tb_bits)], tb_bits)

        ok = same_par and same_bits and xdec
        ok_all &= ok
        print(f"case {i}: A={len(tb_bits):5d} n_re={n_re:5d} R={R} | "
              f"py BG{p['bg']} Z={p['Z']} K={p['K']} C={p['C']} len={len(cp)} | "
              f"mat BG{bgM} Z={ZM} K={KM} C={CM} len={len(cm)} | "
              f"params={'OK' if same_par else 'MISMATCH'} "
              f"bits={'EXACT' if same_bits else f'{nmis} differ'} "
              f"xdecode={'OK' if xdec else 'FAIL'}  -> {'PASS' if ok else 'FAIL'}")
    print("\nOVERALL:", "PASS -- Python port is bit-exact with MATLAB" if ok_all
          else "FAIL -- do not use the port until this passes")
    return 0 if ok_all else 1


if __name__ == '__main__':
    if '--gen' in sys.argv:
        gen()
    elif '--cmp' in sys.argv:
        sys.exit(cmp())
    else:
        print(__doc__)
