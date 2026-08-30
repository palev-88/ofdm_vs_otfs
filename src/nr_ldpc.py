"""3GPP NR LDPC (TS 38.212 §5.2-5.4) — self-contained Python codec.

Base-graph shift tables (TS 38.212 Tables 5.3.2-2/-3) ship alongside as
nr_ldpc_bg.npz; they were machine-extracted from a MATLAB reference
implementation rather than retyped (see eval/validation/extract_bg_tables.py),
and the full rate-matched chain is checked bit-for-bit against MATLAB-encoded
test vectors shipped with this repo
(eval/validation/verify_ldpc_vs_matlab.py + ldpc_tv_*.mat).

Implementation traps in the TS 38.212 chain, and how they are handled here:

  1. 2Z systematic puncture. The encoder emits d[2Z..N-1]; the decoder's
     inverse bit-selection must therefore start its valid-index list at 2Z
     (0-based), NOT 0. Positions 0..2Z-1 stay at LLR 0 (erased).
  2. The 2Z offset bites twice: filler positions must be re-mapped from d
     coordinates into enc_out coordinates via max(0, K_actual - 2Z).
  3. Filler bits sit at the END of the K systematic bits, and their LLRs are
     clamped to +50 at the decoder (they are known zeros).
  4. Repeated circular-buffer LLRs are SUMMED (soft combining), not
     overwritten.
  5. Bit interleaving is column-write / row-read by Qm; the de-interleave is
     the exact transpose inverse.
  6. E_c per code block uses the TS 38.212 §5.4.2/§5.5 distribution formula,
     computed identically in encoder and decoder.
  7. CRC-24B is computed over ALL info slots including NULL padding.
  8. K = Kb_max*Z always; Kb only selects Z (and only varies for BG2).

LLR convention (matches the MATLAB reference implementation):
    L = log P(bit=0) / P(bit=1);  hard bit = (L < 0).

Decoder is block-row layered offset min-sum (layered scheduling, one
block-row at a time), as used in production 5G decoders.
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import os
import numpy as np

_BG_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nr_ldpc_bg.npz")

# TS 38.212 Table 5.3.2-1 lifting sets (0 = unused slot)
Z_SETS = np.array([
    [2,   4,   8,  16,  32,  64, 128, 256],
    [3,   6,  12,  24,  48,  96, 192, 384],
    [5,  10,  20,  40,  80, 160, 320,   0],
    [7,  14,  28,  56, 112, 224,   0,   0],
    [9,  18,  36,  72, 144, 288,   0,   0],
    [11, 22,  44,  88, 176, 352,   0,   0],
    [13, 26,  52, 104, 208,   0,   0,   0],
    [15, 30,  60, 120, 240,   0,   0,   0],
], dtype=np.int64)

_bg_cache = {}
_core_cache = {}


def _bg_table(bg, iLS):
    """(rows, cols, shifts) 1-indexed, for base graph `bg` and lifting set iLS."""
    d = np.load(_BG_NPZ)
    T = d['bg1'] if bg == 1 else d['bg2']
    return T[:, 0], T[:, 1], T[:, 2 + (iLS - 1)]


def bg_params(bg):
    """(K_cb_max, Kb_max, nb, mb) per TS 38.212 Table 5.3.2-1."""
    return (8448, 22, 68, 46) if bg == 1 else (3840, 10, 52, 42)


def build_bg_matrix(bg, Z, iLS):
    """Dense mb x nb shift matrix, -1 where there is no connection."""
    key = (bg, Z, iLS)
    if key in _bg_cache:
        return _bg_cache[key]
    _, Kb_max, nb, mb = bg_params(bg)
    r, c, s = _bg_table(bg, iLS)
    BG = -np.ones((mb, nb), dtype=np.int64)
    BG[r - 1, c - 1] = s % Z
    _bg_cache[key] = BG
    return BG


# ────────────────────────────────────────────────────────────────────
#  CRC  (TS 38.212 §5.1)
# ────────────────────────────────────────────────────────────────────
def crc_generic(bits, poly, clen):
    """Bitwise CRC of `bits` with generator polynomial `poly` (MSB-first); returns the parity bits."""
    p = [(poly >> (clen - i)) & 1 for i in range(clen + 1)]
    reg = [0] * clen
    for b in np.asarray(bits, dtype=np.uint8):
        fb = int(b) ^ reg[0]
        nr = [0] * clen
        for i in range(clen - 1):
            nr[i] = reg[i + 1] ^ (p[i + 1] & fb)
        nr[clen - 1] = fb
        reg = nr
    return np.array(reg, dtype=np.uint8)


CRC24A_POLY, CRC24B_POLY, CRC16_POLY = 0x1864CFB, 0x1800063, 0x11021
crc24a = lambda b: crc_generic(b, CRC24A_POLY, 24)
crc24b = lambda b: crc_generic(b, CRC24B_POLY, 24)
crc16  = lambda b: crc_generic(b, CRC16_POLY, 16)


def _crc_ok(bits_with_crc, fn, clen):
    if len(bits_with_crc) <= clen:
        return False
    return bool(np.all(np.asarray(bits_with_crc[-clen:], np.uint8) ==
                       fn(bits_with_crc[:-clen])))


def bytes_to_bits(data):
    """MSB-first bit expansion."""
    a = np.asarray(data, dtype=np.uint8)
    return np.unpackbits(a)


# ────────────────────────────────────────────────────────────────────
#  GF(2) linear algebra on the 4Z x 4Z core (bitmask rows; cached)
# ────────────────────────────────────────────────────────────────────
def _core_inverse(bg, Z, iLS):
    """Inverse of the 4Z x 4Z core parity submatrix, as bitmask rows.

    The core (base-graph rows 0..3, parity block-columns Kb..Kb+3) is the only
    part that needs elimination; the remaining parity rows have an identity-ish
    diagonal and are obtained by sequential back-substitution.
    """
    key = (bg, Z, iLS)
    if key in _core_cache:
        return _core_cache[key]
    _, Kb_max, nb, mb = bg_params(bg)
    Kb = Kb_max
    BG = build_bg_matrix(bg, Z, iLS)
    n = 4 * Z
    # rows as integers: low n bits = matrix, next n bits = identity (augmented)
    rows = [0] * n
    for i in range(n):
        rows[i] = 1 << (n + i)
    for r in range(4):
        for pc in range(4):
            sh = int(BG[r, Kb + pc])
            if sh < 0:
                continue
            for k in range(Z):
                ri = int(r * Z + k)
                ci = int(pc * Z + ((k + sh) % Z))
                rows[ri] |= (1 << ci)   # Python int shift; numpy ints overflow
    # Gauss-Jordan
    piv_of_col = [-1] * n
    cur = 0
    for col in range(n):
        piv = -1
        for row in range(cur, n):
            if (rows[row] >> col) & 1:
                piv = row
                break
        if piv < 0:
            continue
        rows[cur], rows[piv] = rows[piv], rows[cur]
        pr = rows[cur]
        for row in range(n):
            if row != cur and ((rows[row] >> col) & 1):
                rows[row] ^= pr
        piv_of_col[col] = cur
        cur += 1
    if cur != n:
        raise RuntimeError(f"core matrix singular for bg={bg} Z={Z} (rank {cur}/{n})")
    # inverse row for output column `col` is the augmented half of pivot row
    inv = np.zeros((n, n), dtype=np.uint8)
    for col in range(n):
        aug = rows[piv_of_col[col]] >> n
        for j in range(n):
            inv[col, j] = (aug >> j) & 1
    _core_cache[key] = inv
    return inv


def _lshift(x, v):
    """Left circular shift by v."""
    return x if v == 0 else np.roll(x, -int(v))


# ────────────────────────────────────────────────────────────────────
#  Encoding
# ────────────────────────────────────────────────────────────────────
def encode_cb(info, bg, Z, iLS):
    """One code block. `info` is length K with filler already zeroed.

    Returns enc_out = d[2Z..N-1] of length (nb-2)*Z.
    """
    _, Kb_max, nb, mb = bg_params(bg)
    Kb = Kb_max
    K = Kb * Z
    BG = build_bg_matrix(bg, Z, iLS)
    s_blocks = np.asarray(info, np.uint8)[:K].reshape(Kb, Z)

    # syndromes from the systematic columns
    synd = np.zeros((mb, Z), dtype=np.uint8)
    for i in range(mb):
        acc = np.zeros(Z, dtype=np.uint8)
        for j in range(Kb):
            v = BG[i, j]
            if v >= 0:
                acc ^= _lshift(s_blocks[j], v)
        synd[i] = acc

    parity = np.zeros((mb, Z), dtype=np.uint8)
    # core parity: p_core = Ainv * syndrome(rows 0..3)
    inv = _core_inverse(bg, Z, iLS)
    rhs = synd[:4].reshape(4 * Z).astype(np.int32)
    core = (inv.astype(np.int32) @ rhs) & 1
    parity[:4] = core.astype(np.uint8).reshape(4, Z)

    # extension parity: sequential back-substitution
    for i in range(4, mb):
        acc = synd[i].copy()
        for pp in range(i):
            v = BG[i, Kb + pp]
            if v >= 0:
                acc ^= _lshift(parity[pp], v)
        vd = BG[i, Kb + i]
        parity[i] = _lshift(acc, Z - (vd % Z)) if vd > 0 else acc

    return np.concatenate([np.asarray(info, np.uint8)[2 * Z:K],
                           parity.reshape(mb * Z)])


def rate_match_enc(enc_out, E, fm_enc, Qm):
    """TS 38.212 §5.4.2: circular-buffer selection (rv=0) + bit interleaving."""
    valid = np.flatnonzero(~np.asarray(fm_enc, bool))
    nv = len(valid)
    idx = valid[np.arange(E) % nv]
    e = np.asarray(enc_out, np.uint8)[idx]
    if Qm > 1:
        e = e.reshape(E // Qm, Qm, order='F').T.reshape(-1, order='F')
    return e


def rate_dematch(llr_in, E, N_ldpc, Z, filler_mask_K, Qm):
    """Inverse of rate_match_enc. Repeated positions are SUMMED."""
    x = np.asarray(llr_in, float)
    if Qm > 1:
        x = x.reshape(Qm, E // Qm, order='F').T.reshape(-1, order='F')
    fill_pos = np.flatnonzero(np.asarray(filler_mask_K, bool))
    valid = np.setdiff1d(np.arange(2 * Z, N_ldpc), fill_pos, assume_unique=False)
    nv = len(valid)
    idx = valid[np.arange(E) % nv]
    llr = np.zeros(N_ldpc)
    np.add.at(llr, idx, x)          # soft-combine repetitions
    return llr


def _E_c(c, C, E_total, Qm, nlayers=1):
    """TS 38.212 §5.4.2/§5.5 per-CB rate-matched length (0-based c)."""
    if c <= C - (int(np.floor(E_total / (nlayers * Qm))) % C) - 1:
        return nlayers * Qm * int(np.floor(E_total / (nlayers * Qm * C)))
    return nlayers * Qm * int(np.ceil(E_total / (nlayers * Qm * C)))


def ldpc_encode(tb_bits, code_rate, Qm, n_re):
    """Transport-block encode. `tb_bits` is a 0/1 array (the payload A bits)."""
    tb_bits = np.asarray(tb_bits, np.uint8)
    A = len(tb_bits)
    if A > 3824:
        b = np.concatenate([tb_bits, crc24a(tb_bits)]); crc_len = 24
    else:
        b = np.concatenate([tb_bits, crc16(tb_bits)]);  crc_len = 16

    bg = 2 if (A <= 292 or (A <= 3824 and code_rate <= 0.67) or code_rate <= 0.25) else 1
    K_cb_max, Kb_max, nb, mb = bg_params(bg)

    Bprime = len(b)
    if Bprime > K_cb_max:
        L_cb = 24
        C = int(np.ceil(Bprime / (K_cb_max - L_cb)))
        Bpp = Bprime + C * L_cb
    else:
        C, L_cb, Bpp = 1, 0, Bprime
    K_cb = int(np.ceil(Bpp / C))

    if bg == 1:
        Kb = 22
    else:
        Kb = 10 if Bprime > 640 else (9 if Bprime > 560 else (8 if Bprime > 192 else 6))

    Z, iLS = 0, 0
    for si in range(8):
        for zi in range(8):
            z = int(Z_SETS[si, zi])
            if z == 0:
                break
            if Kb * z >= K_cb:
                if Z == 0 or z < Z:
                    Z, iLS = z, si + 1
                break
    if Z == 0:
        Z, iLS = 256, 1

    K = Kb_max * Z
    N_ldpc = nb * Z
    E_total = n_re * Qm

    filler_mask = np.zeros((C, K), dtype=bool)
    out, offset = [], 0
    for c in range(C):
        if C == 1:
            cb_info = b
        else:
            n_slots = K_cb - L_cb
            end = min(offset + n_slots, len(b))
            pay = np.zeros(n_slots, np.uint8)
            pay[:end - offset] = b[offset:end]
            offset = end
            cb_info = np.concatenate([pay, crc24b(pay)])   # CRC over all slots
        K_actual = len(cb_info)
        info = np.zeros(K, np.uint8)
        info[:K_actual] = cb_info                          # filler at the END
        filler_mask[c, K_actual:] = True

        enc_out = encode_cb(info, bg, Z, iLS)

        fm_enc = np.zeros(len(enc_out), bool)
        if K_actual < K:
            a = max(0, K_actual - 2 * Z)                   # trap #2
            bnd = K - 2 * Z
            if bnd > a:
                fm_enc[a:bnd] = True

        out.append(rate_match_enc(enc_out, _E_c(c, C, E_total, Qm), fm_enc, Qm))

    coded = np.concatenate(out)[:E_total]
    params = dict(bg=bg, Z=Z, iLS=iLS, K=K, C=C, N_ldpc=N_ldpc, Qm=Qm,
                  n_re=n_re, code_rate=code_rate, filler_mask=filler_mask,
                  crc_len=crc_len, B=len(b), A=A, E_total=E_total)
    return coded, params


# ────────────────────────────────────────────────────────────────────
#  Decoding — block-row layered offset min-sum
# ────────────────────────────────────────────────────────────────────
def _layer_index(bg, Z, iLS):
    """Per block-row variable indices, shape (n_conn, Z), cached."""
    key = ('L', bg, Z, iLS)
    if key in _core_cache:
        return _core_cache[key]
    _, Kb_max, nb, mb = bg_params(bg)
    BG = build_bg_matrix(bg, Z, iLS)
    k = np.arange(Z)
    layers = []
    for i in range(mb):
        cols = np.flatnonzero(BG[i] >= 0)
        idx = np.empty((len(cols), Z), dtype=np.int64)
        for t, c in enumerate(cols):
            idx[t] = c * Z + ((k + BG[i, c]) % Z)
        layers.append(idx)
    _core_cache[key] = layers
    return layers


def ldpc_decode_oms(llr_ch, bg, Z, iLS, max_iter=25, beta=0.5):
    """Returns (hard_bits over N_ldpc, iters, converged)."""
    layers = _layer_index(bg, Z, iLS)
    app = np.clip(np.asarray(llr_ch, float), -30, 30).copy()
    m2v = [np.zeros(l.shape) for l in layers]
    converged, used = False, max_iter
    for it in range(1, max_iter + 1):
        for li, idx in enumerate(layers):
            v2c = app[idx] - m2v[li]
            sg = np.where(v2c >= 0, 1.0, -1.0)
            av = np.abs(v2c)
            prod_sign = np.prod(sg, axis=0)
            if av.shape[0] >= 2:
                part = np.partition(av, 1, axis=0)
                mn1, mn2 = part[0], part[1]
            else:
                mn1 = mn2 = av[0]
            is_min = av <= mn1[None, :] + 1e-9
            first = np.cumsum(is_min, axis=0) == 1
            mex = np.where(is_min & first, mn2[None, :], mn1[None, :])
            new = prod_sign[None, :] * sg * np.maximum(mex - beta, 0.0)
            new = np.clip(new, -30, 30)
            m2v[li] = new
            app[idx] = v2c + new
        hard = (app < 0).astype(np.uint8)
        ok = True
        for idx in layers:
            if np.any(hard[idx].sum(axis=0) & 1):
                ok = False
                break
        if ok:
            converged, used = True, it
            break
    return (app < 0).astype(np.uint8), used, converged


def ldpc_decode(llr, params, n_iter=25, offset=0.5):
    """Transport-block decode. Returns dict with decoded bits and CRC flags."""
    llr = np.asarray(llr, float).ravel()
    C, Z, bg, iLS = params['C'], params['Z'], params['bg'], params['iLS']
    Qm, K, N_ldpc = params['Qm'], params['K'], params['N_ldpc']
    E_total = len(llr)

    cbs, cb_ok, tot_it, pos = [], [], 0, 0
    for c in range(C):
        Ec = min(_E_c(c, C, E_total, Qm), len(llr) - pos)
        cb_llr = llr[pos:pos + Ec]; pos += Ec
        fm = params['filler_mask'][c]
        dm = rate_dematch(cb_llr, Ec, N_ldpc, Z, fm, Qm)
        dm[:K][fm] = 50.0                                  # trap #3
        dec, its, _ = ldpc_decode_oms(dm, bg, Z, iLS, n_iter, offset)
        tot_it += its
        info = dec[:K][~fm]
        if C > 1 and len(info) > 24:
            cb_ok.append(_crc_ok(info, crc24b, 24))
            cbs.append(info[:-24])
        else:
            cb_ok.append(True)
            cbs.append(info)

    allb = np.concatenate(cbs)
    if len(allb) > params['B']:
        allb = allb[:params['B']]
    cl = params['crc_len']
    if len(allb) > cl:
        tb_ok = _crc_ok(allb, crc24a if cl == 24 else crc16, cl)
        payload = allb[:-cl]
    else:
        tb_ok, payload = False, allb
    return dict(bits=payload, tb_crc_ok=bool(tb_ok), cb_crc_ok=cb_ok,
                crc_ok=bool(tb_ok) and all(cb_ok), avg_iters=tot_it / C)


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    for (A, Qm, n_re, R) in [(2028, 2, 2028, 0.5), (1200, 2, 1850, 0.5)]:
        tb = rng.integers(0, 2, A).astype(np.uint8)
        coded, p = ldpc_encode(tb, R, Qm, n_re)
        print(f"A={A} -> BG{p['bg']} Z={p['Z']} iLS={p['iLS']} K={p['K']} "
              f"C={p['C']} N={p['N_ldpc']} coded={len(coded)}")
        llr = np.where(coded > 0, -8.0, 8.0)          # noiseless, L>0 => bit 0
        r = ldpc_decode(llr, p)
        print(f"   noiseless: bits match={np.array_equal(r['bits'], tb)} "
              f"tb_crc={r['tb_crc_ok']} iters={r['avg_iters']:.1f}")
