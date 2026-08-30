"""Extract the 3GPP NR LDPC base-graph shift tables from the MATLAB source.

Parsing nr5g_ldpc_bg_tables.m directly (rather than retyping the tables)
guarantees the Python codec uses byte-identical shift values to the verified
MATLAB reference. Writes nr_ldpc_bg.npz with bg1/bg2 arrays of shape
(n_entries, 10) = [row, col, s_iLS1 .. s_iLS8], 1-indexed row/col.
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import re, os, sys
import numpy as np

# Path to the MATLAB reference's nr5g_ldpc_bg_tables.m (not distributed
# with this repository); pass it as the first CLI argument.
# The extracted tables ship with this repo as src/nr_ldpc_bg.npz, so this
# script only needs re-running if the reference tables ever change.
if len(sys.argv) < 2:
    sys.exit("usage: python extract_bg_tables.py <path-to-nr5g_ldpc_bg_tables.m>")
SRC = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nr_ldpc_bg.npz")

txt = open(SRC, 'r', encoding='utf-8', errors='replace').read()


def grab(fn_name):
    """Return the numeric rows of the `T = [ ... ];` block inside a function."""
    i = txt.index(f"function [rows, cols, shifts] = {fn_name}(iLS)")
    j = txt.index("T = [", i)
    k = txt.index("];", j)
    body = txt[j + len("T = ["):k]
    rows = []
    for line in body.splitlines():
        line = line.split('%')[0].strip()          # strip comments
        line = line.rstrip(';').strip()
        if not line:
            continue
        vals = re.findall(r'-?\d+', line)
        if len(vals) != 10:
            raise ValueError(f"{fn_name}: expected 10 ints, got {len(vals)}: {line!r}")
        rows.append([int(v) for v in vals])
    return np.array(rows, dtype=np.int64)


bg1 = grab("bg1_table")
bg2 = grab("bg2_table")

# ---- sanity checks against TS 38.212 ----
assert bg1.shape[1] == 10 and bg2.shape[1] == 10
print(f"BG1: {bg1.shape[0]} entries, rows 1..{bg1[:,0].max()}, cols 1..{bg1[:,1].max()}")
print(f"BG2: {bg2.shape[0]} entries, rows 1..{bg2[:,0].max()}, cols 1..{bg2[:,1].max()}")
assert bg1[:, 0].max() == 46 and bg1[:, 1].max() == 68, "BG1 must be 46x68"
assert bg2[:, 0].max() == 42 and bg2[:, 1].max() == 52, "BG2 must be 42x52"
assert bg1.shape[0] == 316, f"BG1 should have 316 entries, got {bg1.shape[0]}"
assert bg2.shape[0] == 197, f"BG2 should have 197 entries, got {bg2.shape[0]}"
assert (bg1[:, 2:] >= 0).all() and (bg2[:, 2:] >= 0).all(), "shifts must be >= 0"
# no duplicate (row,col) pairs
for nm, t in (("BG1", bg1), ("BG2", bg2)):
    pairs = t[:, 0] * 1000 + t[:, 1]
    assert len(np.unique(pairs)) == len(pairs), f"{nm} has duplicate (row,col)"

np.savez_compressed(OUT, bg1=bg1, bg2=bg2)
print(f"wrote {OUT}")
