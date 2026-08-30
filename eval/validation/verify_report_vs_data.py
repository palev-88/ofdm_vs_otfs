"""Verify the report's coded tables against the datasets on disk.

Parses the sensitivity values out of report.tex and compares them, cell by
cell, against (a) the archived filtered-chain JSON and (b) the current
fixed-rate JSON, to establish unambiguously which dataset the report is
currently showing.
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import os, re, io, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REPORT = os.path.join(ROOT, "report", "report.tex")
CH = ['TDL-A', 'TDL-B', 'TDL-C', 'TDL-D']
FDS = [0, 200, 400, 600, 800, 1000]
METH = ['OFDM', 'PCP-guard', 'PCP-orig']


def knee(pts, tgt=0.10):
    """SNR at which the log-linear BLER interpolation crosses the target (NaN if never bracketed)."""
    s = np.array([p[0] for p in pts]); b = np.clip([p[1] for p in pts], 1e-6, 1)
    lb, lt = np.log10(b), np.log10(tgt)
    for i in range(len(s) - 1):
        if (lb[i] - lt) * (lb[i + 1] - lt) <= 0 and lb[i] != lb[i + 1]:
            return float(s[i] + (lt - lb[i]) / (lb[i + 1] - lb[i]) * (s[i + 1] - s[i]))
    return None


def load(tag):
    out = {}
    for b in ('NB', 'WB'):
        p = os.path.join(HERE, f'final_{b}{tag}.json')
        if not os.path.exists(p):
            continue
        rec = {}
        for k, v in json.load(open(p)).items():
            bw, cm, fd, snr, nm = k.split('|')
            rec.setdefault((bw, cm, float(fd), nm), []).append(
                (float(snr), v['bler']))
        for key, pts in rec.items():
            pts.sort()
            out[key] = knee(pts)
    return out


def parse_report():
    s = io.open(REPORT, encoding='utf-8', errors='replace').read()
    out = {}
    for bw, lbl in (('NB', 'tab:coded_nb'), ('WB', 'tab:coded_wb')):
        i = s.find('label{' + lbl + '}')
        if i < 0:
            continue
        blk = s[i:s.index('bottomrule', i)]
        cur = None
        for line in blk.splitlines():
            mrow = re.search(r'multirow\{6\}\{\*\}\{(TDL-[ABCD])\}', line)
            if mrow:
                cur = mrow.group(1)
            if '&' not in line or 'Channel' in line or 'midrule' in line:
                continue
            cells = [c.strip() for c in line.replace('\\\\', '').split('&')]
            cells = [c for c in cells if c]
            if mrow:
                cells = cells[1:]
            if len(cells) != 4 or not cells[0].isdigit():
                continue
            fd = int(cells[0])
            for nm, c in zip(METH, cells[1:]):
                v = re.sub(r'\\textbf\{|\}', '', c).strip()
                out[(bw, cur, float(fd), nm)] = (
                    None if 'max' in v else float(v))
    return out


rep = parse_report()
filt = load('_filteredchain')
new = load('')
print(f"report cells parsed: {len(rep)}   filtered dataset: {len(filt)}   "
      f"current dataset: {len(new)}\n")

for name, ds in (('FILTERED-CHAIN (archived)', filt), ('CURRENT (fixed-rate)', new)):
    match = diff = missing = 0
    worst = []
    for k, rv in rep.items():
        if k not in ds or ds[k] is None:
            if rv is None and k in ds:
                match += 1
            else:
                missing += 1
            continue
        dv = ds[k]
        if rv is None:
            missing += 1
            continue
        if abs(rv - dv) < 0.02:
            match += 1
        else:
            diff += 1
            worst.append((abs(rv - dv), k, rv, dv))
    worst.sort(reverse=True)
    tot = match + diff + missing
    print(f"--- report vs {name} ---")
    print(f"    exact match: {match}/{tot}   differ: {diff}   "
          f"unavailable: {missing}")
    for d, k, rv, dv in worst[:6]:
        print(f"      {k[0]} {k[1]} fD={k[2]:.0f} {k[3]:10s}: "
              f"report {rv:6.2f}  data {dv:6.2f}   (delta {d:.2f} dB)")
    print()
