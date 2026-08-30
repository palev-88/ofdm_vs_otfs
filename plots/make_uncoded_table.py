"""Build the uncoded-reference table rows (hard-decision QPSK BER at 30 dB)
from the v2 uncoded_ref.json, and print the numbers the prose quotes."""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
d = json.load(open(os.path.join(DATA, 'uncoded_ref.json')))
M = ['OFDM', 'PCP-guard', 'PCP-orig']
SNR = 30.0
FLOOR = 1e-6


def fmt(v):
    """Scientific-notation LaTeX cell; $<10^{-6}$ when below the Monte-Carlo floor."""
    if v is None:
        return '---'
    if v < FLOOR:
        return r'$<10^{-6}$'
    e = 0
    x = v
    while x < 1:
        x *= 10; e -= 1
    return rf'${x:.2f}\times 10^{{{e}}}$'


rows, wanted = [], {}
for bw in ('NB', 'WB'):
    for cm in ('TDL-A', 'TDL-C', 'TDL-D'):
        for fd in (0.0, 1000.0):
            vals = []
            for nm in M:
                k = f'{bw}|{cm}|{fd:g}|{SNR:g}|{nm}'
                vals.append(d[k]['ber'] if k in d else None)
            got = [v for v in vals if v is not None]
            lo = min(got) if got else None
            cells = []
            for v in vals:
                t = fmt(v)
                if v is not None and lo is not None and v <= lo * 1.001:
                    t = r'\textbf{' + t + '}'
                cells.append(t)
            rows.append(f'    {bw} & {cm} & {fd:.0f} & ' + ' & '.join(cells) + r' \\')
            wanted[(bw, cm, fd)] = vals

out = os.path.join(DATA, 'tables', 'tab_uncoded_v2.tex')
open(out, 'w', encoding='utf-8').write('\n'.join(rows) + '\n')
print('\n'.join(rows))
print('\n--- prose numbers ---')
for key in [('NB', 'TDL-C', 1000.0), ('NB', 'TDL-A', 0.0), ('WB', 'TDL-A', 1000.0),
            ('WB', 'TDL-C', 1000.0), ('NB', 'TDL-D', 1000.0)]:
    v = wanted.get(key)
    if v and all(x is not None for x in v):
        r = [f'{x:.2e}' for x in v]
        ratio = v[1] / v[0] if v[0] > 0 else float('inf')
        print(f'  {key}: OFDM {r[0]}  guard {r[1]}  orig {r[2]}   guard/OFDM = {ratio:.1f}x')
