"""Pass-2 fact check: recompute every claim the report makes from the data,
so the prose can be checked against it line by line."""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "BSD-2-Clause"
__copyright__ = "(c) 2026 Panos N. Alevizos"
__date__      = "2026-08-29"
import json, os, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'data')
M = ['OFDM', 'PCP-guard', 'PCP-orig']
CH = ['TDL-A', 'TDL-B', 'TDL-C', 'TDL-D']


def load(tag):
    """Load data/<tag>.json into {(channel, fd, method): [(snr, rec), ...]}."""
    d = json.load(open(os.path.join(DATA, tag + '.json')))
    rec = {}
    for k, v in d.items():
        bw, cm, fd, snr, nm = k.split('|')
        rec.setdefault((cm, float(fd), nm), []).append((float(snr), v))
    return rec


def knee(pts, tgt=0.10):
    """SNR at which the log-linear BLER interpolation crosses the target (NaN if never bracketed)."""
    if not pts:
        return float('nan')
    pts = sorted(pts)
    s = np.array([p[0] for p in pts])
    b = np.clip([p[1]['bler'] for p in pts], 1e-6, 1)
    lb, lt = np.log10(b), np.log10(tgt)
    for i in range(len(s) - 1):
        if (lb[i] - lt) * (lb[i + 1] - lt) <= 0 and lb[i] != lb[i + 1]:
            return float(s[i] + (lt - lb[i]) / (lb[i + 1] - lb[i]) * (s[i + 1] - s[i]))
    return float('nan')


for bw in ('NB', 'WB'):
    r = load(f'final_{bw}')
    fds = sorted({k[1] for k in r if k[0] in CH})
    print(f"\n{'='*66}\n{bw}\n{'='*66}")

    # winner count
    win = {m: 0 for m in M}; tot = 0
    for cm in CH:
        for fd in fds:
            ks = {nm: knee(r.get((cm, fd, nm), [])) for nm in M}
            ks = {k: v for k, v in ks.items() if not math.isnan(v)}
            if ks:
                tot += 1; win[min(ks, key=ks.get)] += 1
    print(f"winners @10%: " + ", ".join(f"{m} {win[m]}/{tot}" for m in M))

    # OFDM Doppler flatness per channel
    print("OFDM Doppler flatness (max-min over fD):")
    for cm in CH:
        v = [knee(r.get((cm, fd, 'OFDM'), [])) for fd in fds]
        v = [x for x in v if not math.isnan(x)]
        if v:
            print(f"   {cm}: {min(v):.2f}-{max(v):.2f}  span {max(v)-min(v):.2f} dB")

    # OFDM-vs-PCPguard margin per channel
    print("OFDM lead over PCP-guard (dB):")
    allg = []
    for cm in CH:
        g = []
        for fd in fds:
            a, b = knee(r.get((cm, fd, 'OFDM'), [])), knee(r.get((cm, fd, 'PCP-guard'), []))
            if not (math.isnan(a) or math.isnan(b)):
                g.append(b - a)
        if g:
            allg += g
            print(f"   {cm}: {min(g):.2f}-{max(g):.2f}")
    if allg:
        print(f"   OVERALL: {min(allg):.2f}-{max(allg):.2f}")

    # missing crossings
    for nm in ('PCP-guard', 'PCP-orig'):
        miss = [(cm, fd) for cm in CH for fd in fds
                if math.isnan(knee(r.get((cm, fd, nm), [])))]
        print(f"{nm} missing crossings: {len(miss)}  {miss if miss else ''}")

    # AWGN anchor (dense where available)
    fp = os.path.join(DATA, 'awgn_fine.json')
    if os.path.exists(fp):
        f = json.load(open(fp))
        rec = {}
        for k, v in f.items():
            b2, cm, fd, snr, nm = k.split('|')
            if b2 == bw and cm == 'AWGN':
                rec.setdefault(nm, []).append((float(snr), v))
        if rec:
            print("AWGN anchor (dense 0.2 dB): " +
                  ", ".join(f"{nm} {knee(rec.get(nm, [])):.2f}" for nm in M))
    print("AWGN anchor (sweep grid): " +
          ", ".join(f"{nm} {knee(r.get(('AWGN', 0.0, nm), [])):.2f}" for nm in M))

# CI half-widths (Wilson) at the knee points
print(f"\n{'='*66}\nCONFIDENCE\n{'='*66}")
for bw in ('NB', 'WB'):
    d = json.load(open(os.path.join(DATA, f'final_{bw}.json')))
    hw = []
    for k, v in d.items():
        n, b = v.get('blocks', 0), v.get('bler', 0)
        if n and 0.03 < b < 0.3:
            se = math.sqrt(b * (1 - b) / n)
            hw.append(1.96 * se / max(b, 1e-9))     # relative
    if hw:
        print(f"{bw}: median relative 95% half-width on BLER near the knee "
              f"{np.median(hw)*100:.1f}%  (n={len(hw)} points)")
