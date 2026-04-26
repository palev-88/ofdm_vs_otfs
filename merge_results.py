"""
merge_results.py -- aggregate per-config BER digests into the unified
hypercube format consumed by plot_results.py.

Bridges the schema gap between:
  Producer:  run_eval.py     -> results/ber_M{M}_SCS{SCS}k_{channel}.json
  Consumer:  plot_results.py <- results/otfs_eval_v1.json

Walks every results/ber_*.json, groups by bandwidth (NB | WB based on the
(M, SCS) tuple), reshapes the flat per-cell BER dict into the per-curve
list-of-records layout that plot_results.py expects, and writes the
aggregated file.

Usage
-----
    python merge_results.py

Inputs
------
results/ber_M*_SCS*_*.json    (any number of files; all are merged)

Output
------
results/otfs_eval_v1.json     (single aggregated file)

Notes
-----
* Bandwidth label assignment (configurable below):
    NB:  M=256, SCS=30 kHz   (the 5G-NR-anchored narrowband config)
    WB:  M=1024, SCS=60 kHz  (the 5G-NR-anchored wideband config)
  Any other (M, SCS) tuple is labelled "OTHER:M<M>_SCS<SCS>k" and still
  emitted -- plot_results.py will simply not include those rows in the
  NB/WB grids.

* Method-name normalisation: run_eval.py uses 'pcp' as a single method;
  plot_results.py distinguishes 'pcp_guard' and 'pcp_orig'. As a
  smoke-test convenience we duplicate the 'pcp' BER curve under both
  labels so plot_results.py renders. For final-eval runs you will want
  to wire run_eval.py to evaluate both PCP variants separately.
"""
__author__    = "Panos N. Alevizos"
__email__     = "bigpan27@gmail.com"
__credits__   = ["Panos N. Alevizos", "Claude Code (Anthropic)"]
__license__   = "CC-BY-4.0"
__copyright__ = "(c) 2026 Panos N. Alevizos"

import glob
import json
import os
import re
import sys
from collections import defaultdict


# ════════════════════════════ Configuration ════════════════════════════

INDIR  = 'results'
OUTFILE = 'results/otfs_eval_v1.json'

# (M, SCS_kHz) -> bandwidth label used by plot_results.py
BW_MAP = {
    (256, 30):  'NB',
    (1024, 60): 'WB',
}

# run_eval method name  ->  list of plot_results method labels.
# After the run_eval.py extension that evaluates pcp_guard and pcp_orig
# separately, this is now an identity-style map for the four methods that
# plot_results.py renders.  mc and deconv are evaluated by run_eval.py
# but excluded from the plotter's main figures (kept in the raw JSON for
# downstream analysis).
METHOD_MAP = {
    'ofdm':      ['ofdm'],
    'zp':        ['zp'],
    'pcp_guard': ['pcp_guard'],
    'pcp_orig':  ['pcp_orig'],
    'mc':        [],
    'deconv':    [],
}


# ═════════════════════════ Per-file parser ═════════════════════════════

def parse_per_config(path):
    """Load one results/ber_M*_SCS*_*.json file.

    Returns
    -------
    cfg : dict           # the 'config' section
    snr_list : list      # SNR sweep points
    ber_dict : dict      # flat {f'{channel}_{fd}_{snr}_{method}': ber} mapping
    """
    with open(path) as f:
        d = json.load(f)
    return d['config'], d['snr_list'], d['ber']


# Key parser: 'TDL-A_500_22.5_pcp_guard' -> ('TDL-A', 500, 22.5, 'pcp_guard')
# Doppler is an integer (Hz); SNR may be int or float (dB); method names may
# contain underscores (e.g. 'pcp_guard', 'pcp_orig').
KEY_RE = re.compile(r'^(?P<ch>[A-Z]+(?:-[A-Z])?)_(?P<fd>\d+)_(?P<snr>[\d\.]+)_(?P<method>[a-z][a-z_]*)$')


def parse_key(k):
    """Decompose a flat per-cell key. Returns (channel, fd_int, snr_float, method)."""
    m = KEY_RE.match(k)
    if m is None:
        return None
    return (m.group('ch'), int(m.group('fd')),
            float(m.group('snr')), m.group('method'))


# ═════════════════════════════ Main ═════════════════════════════════════

def main():
    files = sorted(glob.glob(os.path.join(INDIR, 'ber_M*_SCS*_*.json')))
    if not files:
        sys.exit(f"[merge_results] No {INDIR}/ber_M*_SCS*_*.json files found. "
                 f"Run run_eval.py first.")

    print(f"[merge_results] Found {len(files)} per-config file(s).")

    # records[(bw, method, channel, fd)] -> {snr: ber}
    records = defaultdict(dict)
    snr_set = set()
    channel_set = set()
    doppler_set = set()
    cfg_per_bw = {}    # bw -> config dict (last one wins; they should agree)

    skipped_keys = 0
    for path in files:
        cfg, snr_list, ber_dict = parse_per_config(path)
        snr_set.update(snr_list)

        bw = BW_MAP.get((cfg['M'], int(cfg['SCS_kHz'])),
                        f"OTHER:M{cfg['M']}_SCS{int(cfg['SCS_kHz'])}k")
        cfg_per_bw[bw] = cfg

        for k, ber in ber_dict.items():
            parsed = parse_key(k)
            if parsed is None:
                skipped_keys += 1
                continue
            channel, fd, snr, src_method = parsed
            channel_set.add(channel)
            doppler_set.add(fd)

            for tgt_method in METHOD_MAP.get(src_method, []):
                records[(bw, tgt_method, channel, fd)][snr] = ber

        print(f"  {os.path.basename(path):42s}  "
              f"bw={bw:6s}  cells={len(ber_dict)}")

    if skipped_keys:
        print(f"[merge_results] Skipped {skipped_keys} unparseable keys "
              "(likely a different schema -- inspect manually).")

    # Convert each {snr: ber} dict into a list aligned with snr_list.
    snr_list = sorted(snr_set)
    out_records = []
    for (bw, method, channel, fd), snr_to_ber in records.items():
        ber_list = [snr_to_ber.get(s, float('nan')) for s in snr_list]
        out_records.append({
            'bw_case': bw,
            'method':  method,
            'channel': channel,
            'fd':      fd,
            'ber':     ber_list,
        })

    out = {
        'snr_list': snr_list,
        'configs':  cfg_per_bw,
        'channels': sorted(channel_set),
        'dopplers': sorted(doppler_set),
        'results':  out_records,
    }

    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
    with open(OUTFILE, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\n[merge_results] Wrote {OUTFILE}")
    print(f"  bandwidth cases: {sorted(cfg_per_bw.keys())}")
    print(f"  channels:        {sorted(channel_set)}")
    print(f"  Doppler points:  {sorted(doppler_set)}")
    print(f"  SNR points:      {snr_list}")
    print(f"  total records:   {len(out_records)}")


if __name__ == '__main__':
    main()
