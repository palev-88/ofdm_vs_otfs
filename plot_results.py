"""
plot_results.py — Generate BER plots from JSON evaluation results.

Usage: python3 plot_results.py [output_dir]
  Reads all results/ber_*.json files
  Generates individual BER PNGs in output_dir (default: /mnt/user-data/outputs/)

For each config (M, SCS):
  - 1 AWGN plot
  - 9 fading plots (3 channels × 3 Dopplers)
  - 1 combined 10-panel plot
"""
import numpy as np, matplotlib, json, glob, os, sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import erfc

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/outputs'
os.makedirs(OUT_DIR, exist_ok=True)

SNR = [0, 5, 10, 15, 20, 25, 30]
theory = [0.5 * erfc(np.sqrt(10**(s/10))) for s in SNR]

COLORS = {'ofdm': 'green', 'zp': 'blue', 'pcp': 'red'}
MARKERS = {'ofdm': 's', 'zp': 'D', 'pcp': '^'}
LABELS = {'ofdm': '5G OFDM', 'zp': 'ZP-OTFS', 'pcp': 'PCP-OTFS'}

CH_LABELS = {'TDL-A': 'TDL-A 30ns', 'TDL-D': 'TDL-D 30ns', 'TDL-C': 'TDL-C 100ns'}
DOPPLERS = [0, 100, 1000]


def sp(ax, x, y, f='', **kw):
    x, y = np.array(x), np.array(y, dtype=float)
    m = y > 0
    if np.any(m):
        ax.semilogy(x[m], y[m], f, **kw)
    if np.any(~m):
        for xi in x[~m]:
            ax.plot(xi, 3e-6, 'v', color=kw.get('color', 'k'), markersize=7)


def load_config_results(M, SCS_kHz):
    """Load and merge all channel JSON files for one config."""
    merged = {'ber': {}, 'config': None, 'methods': None}
    for ch in ['AWGN', 'TDL-A', 'TDL-D', 'TDL-C', 'ALL']:
        fname = f"results/ber_M{M}_SCS{SCS_kHz}k_{ch}.json"
        if os.path.exists(fname):
            with open(fname) as f:
                data = json.load(f)
            merged['config'] = data['config']
            merged['methods'] = data['methods']
            merged['ber'].update(data['ber'])
    return merged


def make_single_plot(title, filename, ber_data, methods_info, config_info,
                     channel, fd, show_theory=False):
    """Generate one BER vs SNR plot."""
    fig, ax = plt.subplots(figsize=(9, 6))

    if show_theory:
        sp(ax, SNR, theory, 'k--', label='QPSK Theory', linewidth=1.5)

    for nm in ['ofdm', 'zp', 'pcp']:
        oh = methods_info[nm]['overhead_pct']
        ber_vals = [ber_data.get(f'{channel}_{fd}_{snr}_{nm}', None) for snr in SNR]
        if any(v is not None for v in ber_vals):
            ber_vals = [v if v is not None else 0 for v in ber_vals]
            sp(ax, SNR, ber_vals, f'-{MARKERS[nm]}', color=COLORS[nm],
               label=f'{LABELS[nm]} ({oh:.0f}%OH)', linewidth=2.5, markersize=7)

    M = config_info['M']
    scs = config_info['SCS_kHz']
    nmc = config_info['NMC']
    cp = config_info['CP']

    ax.set_xlabel('Es/N₀ [dB]', fontsize=13)
    ax.set_ylabel('Uncoded BER', fontsize=13)
    ax.set_title(f'{title}\nQPSK | M={M}, SCS={scs:.0f}kHz, CP={cp}(NR) | {nmc} MC',
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim([1e-6, 1] if show_theory else [1e-5, 1])
    ax.annotate('▼=0 errors', xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=8, color='gray')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def make_combined_plot(ber_data, methods_info, config_info, prefix):
    """Generate combined 10-panel plot."""
    fig, axes = plt.subplots(2, 5, figsize=(30, 12))

    # AWGN
    ax = axes[0, 0]
    sp(ax, SNR, theory, 'k--', label='Theory', linewidth=1.5)
    for nm in ['ofdm', 'zp', 'pcp']:
        oh = methods_info[nm]['overhead_pct']
        vals = [ber_data.get(f'AWGN_0_{s}_{nm}', 0) for s in SNR]
        sp(ax, SNR, vals, f'-{MARKERS[nm]}', color=COLORS[nm],
           label=f'{LABELS[nm]}({oh:.0f}%)', linewidth=2.5, markersize=5)
    ax.set_xlabel('SNR'); ax.set_ylabel('BER')
    ax.set_title('AWGN', fontsize=10, fontweight='bold')
    ax.legend(fontsize=6); ax.grid(True, which='both', alpha=0.3)
    ax.set_ylim([1e-6, 1])

    pi = 1
    for fd in DOPPLERS:
        for ch in ['TDL-A', 'TDL-D', 'TDL-C']:
            ax = axes[pi // 5, pi % 5]
            for nm in ['ofdm', 'zp', 'pcp']:
                oh = methods_info[nm]['overhead_pct']
                vals = [ber_data.get(f'{ch}_{fd}_{s}_{nm}', 0) for s in SNR]
                sp(ax, SNR, vals, f'-{MARKERS[nm]}', color=COLORS[nm],
                   label=f'{LABELS[nm]}({oh:.0f}%)', linewidth=2.5, markersize=5)
            ax.set_xlabel('SNR')
            if pi % 5 == 0: ax.set_ylabel('BER')
            ax.set_title(f'{CH_LABELS[ch]} | fD={fd}Hz', fontsize=9, fontweight='bold')
            ax.legend(fontsize=5); ax.grid(True, which='both', alpha=0.3)
            ax.set_ylim([1e-5, 1])
            pi += 1

    M = config_info['M']; scs = config_info['SCS_kHz']; nmc = config_info['NMC']
    fig.suptitle(f'M={M} SCS={scs:.0f}kHz | 5G OFDM vs ZP-OTFS vs PCP-OTFS\n'
                 f'QPSK | {nmc} MC', fontsize=13, y=1.01)
    fig.tight_layout()
    fname = f'{prefix}_combined.png'
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


# ═══ Main ═══
# Discover all configs from result files
configs_found = set()
for f in glob.glob('results/ber_M*_SCS*k_*.json'):
    base = os.path.basename(f).replace('.json', '')
    parts = base.split('_')
    M_str = parts[1][1:]  # M256 -> 256
    SCS_str = parts[2][3:-1]  # SCS60k -> 60
    configs_found.add((int(M_str), int(SCS_str)))

if not configs_found:
    print("No results found in results/. Run run_eval.py first.")
    sys.exit(1)

print(f"Found {len(configs_found)} configs: {sorted(configs_found)}\n")

for M, SCS_kHz in sorted(configs_found):
    print(f"═══ M={M} SCS={SCS_kHz}kHz ═══")
    data = load_config_results(M, SCS_kHz)

    if not data['config']:
        print(f"  No data found, skipping")
        continue

    prefix = f'ber_M{M}_SCS{SCS_kHz}k'
    cfg = data['config']
    methods = data['methods']
    ber = data['ber']

    # AWGN
    if any(f'AWGN_0_{s}_ofdm' in ber for s in SNR):
        make_single_plot('AWGN', f'{prefix}_AWGN.png', ber, methods, cfg,
                         'AWGN', 0, show_theory=True)

    # Fading channels
    for ch in ['TDL-A', 'TDL-D', 'TDL-C']:
        for fd in DOPPLERS:
            key_test = f'{ch}_{fd}_{SNR[0]}_ofdm'
            if key_test in ber:
                make_single_plot(f'{CH_LABELS[ch]} | fD={fd}Hz',
                                 f'{prefix}_{ch}_fD{fd}.png',
                                 ber, methods, cfg, ch, fd)

    # Combined 10-panel
    if any(f'AWGN_0_{s}_ofdm' in ber for s in SNR):
        make_combined_plot(ber, methods, cfg, prefix)

    print()

print("✓ All plots generated")
