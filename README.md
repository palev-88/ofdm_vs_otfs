# OFDM vs OTFS Waveform Comparison Simulator

**Author:** P. Alevizos (Renesas) + Claude (Anthropic)  
**Date:** April 2026  
**Language:** Python 3 (numpy, scipy, matplotlib only)

## Core Modules

| File | Lines | Description |
|------|-------|-------------|
| `channel.py` | 390 | 3GPP TDL-A/B/C/D/E channel model, Jakes fading |
| `qam.py` | 130 | QPSK/16QAM/64QAM modulation/demodulation |
| `ofdm.py` | 590 | 5G NR OFDM TX/RX, LS+linear interp, MMSE, decorrelated DMRS noise estimator |
| `otfs.py` | 700 | CP-OTFS with DD-LMMSE (slow at FFT≥256, reference only) |
| `otfs_zp.py` | 360 | ZP-OTFS: inverse Zak TX, DD impulse pilot, TD sparse LMMSE, ZP-tail noise est |
| `otfs_pcp.py` | 250 | PCP-OTFS: ZC pilot, 2-stage est (ZC corr + GCE-BEM Q=1), FDE, CP noise est |

## Evaluation Scripts

| Script | Usage | Description |
|--------|-------|-------------|
| `run_eval.py` | `python3 run_eval.py 256 60 50 TDL-A` | Full eval: OFDM + ZP + PCP, one config/channel |
| `run_fast.py` | `python3 run_fast.py 256 60 50 TDL-A 0 1000` | Fast eval: OFDM + PCP only, Doppler range |
| `run_single.py` | `python3 run_single.py 256 60 50 TDL-A 0 400` | Like run_fast but includes ZP-OTFS |
| `run_all.sh` | `bash run_all.sh 50` | Master script for all configs |
| `plot_results.py` | `python3 plot_results.py` | Generate PNGs from JSON results |

## Quick Start

```bash
# Run one config (M=256, SCS=60kHz, 20 MC, all channels)
python3 run_eval.py 256 60 20 ALL

# Run OFDM vs PCP only (faster, skip ZP-OTFS)
python3 run_fast.py 256 15 50 TDL-A 0 1000

# Generate plots from results
python3 plot_results.py

# Full evaluation (all 4 configs × all channels)
bash run_all.sh 50
```

## Bandwidth Configurations

| Config | M | SCS (kHz) | Fs (MHz) | CP (NR std) |
|--------|---|-----------|----------|-------------|
| A | 256 | 15 | 3.84 | 18 |
| B | 256 | 30 | 7.68 | 18 |
| C | 256 | 60 | 15.36 | 18 |
| D | 512 | 60 | 30.72 | 36 |

## Key Design Decisions

### OFDM Receiver
- **Channel estimation:** LS at DMRS (comb-2, symbols 2 & 11) → linear freq interp → linear time interp
- **No frequency smoothing, no ICI compensation** (proven unnecessary — interp error dominates ICI by 20 dB)
- **Noise estimator:** Decorrelated DMRS power difference:
  σ² = (1/4) · mean[(|Z₁|²-|Z₂|²)²] / mean[(|Z₁|²+|Z₂|²)/2]
- **Equalization:** Per-subcarrier MMSE

### ZP-OTFS
- DD impulse pilot with 2D guard zone
- ZP-tail noise estimator (cleanest: 0.8-1.2× accuracy)
- Time-domain sparse LMMSE via scipy
- Bottleneck: N=14 Doppler bins → fractional Doppler → BER floor

### PCP-OTFS
- ZC pilot (length L=CP) with delay-domain CP
- Stage 1: ZC correlation → per-subsymbol delay profile
- Stage 2: GCE-BEM (Q=1, 3 basis functions) → temporal smoothing
- CP-redundancy noise estimator
- FDE with pilot cancellation
- Optimized: pilot power 25 dB, guard ±1 Doppler column, overhead 15.7%

## Key Results (SNR=20 dB)

| SCS | fD=1000Hz Winner | Improvement |
|-----|-------------------|-------------|
| 15 kHz | **PCP-OTFS** | 4–37× over OFDM |
| 30 kHz | OFDM | 1.5–3× over PCP |
| 60 kHz | OFDM | 2–4× over PCP |

**Bottom line:** SCS determines the winner, not the waveform.

## Remaining Tasks for Claude Code

1. **Full evaluation:** 4 configs × 5 channels (TDL-A/B/C/D/E) × 11 Dopplers (0:100:1000) × 50 MC
   ```bash
   # Run per channel to manage time:
   for CH in TDL-A TDL-B TDL-C TDL-D TDL-E; do
     python3 run_fast.py 256 15 50 $CH 0 1000
     python3 run_fast.py 256 30 50 $CH 0 1000
     python3 run_fast.py 256 60 50 $CH 0 1000
     python3 run_fast.py 512 60 50 $CH 0 1000
   done
   ```

2. **Regenerate plots** after evaluation completes

3. **Recompile report** with updated figures

## Dependencies

- numpy
- scipy (sparse solvers for ZP-OTFS)
- matplotlib (plotting only)

No MATLAB, no toolboxes, no external packages.
