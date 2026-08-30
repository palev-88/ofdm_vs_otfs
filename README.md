# OFDM vs OTFS — Python Reference Implementation

Link-level comparison of 5G-NR OFDM and OTFS variants under 3GPP TDL fading:
coded (NR-LDPC) receive sensitivity, uncoded BER, receiver complexity, and TX
waveform characterisation.  This is the code behind the technical report
*"OFDM vs OTFS Waveform Comparison"* — every curve and table in the report is
reproducible from this repository.

## Repository layout

```text
src/        All PHY routines (the only place algorithms live)
              channel.py     3GPP TDL-A/B/C/D/E channel, Jakes fading, Farrow FDF
              fdf.py, qam.py Fractional-delay filter; QPSK/QAM mapping
              ofdm.py        5G-NR CP-OFDM transceiver (classical baseline receiver)
              ofdm_td.py     Delay-domain OFDM receiver (CIR fit + residual noise
                             estimate + interpolation-error-aware reliability)
              otfs_pcp.py    PCP-OTFS (ZC pilot, GCE-BEM, FDE)  — evaluated
              otfs.py        CP-OTFS (cross-domain detector)    — documented, excluded
              otfs_zp.py     ZP-OTFS (TD sparse LMMSE)          — documented, excluded
              otfs_mc.py     MC-OTFS (precoded OFDM)            — documented, excluded
              nr_ldpc.py     3GPP TS 38.212 NR LDPC (+ base-graph tables .npz)

eval/       Evaluation drivers (write JSON into data/)
              coded_sweep.py   Adaptive knee-centred coded sensitivity sweep
              coded_eval.py    Waveform wrappers, calibration, LLR chain (library + CLI)
              extend_deep.py   Deep-BLER extension pass
              fill_pass.py     Anchor-SNR completion pass
              refine_pass.py   Anchor-gap refinement pass (9/10/11 dB cells)
              uncoded_eval.py  Uncoded reference (same chain, code removed)
              awgn_fine.py     Dense 0.2 dB AWGN anchor
              ablate_ofdm.py   OFDM estimator ablation (Table 3 of the report)
              ci_knee.py       Bootstrap 95% CIs on the sensitivity knees
              tx_spectrum_papr.py  TX PSD / PAPR / ACLR characterisation
              validation/      One-off validation & audit scripts

plots/      Figure and table generation (read data/, write figures/ and data/tables/)
data/       Evaluation results (JSON) + LaTeX table rows + run logs
figures/    Generated figures
report/     The technical report (LaTeX + PDF)          [not tracked]
results/    Legacy April uncoded-suite results          [kept for reference]
archive/    Superseded scripts and internal history     [not tracked]
third_party/  External reference code and comparisons   [not tracked]
```

## Quick start

```bash
pip install numpy scipy matplotlib
```

Smoke test the coded chain (~1 min):

```bash
python -c "import sys; sys.path[:0]=['eval','src']; from coded_sweep import setup; ctx=setup(dict(tag='NB',M=256,n_act=156,SCS=30e3,N=14),2,quiet=True); print('OK', ctx['names'], 'E =', ctx['E'])"
```

## Reproducing the report

The evaluated configuration is fixed in the drivers: NB = M 256 / 156 active
subcarriers / SCS 30 kHz, WB = M 1024 / 624 / 60 kHz, N = 14, QPSK, NR-LDPC
rate 1/2, iso-block codewords, CSI and noise self-estimated.

**1. Coded sensitivity sweeps** (the report's §12 tables and figures;
NB ≈ 1–2 h, WB ≈ 4–6 h on 8 cores):

```bash
python eval/coded_sweep.py --bw NB --channels AWGN TDL-A TDL-B TDL-C TDL-D --fds 0 200 400 600 800 1000 --coarse-snrs 0 3 6 9 12 15 18 --anchors 4 8 12 --coarse-frames 20 --nmax 2000 --nmin 200 --errtarget 200 --anchor-frames 300 --bank 40 --maxfine 6 --workers 8 --tag final_NB
```

```bash
python eval/coded_sweep.py --bw WB --channels AWGN TDL-A TDL-B TDL-C TDL-D --fds 0 200 400 600 800 1000 --coarse-snrs 0 3 6 9 12 15 18 --anchors 4 8 12 --coarse-frames 20 --nmax 2000 --nmin 200 --errtarget 200 --anchor-frames 300 --bank 40 --maxfine 6 --workers 8 --tag final_WB
```

**2. Completion passes** (deep-BLER points, anchor fill, anchor-gap
refinement):

```bash
python eval/extend_deep.py NB WB
```

```bash
python eval/fill_pass.py NB WB
```

```bash
python eval/refine_pass.py NB WB
```

**2b. Confidence half-widths** (the ±0.19/±0.29 dB figures quoted in the
report; parametric bootstrap over per-point block-error counts):

```bash
python eval/ci_knee.py NB WB
```

**3. Uncoded reference and dense AWGN anchor:**

```bash
python eval/uncoded_eval.py --bw NB WB --frames 200 --workers 8 --tag uncoded_ref
```

```bash
python eval/awgn_fine.py
```

**4. Estimator ablation** (report Table 3):

```bash
python eval/ablate_ofdm.py NB WB
```

**5. TX characterisation** (PSD / PAPR / ACLR; writes to `results/figures/`,
the report's checked-in copies live under `figures/`):

```bash
python eval/tx_spectrum_papr.py
```

**6. Figures and tables** (into `figures/` and `data/tables/`):

```bash
python plots/plot_coded.py final_NB
```

```bash
python plots/plot_coded.py final_WB
```

```bash
python plots/plot_waterfalls.py final_NB
```

```bash
python plots/plot_waterfalls.py final_WB
```

```bash
python plots/plot_awgn.py
```

```bash
python plots/plot_uncoded.py
```

```bash
python plots/make_tables.py NB WB
```

All drivers are seeded; identical commands reproduce identical numbers.

## Data files

| file | content |
|---|---|
| `data/final_NB.json`, `data/final_WB.json` | coded BER / BLER / sensitivity per (channel, Doppler, SNR, method) |
| `data/uncoded_ref.json` | hard-decision QPSK BER, same chain without the code |
| `data/awgn_fine.json` | dense 0.2 dB AWGN anchor (4000 blocks/point) |
| `data/ablate_ofdm.json` | OFDM estimator ablation knees (variants A–D) |
| `data/tables/*.tex` | LaTeX rows for the report's sensitivity tables |
| `data/logs/` | run logs of the sweeps behind the shipped JSONs (provenance) |

## Legacy uncoded suite

The original six-waveform uncoded BER pipeline (`run_eval.py`,
`merge_results.py`, `plot_results.py`, `otfs_deconv.py`) predates the coded
evaluation and is retained outside this repository; its report-grade output
digests remain under `results/`.  It is not needed to reproduce the report.

## License

Code: BSD 2-Clause — (c) 2026 Panos N. Alevizos (see `LICENSE`).  The
accompanying technical report is licensed separately under CC-BY-4.0 (stated
on its title page).  Joint work with Claude Code (Anthropic): analysis,
simulation-harness implementation, numerical results, mathematical
derivations, and report drafting.
