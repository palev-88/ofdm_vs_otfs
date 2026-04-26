# OFDM vs OTFS — Python Reference Implementation

A SISO link-level simulator for comparing OFDM against five OTFS
variants under 3GPP TDL channels with estimated CSI, estimated noise
power, and a calibrated per-RE SNR convention.

This repository is the reference implementation that backs the
technical report *"OFDM vs OTFS Waveform Comparison — A Practical
Assessment Under Estimated CSI, Calibrated Noise, and 3GPP TDL
Channels for High-Mobility Wireless Communications"* (April 2026,
v3.1). Every equation in the report maps to a Python function here;
every BER number in the report is produced by `run_eval.py`.

---

## Waveform architectures

Six transmitter + receiver chains are implemented, each in its own
module:

| Module            | Waveform        | Pilot               | Equaliser                       |
|-------------------|-----------------|---------------------|---------------------------------|
| `ofdm.py`         | 5G-NR OFDM      | DMRS (comb-2, 2 syms) | per-SC MMSE                   |
| `otfs.py`         | CP-OTFS         | DD impulse + guard  | cross-domain iterative LMMSE   |
| `otfs_zp.py`      | ZP-OTFS         | DD impulse + guard  | time-domain sparse LMMSE       |
| `otfs_pcp.py`     | PCP-OTFS (two variants: `orig`, `guard`) | Zadoff–Chu pilot column + delay CP | two-stage ChEst (ZC correlation + GCE-BEM) + FDE |
| `otfs_mc.py`      | MC-OTFS         | DD impulse + guard  | per-SC MMSE in TF (precoded OFDM) |
| `otfs_deconv.py`  | Deconv-OTFS     | DD impulse + guard  | Υ-basis fractional-Doppler estimator + deconvolutional Wiener |

All six share: 3GPP TDL channel, Jakes fading, Farrow fractional-delay
filter, QAM modulation, calibrated per-data-RE SNR convention, and
internal (non-oracle) channel + noise variance estimation.

---

## Quick start

### Install

```bash
pip install numpy scipy matplotlib
```

Tested on Python 3.10+. No GPU required.

### One-liner sanity check (≈ 2 min)

Smoke-test that your install runs end-to-end on a tiny MC budget:

```powershell
# Windows PowerShell
cd C:\path\to\ofdm_vs_otfs
python run_eval.py 256 30 5 TDL-A low
```

Produces `results/ber_M256_SCS30k_TDL-A.json` — a 5-trial BER digest
covering one channel and a narrow Doppler range. Useful only for
"does the pipeline run", not for publishable BER.

---

## How to run each script

### `run_eval.py` — BER evaluation sweep

```text
python run_eval.py <M> <SCS_kHz> [NMC] [channel] [FD_RANGE] [N_ACT]
```

| Arg | Required | Default | Allowed values |
|---|---|---|---|
| `M`        | yes | — | `64`, `128`, `256`, `512`, `1024` (FFT size) |
| `SCS_kHz`  | yes | — | `15`, `30`, `60` (subcarrier spacing in kHz) |
| `NMC`      | no  | `20` | any positive int (Monte-Carlo trials) |
| `channel`  | no  | `ALL` | `TDL-A`, `TDL-B`, `TDL-C`, `TDL-D`, `TDL-E`, `AWGN`, `ALL` |
| `FD_RANGE` | no  | `full` | `low` (0,100,200 Hz), `mid` (300-600), `high` (700-1000), `full` (0-1000 in 100-Hz steps) |
| `N_ACT`    | no  | `≈ 0.61·M` | any int ≤ M − 1 (active subcarriers, 3GPP-PRB aligned) |

**Output**: `results/ber_M<M>_SCS<SCS>k_<channel>.json` — full BER hypercube (method × Doppler × SNR).

**Examples**

```powershell
# Smoke test: 5 trials, single channel, low Doppler              (~2 min)
python run_eval.py 256 30 5 TDL-A low

# Sanity-check NB anchor at default 20 MC across all channels    (~10-15 min)
python run_eval.py 256 30

# Full NB sweep matching the report (NMC=200, ~3-5 hours)
python run_eval.py 256 30 200 ALL full 156

# Full WB sweep matching the report (NMC=200, ~6-12 hours)
python run_eval.py 1024 60 200 ALL full 624

# AWGN-only calibration check, fast
python run_eval.py 256 30 50 AWGN low

# Stress-test fractional Doppler (just the high-Doppler tail)
python run_eval.py 256 30 50 TDL-D high

# Wideband narrow-region debug (LOS path, mid Doppler, 30 MC)
python run_eval.py 1024 60 30 TDL-D mid

# Custom narrowband: M=128, SCS=60 kHz, n_act override to 78
python run_eval.py 128 60 50 ALL full 78
```

### `merge_results.py` — Aggregate per-config JSONs into the plotter's input

`run_eval.py` writes one JSON per `(M, SCS, channel)` slice;
`plot_results.py` consumes a single aggregated hypercube
(`results/otfs_eval_v1.json`). This script bridges the two — walks
every `results/ber_M*_SCS*_*.json` file, groups by bandwidth, normalises
the schema, and writes `results/otfs_eval_v1.json`.

```text
python merge_results.py
```

**No arguments**. Reads everything matching `results/ber_M*_SCS*_*.json`.

**Bandwidth labels** (configurable inside the script via `BW_MAP`):

| `(M, SCS_kHz)` | Label |
|---|---|
| `(256, 30)`  | `NB` |
| `(1024, 60)` | `WB` |
| anything else | `OTHER:M<M>_SCS<SCS>k` (still emitted, plotter will skip) |

**Method-name normalisation** (via `METHOD_MAP`):

| run_eval method | plot_results label(s) |
|---|---|
| `ofdm`        | `ofdm` |
| `zp`          | `zp` |
| `pcp_guard`   | `pcp_guard` |
| `pcp_orig`    | `pcp_orig` |
| `mc`          | (excluded from main figures) |
| `deconv`      | (excluded from main figures) |

Run this between every `run_eval.py` and `plot_results.py` pair.

---

### `plot_results.py` — Generate report-grade figures + CSV digests

```text
python plot_results.py
```

**No arguments**. Reads `results/otfs_eval_v1.json` (the JSON written
by the final-evaluation sweep — different from the per-config
`ber_M*_SCS*_*.json` files written by `run_eval.py` in this checkout).
Outputs land in `results/figures/`.

> **Note**: this plotter targets the *aggregated* `otfs_eval_v1.json`
> format. To produce that, you need to merge the per-channel BER
> digests into a single hypercube. The merge step is part of the
> final-eval pipeline; ad-hoc smoke-test runs from `run_eval.py`
> produce per-config JSONs which this plotter does **not** consume
> directly. (For ad-hoc plots, write your own loader; the older
> simple-plot script is in `sandbox/old_scripts/`.)

**Outputs**

| File | Description |
|---|---|
| `fig_eval_AWGN.png`         | NB and WB AWGN waterfalls side-by-side |
| `fig_eval_BERvsSNR_NB.png`  | 4 channels × 4 Dopplers grid, NB |
| `fig_eval_BERvsSNR_WB.png`  | same grid, WB |
| `fig_eval_BERvsFD_NB.png`   | BER vs Doppler at SNR ∈ {20, 30} dB, NB |
| `fig_eval_BERvsFD_WB.png`   | same, WB |
| `fig_eval_winner_heatmap.png` | per-cell method winner at SNR=20 dB, NB+WB |
| `tab_eval_BERat20dB.csv`    | BER at SNR=20 dB per (bw, channel, fd, method) |
| `tab_eval_SNRforBER.csv`    | SNR (dB) needed for BER=10⁻³ |
| `tab_eval_winner.csv`       | winner method per (bw, channel, fd) |

### `tx_spectrum_papr.py` — TX waveform characterisation

```text
python tx_spectrum_papr.py
```

**No arguments**. Synthesises TX waveforms for every method
(OFDM / CP-OTFS / ZP-OTFS / PCP-orig / PCP-guard / MC-OTFS), passes
each through a 4× upsampler + 3GPP-compliant TX filter, and measures:

- **PSD** via Welch periodogram with Blackman-Harris window
- **PAPR** complementary CDF (full distribution, 0.01 % tail = headline figure)
- **ACLR** (in-band vs adjacent-channel power ratio, dB)

**Outputs** in `results/figures/`:

| File | Description |
|---|---|
| `fig_tx_spectrum_NB-A.png`    | All 5 methods overlaid, narrowband |
| `fig_tx_spectrum_WB-B.png`    | Wideband counterpart |
| `fig_tx_papr_ccdf_NB-A.png`   | PAPR CCDF per method, narrowband |
| `fig_tx_papr_ccdf_WB-B.png`   | Wideband counterpart |

Also prints PAPR@0.01 % and ACLR numbers per method to stdout.

> **Heads-up**: the script mocks `scipy.sparse` *after* importing
> `scipy.signal` so it can run without the sparse-solver dependency
> (the OTFS modules import `scipy.sparse` at module load even though
> this script doesn't exercise the sparse-solver paths).

---

## End-to-end reproduction recipe

To regenerate every report figure from scratch (overnight job):

```powershell
# 1. Full NB sweep (~3-5 hours)
python run_eval.py 256 30 200 ALL full 156

# 2. Full WB sweep (~6-12 hours, can run in parallel terminal)
python run_eval.py 1024 60 200 ALL full 624

# 3. Merge the per-config JSONs into results/otfs_eval_v1.json
python merge_results.py

# 4. Generate the BER figures + CSV digests
python plot_results.py

# 5. Generate the TX waveform figures (independent of BER eval)
python tx_spectrum_papr.py
```

Outputs land in `results/figures/`.

### Smoke-test recipe (~5 minutes)

```powershell
python run_eval.py 256 30 5 AWGN low      # NB AWGN, NMC=5
python run_eval.py 1024 60 5 AWGN low     # WB AWGN, NMC=5
python merge_results.py                     # aggregate
python plot_results.py                      # render fig_eval_AWGN.png + CSVs
python tx_spectrum_papr.py                  # render TX waveform figures
```

Smoke-test outputs only `fig_eval_AWGN.png` (both panels populated) +
the four TX waveform PNGs; the per-channel BER grid figures are skipped
gracefully because no TDL data was generated.

---

## File-by-file documentation

### `__init__.py`

Empty — marks the directory as a Python package so that
`run_eval.py` can `from channel import TDLChannel`, etc. No runtime
code.

---

### `channel.py` — 3GPP TDL channel model

Implements the discrete-time baseband channel per 3GPP TR 38.901 §7.7.2.

**Main objects**

- `TDLChannelConfig(profile, ds_seconds, fd_hz, fs_hz, seed=0)` —
  dataclass. `profile ∈ {TDL-A, TDL-B, TDL-C, TDL-D, TDL-E}`; `ds_seconds`
  = desired delay spread; `fd_hz` = maximum Doppler; `fs_hz` = sampling
  rate.
- `TDLChannel(cfg)` — realises a random multipath channel. Key methods:
  - `apply(x)` → convolves input signal with time-varying channel,
    adds AWGN at the calibrated per-RE SNR convention.
  - `max_delay_samples` → ⌈DS · F_s⌉.
  - Internal: each tap `τᵢ` carries a Jakes-spectrum fading coefficient
    `αᵢ(t)`, generated as a Zheng–Xiao sum-of-sinusoids with 32 oscillators.
    TDL-D/E's first tap additionally carries a deterministic LOS
    component with K-factor 13.3 dB / 22 dB.

**Tap tables** (`TDL_PROFILES` dict at top of file) are reproduced
verbatim from TR 38.901 Tables 7.7.2-1 through 7.7.2-5. Delays are
stored as normalised values; multiply by the desired DS to recover the
actual tap delays.

**Fractional delays** are applied via `FractionalDelayFilter` from
`fdf.py` — no rounding to integer samples.

---

### `fdf.py` — Farrow fractional-delay filter

A modified Farrow-structure FIR filter that interpolates between
integer-sample delays. Ported from KDDI's `FractionalDelayFilter.m`
(arXiv:2010.15396). Uses Lagrange interpolation via Cramer's rule on a
Vandermonde matrix for coefficient computation.

**Main objects**

- `FractionalDelayFilter(filter_order=8)` — constructor.
  - `.init(delays)` precomputes FIR coefficients per tap delay.
  - `.filter(x, tap_idx)` applies the FIR for a specific tap.

Reference: Välimäki & Laakso, *"Fractional Delay Filters — Design and
Applications,"* in *Nonuniform Sampling*, Springer 2001.

---

### `qam.py` — QAM modulation / demodulation

Gray-coded, unit-average-power constellations for QPSK (4-QAM), 16-QAM,
and 64-QAM. Hard-decision demapper for uncoded BER.

**Main functions**

- `qam_constellation(order)` → `(constellation_points, bits_per_symbol)`.
- `qam_modulate(bits, order)` → complex symbols.
- `qam_demodulate(symbols, order)` → bits.
- `generate_random_bits(n_bits, seed=None)` → np.ndarray of 0/1.

Used by every waveform module and by `run_eval.py`.

---

### `ofdm.py` — 5G-NR OFDM transceiver

Full OFDM transmitter and receiver with DMRS-based channel and noise
estimation.

**Main objects**

- `OFDMConfig` dataclass: FFT size, active subcarrier count, SCS, CP
  length, symbols per slot, DMRS positions (comb-2 on symbols 2 and 11,
  per 5G NR TS 38.211).
- `OFDMTransceiver(cfg)`:
  - `.transmit(bits, qam_order)` → time-domain waveform `s[t]`.
  - `.receive(r, qam_order)` → bit estimates + diagnostics (`Ĥ`, `σ̂²`).

**Pipeline**

- TX: bits → QAM → DC-centred subcarrier mapping → IFFT-shift → IFFT → CP
- RX: CP removal → FFT → FFT-shift → channel estimation → MMSE equaliser → demap

**Channel estimation**: least-squares at DMRS positions (symbols 2, 11),
linear frequency interpolation across comb-2 pilot subcarriers, linear
time interpolation between symbols 2 and 11.

**Noise estimation**: decorrelated-DMRS power-difference estimator
(averages `|Z₁|² − |Z₂|²` over pilot subcarriers; the `1/4` normalisation
isolates the noise-variance contribution).

---

### `otfs.py` — CP-OTFS transceiver

OTFS with cyclic prefix per subsymbol and cross-domain iterative
detector.

**Main objects**

- `OTFSConfig` dataclass: FFT size, active subcarrier count, SCS, CP
  length, pilot power boost (dB), 2-D pilot guard (delay × Doppler).
- `OTFSTransceiver(cfg)`:
  - `.transmit(bits)` — ISFFT spreads DD symbols onto TF grid, then
    OFDM-like IFFT+CP produces the time-domain waveform.
  - `.receive(r)` — CP removal + FFT + SFFT to DD, DD pilot estimation
    via parabolic interpolation for fractional Doppler, cross-domain
    iterative LMMSE (TF domain per-subcarrier + SFFT between iterations).

**Key property**: CP-OTFS is bit-compatible with OFDM at the base-station
FFT engine (same IFFT + CP per symbol); the OTFS "uniqueness" comes
from the outer ISFFT.

---

### `otfs_zp.py` — ZP-OTFS transceiver

OTFS with zero-padding (instead of CP) between subsymbols, and a
time-domain sparse LMMSE detector.

**Main objects**

- `ZPOTFSConfig` dataclass: M delay bins, N Doppler bins, SCS, ZP
  length, pilot position in (delay, Doppler), 2-D pilot guard, max DD
  taps retained.
- `ZPOTFSTransceiver(cfg)`:
  - `.transmit(bits)` — DD grid → inverse Zak transform (IDFT across
    Doppler per delay bin) → append ZP → serialise.
  - `.receive(r)` — strip ZP → Zak transform → DD channel estimate from
    pilot region + quadratic Doppler interpolation → build sparse
    banded time-domain channel matrix → `scipy.sparse.linalg.spsolve`
    LMMSE inversion.

**Key property**: the ZP tail is signal-free by construction → provides
a *structurally unbiased* noise-variance estimator (`σ̂²_ZP`), unlike
the CP-based approaches.

---

### `otfs_pcp.py` — PCP-OTFS transceiver (two variants)

OTFS with pilot-CP (Zadoff–Chu sequence on one Doppler column + cyclic
prefix per subsymbol). Two channel-matched CP variants are supported
via the `Mcp` config parameter:

- **`pcp-orig`**: `Mcp = N_CP` (full 5G-NR Normal CP length, matches
  OFDM overhead per subsymbol).
- **`pcp-guard`**: `Mcp = max(⌈10⁻⁶ F_s⌉ + 3, 4)` (channel-matched
  short CP, reduces overhead at the cost of CP rigidity against
  long-delay channels).

**Main objects**

- `PCPOTFSConfig` dataclass: M, N, `Mcp`, SCS, pilot Doppler column
  index, Doppler guard, pilot power (dB), ZC root, BEM order `Q`.
- `PCPOTFSTransceiver(cfg)`:
  - `.transmit(bits)` — ZC pilot length `L = Mcp` placed on one DD
    Doppler column with 1-column guards; inverse Zak then CP per
    subsymbol.
  - `.receive(r)` — CP removal → Stage 1 per-subsymbol LS via ZC
    correlation (fast exploiting ZC flat-spectrum) → Stage 2 GCE-BEM
    temporal smoothing (`Q`-order complex-exponential basis; `Q=1`
    gives ≈ 6.7 dB denoising gain) → FDE with pilot cancellation.

**Noise estimation**: CP-redundancy estimator averages
`|y[l, m'] − y[l, m' + M]|²` over the CP samples — short (one-subsymbol)
observation window keeps the Doppler-induced bias small.

Reference: Sanoopkumar & Farhang, *"A Practical Pilot for Channel
Estimation of OTFS,"* 2023.

---

### `otfs_mc.py` — MC-OTFS transceiver

OTFS as *precoded OFDM* — the TX is identical to CP-OTFS (ISFFT +
IFFT + CP), but the RX performs equalisation per-subcarrier in the
TF domain (scalar MMSE), then returns to DD via SFFT.

**Main objects**

- `MCOTFSConfig` dataclass: same parameters as `OTFSConfig`.
- `MCOTFSTransceiver(cfg)`:
  - `.transmit(bits)` — delegates to the OTFS TX path.
  - `.receive(r)` — CP removal → FFT → SFFT → DD pilot estimation →
    build scalar TF channel `Ĥ_TF[l, n]` → per-subcarrier MMSE → SFFT
    back to DD → demap.

**Key property**: MC-OTFS complexity is `O(NM)` (same as OFDM), not
`O(MN³)` (CP-OTFS cross-domain). The trade-off is a scalar
approximation of the TF channel that breaks down at high Doppler
because off-diagonal Doppler coupling is ignored.

Reference: García Astudillo, *"Performance Comparison of OTFS vs OFDM
Wireless Signals in High-Mobility Environments,"* M.Sc. thesis, UPC,
2026.

---

### `otfs_deconv.py` — Deconv-OTFS transceiver (KDDI method)

CP-OTFS pipeline with the Upsilon-basis fractional-Doppler channel
estimator and a 2-D-FFT deconvolutional Wiener equaliser.

**Main objects**

- `DeconvOTFSConfig` dataclass: adds `dividing_number` (fractional
  Doppler resolution `1/D`, typical `D = 10`), `alpha` (magnitude
  threshold, `1/50`), `beta` (noise-floor threshold, `1/10`).
- `DeconvOTFSTransceiver(cfg)`:
  - `.transmit(bits)` — delegates to `OTFSTransceiver`.
  - `.receive(r)` — OFDM demod → SFFT to DD → Upsilon-basis iterative
    path extraction with dual thresholds → per-delay-bin Wiener via
    2-D FFT.

**Retained in this repo even though the final report excludes
Deconv-OTFS on complexity grounds**; the module is a working reference
implementation of Hashimoto et al., ICC Workshops 2021.

---

### `run_eval.py` — BER evaluation driver

Main evaluation entry point. Iterates Monte-Carlo trials over a fixed
`(M, SCS, channel, Doppler, SNR)` grid and writes BER digests.

```text
Usage: python3 run_eval.py <M> <SCS_kHz> [NMC] [channel] [FD_RANGE] [N_ACT]
  M:        FFT size (64, 128, 256, 512, 1024)
  SCS_kHz:  Subcarrier spacing in kHz (15, 30, 60)
  NMC:      Monte Carlo trials (default 20)
  channel:  TDL-A | TDL-B | TDL-C | TDL-D | TDL-E | AWGN | ALL (default ALL)
  FD_RANGE: low | mid | high | full (default full)
  N_ACT:    Active subcarriers override (default ≈ 0.61·M, 3GPP-like)

Output: results/ber_M<M>_SCS<SCS>k_<channel>.json
```

For each (method × channel × Doppler × SNR) cell, a fresh channel
realisation is drawn from `channel.TDLChannel` with a per-trial seed,
the QAM data is generated, the TX waveform is passed through the
channel, the RX pipeline estimates CSI + noise variance, equalises,
and the BER is recorded. The calibrated noise convention ensures every
data RE experiences exactly the nominal SNR regardless of waveform
loading fraction.

---

### `plot_results.py` — Final-evaluation figures and CSV digests

Generates the report-grade figure set + 3 machine-readable CSV digests
from `results/otfs_eval_v1.json` (the JSON written by the
final-evaluation sweep). Outputs land under `results/figures/`.

```text
Usage: python3 plot_results.py
  Input:  results/otfs_eval_v1.json (full BER hypercube: bw × method × channel × fd × snr)
  Output: results/figures/
```

Figures produced:

| File | Layout | Description |
|---|---|---|
| `fig_eval_AWGN.png`         | NB and WB side-by-side       | AWGN-anchor calibration check |
| `fig_eval_BERvsSNR_NB.png`  | 4 channels × 4 Dopplers grid | BER vs SNR for the four committed methods |
| `fig_eval_BERvsSNR_WB.png`  | same                          | wide-band counterpart |
| `fig_eval_BERvsFD_NB.png`   | 4 channels × 2 SNR rows       | BER vs Doppler at SNR ∈ {20, 30} dB |
| `fig_eval_BERvsFD_WB.png`   | same                          | wide-band counterpart |
| `fig_eval_winner_heatmap.png` | NB and WB heatmaps         | per-(channel, Doppler) cell winner at SNR = 20 dB |

CSV digests:

| File | Columns |
|---|---|
| `tab_eval_BERat20dB.csv`  | `bw, channel, fd, ofdm, zp, pcp_guard, pcp_orig` — BER at SNR = 20 dB per cell |
| `tab_eval_SNRforBER.csv`  | `bw, channel, fd, ofdm, zp, pcp_guard, pcp_orig` — SNR (dB) needed to hit BER = 1e-3, or `> 30` if unreachable |
| `tab_eval_winner.csv`     | `bw, channel, fd, winner` — method with the lowest BER@20 dB per cell |

Uses `matplotlib` with an `Agg` backend (headless). Methods plotted:
`OFDM` (green), `ZP-OTFS` (blue), `PCP-guard` (red), `PCP-orig`
(orange).

---

### `tx_spectrum_papr.py` — TX waveform characterisation (PSD / PAPR / ACLR)

Models a 3GPP-style TX chain and characterises every committed
waveform's spectral and peak-to-average-power properties. Outputs go
to `results/figures/`.

```text
Usage: python tx_spectrum_papr.py
  Input:  none (synthesises TX waveforms internally)
  Output: results/figures/fig_tx_spectrum_NB-A.png
          results/figures/fig_tx_spectrum_WB-B.png
          results/figures/fig_tx_papr_ccdf_NB-A.png
          results/figures/fig_tx_papr_ccdf_WB-B.png
```

**TX chain model**

1. Baseband waveform from each transceiver (OFDM / CP-OTFS / ZP-OTFS / PCP-OTFS / MC-OTFS).
2. 4× upsampling via FFT zero-padding.
3. 3GPP-compliant TX filter: flat passband + raised-cosine roll-off
   in the guard band + zero stopband beyond channel edge (per TS 38.104).
4. PSD via Welch periodogram with a Blackman–Harris window.

**Measurements**

- **PSD overlay**: superimposes all five waveforms at NB-A and WB-B —
  visual proof of equal occupied bandwidth + identical spectral
  roll-off under the common TX filter (i.e., the analog front-end is
  reusable across waveforms).
- **PAPR CCDF**: per-method complementary cumulative distribution of
  peak-to-average power; the 0.01% tail is reported as the headline
  PAPR figure-of-merit.
- **ACLR**: in-band power vs adjacent-channel power ratio, in dB
  (filter-dominated in this report; production ACLR with PA
  non-linearity needs separate spurious-emissions analysis).

**Note**: the script mocks `scipy.sparse` *before* importing the OTFS
modules so it can run without the sparse-solver dependency
(`scipy.sparse` is only needed for the ZP-OTFS receive path, which
this script doesn't exercise).

---

## Reproducing the report's results

The final report's BER tables (`tab_eval_BERat20dB.csv`,
`tab_eval_SNRforBER.csv`, `tab_eval_winner.csv`) were produced by
sweeping:

- **NB case**: `M = 256, SCS = 30 kHz, n_act = 156, F_s = 7.68 MHz`
- **WB case**: `M = 1024, SCS = 60 kHz, n_act = 624, F_s = 61.44 MHz`
- **Channels**: AWGN + TDL-A (30 ns) + TDL-B (100 ns) + TDL-C (300 ns) + TDL-D (30 ns Rician)
- **Dopplers**: 0, 100, 200, …, 1000 Hz (11 points)
- **SNR**: 0 to 30 dB in 2.5 dB steps (13 points)
- **Monte-Carlo trials**: `NMC = 200`

All six architectures were evaluated; the final report reports the
four that pass the complexity filter (OFDM, ZP-OTFS, PCP-orig,
PCP-guard). CP-OTFS, MC-OTFS and Deconv-OTFS are documented in the
report and implemented in this repository but excluded from the final
BER plots on complexity / error-floor grounds.

---

## License

CC-BY-4.0. See `LICENSE`.

---

## Citation

If you use this code, please cite:

```bibtex
@techreport{alevizos2026_ofdmvsotfs,
  author      = {Alevizos, Panos N.},
  title       = {OFDM vs OTFS Waveform Comparison: A Practical Assessment
                 Under Estimated CSI, Calibrated Noise, and 3GPP TDL
                 Channels for High-Mobility Wireless Communications},
  institution = {Zenodo},
  year        = {2026},
  note        = {v3.1. Joint work with Claude Code (Anthropic).}
}
```
