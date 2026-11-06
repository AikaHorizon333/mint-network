# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Room acoustic equalization research project implementing the **MINT** (Multiple-Input/output INverse-filtering Theorem) and related methods to design inverse filters that equalize a measured Room Impulse Response (RIR) into an approximation of a unit impulse.

The RIR (`data/RIR.wav`) is 20944 samples at 48 kHz (~436 ms). Generated Equalized IRs (EIRs) are saved back to `data/` as WAV files.

## Environment & Dependencies

- **Python 3.12**, managed with **`uv`**
- **Virtual environment**: `.venv/`
- **Dependencies** (from `pyproject.toml`): `jupyter`, `librosa`, `matplotlib`, `scipy`

## Commands

```bash
# Install dependencies
uv sync

# Start Jupyter Lab
uv run jupyter lab

# Execute a notebook non-interactively
uv run jupyter nbconvert --to notebook --execute notebooks/ir_analysis.ipynb --output notebooks/ir_analysis.ipynb
```

## Repository Structure

```
data/           # WAV files: RIR.wav (source), *-EIR.wav (equalized outputs), weights/
matlab/         # Reference MATLAB implementations (MINT_EIR.m, PMINT_EIR.m)
notebooks/      # Jupyter notebooks (primary workbench)
notes/          # Reference notes (e.g. TAPS.md on FIR filter design)
src/            # Python modules (currently empty, reserved for future code)
```

## Notebooks

- **`notebooks/ir_analysis.ipynb`** — RIR analysis and multi-EIR comparison: loads `data/RIR.wav`, plots time-domain waveform, frequency response (magnitude spectrum), Hilbert envelope/decay, and overlays multiple EIRs for comparison.

- **`notebooks/mint_analysis.ipynb`** — MINT filter design: implements and compares two inverse-filter approaches:
  - **MINT** (referenced): Toeplitz convolution matrix + `lsqr` least-squares (O(N³), high accuracy)
  - **MINT-2**: FFT-based Tikhonov-regularized solution `W(f) = H*(f) / (|H(f)|² + β)` (O(N log N), trades accuracy for speed)

## MATLAB Reference Scripts

- **`matlab/MINT_EIR.m`** — Canonical MINT implementation: builds a Toeplitz convolution matrix from H and solves `H_conv * W = D` with `lsqr`.
- **`matlab/PMINT_EIR.m`** — Partial MINT (pMINT): windows the RIR to isolate direct sound + early reflections (configurable `early_reflection_ms`), designs the inverse filter against the windowed RIR, then verifies against the full RIR. Saves `data/pMINT-EIR.wav`.

## Key Signal Processing Concepts

- **MINT inverse filter**: `H * W ≈ δ`. Solved as least-squares: `min_W ‖H_conv·W − D‖²`.
- **Regularization**: Tikhonov term `β·‖W‖²` prevents blow-up at spectral nulls. Typical β = `1e-9 × max|H(f)|²`; increasing β trades equalization accuracy for stability.
- **Filter length M**: Set to `N` (= RIR length) by convention. Longer M improves frequency resolution but increases compute. See `notes/TAPS.md` for latency/resolution trade-offs at 48 kHz.
- **FFT padding**: Linear convolution of length `N + M − 1` is zero-padded to the next power-of-2 for `rfft` speed.
- **Direct arrival index `n0`**: First sample where `|h[n]| ≥ 1% × peak`. Used in pMINT windowing.

## Data Files

| File | Description |
|------|-------------|
| `data/RIR.wav` | Source room impulse response (48 kHz, mono, ~436 ms) |
| `data/MINT-EIR.wav` | EIR from Toeplitz MINT (MATLAB reference) |
| `data/MINT-2-EIR.wav` | EIR from FFT-based MINT-2 (Python notebook) |
| `data/pMINT-EIR.wav` | EIR from Partial MINT (MATLAB) |
| `data/CNN1x1_EIR.wav` | EIR from CNN-based method |
| `data/weights/` | Saved model weights |
