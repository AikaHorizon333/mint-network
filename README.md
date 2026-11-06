# MINT-Network

Room acoustic equalization research project. Implements and compares classical
signal-processing inverse filters (MINT, pMINT, RMCLS) in MATLAB against
neural-network approaches trained with PyTorch.

The goal in all cases is to design a FIR inverse filter **W** such that

```
H * W ≈ δ(n − τ)
```

where **H** is a measured Room Impulse Response (RIR) and **δ** is a target
impulse at delay τ. The result, called an Equalized IR (EIR), approximates
anechoic playback in the room.

**Source RIR**: `data/source/RIR.wav` — 20 944 samples at 48 kHz (~436 ms, mono).

---

## Quick start

```bash
uv sync          # install Python deps
uv run jupyter lab
```

---

## Repository layout

```
data/
  source/          RIR.wav (input)
  matlab_eir/      EIR WAVs produced by MATLAB scripts
  DL/              EIR WAVs produced by PyTorch experiments
matlab/            MATLAB reference implementations
notebooks/         Jupyter analysis notebooks
notes/             Design notes and background reading
src/               PyTorch training scripts
```

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/ir_analysis.ipynb` | Load RIR, plot time-domain, magnitude spectrum, Hilbert envelope; overlay multiple EIRs for listening comparison |
| `notebooks/mint_analysis.ipynb` | Implement and compare Toeplitz MINT vs FFT-based Tikhonov MINT-2 |

---

## Signal-processing methods (MATLAB)

All MATLAB scripts live in `matlab/` and read `data/source/RIR.wav`. They share
the same filter length `M = N` (RIR length) and use `lsqr` to solve the
least-squares system.

### MINT — Multiple-Input/output INverse-filtering Theorem

> Miyoshi & Kaneda, *IEEE Trans. Acoust. Speech Signal Process.*, 36(2), 1988.

Classic full-RIR inversion. Builds an `N × M` Toeplitz convolution matrix from
H and solves `min ‖H_conv · W − D‖²`. D is a unit impulse at sample τ.
Works well in theory but is sensitive to spectral nulls (ill-conditioned
matrix) and requires good regularization in practice.

| Script | Description |
|---|---|
| `matlab/mint/MINT_EIR.m` | Baseline MINT, τ = 0, no regularization |
| `matlab/mint/MINT_TAU_EIR.m` | Tau sweep (τ ∈ {0, N/8, N/4, 3N/8, N/2}) — shows how equalization degrades as the target delay increases |
| `matlab/mint/MINT_TK_TAU_EIR.m` | MINT with Tikhonov regularization (`β·‖W‖²`) and a Wiener circulant preconditioner; τ aligned to direct-sound peak; compares base vs regularized |

### pMINT — Partial MINT

Addresses the ill-conditioning of full MINT by inverting only the **early part**
of the RIR. The convolution matrix is built from a windowed copy `H_w` (direct
sound + first 50 ms of early reflections), making the system better conditioned.
The filter is then verified on the full H. Equivalent to RMCLS with δ = 0.

| Script | Description |
|---|---|
| `matlab/pmint/PMINT_EIR.m` | pMINT, no regularization; rectangular window from onset to ER end (50 ms after peak) |
| `matlab/pmint/PMINT_TK_EIR.m` | pMINT + Tikhonov regularization |

### RMCLS — Relaxed Multichannel Least-Squares

> Kodrasi, Goetze & Doclo, *IEEE Trans. Signal Process.*, 61(23), 2013.

Generalizes pMINT by working with the **full** convolution matrix but applying
a diagonal weight matrix Λ: rows inside the ER window have weight 1 (hard
constraint), rows in the tail have weight √δ (soft relaxation):

```
J_RMCLS(W) = ‖Λ·(H_conv·W − D)‖²

δ = 0  →  pMINT   (tail completely ignored)
δ = 1  →  MINT    (tail and window weighted equally)
```

| Script | Description |
|---|---|
| `matlab/rmcls/RMCLS_EIR.m` | RMCLS, δ = 0, no regularization |
| `matlab/rmcls/RMCLS_TK_EIR.m` | RMCLS + Tikhonov regularization |
| `matlab/rmcls/RMCLS_DELTA_EIR.m` | Delta sweep (δ ∈ {0, 0.25, 0.5, 0.75, 1}) — shows the pMINT→MINT continuum |

### Comparisons

| Script | Description |
|---|---|
| `matlab/comparison/MINT_PMINT_RMCLS_comparison.m` | Side-by-side overlay: MINT, pMINT, RMCLS(δ=0) — same RIR, same M, same τ, no regularization |
| `matlab/comparison/MINT_PMINT_RMCLS_TK_comparison.m` | Same comparison with Tikhonov regularization added to all three methods |

---

## Neural-network experiments (PyTorch)

Scripts live in `src/` and are self-contained Google Colab notebooks (`.py`
files organized as cells). Each script trains a network to directly regress the
inverse filter taps.

File naming convention: `cnn_<arch>_<optimizer>_<scheduler>[_variant].py`

### Architectures

#### TinyCNN1x1 (`src/tiny/`)

> Baseline from Pepe et al., *Applied Sciences* 10(7), 2020.

Three `Conv1d` layers with kernel size 1 (point-wise feature extractor) followed
by two fully-connected layers. Outputs `half = (T+1)//2` taps that are mirrored
to produce a linear-phase symmetric filter of length T = 1025.

Small, fast, but limited: receptive field is ~25 samples out of 20 944,
meaning the time-domain features are essentially local statistics of H
rather than global structure.

#### MultiScaleCNN (`src/multiscale/`)

Architecture v4 — designed to address the receptive-field bottleneck.

**Time branch** — two sequential dilation cycles, each consisting of 10
`DilatedResBlock` modules (dilations 1, 2, 4, …, 512). After each cycle an
`AttentionPool` module computes a learned weighted average over the time axis,
yielding a fixed-size vector. The two pooled vectors (t1, t2) capture features
at different temporal scales.

**Frequency branch** — a strided `Conv1d` encoder applied to the
log-magnitude spectrum of H; hierarchically extracts spectral patterns with
~30× fewer parameters than a plain linear layer.

**Head** — `cat(t1, t2, freq_features)` → FC(256→512) → GELU → FC(512→512)
→ GELU → FC(512→T). Near-zero init on the output layer prevents large loss
spikes at initialization.

Activations: GELU throughout. Training uses a 1000-iteration linear LR warmup.

#### DriftCNN (`src/drift/`)

Similar depth to MultiScaleCNN but with a different aggregation strategy and
an explicit **window conditioning** signal.

**Time branch** — 20 `DilatedResBlock` modules (two cycles of dilations
1,2,…,512) followed by global average pool (no attention).

**Frequency branch** — a single linear layer mapping the log-magnitude
spectrum to a 128-dimensional vector.

**Window conditioning** — a 3-element vector `[peak/L, pre_peak/L, er_end/L]`
encoding the direct-sound peak, onset, and early-reflection boundary (all
normalized to [0,1]). This injects RMCLS window geometry directly into the
network.

**Head** — `cat(time, freq, cond)` → FC(→256) → ReLU → FC(256→half).
Activations: ReLU throughout. Outputs half-taps, symmetrized into a linear-
phase filter.

---

### Loss functions

#### SELF — Spectral Euclidean Loss Function

```
J = w · Σ_f  (|H(f)·G(f)| − 1)²        w = 100
```

Minimizes the deviation of the EIR magnitude spectrum from unity (0 dB) at
every frequency bin in [20 Hz, fs/2]. All bins are weighted equally.
Reference: Pepe et al. 2020.

#### LFWLF v1 — Log-Frequency Weighted Loss Function

```
w_f = (1/f) · (1/|H(f)|),  normalized
J   = w · Σ_f  w_f · (|H(f)·G(f)| − 1)²
```

Extension of SELF with per-bin weights that emphasize low frequencies and
spectral nulls. The `1/f` factor mimics perceptual (log-frequency) resolution;
the `1/|H(f)|` factor increases the gradient at frequencies where H is small
(i.e., where equalization is hardest).

#### LFWLF v2 — Log-Magnitude Squared + Improved LF Weighting

```
error_f = 20·log10(|H(f)·G(f)|)           (dB, symmetric around 0 dB)
w_f     = 1/f²  below 500 Hz,  1/f·1/|H(f)|  above
J       = w · Σ_f  w_f · error_f²
```

Experimental variant that computes the error in decibels (symmetric around
the 0 dB target) and uses stronger low-frequency weighting. **Failed in
practice**: the joint renormalization after mixing `1/f²` and `1/f·1/|H(f)|`
collapsed the mid/high-frequency gradient, causing a +10 dB DC drift.

#### LFWLF v3 — Additive LF Log-Magnitude Term

```
J = SELF_v1  +  lf_boost · Σ_{f < 200 Hz} (20·log10(|H(f)·G(f)|))²
```

Fixes v2 by keeping the v1 base loss intact (mid/HF gradient preserved) and
adding an independent log-magnitude penalty only below 200 Hz. A single
`lf_boost` knob controls the trade-off.

#### LFWLF v3.1 — v3 with dB floor

Same as v3, but the log inside the LF term is clamped to −30 dB:

```
LF term: lf_boost · Σ_{f < 200 Hz} clamp(20·log10(|H·G|), min=−30)²
```

Prevents room-mode nulls from producing explosive gradients
(log(1e-8) = −160 dB → loss spike ~1e8 per bin).

#### RMCLS Loss — Time-Domain Weighted Least-Squares

```
J(g) = ‖ λ · (H*g − D) ‖²

λ_i = 1         for i ∈ [pre_peak, er_end]   (direct + early reflections)
λ_i = √δ        otherwise                     (late-reverb relaxation)
```

Converts the RMCLS objective from the MATLAB domain into a differentiable
PyTorch loss. δ is either fixed or annealed via a **curriculum schedule**
(δ: 0.01 → 0.05 → 0.10 across training iterations) to gradually introduce
the tail constraint.

---

### Optimizers and schedulers

| Token | Description |
|---|---|
| `adam` | Adam (lr = 1e-3) |
| `adamw` | AdamW (lr = 1e-3, weight_decay = 1e-4) |
| `setlr` | Inverse-time decay: `lr_t = lr / (1 + 3e-8 · t)` |
| `coslr` | Cosine annealing |
| `coswr` | Cosine annealing with warm restarts |
| `plateau` | ReduceLROnPlateau (factor 0.5, patience 8 000 iters) |

---

### Experiment index

The directory tree under `src/` follows:

```
src/<arch>/<loss>/<loss_version>/<optimizer>/  script.py
```

#### TinyCNN (`src/tiny/`)

| Path | Loss | Optimizer | Scheduler | Notes |
|---|---|---|---|---|
| `SELF/Adam/cnn_t_adam_setlr.py` | SELF | Adam | set_lr | Baseline (Pepe 2020) |
| `SELF/Adam/cnn_t_adam_coslr.py` | SELF | Adam | cosine | |
| `SELF/Adam/cnn_t_adam_coswr.py` | SELF | Adam | cos+WR | |
| `SELF/Adam/cnn_t_adam_plateau.py` | SELF | Adam | plateau | |
| `SELF/Adam/cnn_t2x_adam_setlr.py` | SELF | Adam | set_lr | 2× filter taps (T=2049) |
| `SELF/AdamW/cnn_t_adamw_setlr.py` | SELF | AdamW | set_lr | |
| `SELF/AdamW/cnn_t_adamw_coslr.py` | SELF | AdamW | cosine | |
| `SELF/AdamW/cnn_t_adamw_coswr.py` | SELF | AdamW | cos+WR | |
| `SELF/AdamW/cnn_t_adamw_plateau.py` | SELF | AdamW | plateau | |
| `LFWLF/v1/Adam/cnn_t_lfwlf_adam_setlr.py` | LFWLF v1 | Adam | set_lr | Log-freq + notch weights |
| `LFWLF/v1/Adam/cnn_t_lfwlf_adam_coslr.py` | LFWLF v1 | Adam | cosine | |
| `LFWLF/v1/Adam/cnn_t_lfwlf_adam_coswr.py` | LFWLF v1 | Adam | cos+WR | |
| `LFWLF/v1/Adam/cnn_t_lfwlf_adam_plateau.py` | LFWLF v1 | Adam | plateau | |
| `LFWLF/v1/AdamW/cnn_t_lfwlf_adamw_setlr.py` | LFWLF v1 | AdamW | set_lr | |
| `LFWLF/v1/AdamW/cnn_t_lfwlf_adamw_coslr.py` | LFWLF v1 | AdamW | cosine | |
| `LFWLF/v1/AdamW/cnn_t_lfwlf_adamw_coswr.py` | LFWLF v1 | AdamW | cos+WR | |
| `LFWLF/v1/AdamW/cnn_t_lfwlf_adamw_plateau.py` | LFWLF v1 | AdamW | plateau | |
| `LFWLF/v2/Adam/cnn_t_lfwlf_v2_adam_setlr.py` | LFWLF v2 | Adam | set_lr | dB-error + 1/f²; unstable |
| `LFWLF/v2/Adam/cnn_t_lfwlf_v2_adam_coslr.py` | LFWLF v2 | Adam | cosine | |
| `LFWLF/v2/Adam/cnn_t_lfwlf_v2_adam_coswr.py` | LFWLF v2 | Adam | cos+WR | |
| `LFWLF/v2/Adam/cnn_t_lfwlf_v2_adam_plateau.py` | LFWLF v2 | Adam | plateau | |
| `LFWLF/v2/AdamW/cnn_t_lfwlf_v2_adamw_setlr.py` | LFWLF v2 | AdamW | set_lr | |
| `LFWLF/v2/AdamW/cnn_t_lfwlf_v2_adamw_coslr.py` | LFWLF v2 | AdamW | cosine | |
| `LFWLF/v2/AdamW/cnn_t_lfwlf_v2_adamw_coswr.py` | LFWLF v2 | AdamW | cos+WR | |
| `LFWLF/v2/AdamW/cnn_t_lfwlf_v2_adamw_plateau.py` | LFWLF v2 | AdamW | plateau | |
| `LFWLF/v3/Adam/cnn_t_lfwlf_v3_adam_setlr.py` | LFWLF v3 | Adam | set_lr | Additive LF dB term |
| `LFWLF/v3/Adam/cnn_t_lfwlf_v3_1_adam_setlr.py` | LFWLF v3.1 | Adam | set_lr | v3 + dB floor at −30 dB |
| `RMCLS/Adam/cnn_t_rmcls_adam_base.py` | RMCLS | Adam | — | Time-domain weighted LS; original training method |

#### MultiScaleCNN (`src/multiscale/`)

| Path | Loss | Optimizer | Scheduler | Notes |
|---|---|---|---|---|
| `SELF/Adam/cnn_ms_adam_setlr.py` | SELF | Adam | set_lr | |
| `SELF/Adam/cnn_ms_adam_setlr_2xt.py` | SELF | Adam | set_lr | 2× filter taps (T=2049) |
| `SELF/Adam/cnn_ms_adam_coslr.py` | SELF | Adam | cosine | |
| `SELF/Adam/cnn_ms_adam_coswr.py` | SELF | Adam | cos+WR | |
| `SELF/Adam/cnn_ms_adam_plateau.py` | SELF | Adam | plateau | |
| `SELF/AdamW/cnn_ms_adamw_setlr.py` | SELF | AdamW | set_lr | |
| `SELF/AdamW/cnn_ms_adamw_coslr.py` | SELF | AdamW | cosine | |
| `SELF/AdamW/cnn_ms_adamw_coswr.py` | SELF | AdamW | cos+WR | |
| `SELF/AdamW/cnn_ms_adamw_plateau.py` | SELF | AdamW | plateau | |
| `LFWLF/v1/Adam/cnn_ms_lfwlf_adam_setlr.py` | LFWLF v1 | Adam | set_lr | |
| `LFWLF/v1/Adam/cnn_ms_lfwlf_adam_coslr.py` | LFWLF v1 | Adam | cosine | |
| `LFWLF/v1/Adam/cnn_ms_lfwlf_adam_coswr.py` | LFWLF v1 | Adam | cos+WR | |
| `LFWLF/v1/Adam/cnn_ms_lfwlf_adam_plateau.py` | LFWLF v1 | Adam | plateau | |
| `LFWLF/v1/AdamW/cnn_ms_lfwlf_adamw_setlr.py` | LFWLF v1 | AdamW | set_lr | |
| `LFWLF/v1/AdamW/cnn_ms_lfwlf_adamw_coslr.py` | LFWLF v1 | AdamW | cosine | |
| `LFWLF/v1/AdamW/cnn_ms_lfwlf_adamw_coswr.py` | LFWLF v1 | AdamW | cos+WR | |
| `LFWLF/v1/AdamW/cnn_ms_lfwlf_adamw_plateau.py` | LFWLF v1 | AdamW | plateau | |
| `LFWLF/v3/Adam/cnn_ms_lfwlf_v3_adam_setlr.py` | LFWLF v3 | Adam | set_lr | Additive LF dB term |
| `cnn.py` | — | — | — | Standalone MultiScaleCNN v4 prototype (AdamW + plateau) |

#### DriftCNN (`src/drift/`)

| Path | Loss | Optimizer | Scheduler | Notes |
|---|---|---|---|---|
| `SELF/Adam/cnn_d_adam_setlr.py` | SELF | Adam | set_lr | Window-conditioned |
| `RMCLS/Adam/cnn_d_rmcls_adam_setlr.py` | RMCLS δ=0.1 | Adam | set_lr | Fixed delta |
| `RMCLS/Adam/cnn_d_rmcls_adam_plateau.py` | RMCLS δ=0.1 | Adam | plateau | |
| `RMCLS/Adam/cnn_d_rmcls_adam_setlr_curriculum.py` | RMCLS curriculum | Adam | set_lr | δ: 0.01→0.05→0.10 |
| `RMCLS/Adam/cnn_d_rmcls_adam_plateau_curriculum.py` | RMCLS curriculum | Adam | plateau | |
| `RMCLS/AdamW/cnn_d_rmcls_adamw_setlr.py` | RMCLS δ=0.1 | AdamW | set_lr | |
| `RMCLS/AdamW/cnn_d_rmcls_adamw_plateau.py` | RMCLS δ=0.1 | AdamW | plateau | |
| `RMCLS/AdamW/cnn_d_rmcls_adamw_setlr_curriculum.py` | RMCLS curriculum | AdamW | set_lr | |
| `RMCLS/AdamW/cnn_d_rmcls_adamw_plateau_curriculum.py` | RMCLS curriculum | AdamW | plateau | |
| `LFWLF/v3/Adam/cnn_d_lfwlf_v3_adam_setlr.py` | LFWLF v3 | Adam | set_lr | Window-conditioned + LF dB term |

---

## Key concepts

**Filter length M**: Set to N (RIR length = 20 944 samples) by convention.
Longer filters improve frequency resolution but increase compute and the risk
of overfitting to spectral nulls. See `notes/TAPS.md`.

**Regularization (β)**: Tikhonov term `β·‖W‖²` prevents blow-up at spectral
nulls. Typical β = `1e-9 × max|H(f)|²`; increasing β trades equalization
accuracy for stability.

**Direct arrival index n₀**: First sample where `|h[n]| ≥ 1% × peak`.
Used in pMINT/RMCLS windowing to locate the onset of the direct sound.

**FFT padding**: Linear convolution of length `N + M − 1` is zero-padded to
the next power-of-2 for `rfft` efficiency.

---

## References

- Miyoshi & Kaneda (1988) — MINT: *IEEE Trans. Acoust. Speech Signal Process.*, 36(2).
- Kodrasi, Goetze & Doclo (2013) — RMCLS: *IEEE Trans. Signal Process.*, 61(23).
- Pepe et al. (2020) — TinyCNN1x1 baseline: *Applied Sciences*, 10(7), 2483.
