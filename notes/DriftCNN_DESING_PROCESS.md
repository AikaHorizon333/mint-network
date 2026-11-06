# CNN Architecture Design Process
## Single-Source RIR Equalization with RMCLS Loss

---

## 1. Starting Point — Paper Replication (TinyCNN1x1)

The baseline comes from **Pepe et al. (2020)** — *"Designing Audio Equalization Filters by Deep Neural Networks"*, Applied Sciences 10(7), 2483.

The paper trains a CNN to output FIR filter coefficients that equalize the frequency response of a room impulse response (RIR). Key design decisions from the paper:

| Parameter | Value |
|-----------|-------|
| Filter length T | 1025 taps (odd, linear phase) |
| Optimizer | Adam, lr=1e-3, decay=3e-8 |
| Iterations | 200,000 |
| Loss | Spectral Euclidean distance, w=100 |
| Desired response | Flat magnitude over [20 Hz .. Nyquist] |

The **loss function** is frequency-domain:
```
J = w * sum_f ( |H(f) * G(f)| - 1 )²
```
where `H(f)` is the RIR spectrum, `G(f)` is the filter spectrum, and the target is flat (= 1).

### TinyCNN1x1 Architecture

```
Input: h [L]
  → Conv1d(1→16, k=9)  → ReLU
  → Conv1d(16→32, k=9) → ReLU
  → Conv1d(32→16, k=9) → ReLU
  → AdaptiveAvgPool1d(256)
  → Flatten → FC(4096→256) → ReLU
  → FC(256→half)
Output: half_taps [half=513]  →  symmetrize()  →  g [T=1025]
```

This works with the spectral loss because the frequency-domain loss only requires the network to learn the spectral envelope — even a small receptive field can capture that through global average pooling.

---

## 2. First Problem — RMCLS Loss Broke TinyCNN1x1

The RMCLS (Relaxed Multi-Channel Least-Squares) loss is a **time-domain** alternative:
```
J(g) = || lambda_vec * (H * g - D) ||²
```

where:
- `D` is a target impulse at the direct-sound peak index
- `lambda_vec = 1` inside the early reflection window `[pre_peak .. er_end]`
- `lambda_vec = sqrt(delta)` outside the window (late reverb relaxation)

The window is detected from the RIR:
- `peak_idx` = `argmax(|h|)`
- `pre_peak_idx` = first sample exceeding `onset_thresh × peak`
- `er_end_idx` = `peak_idx + early_reflection_ms × fs`

### Why RMCLS failed with TinyCNN1x1

| | Spectral loss | RMCLS |
|---|---|---|
| Domain | Frequency | Time |
| What the network needs to learn | Spectral envelope (global) | Time-domain structure of full RIR |
| Receptive field needed | Small (global pool captures it) | Large — needs to "see" direct sound + reverb tail |
| TinyCNN1x1 receptive field | ~25 samples | ~25 samples |
| RIR length | 20,944 samples | 20,944 samples |

**Result**: TinyCNN1x1 + RMCLS collapsed to the trivial solution (near-zero filter taps).

The trivial solution loss: if `g ≈ 0`, then `H*g ≈ 0` and:
```
J = sum(lambda_vec * D)² = lambda_vec[peak_idx]² * 1 = 1.0
```
Training logs confirmed the loss never escaped ~0.926 (just below 1.0).

---

## 3. Architecture Redesign — DriftCNN

Three improvements were added to address the receptive field problem:

### 3.1 Dilated Residual Convolutions (Time Branch)

Standard convolutions with kernel size 9 and 3 layers give a receptive field of ~25 samples. The RIR is 20,944 samples (~436 ms at 48 kHz). The network needs to see at least the direct sound + early reflections window (~50–85 ms = 2400–4000 samples).

**Dilated convolutions** insert gaps between kernel elements controlled by a `dilation` parameter:

```
dilation=1:  [x . x . x]         RF = 3
dilation=2:  [x . . x . . x]     RF = 5
dilation=4:  [x . . . x . . . x] RF = 9
...
dilation=512:                     RF = 1025
```

Stacking one cycle `[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]` gives:
```
RF = 2 * sum(dilations) + 1 = 2 * 1023 + 1 = 2047 samples  (~42 ms at 48 kHz)
```

Using **two cycles** doubles coverage to ~4094 samples (~85 ms) — enough to cover the direct sound and early reflections window used by RMCLS.

**Residual connections** stabilize training when stacking 20 blocks:
```python
def forward(self, x):
    return x + F.relu(GroupNorm(Conv1d(x)))  # residual connection
```

GroupNorm is used instead of BatchNorm because batch size is 1 (single RIR at a time).

### 3.2 Frequency Branch (Dual-Path)

Even with dilated convolutions, spectral nulls (deep notches in `|H(f)|`) are hard to detect in the time domain alone. A second branch processes the **log-magnitude spectrum** directly:

```
H_logmag = log(|rfft(h)| + ε)   [FREQ_IN = L//2 + 1 bins]
  → Linear(FREQ_IN → 128)
  → ReLU
  → [1, FREQ_DIM=128]
```

This gives the network explicit access to spectral structure without relying on the time-domain branch to infer it.

### 3.3 Window Conditioning

The RMCLS loss is built around three key indices: `peak_idx`, `pre_peak_idx`, `er_end_idx`. Without this information the network must infer the direct-sound location from the waveform alone.

A conditioning vector passes these indices (normalized by `L`) directly as input:
```python
window_cond = [peak_idx/L, pre_peak_idx/L, er_end_idx/L]  # [3]
```

This aligns the network's forward pass with the loss structure — the network knows where the direct sound is before computing filter taps.

### Full Architecture

```
Input: h [L],  cond [3]

  TIME BRANCH                  FREQ BRANCH           CONDITIONING
  h → Conv1d(1→64, k=1)        h → rfft → log|·|     cond [3]
    → 20× DilatedResBlock(64)    → Linear(FREQ_IN→128)
    → mean(dim=-1)               → ReLU
    → [1, 64]                    → [1, 128]

  Concatenate: [1, 64 + 128 + 3] = [1, 195]
    → FC(195 → 256) → ReLU
    → FC(256 → half=513)
    → symmetrize()
  Output: g [T=1025]
```

**Parameter count**: ~3.5M (dominated by `freq_fc: Linear(10473 → 128)`)

---

## 4. Optimizer Journey

### 4.1 Original Paper Optimizer (Problem)

The paper uses Adam with inverse-time decay:
```
lr_t = lr / (1 + decay * t)
decay = 3e-8,  lr = 1e-3,  t_max = 200,000
```

At `t = 200,000`:
```
lr_200k = 1e-3 / (1 + 3e-8 × 200,000) = 1e-3 / 1.006 ≈ 0.994e-3
```

**The decay is essentially zero** — the learning rate never meaningfully changes. Once the optimizer stalls in a flat region of the loss landscape it stays stalled.

Observed in training: loss plateaued at ~0.922 from iter ~33,000 onward, with the LR unchanged at `~1e-3` for all 200,000 iterations.

### 4.2 First Fix — ReduceLROnPlateau

Replaced the manual decay with `torch.optim.lr_scheduler.ReduceLROnPlateau`:
- Halves the LR (`factor=0.5`) when loss stops improving
- Intended LR sequence: `1e-3 → 5e-4 → 2.5e-4 → 1.25e-4 → ... → 1e-6`

**New problem**: initial `patience=3000` was too short. With `scheduler.step(lval)` (instantaneous noisy loss), false plateau triggers fired every ~3,500 iterations. The LR burned through four reductions in 11,000 iterations, reaching `3.13e-05` by iter 63,000.

Root cause: instantaneous loss fluctuates — e.g., `loss=0.9226` at iter 53,500 while `best=0.9225`. The scheduler saw these up-ticks as stagnation.

### 4.3 Second Fix — Track `best_loss` in Scheduler

Changed `scheduler.step(lval)` → `scheduler.step(best_loss)`.

`best_loss` is monotonically non-increasing, so the scheduler only triggers when the true best genuinely stops improving — not on noisy up-ticks in the current iteration's loss.

### 4.4 Final Optimizer Setup (cnn_improved_v2.py)

| Component | Choice | Reason |
|-----------|--------|--------|
| Optimizer | AdamW | Decoupled weight decay — L2 reg on weights directly, not via adaptive LR |
| weight_decay | 1e-4 | Prevents weight blow-up in deep dilated stack |
| LR schedule | ReduceLROnPlateau | Adapts to loss landscape automatically |
| patience | 8,000 | Long enough to distinguish genuine plateau from noise |
| factor | 0.5 | Halve LR at each reduction |
| min_lr | 1e-6 | Hard floor — stops reducing when further reduction is useless |
| Scheduler input | `best_loss` | Monotonic — no false triggers from noisy current loss |
| Gradient clipping | max_norm=1.0 | Prevents large gradient steps after LR reductions |

---

## 5. Delta Curriculum

### Problem

`delta` controls the late-reverb penalty weight:
```
lambda_vec[outside window] = sqrt(delta)
lambda_vec[inside window]  = 1.0
```

With `delta=0.1`, `sqrt(0.1) ≈ 0.316` — the tail (most of the 20,944 samples) dominates the gradient. The loss landscape is difficult: the network must suppress reverb over hundreds of milliseconds simultaneously with matching the direct sound peak.

Starting with a hard problem slows convergence. The network gets stuck in a flat region before it has learned the basic structure of the filter.

### Solution — Progressive Difficulty

Start with a soft tail penalty and gradually increase:

| Iteration | delta | lambda_tail | Character |
|-----------|-------|-------------|-----------|
| 0 | 0.01 | 0.10 | Nearly pMINT — focus only on early window |
| 100,000 | 0.05 | 0.22 | Moderate tail suppression |
| 200,000 | 0.10 | 0.32 | Full RMCLS target |

At each transition:
1. `lambda_vec` is updated **in-place** (picked up immediately by `rmcls_loss`)
2. `best_loss` is reset to `inf` (loss scale changes with delta)
3. Scheduler is reset (avoids immediate trigger from the loss jump)
4. Model weights are **not** reset — learning is cumulative

### Rationale

This is analogous to curriculum learning in other deep learning domains:
- Stage 1 teaches the network to align the direct sound (easy target)
- Stage 2 introduces light tail suppression (medium difficulty)
- Stage 3 enforces the full RMCLS objective (target)

Each stage builds on the previous, rather than requiring the network to solve the full problem from random initialization.

---

## 6. File Summary

| File | Loss | Architecture | Optimizer | Status |
|------|------|--------------|-----------|--------|
| `cnn.py` | Spectral (paper) | TinyCNN1x1 | Adam + inverse decay | Paper replication |
| `rmcls_cnn.py` | RMCLS | TinyCNN1x1 | Adam + inverse decay | Baseline (stuck at ~0.926) |
| `cnn_improved.py` | RMCLS | ImprovedCNN (dilated) | Adam + inverse decay | Plateaued at ~0.922 |
| `cnn_improved_v2.py` | RMCLS | ImprovedCNN (dilated) | AdamW + ReduceLROnPlateau + curriculum | **Current** |
| `cnn_multi_independent.py` | RMCLS (per source) | TinyCNN Multi | Adam | Multi-source Option A |
| `cnn_multi_joint.py` | RMCLS (joint) | TinyCNN Multi Joint | Adam | Multi-source Option B |

---

## 7. Key References

1. **Pepe et al. (2020)** — Architecture and spectral loss baseline.
   *Designing Audio Equalization Filters by Deep Neural Networks.*
   Applied Sciences, 10(7), 2483.

2. **Kodrasi (2015)** — RMCLS theoretical foundation.
   *Dereverberation and Noise Reduction Techniques based on Acoustic
   Multi-Channel Equalization.* PhD Thesis, Universität Oldenburg.
