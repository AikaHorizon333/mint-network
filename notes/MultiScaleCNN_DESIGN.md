# MultiScaleCNN Design Document
## Spectral Inverse Filter Design via Dual-Path Multi-Scale Convolution

---

## 1. Context — What Came Before

MultiScaleCNN (v4) is the evolution of a series of architectures all targeting the same goal: given a room impulse response `h`, predict an FIR inverse filter `g` such that:

```
h * g ≈ δ   (unit impulse — flat frequency response)
```

The spectral loss used throughout is:

```
J = w * sum_f [ freq_weight(f) * ( |H(f) * G(f)| - 1 )² ]
```

The key bottlenecks identified in earlier architectures (`ImprovedCNN`, `cnn_improved_v2.py`):

| Problem | Root Cause | Consequence |
|---------|-----------|-------------|
| Limited spectral feature extraction | A single `Linear(10473, 128)` layer mapped the entire log-spectrum to a 128-dim vector | 10.4 M parameters just for the freq branch; brittle to unseen RIRs |
| Global average pooling loses detail | Single pool over all time steps weighted every sample equally | Time-domain features at different depths were collapsed identically |
| Shallow aggregation head | Only 2 FC layers (e.g. 195 → 256 → 513) | Insufficient nonlinear capacity for the feature-to-filter mapping |
| ReLU activations | Dead-neuron risk during 300k-iter training runs | Gradient flow inconsistency across deep dilated stacks |

MultiScaleCNN addresses all four with targeted architectural changes.

---

## 2. Architecture Overview

```
                    h  (RIR, L = 20,944 samples)
                   / \
                  /   \
      [Time Branch]   [Frequency Branch]
            |                  |
    Conv1d(1→64, k=1)    rfft(h) → log|·|
    + GELU                     |
            |            FreqEncoder
      Cycle 1                  |
    (10 DilatedResBlocks)   f [128]
            |
       AttentionPool → t1 [64]
            |
      Cycle 2
    (10 DilatedResBlocks)
            |
       AttentionPool → t2 [64]
            |
            +------- cat(t1, t2, f) -------+
                              |
                         [256 features]
                              |
                    FC(256 → 512) + GELU
                    FC(512 → 512) + GELU
                    FC(512 → out_dim)
                              |
                    g  [T = 1025 taps]
```

**Hyperparameters:**

| Symbol | Value | Description |
|--------|-------|-------------|
| `T` | 1025 | Inverse filter length (odd) |
| `L` | 20,944 | RIR length (samples at 48 kHz) |
| `TIME_DIM` | 64 | Channels in the time branch |
| `FREQ_DIM` | 128 | Output dimension of the frequency encoder |
| `DILATIONS_PER_CYCLE` | [1,2,4,8,16,32,64,128,256,512] | Dilation schedule per cycle |
| `SYMMETRIC` | False | Full T-tap output (no linear-phase constraint) |

---

## 3. Building Blocks

### 3.1 `DilatedResBlock`

```python
class DilatedResBlock(nn.Module):
    def __init__(self, channels, dilation, kernel=3):
        self.conv = nn.Conv1d(channels, channels,
                              kernel_size=kernel,
                              dilation=dilation,
                              padding=dilation * (kernel - 1) // 2)
        self.norm = nn.GroupNorm(num_groups=4, num_channels=channels)

    def forward(self, x):
        return x + F.gelu(self.norm(self.conv(x)))
```

A dilated 1D convolution with a residual connection. The key properties:

- **Same-length output**: padding = `dilation × (kernel - 1) // 2` keeps the time dimension unchanged, so no reshaping is needed between blocks.
- **GroupNorm (4 groups)**: preferred over BatchNorm because training uses a batch size of 1 (single RIR). BatchNorm would compute statistics over 1 sample, which is undefined.
- **GELU activation**: smoother gradient landscape than ReLU. No hard zero cutoff means no dead neurons during long training runs.
- **Residual connection**: `x + f(x)` prevents the gradient from vanishing across 20 stacked blocks.

**Receptive field** of a single block with kernel 3 and dilation `d`:

```
RF = d × (kernel - 1) + 1 = 2d + 1
```

| Dilation | RF of that block |
|----------|-----------------|
| 1 | 3 |
| 2 | 5 |
| 4 | 9 |
| 8 | 17 |
| 16 | 33 |
| 32 | 65 |
| 64 | 129 |
| 128 | 257 |
| 256 | 513 |
| 512 | 1025 |

Stacked cumulatively, one cycle reaches a combined receptive field of:

```
RF_cycle = 2 × sum([1,2,4,...,512]) + 1 = 2 × 1023 + 1 = 2047 samples  (~42 ms at 48 kHz)
```

Two cycles: **~4094 samples** (~85 ms) — sufficient to cover the direct sound and early reflections of the RIR.

---

### 3.2 `AttentionPool`

```python
class AttentionPool(nn.Module):
    def __init__(self, channels):
        self.query = nn.Linear(channels, 1)

    def forward(self, x):              # x: [B, C, T]
        logits  = self.query(x.transpose(1, 2))        # [B, T, 1]
        weights = F.softmax(logits, dim=1)              # [B, T, 1]
        return (x * weights.transpose(1, 2)).sum(dim=-1)  # [B, C]
```

Replaces global average pooling (GAP) used in earlier architectures.

**Why GAP is insufficient**: GAP averages all time steps with equal weight. For a 20,944-sample RIR, the direct sound peak (a sharp impulse at one location) is diluted by thousands of low-energy reverberant samples. The network loses information about where the dominant energy is.

**What AttentionPool does**: a single learned linear layer produces a scalar score per time step. After softmax normalization, these scores become a probability distribution over time, and the output is the **learned weighted average** of the feature vectors. The network can concentrate attention on the few time steps that carry the most useful information.

There are two pools in MultiScaleCNN:
- `attn1` taps after Cycle 1 → `t1`: shallower features, more local patterns.
- `attn2` taps after Cycle 2 → `t2`: deeper features, more global patterns.

This is the "multi-scale" aspect: two independent summaries of the temporal features at different processing depths.

---

### 3.3 `FreqEncoder`

```python
class FreqEncoder(nn.Module):
    def __init__(self, freq_dim=128):
        self.net = nn.Sequential(
            nn.Conv1d(1,  32,        kernel_size=9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(32, 64,        kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv1d(64, freq_dim,  kernel_size=5, stride=4, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, log_mag):    # log_mag: [B, FREQ_IN]
        x = log_mag.unsqueeze(1)   # [B, 1, FREQ_IN]
        return self.net(x).squeeze(-1)   # [B, freq_dim]
```

Encodes the log-magnitude spectrum into a 128-dim vector using a 3-layer strided CNN.

**Input**: `log(|rfft(h)| + ε)` — the log-magnitude spectrum. For `L = 20,944` samples, `rfft` produces `L//2 + 1 = 10,473` frequency bins.

**Strided convolutions**: each layer downsamples by 4×, so the spatial dimension is reduced:
```
10,473 → 2,619 → 655 → 164 → (AdaptiveAvgPool) → 1
```
The final `AdaptiveAvgPool1d(1)` collapses the remaining spatial dimension regardless of input size.

**Comparison to the previous approach**:

| | Old (`Linear`) | New (`FreqEncoder`) |
|--|--|--|
| Layer | `Linear(10473, 128)` | 3× `Conv1d` + pool |
| Parameters | 1,340,544 | ~46,000 |
| Reduction | — | **~30×** |
| Spectral pattern extraction | flat — each bin independent | hierarchical — local spectral patterns at multiple scales |

**Why log-magnitude**: the raw FFT magnitude spans several orders of magnitude. Taking the log compresses the dynamic range, making spectral nulls (near-zero magnitude) finite negative values rather than near-zero floats that gradient descent struggles with.

---

## 4. Improved Spectral Loss

The loss function is an enhancement of the paper's original spectral Euclidean distance, with two additional frequency weighting terms:

```
J = w * sum_f [ freq_weight(f) * ( |H(f) * G(f)| - 1 )² ]
```

`freq_weight(f)` is a combined weight applied per frequency bin:

```python
log_weight   = 1.0 / (freqs + 1e-6)          # emphasize low frequencies
notch_weight = 1.0 / (H_mag + eps_notch)      # emphasize spectral nulls

combined = log_weight * notch_weight
combined = combined / combined.sum() * n_active  # normalize to unit mean
```

| Weight term | Purpose |
|-------------|---------|
| `1 / (f + ε)` | Logarithmic frequency emphasis — lower frequencies are perceptually more important for room acoustics |
| `1 / (|H(f)| + ε)` | Notch emphasis — spectral nulls are hardest to equalize and most damaging; this forces the network to pay attention to them |
| Normalization | Keeps total loss magnitude scale-invariant to the number of active bins |

The active frequency range is `[20 Hz, Nyquist]`. Bins outside this range receive zero weight.

---

## 5. Full Network: `MultiScaleCNN`

### Forward Pass (annotated)

```python
def forward(self, h):            # h: [L]

    # --- Time branch ---
    x  = h.view(1, 1, -1)       # [1, 1, L]
    x  = F.gelu(self.input_proj(x))        # [1, 64, L]
    x  = self.cycle1(x)                    # [1, 64, L]  — 10 DilatedResBlocks
    t1 = self.attn1(x)                     # [1, 64]     — pool after cycle 1
    x  = self.cycle2(x)                    # [1, 64, L]  — 10 more DilatedResBlocks
    t2 = self.attn2(x)                     # [1, 64]     — pool after cycle 2

    # --- Frequency branch ---
    H_logmag = log(|rfft(h, n=L)| + 1e-8) # [L//2 + 1]
    f = self.freq_enc(H_logmag.unsqueeze(0))  # [1, 128]

    # --- Aggregate ---
    z = cat([t1, t2, f], dim=-1)  # [1, 64+64+128] = [1, 256]
    return self.head(z).squeeze(0)   # [1025]
```

**Note**: Cycle 2 receives the output of Cycle 1 as input — the two cycles are **sequential**, not parallel. Cycle 2 builds on the representations already computed by Cycle 1. The two attention pools extract summaries at two different levels of abstraction from the same processing pipeline.

### Aggregation Head

```
cat(t1, t2, f)  →  [256]
  → Linear(256, 512)  → GELU
  → Linear(512, 512)  → GELU
  → Linear(512, 1025)
```

Three fully-connected layers (was two in the previous version). The 256 → 512 expansion gives the head more capacity to learn the mapping from the compressed feature vector to 1025 filter coefficients.

### Near-Zero Output Initialization

```python
nn.init.uniform_(self.head[-1].weight, -1e-4, 1e-4)
nn.init.zeros_(self.head[-1].bias)
```

At the start of training, the output layer is initialized to produce ~zero taps. This means `g ≈ 0` and the initial EIR is near-zero, giving a well-defined, moderate starting loss rather than a random large-loss spike that can destabilize the optimizer in the first few iterations.

### Linear-Phase Option

```python
SYMMETRIC = False   # Full T taps (mixed phase allowed)
# SYMMETRIC = True  → output half=(T+1)//2=513 taps, then mirror to T=1025
```

Setting `SYMMETRIC = True` enforces a linear-phase filter by constructing the output as a palindrome:

```python
def symmetrize(half_taps):
    return torch.cat([half_taps, half_taps[:-1].flip(0)], dim=0)
```

This halves the output dimension (513 instead of 1025) and constrains the solution space. It was shown in the reference paper that linear phase emerges naturally even without this constraint, so `SYMMETRIC = False` is the default.

---

## 6. Training Setup

### Optimizer

| Component | Value | Reason |
|-----------|-------|--------|
| Optimizer | AdamW | Decoupled weight decay — L2 regularization applied directly to weights, not through the adaptive LR |
| `lr` | 1e-3 | Standard starting point for AdamW on audio tasks |
| `weight_decay` | 1e-4 | Prevents weight blow-up in 20-block dilated stack |
| Gradient clipping | `max_norm=1.0` | Prevents large steps after LR reductions |

### Learning Rate Schedule

**Linear warmup** for the first 1000 iterations:

```python
if t <= warmup_iters:
    warmup_lr = lr * t / warmup_iters   # ramps from lr/1000 → lr
```

This prevents large gradient steps when the model output is near-zero (near-zero output layer weights + non-zero input → near-zero activations → large relative gradients in early layers).

After warmup: **ReduceLROnPlateau** tracking `best_loss`:

```python
scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5,
                              patience=8000, min_lr=1e-6)
scheduler.step(best_loss)   # NOT lval (instantaneous loss)
```

Using `best_loss` (monotonically non-increasing) instead of the instantaneous loss prevents false plateau triggers from noisy single-iteration fluctuations. See `DriftCNN_DESING_PROCESS.md §4.3` for the full analysis of why instantaneous loss tracking causes premature LR reductions.

### Training Loop Summary

```
for t in 1 .. 300,000:
    1. Apply warmup LR (first 1000 iters) or step scheduler
    2. Forward pass: taps = model(h)
    3. Optionally symmetrize: g = symmetrize(taps) or taps
    4. Compute loss: J = improved_spectral_loss(h, g)
    5. Backward + clip gradients + optimizer step
    6. Track best_loss and save checkpoint if improved
    7. Log every 500 iters; save periodic checkpoint every 10,000 iters
```

---

## 7. Design Decision Summary

| Decision | Alternative Considered | Why This Choice |
|----------|----------------------|-----------------|
| Two attention pools (multi-scale) | Single pool after all 20 blocks | Preserves information at two processing depths; shallower pool retains local structure, deeper pool captures global context |
| Strided Conv1d FreqEncoder | `Linear(FREQ_IN, FREQ_DIM)` | 30× fewer parameters; hierarchical spectral pattern extraction vs. flat projection |
| Sequential cycles (not parallel) | Two independent parallel branches merged at the end | Cycle 2 benefits from Cycle 1 representations; parameters are shared efficiently |
| GELU throughout | ReLU | Smoother gradient flow; no dead-neuron risk over 300k iterations |
| Full T-tap output | Symmetrized half-tap | Allows mixed-phase solutions; linear phase emerges naturally |
| Near-zero output init | Random init | Stable low initial loss; avoids large gradient spikes at iteration 1 |
| GroupNorm (4 groups) | BatchNorm | Batch size is 1 (single RIR); BatchNorm is undefined for B=1 |
| `best_loss` in scheduler | Instantaneous `lval` | Monotonic signal prevents false plateau triggers from noisy per-iter losses |
| LR warmup (1000 iters) | Immediate full LR | Near-zero init + full LR → large relative gradients at iter 1; warmup smooths entry |

---

## 8. File Reference

| File | Role |
|------|------|
| `src/multiscale/cnn.py` | Full MultiScaleCNN v4 implementation (architecture + training) |
| `src/multiscale/SELF/Adam/cnn_ms_adam_setlr.py` | Variant with fixed LR schedule |
| `src/multiscale/SELF/Adam/cnn_ms_adam_coslr.py` | Variant with cosine LR annealing |
| `src/multiscale/SELF/Adam/cnn_ms_adam_setlr_2xt.py` | Variant with 2× TIME_DIM |
| `data/CNN1x1_EIR.wav` | EIR produced by the CNN-based method |
| `data/weights/` | Saved model checkpoints |

---

## 9. Key References

1. **Pepe et al. (2020)** — Original architecture and spectral loss.
   *Designing Audio Equalization Filters by Deep Neural Networks.*
   Applied Sciences, 10(7), 2483.

2. **van den Oord et al. (2016)** — Dilated causal convolutions for audio (WaveNet).
   *WaveNet: A Generative Model for Raw Audio.* arXiv:1609.03499.

3. **Kodrasi (2015)** — RMCLS theoretical foundation (context for prior architectures).
   *Dereverberation and Noise Reduction Techniques based on Acoustic
   Multi-Channel Equalization.* PhD Thesis, Universität Oldenburg.
