# Low-Frequency Equalization — Limits and Strategies

## The Problem

Room modes below ~200 Hz produce deep resonant peaks and nulls in the RIR.
A short FIR equalizer filter cannot fully invert them. The result is the
characteristic oscillating pattern (±10–20 dB) visible in the equalized
magnitude response at LF.

---

## Why a Short FIR Cannot Invert Room Modes

### The Room as a Pole-Zero System

A room mode at frequency `f₀` with decay time `τ` corresponds to a **pole
near the unit circle** in the Z-domain. To cancel a pole, a FIR filter must
place a zero at exactly that location — and the closer the pole is to the unit
circle (= the longer it rings), the longer the FIR must be.

### Required Filter Length

For a mode with decay time `τ` (seconds), sampling rate `fs`, and target
cancellation accuracy `ε`:

```
T_required ≈ τ · fs · log(1/ε) / π
```

Example: a 200 ms room mode at 48 kHz, cancellation to 1%:

```
T_required ≈ 0.2 × 48000 × log(100) / π ≈ 14,000 taps
```

At T=1025 this is ~14× too short.

### What Happens When You Try Anyway

The optimizer partially compensates modal peaks and creates dips at neighboring
frequencies, and vice versa. The filter thrashes — gradient descent keeps
adjusting taps but the solution requires more degrees of freedom than available.
This wastes training capacity on an unsolvable problem.

---

## Real-Time Latency vs. Filter Length

The `symmetrize()` function enforces a **linear-phase symmetric FIR**, which
introduces a fixed group delay of `(T−1)/2` samples — unavoidable by design.

| T    | Group Delay   | Latency @48kHz | Frequency Resolution |
|------|---------------|----------------|----------------------|
| 512  | 256 samples   | ~5.3 ms        | ~94 Hz               |
| 1025 | 512 samples   | ~10.7 ms       | ~47 Hz               |
| 2048 | 1024 samples  | ~21.3 ms       | ~23 Hz               |
| 4096 | 2048 samples  | ~42.7 ms       | ~12 Hz               |

### Latency Tolerance by Use Case

| Use Case                         | Max Acceptable Latency | Max T     |
|----------------------------------|------------------------|-----------|
| Live instrument monitoring       | ~10 ms                 | ~1025     |
| Studio mixing / DAW insert       | ~20 ms                 | ~2048     |
| Room correction for playback     | ~40–50 ms              | ~4096     |
| Offline rendering                | unlimited              | unlimited |

**T=1025 was deliberately chosen** to sit at the real-time boundary (~10.7 ms),
which is standard practice for near-RT room EQ.

---

## Loss Function Strategies for Better LF Accuracy

Given T=1025 is fixed by the RT constraint, the loss function is the primary
lever for improving LF equalization.

### Strategy 1 — Stronger Log-Frequency Weighting (`1/f²`)

Replace the `1/f` term with `1/f²` to give LF bins dominant gradient weight:

```python
log_weight = 1.0 / (freqs + 1e-6) ** 2
```

The per-bin ratio between 50 Hz and 1 kHz becomes **400×** instead of 20×.

**Note on the notch weight `1/|H(f)|`:** At LF, `|H(f)|` has both large peaks
(modes) and deep nulls (between modes). The notch weight reduces emphasis at
peaks and amplifies at nulls — but the filter cannot fill deep nulls anyway.
Consider decoupling them for LF:

```python
lf_region    = freqs < 500
log_weight   = np.where(lf_region, 1/(freqs+1e-6)**2, 1/(freqs+1e-6))
notch_weight = np.where(lf_region, 1.0, 1/(H_mag_np + eps_notch))
combined     = log_weight * notch_weight
```

### Strategy 2 — Log-Magnitude Loss

The linear loss `(|H(f)G(f)| − 1)²` is asymmetric in dB:
- −15 dB error → `(0.18 − 1)² = 0.67`
- +15 dB error → `(5.6  − 1)² = 21.2`

Large positive dB errors dominate the gradient. Switching to log-magnitude
makes the loss **symmetric and uniform in dB**:

```python
def log_frequency_weighted_loss(h, g):
    H      = torch.fft.rfft(h, n=N_fft)
    G      = torch.fft.rfft(g, n=N_fft)
    mag    = torch.abs(H * G).clamp(min=1e-8)
    err_db = 20.0 * torch.log10(mag)   # target = 0 dB
    return w * torch.sum(freq_weight * err_db ** 2)
```

Use `.clamp(min=1e-8)` to avoid log(0) instability. Prefer starting from a
pre-trained checkpoint rather than random init.

### Strategy 3 — Explicit LF Sub-Band Penalty

Directly add a high-weight term for the LF band:

```python
lf_mask  = torch.tensor((freqs >= 20) & (freqs <= 300), dtype=torch.bool, device=device)
lf_boost = 200.0

def log_frequency_weighted_loss(h, g):
    H   = torch.fft.rfft(h, n=N_fft)
    G   = torch.fft.rfft(g, n=N_fft)
    mag = torch.abs(H * G)
    err = mag - 1.0
    base = w * torch.sum(freq_weight * err ** 2)
    lf   = lf_boost * torch.sum(err[lf_mask] ** 2)
    return base + lf
```

Simple and easy to tune (`lf_boost`, `lf_cutoff`). Best combined with the
log-magnitude loss to fix the asymmetry.

### Strategy 4 — Smoothed LF Target (Don't Fight the Physics)

Instead of targeting flat 0 dB at LF (unachievable with T=1025), target a
**1/3-octave smoothed version of H(f)**. The filter is only asked to correct
the broad spectral envelope — which it can achieve — not the individual modal
peaks and nulls — which it cannot.

```python
from scipy.ndimage import uniform_filter1d

H_mag_np     = np.abs(np.fft.rfft(h_np, n=N_fft))
H_mag_smooth = uniform_filter1d(H_mag_np, size=30)          # ~1/3 oct at LF
H_target     = H_mag_smooth / (H_mag_smooth.max() + 1e-12)
H_target_t   = torch.tensor(H_target, dtype=torch.float32, device=device)

lf_mask = torch.tensor((freqs >= 20) & (freqs <= 500), dtype=torch.bool, device=device)

def log_frequency_weighted_loss(h, g):
    H   = torch.fft.rfft(h, n=N_fft)
    G   = torch.fft.rfft(g, n=N_fft)
    mag = torch.abs(H * G)
    target = torch.where(lf_mask, H_target_t, torch.ones_like(H_target_t))
    err    = mag - target
    return w * torch.sum(freq_weight * err ** 2)
```

**Trade-off:** The equalized LF response will follow the room's average
spectral shape rather than being flat. This is perceptually valid and
analogous to what pMINT does (windowing the RIR to only model what the filter
can achieve).

---

## Recommended Combination

| Priority | Strategy                        | Why                                      |
|----------|---------------------------------|------------------------------------------|
| 1st      | Log-magnitude loss              | Fixes asymmetric dB scaling — always use |
| 2nd      | `1/f²` + decouple notch at LF   | Gives LF bins gradient dominance         |
| 3rd      | Smoothed LF target              | Stops wasting capacity on the impossible |
| Optional | Explicit LF sub-band penalty    | Fast empirical boost, easy to tune       |

---

## Connection to pMINT

The smoothed-target approach is conceptually equivalent to **pMINT** (Partial
MINT): instead of inverting the full 436 ms RIR, pMINT windows the RIR to
isolate direct sound + early reflections, then designs the inverse filter
against that windowed version. Both approaches accept that a short filter
cannot invert the full room decay and design accordingly.

See `matlab/PMINT_EIR.m` for the reference implementation.
