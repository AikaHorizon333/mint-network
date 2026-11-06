"""
TinyCNN1x1 — Single-Source RIR Equalization with RMCLS Loss  [rmcls_cnn.py]
=============================================================================
Architecture: 3-layer Conv1d feature extractor + 2 FC layers.
Loss:         RMCLS — time-domain weighted least-squares.
              J(g) = || lambda_vec * (H*g - D) ||^2
              lambda_vec = 1  inside [pre_peak .. er_end]  (direct + early reflections)
              lambda_vec = sqrt(delta)  outside             (late reverb relaxation)

Limitation: receptive field ~25 samples out of 20944 (RIR length).
            Use cnn_improved.py for better results (dilated convolutions).

ORIGINAL TRAINING METHOD

"""


# ============================================================
# CELL 1 — Imports
# ============================================================

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# soundfile reads WAV/FLAC/OGG; install once per Colab session:
# !pip install -q soundfile
import soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# ============================================================
# CELL 2 — Load RIR
# Upload your WAV file to Colab first (Files panel → Upload),
# then update rir_path below.
# ============================================================

rir_path = "/content/RIR.wav"

data, fs = sf.read(rir_path, dtype="float32")
assert fs == 48000, f"Expected 48 kHz, got {fs}"

if data.ndim > 1:
    data = data[:, 0]          # take first channel if stereo/multi-channel

# Peak normalization: scale so max(|h|) = 1
h_np = data / np.max(np.abs(data))

L = h_np.shape[0]
print(f"RIR loaded: fs={fs} Hz, L={L} samples, duration={L/fs:.3f} s")

# Move to GPU/CPU tensor — shape [L]
h = torch.from_numpy(h_np).to(device)


# ============================================================
# CELL 3 — Filter Hyperparameters
# T must be odd so the symmetric FIR has a single center tap.
# half = (T+1)//2 independent taps; the full filter is 2*half-1 = T.
# ============================================================

T    = 1025          # inverse filter length (odd)
half = (T + 1) // 2  # number of independent (half-side) taps = 513

print(f"Filter length T={T},  half={half}")

def symmetrize(half_taps: torch.Tensor) -> torch.Tensor:
    """
    Build an odd-length symmetric FIR from its first (half) taps.
    half_taps : [half]   — left side including the center tap
    returns g : [T]      — full symmetric filter  [left | mirrored_left]
    """
    left  = half_taps                    # [half]
    right = half_taps[:-1].flip(0)       # mirror excluding the center tap
    return torch.cat([left, right], dim=0)   # [T = 2*half - 1]


# ============================================================
# CELL 4 — RMCLS Setup
# Detects the direct-sound window once from the RIR, then builds:
#   D          — target impulse (1 at peak_idx, 0 elsewhere)
#   lambda_vec — per-sample weight for the loss
# ============================================================

# --- Tunable parameters ---
delta               = 0.1   # late-reverb relaxation factor
                             # 0 → pMINT-like (only early window matters)
                             # 1 → full MINT-like (every sample equally weighted)
onset_thresh        = 0.01  # fraction of peak amplitude to detect pre-peak onset
early_reflection_ms = 50    # ms after peak included in the early-reflection window

# --- Window detection (CPU, done once before training) ---
peak_idx = int(np.argmax(np.abs(h_np)))

# First sample before the peak that exceeds onset_thresh × peak
pre_peak     = np.where(np.abs(h_np[:peak_idx]) > onset_thresh * np.abs(h_np).max())[0]
pre_peak_idx = int(pre_peak[0]) if len(pre_peak) > 0 else peak_idx

# Last sample in the early-reflection window
er_end_idx = min(peak_idx + round(early_reflection_ms / 1000 * fs), L - 1)

print(f"Peak:     sample {peak_idx}  ({peak_idx/fs*1000:.2f} ms)")
print(f"Pre-peak: sample {pre_peak_idx}  ({pre_peak_idx/fs*1000:.2f} ms)")
print(f"ER end:   sample {er_end_idx}  ({er_end_idx/fs*1000:.2f} ms)")
print(f"delta:    {delta}")

# --- Target impulse D: 1 at peak, 0 elsewhere ---
D = torch.zeros(L, device=device)
D[peak_idx] = 1.0

# --- Row-weight vector lambda_vec ---
# Inside [pre_peak_idx .. er_end_idx]: weight = 1  (full penalty)
# Outside:                             weight = sqrt(delta)  (relaxed penalty)
lambda_vec = torch.full((L,), math.sqrt(delta), device=device)
lambda_vec[pre_peak_idx : er_end_idx + 1] = 1.0


# ============================================================
# CELL 5 — RMCLS Loss Function
# J(g) = || lambda_vec * (H*g − D) ||^2
# H*g is computed via F.conv1d (linear convolution, truncated to L).
# ============================================================

def rmcls_loss(h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """
    h : [L]   — room impulse response
    g : [T]   — inverse filter
    Returns scalar loss.
    """
    x = h.view(1, 1, -1)                                    # [1, 1, L]
    k = g.view(1, 1, -1)                                    # [1, 1, T]
    # Full linear convolution length = L + T - 1; keep first L samples
    y        = F.conv1d(x, k, padding=k.shape[-1] - 1).view(-1)[:L]
    residual = y - D                                        # [L]
    return torch.sum((lambda_vec * residual) ** 2)          # scalar


# ============================================================
# CELL 6 — Model Architecture (TinyCNN1x1)
# 3 × Conv1d(kernel=9) → AdaptiveAvgPool → 2 × Linear → half taps
# Receptive field: ~25 samples (small relative to L=20944).
# For a larger receptive field use cnn_improved.py instead.
# ============================================================

class TinyCNN1x1(nn.Module):
    def __init__(self, L: int, half: int):
        super().__init__()
        # Feature extraction: 1-channel input (mono RIR), progressively wider channels
        self.conv1 = nn.Conv1d(1,  16, kernel_size=9, padding=4)   # [1, 16, L]
        self.conv2 = nn.Conv1d(16, 32, kernel_size=9, padding=4)   # [1, 32, L]
        self.conv3 = nn.Conv1d(32, 16, kernel_size=9, padding=4)   # [1, 16, L]

        # Compress time axis to fixed size 256 regardless of L
        self.pool = nn.AdaptiveAvgPool1d(256)                       # [1, 16, 256]

        # Regression head: flatten → 256 hidden units → half taps
        self.fc1 = nn.Linear(16 * 256, 256)
        self.fc2 = nn.Linear(256, half)

    def forward(self, h_1d: torch.Tensor) -> torch.Tensor:
        """
        h_1d : [L]      — mono RIR
        returns          — half_taps [half]
        """
        x = h_1d.view(1, 1, -1)       # add batch + channel dims  [1, 1, L]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)               # [1, 16, 256]
        x = x.flatten(1)               # [1, 16*256 = 4096]
        x = F.relu(self.fc1(x))        # [1, 256]
        half_taps = self.fc2(x).squeeze(0)   # [half]
        return half_taps

model = TinyCNN1x1(L=L, half=half).to(device)
print("Parameters:", sum(p.numel() for p in model.parameters()))


# ============================================================
# CELL 7 — Training
# Adam optimizer with inverse-time learning-rate decay.
# Best filter (lowest loss) is saved to disk after every improvement.
# Full checkpoints (model + optimizer state) saved every 10 000 iters
# so training can be resumed if the Colab session disconnects.
# ============================================================

lr         = 1e-3     # initial learning rate
adam_decay = 3e-8     # inverse-time decay coefficient
iters      = 200_000  # total training iterations
log_every  = 200      # print a log line every N iterations

checkpoint_dir       = "/content/checkpoints"
checkpoint_frequency = 10_000   # full checkpoint every N iterations

os.makedirs(checkpoint_dir, exist_ok=True)

opt = torch.optim.Adam(model.parameters(), lr=lr)

def set_lr(optimizer, base_lr, decay, t):
    """Inverse-time decay: lr_t = base_lr / (1 + decay * t)."""
    lr_t = base_lr / (1.0 + decay * t)
    for pg in optimizer.param_groups:
        pg["lr"] = lr_t
    return lr_t

best_loss = float("inf")
best_g    = None

model.train()
for t in range(1, iters + 1):
    lr_t = set_lr(opt, lr, adam_decay, t)

    opt.zero_grad()
    half_taps = model(h)              # forward pass → [half]
    g         = symmetrize(half_taps) # build full symmetric filter [T]
    loss      = rmcls_loss(h, g)      # RMCLS loss (scalar)
    loss.backward()
    opt.step()

    lval = float(loss.detach().cpu())

    # Track and save the best filter found so far
    if lval < best_loss:
        best_loss = lval
        best_g    = g.detach().clone()
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
        torch.save(best_g,             os.path.join(checkpoint_dir, "best_g.pt"))

    if t % log_every == 0:
        print(f"iter={t:>7d}  lr={lr_t:.6e}  loss={lval:.6e}  best={best_loss:.6e}")

    # Full checkpoint for resuming training
    if t % checkpoint_frequency == 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{t:07d}.pth")
        torch.save({
            'iteration':            t,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'best_loss':            best_loss,
            'best_g':               best_g.cpu() if best_g is not None else None,
        }, checkpoint_path)
        print(f"Checkpoint saved at iter {t}: {checkpoint_path}")

print("Done. best_loss:", best_loss)


# ============================================================
# CELL 8 — Inspect Best Filter
# Run this after training to verify whether the filter is
# meaningful (large norm) or trivial (near-zero output).
# ============================================================

import matplotlib.pyplot as plt

g_np  = best_g.cpu().numpy()
eir   = np.convolve(h_np, g_np)[:L]          # equalized IR

print(f"Filter norm:  {np.linalg.norm(g_np):.4f}")
print(f"Filter max:   {np.abs(g_np).max():.4f}")
print(f"EIR peak:     {np.abs(eir).max():.4f}  at sample {np.argmax(np.abs(eir))}")

fig, axes = plt.subplots(3, 1, figsize=(12, 8))

axes[0].plot(g_np)
axes[0].set_title("Best Inverse Filter (g)")
axes[0].set_xlabel("Sample"); axes[0].set_ylabel("Amplitude"); axes[0].grid(True)

axes[1].plot(eir)
axes[1].axvline(peak_idx, color="r", linestyle="--", label=f"target peak ({peak_idx})")
axes[1].set_title("Equalized IR  (H * g)")
axes[1].set_xlabel("Sample"); axes[1].set_ylabel("Amplitude")
axes[1].legend(); axes[1].grid(True)

axes[2].plot(20 * np.log10(np.abs(np.fft.rfft(eir)) + 1e-12))
axes[2].set_title("Equalized IR — Magnitude Spectrum (dB)")
axes[2].set_xlabel("Frequency bin"); axes[2].set_ylabel("dB"); axes[2].grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# CELL 9 — Save Results as WAV
# Saves the inverse filter (W) and the equalized IR (EIR)
# to the checkpoint directory as normalized float32 WAV files.
# ============================================================

from scipy.io import wavfile

def save_results(g: torch.Tensor, tag: str = "best"):
    g_np  = g.cpu().numpy()
    eir   = np.convolve(h_np, g_np)[:L]

    # Normalize to [-1, 1] for WAV
    g_norm   = g_np / (np.max(np.abs(g_np))  + 1e-12)
    eir_norm = eir  / (np.max(np.abs(eir))   + 1e-12)

    wavfile.write(os.path.join(checkpoint_dir, f"{tag}_W.wav"),
                  fs, g_norm.astype(np.float32))
    wavfile.write(os.path.join(checkpoint_dir, f"{tag}_EIR.wav"),
                  fs, eir_norm.astype(np.float32))

    err = np.linalg.norm(eir - D.cpu().numpy())
    print(f"Verification error (||EIR - D||): {err:.6e}")
    print(f"Saved: {tag}_W.wav, {tag}_EIR.wav  →  {checkpoint_dir}/")

save_results(best_g, tag="best")
