"""
TinyCNN1x1 — AdamW + ReduceLROnPlateau + LFWLF-v2
===================================================
Same architecture as the baseline (Pepe et al. 2020).
Uses AdamW with adaptive plateau-based LR reduction.

Optimizer : AdamW  (lr=1e-3, weight_decay=1e-4)
Scheduler : ReduceLROnPlateau(factor=0.5, patience=8000, min_lr=1e-6)

Loss: LFWLF-v2 — Log-Magnitude Squared + Improved LF Weighting
      Weighting : 1/f² below 500 Hz (no notch), 1/f * 1/|H(f)| above
      Error     : (20 * log10(|H(f)*G(f)|))²  — symmetric in dB, target = 0 dB
"""


# ============================================================
# CELL 1 — Imports
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.io import wavfile

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# ============================================================
# CELL 2 — Experiment Config
# ============================================================

EXP_TAG        = "cnn_t_lfwlf_v2_adamw_plateau"
checkpoint_dir = f"/content/checkpoints/{EXP_TAG}"
os.makedirs(checkpoint_dir, exist_ok=True)


# ============================================================
# CELL 3 — Load RIR
# ============================================================

rir_path = "/content/RIR.wav"

data, fs = sf.read(rir_path, dtype="float32")
assert fs == 48000, f"Expected 48 kHz, got {fs}"

if data.ndim > 1:
    data = data[:, 0]

h_np = data / np.max(np.abs(data))
L    = h_np.shape[0]
print(f"RIR loaded: fs={fs} Hz, L={L} samples, duration={L/fs:.3f} s")

h = torch.from_numpy(h_np).to(device)


# ============================================================
# CELL 4 — Filter Hyperparameters
# ============================================================

T    = 1025
half = (T + 1) // 2

print(f"Filter length T={T},  half={half}")

def symmetrize(half_taps: torch.Tensor) -> torch.Tensor:
    left  = half_taps
    right = half_taps[:-1].flip(0)
    return torch.cat([left, right], dim=0)


# ============================================================
# CELL 5 — LFWLF-v2: Log-Magnitude Loss + Improved LF Weighting
#
#   Weighting (two regimes):
#     f < lf_split : w_f = 1/f²          (no notch — LF nulls are unachievable)
#     f >= lf_split: w_f = 1/f * 1/|H(f)| (original notch weight for HF)
#   Both regimes normalized jointly so weights sum to n_active.
#
#   Loss: J = w * sum_f [ w_f * (20*log10(|H(f)*G(f)|))² ]
#   Log-magnitude error is symmetric in dB: +15 dB and -15 dB cost equally.
# ============================================================

w        = 1.0
freq_lo  = 20
freq_hi  = fs // 2
lf_split = 500   # Hz — boundary between LF and HF weighting regimes

N_fft   = L + T - 1
freqs   = np.fft.rfftfreq(N_fft, d=1.0/fs)

freq_mask = torch.tensor(
    (freqs >= freq_lo) & (freqs <= freq_hi),
    dtype=torch.bool, device=device
)

H_mag_np  = np.abs(np.fft.rfft(h_np, n=N_fft))
lf_region = freqs < lf_split

# 1/f² for LF, 1/f for HF
log_weight = np.where(lf_region,
    1.0 / (freqs + 1e-6) ** 2,
    1.0 / (freqs + 1e-6))

# Notch weight only for HF — applying it at LF penalises impossible-to-fix nulls
eps_notch    = 1e-3 * H_mag_np.max()
notch_weight = np.where(lf_region,
    1.0,
    1.0 / (H_mag_np + eps_notch))

combined = log_weight * notch_weight
combined[~((freqs >= freq_lo) & (freqs <= freq_hi))] = 0.0

n_active = freq_mask.sum().item()
combined = combined / (combined.sum() + 1e-12) * n_active

freq_weight = torch.tensor(combined, dtype=torch.float32, device=device)

print(f"Loss freq range : {freq_lo}–{freq_hi} Hz  ({n_active} active bins)")
print(f"LF split        : {lf_split} Hz  |  1/f² below, 1/f*notch above")
print(f"Notch eps       : {eps_notch:.4e}")

def log_frequency_weighted_loss(h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    H      = torch.fft.rfft(h, n=N_fft)
    G      = torch.fft.rfft(g, n=N_fft)
    mag    = torch.abs(H * G).clamp(min=1e-8)
    err_db = 20.0 * torch.log10(mag)   # target = 0 dB
    return w * torch.sum(freq_weight * err_db ** 2)


# ============================================================
# CELL 6 — Model Architecture (TinyCNN1x1)
# ============================================================

class TinyCNN1x1(nn.Module):
    def __init__(self, L: int, half: int):
        super().__init__()
        self.conv1 = nn.Conv1d(1,  16, kernel_size=9, padding=4)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=9, padding=4)
        self.conv3 = nn.Conv1d(32, 16, kernel_size=9, padding=4)
        self.pool  = nn.AdaptiveAvgPool1d(256)
        self.fc1   = nn.Linear(16 * 256, 256)
        self.fc2   = nn.Linear(256, half)

    def forward(self, h_1d: torch.Tensor) -> torch.Tensor:
        x = h_1d.view(1, 1, -1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(0)

model = TinyCNN1x1(L=L, half=half).to(device)
print("Parameters:", sum(p.numel() for p in model.parameters()))


# ============================================================
# CELL 7 — Optimizer & Scheduler
# AdamW + ReduceLROnPlateau: halves LR when loss plateaus.
# ============================================================

lr           = 1e-3
weight_decay = 1e-4
iters        = 200_000
log_every    = 200
checkpoint_frequency = 10_000

opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt,
    mode     = "min",
    factor   = 0.5,
    patience = 8000,
    min_lr   = 1e-6,
)

best_loss = float("inf")
best_g    = None


# ============================================================
# CELL 8 — Training
# ============================================================

model.train()
for t in range(1, iters + 1):
    opt.zero_grad()
    half_taps = model(h)
    g         = symmetrize(half_taps)
    loss      = log_frequency_weighted_loss(h, g)
    loss.backward()
    opt.step()

    lval   = float(loss.detach().cpu())
    scheduler.step(lval)
    cur_lr = opt.param_groups[0]["lr"]

    if lval < best_loss:
        best_loss = lval
        best_g    = g.detach().clone()
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
        torch.save(best_g,             os.path.join(checkpoint_dir, "best_g.pt"))

    if t % log_every == 0:
        print(f"iter={t:>7d}  lr={cur_lr:.6e}  loss={lval:.6e}  best={best_loss:.6e}")

    if t % checkpoint_frequency == 0:
        torch.save({
            'iteration':            t,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss':            best_loss,
            'best_g':               best_g.cpu() if best_g is not None else None,
        }, os.path.join(checkpoint_dir, f"checkpoint_{t:07d}.pth"))

print(f"Done. best_loss: {best_loss:.6e}")


# ============================================================
# CELL 9 — Plot Results
# ============================================================

def smooth_log(freqs, mag_db_vals, frac=1/6):
    """Fractional-octave smoothing in log-frequency domain (visualization only)."""
    smoothed = np.zeros_like(mag_db_vals)
    for i, f0 in enumerate(freqs):
        if f0 <= 0:
            smoothed[i] = mag_db_vals[i]
            continue
        lo  = f0 / (2 ** (frac / 2))
        hi  = f0 * (2 ** (frac / 2))
        idx = (freqs >= lo) & (freqs <= hi)
        smoothed[i] = np.mean(mag_db_vals[idx])
    return smoothed

g_np = best_g.cpu().numpy()
eir  = np.convolve(h_np, g_np)[:L]

print(f"Filter norm: {np.linalg.norm(g_np):.4f}")
print(f"Filter max:  {np.abs(g_np).max():.4f}")
print(f"EIR peak:    {np.abs(eir).max():.4f}  at sample {np.argmax(np.abs(eir))}")

fig, axes = plt.subplots(2, 1, figsize=(12, 6))

axes[0].plot(g_np)
axes[0].set_title(f"Learned Inverse Filter (g) — {EXP_TAG}")
axes[0].set_xlabel("Sample"); axes[0].set_ylabel("Amplitude"); axes[0].grid(True)

axes[1].plot(eir)
axes[1].set_title("Equalized IR  (H * g)")
axes[1].set_xlabel("Sample"); axes[1].set_ylabel("Amplitude"); axes[1].grid(True)

plt.tight_layout()
plt.show()

H_mag_db = 20 * np.log10(np.abs(np.fft.rfft(h_np, n=N_fft)) + 1e-12)
Y_mag_db = 20 * np.log10(np.abs(np.fft.rfft(eir,  n=N_fft)) + 1e-12)

H_smooth = smooth_log(freqs, H_mag_db)
Y_smooth = smooth_log(freqs, Y_mag_db)

plt.figure(figsize=(10, 6))
plt.axvspan(freq_lo, freq_hi, color="gray", alpha=0.08, label="Optimization band")
plt.axvline(lf_split, color="green", linestyle="--", linewidth=1, alpha=0.6, label=f"LF split ({lf_split} Hz)")
plt.semilogx(freqs, H_mag_db, color="tab:blue",   alpha=0.15)
plt.semilogx(freqs, Y_mag_db, color="tab:orange",  alpha=0.15)
plt.semilogx(freqs, H_smooth, color="tab:blue",   linewidth=2, label="Original |H(f)|")
plt.semilogx(freqs, Y_smooth, color="tab:orange",  linewidth=2, label="Equalized |H(f)G(f)|")
plt.axhline(0, linestyle="--", linewidth=1, color="black", label="Target (0 dB)")
plt.xlim([20, fs / 2])
plt.ylim([-50, 30])
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title(f"Magnitude Response — {EXP_TAG}")
plt.grid(True, which="both", linestyle=":", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# CELL 10 — Save Artifacts
# ============================================================

def save_results(g: torch.Tensor, tag: str = "best"):
    g_np  = g.cpu().numpy()
    eir   = np.convolve(h_np, g_np)[:L]

    g_norm   = g_np / (np.max(np.abs(g_np))  + 1e-12)
    eir_norm = eir  / (np.max(np.abs(eir))   + 1e-12)

    wavfile.write(os.path.join(checkpoint_dir, f"{tag}_W.wav"),
                  fs, g_norm.astype(np.float32))
    wavfile.write(os.path.join(checkpoint_dir, f"{tag}_EIR.wav"),
                  fs, eir_norm.astype(np.float32))

    H_mag = np.abs(np.fft.rfft(h_np, n=N_fft))
    G_mag = np.abs(np.fft.rfft(g_np, n=N_fft))
    mask  = (freqs >= freq_lo) & (freqs <= freq_hi)
    mse   = np.mean((H_mag[mask] * G_mag[mask] - 1.0) ** 2)
    print(f"MSE (spectral, paper metric): {mse:.6e}")
    print(f"Saved: {tag}_W.wav, {tag}_EIR.wav  →  {checkpoint_dir}/")

save_results(best_g, tag="best")
