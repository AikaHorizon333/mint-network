"""
TinyCNN1x1 — Adam + CosineAnnealingLR
=======================================
Same architecture and loss as the baseline (Pepe et al. 2020).
Uses monotonic cosine decay (no restarts).

Optimizer : Adam  (lr=1e-3)
Scheduler : CosineAnnealingLR(T_max=200k, eta_min=1e-6)
            Smooth monotonic decay — LR stays high for first ~half,
            then decays for fine-tuning.

Loss: Spectral Euclidean distance
      J = w * sum( | |H(f)*G(f)| - 1 |^2 ),  w=100
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

EXP_TAG        = "cnn_t_adam_coslr"
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
# CELL 5 — Spectral Loss
# ============================================================

w        = 100.0
freq_lo  = 20
freq_hi  = fs // 2

N_fft   = L + T - 1
freqs   = np.fft.rfftfreq(N_fft, d=1.0/fs)
freq_mask = torch.tensor(
    (freqs >= freq_lo) & (freqs <= freq_hi),
    dtype=torch.bool, device=device
)

print(f"Loss freq range: {freq_lo}–{freq_hi} Hz  ({freq_mask.sum().item()} bins / {len(freqs)} total)")
print(f"Loss scale w={w}")

def spectral_euclidean_loss(h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    H   = torch.fft.rfft(h, n=N_fft)
    G   = torch.fft.rfft(g, n=N_fft)
    mag = torch.abs(H * G)
    err = mag[freq_mask] - 1.0
    return w * torch.sum(err ** 2)


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
# Adam + CosineAnnealingLR: monotonic cosine decay over full run.
# ============================================================

lr    = 1e-3
iters = 200_000
log_every = 200
checkpoint_frequency = 10_000

opt = torch.optim.Adam(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt,
    T_max   = iters,
    eta_min = 1e-6,
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
    loss      = spectral_euclidean_loss(h, g)
    loss.backward()
    opt.step()
    scheduler.step()

    lval   = float(loss.detach().cpu())
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
