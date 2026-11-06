"""
DriftCNN — AdamW + ReduceLROnPlateau — RMCLS Loss — Fixed Delta
================================================================
DriftCNN architecture with dilated residual convolutions, dual-path
(time + frequency), and window conditioning.

Optimizer : AdamW  (lr=1e-3, weight_decay=1e-4)
Scheduler : ReduceLROnPlateau(factor=0.5, patience=8000, min_lr=1e-6)
Loss      : RMCLS — time-domain weighted least-squares
            J(g) = || lambda_vec * (H*g - D) ||^2
Delta     : Fixed at 0.1
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
from scipy.io import wavfile
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ============================================================
# CELL 2 — Experiment Config
# ============================================================

EXP_TAG        = "cnn_d_rmcls_adamw_plateau"
checkpoint_dir = f"/content/checkpoints/{EXP_TAG}"
os.makedirs(checkpoint_dir, exist_ok=True)


# ============================================================
# CELL 3 — Load RIR
# ============================================================

rir_path = "/content/RIR.wav"

rate, data = wavfile.read(rir_path)
fs   = rate
data = data.astype(np.float32)
if data.ndim > 1:
    data = data[:, 0]
data = data / (np.max(np.abs(data)) + 1e-12)

L    = len(data)
h_np = data
h    = torch.tensor(data, dtype=torch.float32, device=device)

print(f"RIR loaded: fs={fs} Hz, L={L} samples, duration={L/fs:.3f} s")


# ============================================================
# CELL 4 — Filter Hyperparameters
# ============================================================

T    = 1025
half = (T + 1) // 2

print(f"Filter length T={T}, half={half}")

def symmetrize(half_taps: torch.Tensor) -> torch.Tensor:
    return torch.cat([half_taps.flip(0)[:-1], half_taps])


# ============================================================
# CELL 5 — RMCLS Setup + Loss
# ============================================================

delta               = 0.1
onset_thresh        = 0.01
early_reflection_ms = 50

peak_idx     = int(np.argmax(np.abs(h_np)))
pre_peak     = np.where(np.abs(h_np[:peak_idx]) > onset_thresh * np.abs(h_np).max())[0]
pre_peak_idx = int(pre_peak[0]) if len(pre_peak) > 0 else peak_idx
er_end_idx   = min(peak_idx + round(early_reflection_ms / 1000 * fs), L - 1)

print(f"Peak:     sample {peak_idx}  ({peak_idx/fs*1000:.2f} ms)")
print(f"Pre-peak: sample {pre_peak_idx}  ({pre_peak_idx/fs*1000:.2f} ms)")
print(f"ER end:   sample {er_end_idx}  ({er_end_idx/fs*1000:.2f} ms)")
print(f"delta:    {delta}")

D = torch.zeros(L, device=device)
D[peak_idx] = 1.0

lambda_vec = torch.full((L,), math.sqrt(delta), device=device)
lambda_vec[pre_peak_idx : er_end_idx + 1] = 1.0

window_cond = torch.tensor(
    [peak_idx / L, pre_peak_idx / L, er_end_idx / L],
    dtype=torch.float32, device=device
)


def rmcls_loss(h: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    x = h.view(1, 1, -1)
    k = g.view(1, 1, -1)
    y = F.conv1d(x, k, padding=k.shape[-1] - 1).view(-1)[:L]
    residual = y - D
    return torch.sum((lambda_vec * residual) ** 2)


# ============================================================
# CELL 6 — Building Blocks
# ============================================================

class DilatedResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel,
            dilation=dilation,
            padding=dilation * (kernel - 1) // 2
        )
        self.norm = nn.GroupNorm(num_groups=4, num_channels=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.relu(self.norm(self.conv(x)))


# ============================================================
# CELL 7 — Architecture
# ============================================================

TIME_DIM = 64
FREQ_DIM = 128

DILATIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
             1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

FREQ_IN = L // 2 + 1


class DriftCNN(nn.Module):
    def __init__(self, L: int, half: int):
        super().__init__()
        self.L    = L
        self.half = half

        self.input_proj = nn.Conv1d(1, TIME_DIM, kernel_size=1)
        self.dilated_blocks = nn.Sequential(*[
            DilatedResBlock(TIME_DIM, d) for d in DILATIONS
        ])

        self.freq_fc = nn.Linear(FREQ_IN, FREQ_DIM)

        agg_in = TIME_DIM + FREQ_DIM + 3
        self.fc1 = nn.Linear(agg_in, 256)
        self.fc2 = nn.Linear(256, half)

    def forward(self,
                h: torch.Tensor,
                cond: torch.Tensor
                ) -> torch.Tensor:
        x = h.view(1, 1, -1)
        x = F.relu(self.input_proj(x))
        x = self.dilated_blocks(x)
        x = x.mean(dim=-1)

        H_fft    = torch.fft.rfft(h, n=L)
        H_logmag = torch.log(torch.abs(H_fft) + 1e-8)
        f = F.relu(self.freq_fc(H_logmag.unsqueeze(0)))

        c = cond.unsqueeze(0)

        z = torch.cat([x, f, c], dim=-1)
        z = F.relu(self.fc1(z))
        half_taps = self.fc2(z).squeeze(0)
        return half_taps


model = DriftCNN(L=L, half=half).to(device)
print("Parameters:", sum(p.numel() for p in model.parameters()))


# ============================================================
# CELL 8 — Optimizer & Scheduler
# ============================================================

lr           = 1e-3
weight_decay = 1e-4
iters        = 200_000
log_every    = 500
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
# CELL 9 — Training
# ============================================================

model.train()
for t in range(1, iters + 1):
    opt.zero_grad()
    half_taps = model(h, window_cond)
    g         = symmetrize(half_taps)
    loss      = rmcls_loss(h, g)
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
# CELL 10 — Plot Results
# ============================================================

N_fft   = L + T - 1
freqs   = np.fft.rfftfreq(N_fft, d=1.0/fs)
freq_lo = 20
freq_hi = fs // 2

def smooth_log(freqs, mag_db_vals, frac=1/6):
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
# CELL 11 — Save Artifacts
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
    print(f"MSE (spectral): {mse:.6e}")
    print(f"Saved: {tag}_W.wav, {tag}_EIR.wav  →  {checkpoint_dir}/")

save_results(best_g, tag="best")
